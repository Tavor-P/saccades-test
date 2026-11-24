"""How the experiment reads and writes its own files: one JSON reader that answers None for a missing or damaged file, and
the callers that must not turn a damaged file into silent data loss. (Encodings: see the second half of this file.)"""

import json

import numpy as np
import pytest
from synthetic_runs import write_old_column_run
from test_analyze_session import ONSETS, _write_session

from src.dashboard import app as app_module
from src.dashboard import data_access
from src.experiment import analyze_session as an
from src.experiment import session_files
from src.experiment import settings as settings_module

# -- read_json ------------------------------------------------------------------------------------


def test_read_json_returns_the_object_or_none_for_missing_empty_corrupt_and_non_object_files(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert session_files.read_json(good) == {"a": 1}
    assert session_files.read_json(str(good)) == {"a": 1}  # a str path works too
    assert session_files.read_json(tmp_path / "missing.json") is None
    assert session_files.read_json(tmp_path) is None  # a folder is not a file
    for name, text in [("empty.json", ""), ("cut.json", '{"participant": {"id": "p1", "start'), ("list.json", "[1, 2]"), ("number.json", "3")]:
        (tmp_path / name).write_text(text, encoding="utf-8")
        assert session_files.read_json(tmp_path / name) is None, name
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe\x00\x81{")
    assert session_files.read_json(tmp_path / "binary.json") is None


def test_a_run_whose_session_json_is_damaged_keeps_its_key_from_the_folder_name(tmp_path):
    folder = write_old_column_run(tmp_path, 1700000500, [], ["trial_index"])
    key = session_files.find_session(tmp_path, folder.name).key
    (folder / "session.json").write_text("{not json", encoding="utf-8")
    assert session_files.find_session(tmp_path, folder.name).key == key  # falls back to the folder's own timestamp
    (folder / "session.json").write_text('{"run": {"unix_timestamp": "soon"}}', encoding="utf-8")
    assert session_files.find_session(tmp_path, folder.name).key == key
    (folder / "session.json").unlink()
    assert session_files.find_session(tmp_path, folder.name).key == key


# -- callers that must fail loudly instead of overwriting or guessing ------------------------------


def test_saving_participant_details_never_overwrites_a_damaged_session_json(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings_module, "DATA_DIR", tmp_path)
    folder = write_old_column_run(tmp_path, 1700000600, [], ["trial_index"])
    damaged = '{"participant": {"id": "p1"}, "settings": {"calibr'
    (folder / "session.json").write_text(damaged, encoding="utf-8")
    with pytest.raises(data_access.DamagedSessionFile, match="damaged"):
        data_access.save_participant_info("1700000600", "f", "31")
    assert (folder / "session.json").read_text(encoding="utf-8") == damaged

    app_module.app.config.update(TESTING=True)
    response = app_module.app.test_client().post("/sessions/1700000600", json={"gender": "f", "age": "31"})
    assert response.status_code == 409 and "damaged" in response.get_json()["error"]
    assert (folder / "session.json").read_text(encoding="utf-8") == damaged


def test_analysis_without_a_readable_session_json_asks_for_the_offset_instead_of_guessing(tmp_path):
    stamp = _write_session(tmp_path, ONSETS, [(0, "correct", 0.1)])
    meta_path = session_files.find_session(tmp_path, stamp).meta_path
    for damage in (lambda: meta_path.write_text("{oops", encoding="utf-8"), lambda: meta_path.unlink()):
        damage()
        with pytest.raises(ValueError, match="offset"):
            an.analyze(stamp, data_dir=tmp_path)
        run = an.analyze(stamp, data_dir=tmp_path, clock_offset_ms=250.0, out_dir=tmp_path / "out")  # an explicit offset needs no session.json
        assert run["summary"]["flashes"] == 1


def test_a_damaged_settings_file_reads_as_empty_and_is_replaced_on_the_next_save(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_module, "DATA_DIR", tmp_path)
    assert settings_module.load_last_startup() == {}  # no file
    (tmp_path / "settings.json").write_text("[1, 2", encoding="utf-8")
    assert settings_module.load_last_startup() == {}
    settings_module.save_last_startup({"participant_id": "p1"})
    assert settings_module.load_last_startup() == {"participant_id": "p1"}
    (tmp_path / "settings.json").write_text(json.dumps({"other": 1, "last_startup": {"a": 1}}), encoding="utf-8")
    settings_module.save_last_startup({"b": 2})  # a readable file keeps its other keys
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")) == {"other": 1, "last_startup": {"b": 2}}


# -- encodings ---------------------------------------------------------------------------------------
# Files are written as UTF-8; runs recorded by older versions used the Windows locale encoding (cp1252) and must still read.

NON_ASCII = "Ana María Müller – ñ"


def test_text_and_csv_readers_take_utf8_and_fall_back_to_cp1252(tmp_path):
    from src import textio

    utf8, cp1252, bom = tmp_path / "utf8.csv", tmp_path / "cp1252.csv", tmp_path / "bom.csv"
    text = f"name,value\r\n{NON_ASCII.replace('–', '-')},1\r\n"
    utf8.write_bytes(text.encode("utf-8"))
    cp1252.write_bytes(text.encode("cp1252"))
    bom.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    for path in (utf8, cp1252, bom):
        assert list(textio.read_csv(path)) == [{"name": NON_ASCII.replace("–", "-"), "value": "1"}], path.name
    assert textio.read_text(cp1252) == text
    assert list(textio.read_csv(utf8).fieldnames) == ["name", "value"]  # a BOM does not leak into the first header
    with pytest.raises(OSError):
        textio.read_csv(tmp_path / "missing.csv")
    (tmp_path / "newline.csv").write_bytes(b'a,b\r\n"line1\r\nline2",2\r\n')  # an embedded newline survives (csv does its own newline handling)
    assert list(textio.read_csv(tmp_path / "newline.csv"))[0]["a"] == "line1\r\nline2"


def test_read_json_reads_utf8_and_cp1252_files(tmp_path):
    (tmp_path / "u.json").write_bytes(json.dumps({"note": NON_ASCII}, ensure_ascii=False).encode("utf-8"))
    (tmp_path / "c.json").write_bytes(json.dumps({"note": NON_ASCII.replace("–", "-")}, ensure_ascii=False).encode("cp1252"))
    assert session_files.read_json(tmp_path / "u.json") == {"note": NON_ASCII}
    assert session_files.read_json(tmp_path / "c.json") == {"note": NON_ASCII.replace("–", "-")}


def test_an_old_cp1252_trials_file_and_gaze_trace_still_load(tmp_path):
    from src.experiment.logger import read_trial_rows
    from src.eye_tracking.gaze_trace import load_trace

    folder = write_old_column_run(tmp_path, 1700000700, [], ["trial_index"], participant="Ana María")
    (folder / "trials.csv").write_bytes("trial_index,phase,grating_shown,contrast,note\r\n0,saccade,True,0.05,año\r\n".encode("cp1252"))
    (folder / "gaze_trace.csv").write_bytes("t_s,device_unix_s,x_px,worn,comment\r\n0.0,100.0,500,1,señal\r\n".encode("cp1252"))
    assert list(read_trial_rows(folder / "trials.csv"))[0]["note"] == "año"
    (row,) = load_trace(folder / "gaze_trace.csv")
    assert row.device_unix_s == 100.0 and row.x_px == 500.0


def test_non_ascii_participant_ids_notes_and_group_labels_round_trip(tmp_path, monkeypatch):
    from src import textio
    from src.experiment import logger as logger_module
    from src.experiment import participants
    from src.experiment.logger import ResultLogger, read_trials

    # participant label file: group and note are free text
    participants.set_label("Ana María", "Grüppe Ñ", note=NON_ASCII, data_dir=tmp_path)
    assert participants.list_labels(tmp_path)["Ana_María"] == {"group": "Grüppe Ñ", "note": NON_ASCII}
    assert participants.participants_in_group("grüppe ñ", tmp_path) == ["Ana_María"]

    # a run recorded for such a participant: folder name, session.json, trials.csv and the analysis outputs
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger("Ana María", test_mode=True)
    log.close()
    assert log.folder.parent.name == "Ana_María"
    meta = session_files.read_json(log.folder / "session.json")
    assert meta["participant"]["id"] == "Ana María"
    assert read_trials(log.folder / "trials.csv") == []
    assert session_files.find_session(tmp_path, str(log._timestamp)).participant == "Ana_María"

    placed = [an.Placed(an.Flash(0, 1000.0, 0.06, "above", "correct"), 10.0, False)]
    gates = an.classify_flashes(placed, [], np.array([]), np.array([]), 0.0, [None])

    flashes = an.write_flash_rows(tmp_path / "flashes.csv", placed, gates, extra={"run": ["Ana_María/2026-11-24_10-00-00"]})
    assert list(textio.read_csv(flashes))[0]["run"] == "Ana_María/2026-11-24_10-00-00"
    assert "Ana_María".encode() in flashes.read_bytes()  # written as UTF-8 whatever the Windows locale is
