"""How the experiment reads and writes its own files: one JSON reader that answers None for a missing or damaged file, and
the callers that must not turn a damaged file into silent data loss. (Encodings: see the second half of this file.)"""

import json

import pytest
from synthetic_runs import write_old_column_run

from src.dashboard import app as app_module
from src.dashboard import data_access
from src.experiment import analyze_session as an
from src.experiment import session_files
from src.experiment import settings as settings_module
from test_analyze_session import ONSETS, _write_session

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
