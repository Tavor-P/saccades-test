"""The on-disk layout: data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/ for new runs, old loose results_<ts>.* files still readable."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from include.experiment.types import Side, Target, TrialResult
from src.dashboard import data_access
from src.experiment.logger import read_trials
from src.experiment import analyze_session as an
from src.experiment import logger as logger_module
from src.experiment import session_files
from src.experiment.logger import ResultLogger
from test_analyze_session import CLOCK_OFFSET_S, ONSETS, _trace, _write_session

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "old_layout"


# -- naming ---------------------------------------------------------------------------------------


def test_run_folder_naming_safe_ids_local_time_test_suffix_and_same_second_runs(tmp_path, monkeypatch):
    import time

    for raw, expected in [
        ("p01", "p01"), ("", "anonymous"), ("   ", "anonymous"), (None, "anonymous"), ("Ana María", "Ana_María"),
        ('a/b\\c:d*e?f"g<h>i|j', "a_b_c_d_e_f_g_h_i_j"), ("trailing dot.", "trailing_dot"), ("..", "anonymous"),
        ("CON", "CON_"), ("nul.txt", "nul.txt_"), ("x" * 200, "x" * 60),
    ]:
        assert session_files.sanitize_participant_id(raw) == expected, raw

    ts = time.mktime((2026, 10, 25, 14, 3, 11, 0, 0, -1))
    assert session_files.session_folder_name(ts) == "2026-10-25_14-03-11"
    assert session_files.session_folder_name(ts, test_mode=True) == "2026-10-25_14-03-11_test"

    a = session_files.create_session_folder(tmp_path, "p", 1_800_000_000, False)
    b = session_files.create_session_folder(tmp_path, "p", 1_800_000_000, False)
    assert a != b and a.parent == b.parent and b.name.endswith("_2")

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    ResultLogger("anonymous", test_mode=True).close()
    assert (tmp_path / "anonymous").is_dir()


# -- finding sessions in both layouts -------------------------------------------------------------


def _new_run(tmp_path, monkeypatch, participant="p1", **kwargs):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger(participant, **kwargs)
    log.close()
    return log


def test_find_session_by_any_identifier_newest_across_layouts_and_key_without_session_json(tmp_path, monkeypatch):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)  # old sessions from 2026-10-22
    log = _new_run(tmp_path, monkeypatch)
    key, folder = str(log._timestamp), log.folder
    for ident in (key, folder.name, f"p1/{folder.name}", str(folder), str(folder / "trials.csv")):
        ref = session_files.find_session(tmp_path, ident)
        assert ref is not None and ref.folder == folder and ref.key == key, ident
    assert session_files.find_session(tmp_path, "999") is None

    refs = session_files.list_sessions(tmp_path)
    assert refs[0].key == key and not refs[0].is_legacy
    assert {r.key for r in refs} >= {"1792967858", "1792712654"}
    assert session_files.newest_session(tmp_path, lambda r: r.is_legacy).key == "1792967858"

    (folder / "session.json").unlink()
    assert session_files.find_session(tmp_path, folder.name).key == key  # the key survives a lost session.json


# -- old sessions still load ----------------------------------------------------------------------


def test_old_layout_grating_and_line_rows_load_and_unknown_values_read_as_none(tmp_path, monkeypatch):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    sessions = {s.timestamp: s for s in data_access.discover_sessions()}
    assert set(sessions) == {"1792967858", "1792712654"}
    assert sessions["1792967858"].trial_counts == {"rt_test": 6, "saccade": sessions["1792967858"].trial_counts["saccade"]}
    assert sessions["1792712654"].metadata["participant_id"] is None  # no meta file at all
    trials = read_trials(data_access.csv_path_for("1792712654"))
    shown = [t for t in trials if t.grating_shown]
    assert [t.orientation for t in shown[:3]] == [Side.BELOW, Side.BELOW, Side.ABOVE]  # old 'horizontal'/'vertical' values: the documented relabel
    assert trials[0].source is Target.DOT

    flashes = [t for t in read_trials(data_access.csv_path_for("1792967858")) if t.grating_shown]
    assert [t.orientation for t in flashes[:2]] == [Side.ABOVE, Side.BELOW]
    assert flashes[0].outcome == "miss" and flashes[0].contrast == pytest.approx(0.02)
    path = tmp_path / "results_1792967858.csv"
    path.write_text(path.read_text().replace(",above,", ",diagonal,", 1))  # a value no version of the experiment writes
    assert read_trials(data_access.csv_path_for("1792967858"))[6].orientation is None


def test_dashboard_lists_edits_and_deletes_new_and_old_sessions(tmp_path, monkeypatch):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    log = _new_run(tmp_path, monkeypatch, participant="alice", test_mode=True)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    key = str(log._timestamp)
    sessions = data_access.discover_sessions()
    assert sessions[0].timestamp == key
    assert sessions[0].metadata["participant_id"] == "alice" and sessions[0].metadata["test_mode"] is True
    data_access.save_participant_info(key, "f", "31")
    assert data_access.load_metadata(key)["age"] == 31
    assert json.loads((log.folder / "session.json").read_text())["participant"]["gender"] == "f"
    data_access.save_participant_info("1792967858", "", "31")
    assert json.loads((tmp_path / "results_1792967858_meta.json").read_text())["age"] == 31

    (log.glasses_dir / "video.mp4").write_bytes(b"x")
    data_access.delete_session(key)  # a new-layout delete removes the whole run folder, glasses video included
    assert not log.folder.exists() and not (tmp_path / "alice").exists()


# -- analyze_session --------------------------------------------------------------------------------


def _new_layout_session(tmp_path, monkeypatch, plan):
    """The same planted-dip session as test_analyze_session, written through the real logger."""
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger("p7")
    t, x = _trace(ONSETS)
    with log.gaze_trace_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "device_unix_s", "x_px", "worn"])
        for ti, xi in zip(t, x):
            w.writerow([ti, ti - CLOCK_OFFSET_S, xi * NEON_SCENE_WIDTH, 1])
    log.set_gaze_quality({"clock_offset_ms": CLOCK_OFFSET_S * 1000})
    for i, (rel_ms, outcome, contrast) in enumerate(plan):
        log.log(
            TrialResult(
                index=i, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=None, grating_shown=True, contrast=contrast,
                orientation=Side.ABOVE, responded=outcome != "miss", response_orientation=None, response_time_ms=None, outcome=outcome,
                grating_shown_at_unix_ms=(ONSETS[i] + rel_ms / 1000) * 1000, scheduled_delay_ms=200.0,
            )
        )
    log.close()
    return log


PLAN = [(-120, "correct", 0.1), (-30, "miss", 0.1), (-10, "incorrect", 0.15), (5, "miss", 0.15), (20, "miss", 0.2), (45, "correct", 0.2), (110, "correct", 0.1), (350, "correct", 0.15)]


def test_analysis_of_a_new_run_writes_into_its_analysis_folder_and_gives_the_old_numbers(tmp_path, monkeypatch):
    log = _new_layout_session(tmp_path, monkeypatch, PLAN)
    new = an.analyze(None, data_dir=tmp_path)  # the newest session, found without being named
    assert new["png"].parent == log.analysis_dir
    assert {p.name for p in log.analysis_dir.iterdir()} == {"report.txt", "plot.png", "flashes.csv", "suppression.json"}
    assert new["summary"]["near_onset"] == (1, 5) and new["summary"]["flashes"] == 8  # same as the old layout's planted-dip test
    assert (log.analysis_dir / "report.txt").read_text().startswith("Saccade-phase analysis")

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    stamp = _write_session(old_dir, ONSETS, PLAN)
    old = an.analyze(stamp, data_dir=old_dir)
    assert old["png"].name == f"analysis_{stamp}.png" and old["png"].parent == old_dir  # old sessions keep their file names
    assert old["summary"] == new["summary"]
    assert old["report"].splitlines()[1:] == new["report"].splitlines()[1:]  # identical apart from the session label line


def test_analysis_accepts_the_folder_name_and_latest_looks_in_both_layouts(tmp_path, monkeypatch):
    log = _new_layout_session(tmp_path, monkeypatch, PLAN)
    assert an.latest_timestamp(tmp_path) == str(log._timestamp)
    by_name = an.analyze(log.folder.name, data_dir=tmp_path)
    assert by_name["summary"]["flashes"] == 8
    stamp = _write_session(tmp_path, ONSETS, PLAN[:2], stamp="1700000001")  # an older loose-file session
    assert an.latest_timestamp(tmp_path) == str(log._timestamp)
    assert an.analyze(stamp, data_dir=tmp_path)["summary"]["flashes"] == 2
