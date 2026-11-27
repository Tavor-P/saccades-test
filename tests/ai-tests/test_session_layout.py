"""The on-disk layout: data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/, and run folders whose trials.csv still has the grating-era column names."""

import csv
import json
from pathlib import Path

import pytest
from synthetic_runs import write_old_column_run
from test_analyze_session import CLOCK_OFFSET_S, ONSETS, _trace, _write_session

from include.experiment.types import Side, Target, TrialResult
from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.dashboard import data_access
from src.experiment import analyze_session as an
from src.experiment import logger as logger_module
from src.experiment import session_files
from src.experiment.logger import ResultLogger, read_trials

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "old_columns"  # trials.csv with the grating-era column names and 'vertical'/'horizontal' sides


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


# -- finding sessions -----------------------------------------------------------------------------


def _new_run(tmp_path, monkeypatch, participant="p1", **kwargs):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger(participant, **kwargs)
    log.close()
    return log


def test_find_session_by_any_identifier_newest_first_and_key_without_session_json(tmp_path, monkeypatch):
    write_old_column_run(tmp_path, 1792967858, FIXTURES / "grating_names_b.csv", participant="anonymous")
    write_old_column_run(tmp_path, 1792712654, FIXTURES / "grating_names_a.csv", participant="anonymous")
    log = _new_run(tmp_path, monkeypatch)
    key, folder = str(log._timestamp), log.folder
    for ident in (key, folder.name, f"p1/{folder.name}", str(folder), str(folder / "trials.csv")):
        ref = session_files.find_session(tmp_path, ident)
        assert ref is not None and ref.folder == folder and ref.key == key, ident
    assert session_files.find_session(tmp_path, "999") is None

    refs = session_files.list_sessions(tmp_path)
    assert refs[0].key == key and refs[0].participant == "p1"
    assert {r.key for r in refs} >= {"1792967858", "1792712654"}
    assert session_files.newest_session(tmp_path, lambda r: r.participant == "anonymous").key == "1792967858"
    assert session_files.find_session(tmp_path, "results_1792967858.csv") is None  # the loose layout is gone

    (folder / "session.json").unlink()
    assert session_files.find_session(tmp_path, folder.name).key == key  # the key survives a lost session.json


# -- old column names still load ------------------------------------------------------------------


def test_old_column_names_grating_and_line_rows_load_and_unknown_values_read_as_none(tmp_path, monkeypatch):
    write_old_column_run(tmp_path, 1792967858, FIXTURES / "grating_names_b.csv", participant="anonymous")
    write_old_column_run(tmp_path, 1792712654, FIXTURES / "grating_names_a.csv", participant="anonymous")
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    sessions = {s.timestamp: s for s in data_access.discover_sessions()}
    assert set(sessions) == {"1792967858", "1792712654"}
    assert sessions["1792967858"].trial_counts == {"rt_test": 6, "saccade": sessions["1792967858"].trial_counts["saccade"]}
    assert sessions["1792712654"].metadata["participant_id"] is None  # session.json without a participant section
    trials = read_trials(data_access.csv_path_for("1792712654"))
    shown = [t for t in trials if t.flash_shown]
    assert [t.line_side for t in shown[:3]] == [Side.BELOW, Side.BELOW, Side.ABOVE]  # old 'horizontal'/'vertical' values: the documented relabel
    assert trials[0].source is Target.DOT

    flashes = [t for t in read_trials(data_access.csv_path_for("1792967858")) if t.flash_shown]
    assert [t.line_side for t in flashes[:2]] == [Side.ABOVE, Side.BELOW]
    assert flashes[0].outcome == "miss" and flashes[0].contrast == pytest.approx(0.02)
    path = data_access.csv_path_for("1792967858")
    path.write_text(path.read_text().replace(",above,", ",diagonal,", 1))  # a value no version of the experiment writes
    assert read_trials(data_access.csv_path_for("1792967858"))[6].line_side is None


def test_dashboard_lists_edits_and_deletes_a_session(tmp_path, monkeypatch):
    log = _new_run(tmp_path, monkeypatch, participant="alice", test_mode=True)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    key = str(log._timestamp)
    sessions = data_access.discover_sessions()
    assert sessions[0].timestamp == key
    assert sessions[0].metadata["participant_id"] == "alice" and sessions[0].metadata["test_mode"] is True
    data_access.save_participant_info(key, "f", "31")
    assert data_access.load_metadata(key)["age"] == 31
    assert json.loads((log.folder / "session.json").read_text())["participant"]["gender"] == "f"
    (log.glasses_dir / "video.mp4").write_bytes(b"x")
    data_access.delete_session(key)  # a delete removes the whole run folder, glasses video included
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
                index=i, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=None, flash_shown=True, contrast=contrast,
                line_side=Side.ABOVE, responded=outcome != "miss", answer_side=None, response_time_ms=None, outcome=outcome,
                flash_unix_ms=(ONSETS[i] + rel_ms / 1000) * 1000, scheduled_delay_ms=200.0,
            )
        )
    log.close()
    return log


PLAN = [(-120, "correct", 0.1), (-30, "miss", 0.1), (-10, "incorrect", 0.15), (5, "miss", 0.15), (20, "miss", 0.2), (45, "correct", 0.2), (110, "correct", 0.1), (350, "correct", 0.15)]


def test_analysis_of_a_new_run_writes_into_its_analysis_folder_and_gives_the_old_numbers(tmp_path, monkeypatch):
    log = _new_layout_session(tmp_path, monkeypatch, PLAN)
    new = an.analyze(None, data_dir=tmp_path)  # the newest session, found without being named
    assert new["png"].parent == log.analysis_dir
    assert {p.name for p in log.analysis_dir.iterdir()} == {"report.txt", "plot.png", "timecourse_by_contrast.png", "gaze_timeline.png", "flashes.csv", "suppression.json"}
    assert new["summary"]["near_onset"] == (1, 5) and new["summary"]["flashes"] == 8  # same as the old-column planted-dip test
    assert (log.analysis_dir / "report.txt").read_text().startswith("Saccade-phase analysis")

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    stamp = _write_session(old_dir, ONSETS, PLAN)
    old = an.analyze(stamp, data_dir=old_dir)
    assert old["png"].name == "plot.png" and old["png"].parent == session_files.find_session(old_dir, stamp).analysis_dir
    # Saccade metrics need the go cue (flash_ms_after_go_cue) and the screen geometry, which grating-era runs lack: compared without them.
    assert _without_saccade_metrics(old["summary"]) == _without_saccade_metrics(new["summary"])
    old_lines, new_lines = (_without_metrics_section(r["report"]) for r in (old, new))
    assert old_lines[1:] == new_lines[1:]  # identical apart from the session label line


def _without_saccade_metrics(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k != "saccade_metrics"}


def _without_metrics_section(report: str) -> list[str]:
    lines = report.splitlines()
    start, end = lines.index("SACCADE METRICS"), lines.index("TECHNICAL DETAILS")
    return lines[:start] + lines[end:]


def test_analysis_accepts_the_folder_name_and_latest_looks_at_every_run(tmp_path, monkeypatch):
    log = _new_layout_session(tmp_path, monkeypatch, PLAN)
    assert an.latest_timestamp(tmp_path) == str(log._timestamp)
    by_name = an.analyze(log.folder.name, data_dir=tmp_path)
    assert by_name["summary"]["flashes"] == 8
    stamp = _write_session(tmp_path, ONSETS, PLAN[:2], stamp="1700000001")  # an older run with the old column names
    assert an.latest_timestamp(tmp_path) == str(log._timestamp)
    assert an.analyze(stamp, data_dir=tmp_path)["summary"]["flashes"] == 2


def test_every_module_that_uses_data_dir_shares_the_one_definition():
    """DATA_DIR/REPO_ROOT are defined once in session_files; each module keeps its own importable name (tests patch it per module)."""
    from src.dashboard import data_access as da
    from src.experiment import import_glasses_video, run_summary, settings

    for module in (da, an, import_glasses_video, logger_module, run_summary, settings):
        assert module.DATA_DIR is session_files.DATA_DIR, module.__name__
    assert session_files.DATA_DIR == session_files.REPO_ROOT / "data" and (session_files.REPO_ROOT / "src").is_dir()
