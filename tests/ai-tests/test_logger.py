import csv
import json

from include.experiment.types import Side, Target, TrialResult
from src.experiment import logger as logger_module
from src.experiment.logger import FIELDNAMES, TRIAL_COLUMNS, ResultLogger


def _make_logger(tmp_path, monkeypatch, participant_id="p1", **kwargs):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    return ResultLogger(participant_id, **kwargs)


def _meta(tmp_path) -> dict:
    return json.loads(next(tmp_path.glob("*/*/session.json")).read_text())


def _rows(tmp_path) -> list[dict]:
    with next(tmp_path.glob("*/*/trials.csv")).open(newline="") as f:
        return list(csv.DictReader(f))


def _trial(**overrides) -> TrialResult:
    values = dict(
        index=0, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=None, grating_shown=True,
        contrast=0.05, orientation=Side.ABOVE, responded=True, response_orientation=Side.ABOVE,
        response_time_ms=250.0, outcome="correct",
    )
    values.update(overrides)
    return TrialResult(**values)


def test_writes_the_run_folder_with_its_files_and_the_trial_columns_in_order(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch, participant_id="alice", test_mode=True)
    log.close()
    (participant,) = [p for p in tmp_path.iterdir()]
    assert participant.name == "alice"
    (run,) = list(participant.iterdir())
    assert run.name.endswith("_test")
    assert (run / "trials.csv").exists() and (run / "session.json").exists()
    assert (run / "analysis").is_dir() and (run / "glasses").is_dir()
    assert log.gaze_trace_path == run / "gaze_trace.csv"
    assert log.analysis_dir == run / "analysis" and log.glasses_dir == run / "glasses"

    header = (run / "trials.csv").read_text().splitlines()[0]
    assert header.split(",") == FIELDNAMES == [c.name for c in TRIAL_COLUMNS]
    assert FIELDNAMES == [
        "phase", "trial", "contrast_pct", "flash_shown", "flash_during_saccade", "answered", "correct", "outcome",
        "line_side", "answer_side", "flash_leg",
        "flash_ms_after_go_cue", "jump_back_ms_after_go_cue", "reaction_ms", "return_reaction_ms", "answer_ms", "flash_unix_ms",
        "jump_back_forced", "tracking_ok",
    ]


def test_session_json_sections_and_what_the_setters_record(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch, participant_id="alice")
    log.close()
    meta = _meta(tmp_path)
    assert list(meta) == ["participant", "run", "settings", "hardware", "results", "gaze_quality", "head_motion", "ignored_space_presses", "glasses_video"]
    assert meta["participant"]["id"] == "alice"
    assert meta["run"]["unix_timestamp"] > 0 and meta["run"]["mode"] is None
    assert "viewing_distance_cm" in meta["settings"] and meta["settings"]["calibration"] is None
    assert "constants" not in meta  # the giant constants snapshot is gone

    other = tmp_path / "other"
    log = _make_logger(other, monkeypatch, camera_label="Camera 2", test_mode=True)
    log.set_calibration(0.3, 0.5, 0.7)
    log.set_saccade_contrasts((0.10, 0.15, 0.20))
    log.set_gaze_quality({"samples": 10, "latency_ms_median": 16.0})
    log.update_glasses_video(recording="started", recording_id="abc")
    log.add_sync_flash("start 1", 1_700_000_000_000.04)
    log.add_sync_flash("start 2", 1_700_000_000_450.0)
    log.close()
    meta = _meta(other)
    assert meta["settings"]["gaze_source"] == "Camera 2"
    assert meta["run"]["mode"] == "test"
    assert meta["settings"]["calibration"] == {"left_ratio": 0.3, "center_ratio": 0.5, "right_ratio": 0.7}
    assert meta["settings"]["saccade_contrasts_pct"] == [10.0, 15.0, 20.0]
    assert meta["gaze_quality"] == {"samples": 10, "latency_ms_median": 16.0}
    block = meta["glasses_video"]
    assert block["recording"] == "started" and block["recording_id"] == "abc"
    assert block["sync_flashes"] == [{"label": "start 1", "pc_unix_ms": 1_700_000_000_000.0}, {"label": "start 2", "pc_unix_ms": 1_700_000_000_450.0}]


def test_log_writes_plain_words_and_units_for_saccade_rt_test_and_timeout_rows(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch)
    log.log(_trial(timing_offset_ms=-40, scheduled_delay_ms=185.04, reaction_latency_ms=210.5, grating_shown_at_unix_ms=1_700_000_000_123.46, flash_during_saccade=True, tracking_ok=True))
    log.log(_trial(index=1, phase="rt_test", grating_shown=False, contrast=None, orientation=None, responded=False, response_orientation=None, response_time_ms=None, outcome="detected", reaction_latency_ms=210.5))
    log.log(_trial(index=2, grating_shown=False, contrast=None, orientation=None, responded=False, response_orientation=None, outcome="timeout"))
    log.close()
    row, rt_row, timeout_row = _rows(tmp_path)
    assert row["phase"] == "saccade" and row["trial"] == "0"
    assert "from" not in row and "to" not in row  # direction = trial parity, so source/target are not written
    assert row["flash_shown"] == "yes" and row["contrast_pct"] == "5.0"
    assert row["line_side"] == "above" and row["answer_side"] == "above"
    assert "planned_offset_ms" not in row and row["flash_ms_after_go_cue"] == "185.0" and row["reaction_ms"] == "210.5"
    assert row["flash_unix_ms"] == "1700000000123.5"
    assert row["answered"] == "yes" and row["answer_ms"] == "250.0"
    assert row["outcome"] == "correct" and row["correct"] == "yes"
    assert row["tracking_ok"] == "yes" and row["flash_during_saccade"] == "yes"
    assert rt_row["phase"] == "rt_test" and rt_row["reaction_ms"] == "210.5" and rt_row["outcome"] == "detected"
    assert timeout_row["outcome"] == "timeout" and timeout_row["correct"] == ""  # a timeout has no correct value


def test_outcome_words_and_old_baseline_and_catch_trial_rows_still_write_and_load(tmp_path, monkeypatch):
    # Sessions recorded before the baseline phase / catch trials were removed hold presaccade, false_alarm, correct_rejection rows.
    log = _make_logger(tmp_path, monkeypatch)
    log.log(_trial(phase="presaccade", source=None, target=None, outcome="miss", responded=False, response_orientation=None))
    log.log(_trial(index=1, outcome="incorrect", response_orientation=Side.BELOW))
    log.log(_trial(index=2, grating_shown=False, contrast=None, orientation=None, outcome="correct_rejection", responded=False, response_orientation=None))
    log.log(_trial(index=3, grating_shown=False, contrast=None, orientation=None, outcome="false_alarm"))
    log.close()
    first, second, third, fourth = _rows(tmp_path)
    assert first["phase"] == "baseline" and first["outcome"] == "no_answer" and first["correct"] == "no"
    assert second["outcome"] == "wrong" and second["correct"] == "no"
    assert third["outcome"] == "correct_rejection" and third["correct"] == "yes" and third["flash_shown"] == "no"
    rows = logger_module.read_trials(next(tmp_path.glob("*/*/trials.csv")))
    assert [r.outcome for r in rows] == ["miss", "incorrect", "correct_rejection", "false_alarm"]
    assert rows[0].phase == "presaccade"
    assert not any(r.grating_shown for r in rows[2:])


def test_results_summary_and_dropped_diagnostics_go_to_session_json(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch)
    log.log(_trial(saccade_duration_ms=80.0, onset_detection_lag_ms=12.0, reaction_latency_ms=200.0))
    log.log(_trial(index=1, outcome="miss", responded=False, response_orientation=None, saccade_duration_ms=100.0, onset_detection_lag_ms=16.0, reaction_latency_ms=220.0))
    log.close()
    saccade = _meta(tmp_path)["results"]["phases"]["saccade"]
    assert saccade["flashes_shown"] == 2 and saccade["flashes_correct"] == 1 and saccade["accuracy_on_flashes"] == 0.5
    assert saccade["median_saccade_duration_ms_live"] == 90.0 and saccade["median_onset_detection_lag_ms_live"] == 14.0
    assert saccade["median_reaction_ms"] == 210.0

