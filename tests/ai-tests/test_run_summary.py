import json

from include.experiment.types import Side, Target, TrialResult
from src.experiment import logger as logger_module
from src.experiment import run_summary
from src.experiment.logger import ResultLogger
from src.experiment.run_summary import build_run_summary, finish_run_summary, gaze_quality_lines, trial_counts

GOOD_QUALITY = {"sample_rate_hz": 160.0, "latency_ms_median": 95.0, "longest_gap_ms": 47.0, "gaps_over_threshold": 3}


def _trial(index, **overrides) -> TrialResult:
    values = dict(
        index=index, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=None, grating_shown=True,
        contrast=0.06, orientation=Side.ABOVE, responded=True, response_orientation=Side.ABOVE, response_time_ms=250.0,
        outcome="correct", reaction_latency_ms=200.0 + index, tracking_ok=True, scheduled_delay_ms=150.0,
    )
    values.update(overrides)
    return TrialResult(**values)


def _make_run(tmp_path, monkeypatch, trials, quality=GOOD_QUALITY, test_mode=True, glasses=None):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger("alice", test_mode=test_mode)
    for t in trials:
        log.log(t)
    if quality is not None:
        log.set_gaze_quality(quality)
    log.set_ignored_space_presses("saccade", {"during a trial": 2, "foreperiod": 1})
    if glasses:
        log.update_glasses_video(**glasses)
    log.close()
    return log.folder


def _ten_trials(**overrides):
    return [_trial(i, **overrides) for i in range(10)]


def test_summary_of_a_good_run(tmp_path, monkeypatch):
    trials = _ten_trials()
    trials[0] = _trial(0, outcome="wrong", response_orientation=Side.BELOW)
    trials[1] = _trial(1, responded=False, response_orientation=None, outcome="miss")
    folder = _make_run(tmp_path, monkeypatch, trials, glasses={"recording": "saved"})
    text = build_run_summary(folder)
    assert "Participant: alice" in text and "Mode: test" in text
    assert "Flashes shown: 10; answered: 9; correct: 8 (89 % of answered)" in text
    assert "Missing answers: 1" in text
    assert "Average (median) reaction time: 204 ms" in text  # median of 200..209 is 204.5, rounded half to even
    assert "Gaze rate: 160 Hz (OK)" in text and "Gaze delay: 95 ms (OK)" in text and "Gaze gaps: longest 47 ms (OK)" in text
    assert "Phone recording: saved on the phone." in text
    assert "Ignored SPACE presses: 3" in text
    assert "import_glasses_video" in text and "analyze_session" in text and "alice" in text
    assert "Very few trials" in text  # 10 flashes of a 30-trial test run is under half


def test_write_and_finish_create_the_file_and_print(tmp_path, monkeypatch, capsys):
    folder = _make_run(tmp_path, monkeypatch, _ten_trials())
    assert finish_run_summary(folder) is not None
    assert (folder / "RUN_SUMMARY.txt").read_text(encoding="utf-8") == build_run_summary(folder)
    assert "RUN SUMMARY" in capsys.readouterr().out


def test_finish_never_raises(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(run_summary, "build_run_summary", lambda folder: 1 / 0)
    assert finish_run_summary(tmp_path / "nowhere") is None
    assert "Could not write the run summary" in capsys.readouterr().err
    assert finish_run_summary(tmp_path / "nowhere") is None


def test_empty_and_missing_folders_still_print_something(tmp_path):
    (tmp_path / "p" / "2026-01-01_10-00-00").mkdir(parents=True)
    text = build_run_summary(tmp_path / "p" / "2026-01-01_10-00-00")
    assert "trials.csv is missing" in text and "session.json is missing" in text and "NEXT STEPS" in text
    assert "RUN SUMMARY" in build_run_summary(tmp_path / "does-not-exist")


def test_damaged_session_json_and_empty_trials(tmp_path, monkeypatch):
    folder = _make_run(tmp_path, monkeypatch, [], quality=None)
    (folder / "session.json").write_text("{not json")
    text = build_run_summary(folder)
    assert "session.json is missing or unreadable" in text
    assert "No flashes were shown" in text
    assert "Mode: test" in text  # from the _test folder name


def test_old_layout_rows_are_read(tmp_path):
    folder = tmp_path / "bob" / "2026-01-01_10-00-00"
    folder.mkdir(parents=True)
    (folder / "trials.csv").write_text(
        "trial_index,phase,grating_shown,responded,outcome,reaction_latency_ms,tracking_ok\n"
        "0,saccade,True,True,correct,200,True\n1,saccade,True,False,miss,180,False\n2,rt_test,False,False,detected,220,True\n"
    )
    text = build_run_summary(folder)
    assert "Flashes shown: 2; answered: 1; correct: 1" in text
    assert "Missing answers: 1" in text
    assert "reaction time: 200 ms" in text  # median of 200, 180, 220 (the rt_test row counts)


def test_warning_texts(tmp_path, monkeypatch):
    lost = _ten_trials(tracking_ok=False)
    text = build_run_summary(_make_run(tmp_path, monkeypatch, lost))
    assert "Tracking was lost near 10 of 10 flashes (100 %)" in text
    unanswered = _ten_trials(responded=False, response_orientation=None, outcome="miss")
    text = build_run_summary(_make_run(tmp_path / "b", monkeypatch, unanswered))
    assert "10 of 10 flashes got no answer" in text
    real = build_run_summary(_make_run(tmp_path / "c", monkeypatch, _ten_trials(), test_mode=False))
    assert "10 flashes shown, a full real run has about 150" in real


def test_gaze_quality_wording():
    lines, warnings = gaze_quality_lines({"sample_rate_hz": 60.0, "latency_ms_median": 300.0, "longest_gap_ms": 500.0, "gaps_over_threshold": 9})
    assert "Gaze rate: 60 Hz (check, expected 140+)" in lines
    assert "Gaze delay: 300 ms (check, typical 90-100)" in lines
    assert "Gaze gaps: longest 500 ms, 9 over the threshold (check)" in lines
    assert len(warnings) == 3
    lines, warnings = gaze_quality_lines(None)
    assert "not recorded" in lines[0] and warnings
    lines, warnings = gaze_quality_lines(GOOD_QUALITY)
    assert not warnings and all("(OK)" in line for line in lines)


def test_recording_states(tmp_path, monkeypatch):
    failed = build_run_summary(_make_run(tmp_path, monkeypatch, _ten_trials(), glasses={"recording": "failed (no wearer)"}))
    assert "Phone recording: failed (no wearer)." in failed and "Phone recording problem" in failed
    started = build_run_summary(_make_run(tmp_path / "b", monkeypatch, _ten_trials(), glasses={"recording": "started"}))
    assert "not confirmed saved" in started
    never = build_run_summary(_make_run(tmp_path / "c", monkeypatch, _ten_trials()))
    assert "Phone recording: not started." in never and "No phone recording was made." in never


def test_trial_counts_ignore_rt_test_rows_for_flashes():
    rows = [{"phase": "rt_test", "grating_shown": "False", "reaction_latency_ms": "300"}, {"phase": "saccade", "grating_shown": "True", "responded": "True", "outcome": "correct"}]
    counts = trial_counts(rows)
    assert counts["shown"] == 1 and counts["correct"] == 1 and counts["median_reaction_ms"] == 300


def test_cli_resolves_the_newest_and_a_named_session(tmp_path, monkeypatch, capsys):
    older = _make_run(tmp_path, monkeypatch, _ten_trials())
    # make the second run later and distinguishable
    meta = json.loads((older / "session.json").read_text())
    meta["run"]["unix_timestamp"] -= 1000
    (older / "session.json").write_text(json.dumps(meta))
    monkeypatch.setattr(logger_module.timebase, "unix_now", lambda: meta["run"]["unix_timestamp"] + 5000.0)
    newer = _make_run(tmp_path, monkeypatch, _ten_trials()[:4])
    assert newer != older
    assert run_summary.main(["--data-dir", str(tmp_path)]) == 0
    assert "Flashes shown: 4;" in capsys.readouterr().out
    assert run_summary.main([older.name, "--data-dir", str(tmp_path)]) == 0
    assert "Flashes shown: 10;" in capsys.readouterr().out
    assert run_summary.main(["nonexistent", "--data-dir", str(tmp_path)]) == 1
    assert run_summary.main(["--data-dir", str(tmp_path / "empty")]) == 1


def test_cli_old_loose_session(tmp_path, capsys):
    (tmp_path / "results_1700000000.csv").write_text(
        "trial_index,phase,grating_shown,responded,outcome\n0,saccade,True,True,correct\n"
    )
    assert run_summary.main(["--data-dir", str(tmp_path)]) == 0
    assert "Flashes shown: 1" in capsys.readouterr().out
