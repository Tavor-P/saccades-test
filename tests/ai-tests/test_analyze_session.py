"""The post-hoc saccade analysis (src/experiment/analyze_session.py) on a synthetic session whose saccade
onsets and flash times are known exactly, plus the startup dialog's saccade-contrast settings."""

import csv
import json

import numpy as np
import pytest
import synthetic_runs as sr

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.experiment import analyze_session as an
from src.experiment.settings import saccade_contrast_levels, validate_saccade_contrast_range

CLOCK_OFFSET_S = 0.25
RATE_HZ = 200
T0 = 1_000_000.0  # PC unix seconds of the first sample


def _trace(onsets, duration_s=0.08, lo=0.40, hi=0.70, length_s=20.0):
    """200 Hz gaze that alternates between lo and hi: a smooth (sigmoid) saccade of `duration_s` starting at each
    onset (PC unix time). Returns (phone capture times, x) as the trace file would store them."""
    t = np.arange(0, length_s, 1 / RATE_HZ) + T0
    x = np.full_like(t, lo)
    level = lo
    for onset in onsets:
        target = hi if level == lo else lo
        u = np.clip((t - onset) / duration_s, 0, 1)
        smooth = 3 * u**2 - 2 * u**3
        x = np.where(t >= onset, level + (target - level) * smooth, x)
        x = np.where(t > onset + duration_s, target, x)
        level = target
    return t, x


def _write_session(tmp_path, onsets, flashes, stamp="1700000000"):
    """A run folder under tmp_path (participant p1, key `stamp`) whose trials.csv keeps the grating-era column names.
    flashes: (time relative to the matching onset in ms, outcome, contrast); onset index i is used in order."""
    t, x = _trace(onsets)
    # phone clock = PC clock - offset
    trace = [[ti, ti - CLOCK_OFFSET_S, xi * NEON_SCENE_WIDTH, 1] for ti, xi in zip(t, x)]
    fields = ["trial_index", "phase", "grating_shown", "contrast", "orientation", "outcome", "grating_shown_at_unix_ms"]
    rows = [
        {"trial_index": i, "phase": "saccade", "grating_shown": True, "contrast": contrast, "orientation": "vertical", "outcome": outcome, "grating_shown_at_unix_ms": (onsets[i] + rel_ms / 1000) * 1000}
        for i, (rel_ms, outcome, contrast) in enumerate(flashes)
    ]
    rows.append({"trial_index": 99, "phase": "saccade", "grating_shown": False, "outcome": "correct_rejection"})  # a catch row from an old session: ignored
    rows.append({"trial_index": 100, "phase": "presaccade", "grating_shown": True, "contrast": 0.01, "outcome": "correct", "grating_shown_at_unix_ms": 1})  # other phase: ignored
    sr.write_old_column_run(tmp_path, stamp, rows, fields, meta={"gaze_quality": {"clock_offset_ms": CLOCK_OFFSET_S * 1000}}, gaze_trace_rows=trace)
    return stamp


def _meta_path(tmp_path, stamp):
    return an.session_files.find_session(tmp_path, stamp).meta_path


ONSETS = [T0 + 2 + 2 * i for i in range(8)]


def test_detection_finds_each_saccade_with_the_right_onset_and_duration():
    t, x = _trace(ONSETS)
    found = an.detect_saccades(t, x)
    assert len(found) == len(ONSETS)
    for expected, s in zip(ONSETS, found):
        assert s.onset == pytest.approx(expected, abs=0.015)
        assert 40 <= s.duration_ms <= 110
        assert s.amplitude == pytest.approx(0.30, abs=0.03)


def test_small_movements_and_noise_are_not_saccades():
    t = np.arange(0, 5, 1 / RATE_HZ) + T0
    x = 0.5 + 0.004 * np.sin(40 * t)  # fixational jitter
    x[600:] += 0.05  # a slow 0.05 drift step is below the minimum amplitude
    assert an.detect_saccades(t, x) == []


def test_flashes_are_placed_against_the_nearest_onset_and_inside_flags_are_right():
    t, x = _trace(ONSETS)
    saccades = an.detect_saccades(t, x)
    flashes = [an.Flash(i, onset + rel / 1000 + 0.0, 0.1, "vertical", "correct") for i, (onset, rel) in enumerate(zip(ONSETS, [-120, -30, 10, 40, 70, 200, 330, 5]))]
    placed = an.place_flashes(flashes, saccades)
    rels = [p.rel_ms for p in placed]
    assert rels == pytest.approx([-120, -30, 10, 40, 70, 200, 330, 5], abs=15)
    inside = [p.inside for p in placed]
    assert inside[2] and inside[3] and inside[7]  # 10/40/5 ms after onset: mid-saccade
    assert not inside[0] and not inside[1] and not inside[5] and not inside[6]


def test_flash_delay_shifts_the_flash_later():
    t, x = _trace(ONSETS)
    saccades = an.detect_saccades(t, x)
    flash = an.Flash(0, ONSETS[0] + 0.02, 0.1, "vertical", "correct")
    assert an.place_flashes([flash], saccades, flash_delay_ms=15)[0].rel_ms == pytest.approx(an.place_flashes([flash], saccades)[0].rel_ms + 15)


def test_full_analysis_reproduces_a_planted_suppression_dip(tmp_path):
    # Misses planted exactly where flashes fall within +-50 ms of onset; everything else is answered.
    plan = [(-120, "correct", 0.1), (-30, "miss", 0.1), (-10, "incorrect", 0.15), (5, "miss", 0.15), (20, "miss", 0.2), (45, "correct", 0.2), (110, "correct", 0.1), (350, "correct", 0.15)]
    stamp = _write_session(tmp_path, ONSETS, plan)
    result = an.analyze(stamp, data_dir=tmp_path, out_dir=tmp_path / "out")
    s = result["summary"]
    assert s["flashes"] == 8 and s["saccades"] == len(ONSETS)  # legacy catch + presaccade rows excluded
    assert s["near_onset"] == (1, 5)  # -30, -10, 5, 20, 45 ms: only the 45 ms flash was seen
    assert s["beyond"] == (2, 2) and s["far"] == (1, 1)  # -120 and +110 ms; and the +350 ms flash is the fixation control
    assert s["misses"] == 3
    assert s["median_saccade_ms"] == pytest.approx(80, abs=25)
    assert "Saccade-phase analysis" in result["report"]
    assert result["png"].exists() and result["png"].stat().st_size > 10_000
    rows = list(csv.DictReader(result["csv"].open()))
    assert len(rows) == 8 and float(rows[0]["flash_rel_onset_ms"]) == pytest.approx(-120, abs=15)


def test_latest_timestamp_picks_the_newest_session_with_flashes(tmp_path):
    _write_session(tmp_path, ONSETS, [(0, "correct", 0.1)], stamp="1700000001")
    assert an.latest_timestamp(tmp_path) == "1700000001"
    with pytest.raises(FileNotFoundError):
        an.latest_timestamp(tmp_path / "empty")


def test_a_session_without_a_clock_offset_asks_for_one(tmp_path):
    stamp = _write_session(tmp_path, ONSETS, [(0, "correct", 0.1)])
    _meta_path(tmp_path, stamp).write_text(json.dumps({"gaze_quality": {}}))
    with pytest.raises(ValueError, match="offset"):
        an.analyze(stamp, data_dir=tmp_path)
    assert an.analyze(stamp, data_dir=tmp_path, clock_offset_ms=250.0, out_dir=tmp_path / "o")["summary"]["flashes"] == 1


# -- the startup dialog's saccade contrast range --------------------------------------------------


def test_contrast_levels_are_evenly_spaced_between_min_and_max():
    assert saccade_contrast_levels(10, 20) == pytest.approx((0.10, 0.15, 0.20))  # other values still work
    assert saccade_contrast_levels(10, 20, count=2) == pytest.approx((0.10, 0.20))
    assert saccade_contrast_levels(15, 15) == (0.15,)


def test_invalid_contrast_ranges_are_rejected():
    for bad in ((20, 10), (0.0, 10), (10, 150)):
        with pytest.raises(ValueError):
            validate_saccade_contrast_range(*bad)


def test_session_uses_the_dialog_contrasts_instead_of_the_default(fake_gaze, fake_time):
    from test_session import _make_session

    session = _make_session(fake_gaze, fixed_contrasts=(0.07,))
    assert {session._fixed_contrast_bag.draw() for _ in range(4)} == {0.07}
    with pytest.raises(ValueError):  # an empty set is refused (the staircase it used to select is gone)
        _make_session(fake_gaze, fixed_contrasts=())


# -- suppression index ---------------------------------------------------------------------------


def _placed(rel_ms, contrast, correct, n=1):
    return [an.Placed(an.Flash(i, 0.0, contrast, "vertical", "correct" if i < correct else "incorrect"), rel_ms, False) for i in range(n)]


def _synthetic(threshold_sac, threshold_fix, contrasts=(0.01, 0.02, 0.04, 0.08, 0.16), n=60, slope=3.0, seed=1):
    """Binomial answers from a known Weibull threshold per condition: saccade flashes at +10 ms, fixation at +500 ms."""
    rng = np.random.default_rng(seed)
    placed = []
    for rel, thr in ((10.0, threshold_sac), (500.0, threshold_fix)):
        for c in contrasts:
            k = int(rng.binomial(n, float(an._weibull(np.asarray(c), np.asarray(thr), np.asarray(slope)))))
            placed += _placed(rel, c, k, n)
    return placed


def test_suppression_index_recovers_known_suppression():
    r = an.suppression_index(_synthetic(0.04, 0.02), n_boot=300)  # sensitivity halved: index 0.5
    assert r["method"] == "psychometric"
    assert r["index"] == pytest.approx(0.5, abs=0.12)
    assert r["ci"][0] < 0.5 < r["ci"][1]
    assert (r["n_saccade"], r["n_fixation"]) == (300, 300)


def test_suppression_index_is_near_zero_without_suppression():
    r = an.suppression_index(_synthetic(0.03, 0.03, seed=2), n_boot=300)
    assert r["method"] == "psychometric"
    assert r["ci"][0] < 0 < r["ci"][1]


def test_window_definitions_are_parameters():
    placed = _placed(80.0, 0.1, 5, 10) + _placed(500.0, 0.1, 10, 10)
    assert an.suppression_index(placed)["n_saccade"] == 0  # 80 ms is outside the default +-50 ms
    assert an.suppression_index(placed, near_ms=100)["n_saccade"] == 10
    assert an.suppression_index(placed, near_ms=100, far_ms=600)["n_fixation"] == 0


def test_too_few_flashes_near_onset_is_reported_not_raised():
    r = an.suppression_index(_placed(500.0, 0.1, 9, 10))
    assert r["index"] is None and r["method"] is None
    assert "cannot compute" in r["notes"][0]
    assert "cannot be computed" in an.format_suppression(r)
    assert an.suppression_index([])["index"] is None
    assert an.suppression_index([an.Placed(_placed(0, 0.1, 1)[0].flash, None, False)])["index"] is None  # no saccades at all


def test_ceiling_falls_back_to_a_labelled_accuracy_index():
    placed = _placed(0.0, 0.1, 6, 10) + _placed(500.0, 0.1, 10, 10)  # one contrast, so no psychometric fit
    r = an.suppression_index(placed)
    assert r["method"] == "accuracy"
    assert r["index"] == pytest.approx(0.4)
    assert r["ci"][0] < 0.4 < r["ci"][1]
    assert any("psychometric index not computed" in n for n in r["notes"])
    assert "accuracy fallback" in an.format_suppression(r)


def test_all_correct_everywhere_is_flagged_as_ceiling_not_a_threshold():
    placed = []
    for c in (0.1, 0.2, 0.4):
        placed += _placed(0.0, c, 10, 10) + _placed(500.0, c, 10, 10)
    r = an.suppression_index(placed)
    assert r["method"] == "accuracy" and r["index"] == pytest.approx(0.0)
    assert any("ceiling" in n for n in r["notes"])


def test_fixation_at_floor_is_not_computable():
    placed = []
    for c in (0.1, 0.2):
        placed += _placed(0.0, c, 5, 10) + _placed(500.0, c, 5, 10)  # 50 % everywhere: nothing visible
    r = an.suppression_index(placed)
    assert r["index"] is None
    assert any("floor" in n for n in r["notes"])


def test_gates_decide_the_conditions_and_excluded_flashes_are_left_out():
    from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE, GateResult

    # rel_ms says "near onset" for everything, but the gates say otherwise: they win.
    placed = _placed(0.0, 0.1, 5, 10) + _placed(0.0, 0.1, 10, 10) + _placed(0.0, 0.1, 1, 10)
    gates = [GateResult(SACCADE, "")] * 10 + [GateResult(FIXATION, "")] * 10 + [GateResult(EXCLUDED, "x")] * 10
    r = an.suppression_index(placed, gates=gates)
    assert (r["n_saccade"], r["n_fixation"], r["conditions_from"]) == (10, 10, "gates")
    assert r["index"] == pytest.approx(0.5)
    assert an.suppression_index(placed)["conditions_from"] == "windows"


def test_identical_conditions_give_a_finite_ci_or_none():
    placed = []
    for c, k in ((0.10, 3), (0.15, 4), (0.20, 5)):
        placed += _placed(0.0, c, k, 5) + _placed(500.0, c, k, 5)
    r = an.suppression_index(placed)
    assert r["index"] is not None
    if r["ci"] is None:
        assert any("no CI" in n for n in r["notes"])
    else:
        assert all(-1.5 <= v <= 1.0 for v in r["ci"])
    an.format_suppression(r)


def test_missing_ci_is_formatted():
    r = an.suppression_index(_synthetic(0.04, 0.02), n_boot=300)
    r["ci"] = None
    assert "CI not available" in an.format_suppression(r)
    json.dumps(r)


def test_wilson_interval_contains_the_proportion_for_every_k():
    for n in range(1, 120):
        for k in range(n + 1):
            lo, hi = an.wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0, (k, n, lo, hi)
        assert an.wilson(0, n)[0] == 0.0 and an.wilson(n, n)[1] == 1.0
    assert an.wilson(0, 0) == (0.0, 1.0)
    assert (an.wilson(2, 2)[1] - an.wilson(2, 2)[0]) > (an.wilson(40, 40)[1] - an.wilson(40, 40)[0])  # wide for small n


@pytest.mark.parametrize("outcomes", [["correct"] * 6, ["incorrect"] * 11, ["correct"], ["incorrect"], ["correct", "incorrect", "miss"]])
def test_plot_report_handles_all_correct_all_wrong_and_single_flash_bins(tmp_path, outcomes):
    """Regression: k == n (or k == 0) used to give an error bar of -1e-16 and matplotlib refused to draw it."""
    placed = [an.Placed(an.Flash(i, 1000.0 + i, 0.06, "above", o), 10.0 * i - 20, False) for i, o in enumerate(outcomes)]
    summary = an.summarize(placed, [])
    png = an.plot_report(placed, [], summary, tmp_path / "plot.png")
    assert png.exists()


def test_a_timeout_row_whose_line_was_shown_and_answered_is_scored_from_the_answer(tmp_path):
    """The live session labels a trial 'timeout' when no eye movement was detected, even if the flash fired and was answered."""
    path = tmp_path / "trials.csv"
    fields = ["trial_index", "phase", "grating_shown", "contrast", "orientation", "responded", "response_orientation", "outcome", "grating_shown_at_unix_ms"]
    rows = [
        (0, "True", "above", "timeout"),  # answered correctly
        (1, "True", "below", "timeout"),  # answered wrongly
        (2, "False", "", "timeout"),  # never answered
        (3, "True", "above", "correct"),  # not a timeout: untouched
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, responded, answer, outcome in rows:
            w.writerow({"trial_index": i, "phase": "saccade", "grating_shown": "True", "contrast": 0.06, "orientation": "above", "responded": responded, "response_orientation": answer, "outcome": outcome, "grating_shown_at_unix_ms": 1_000_000 + i})
    assert [f.outcome for f in an.load_flashes(path)] == ["correct", "incorrect", "miss", "correct"]


def test_clock_offset_history_is_interpolated_over_the_trace(tmp_path):
    """The phone clock drifts against the PC's: the offset at the start of the trace differs from the final one."""
    path = tmp_path / "trace.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "device_unix_s", "x_px", "worn"])
        for phone_t in (1000.0, 1060.0, 1120.0, 1180.0):
            w.writerow([0, phone_t, 500, 1])
    history = [{"phone_unix_s": 1000.0, "offset_ms": 250.0}, {"phone_unix_s": 1120.0, "offset_ms": 270.0}]
    t, _ = an.load_trace(path, 0.999, history)
    assert t - np.asarray([1000.0, 1060.0, 1120.0, 1180.0]) == pytest.approx([0.250, 0.260, 0.270, 0.270])
    t, _ = an.load_trace(path, 0.25, [history[0]])  # a single entry (or none) falls back to the scalar offset
    assert t - np.asarray([1000.0, 1060.0, 1120.0, 1180.0]) == pytest.approx([0.25] * 4)


# -- the human-first report: the same layout every run -----------------------------------------------------------------------------

SECTIONS = ["DATA QUALITY", "INDEX FIT", "ACCURACY BY FLASH TIME", "ACCURACY BY CONTRAST", "SUPPRESSION INDEX BY LEG", "RETURN LATENCY", "TECHNICAL DETAILS"]
GOOD_META = {"gaze_quality": {"latency_ms_median": 14.0, "longest_gap_ms": 20.0, "clock_offset_status": "ok"}, "head_motion": {"status": "ok", "flagged_trials": 0}}


def test_report_has_the_three_line_top_block_then_the_same_sections_every_run(tmp_path):
    stamp = _write_session(tmp_path, ONSETS, [(rel, "correct", c) for rel in (0, 10, -10, 20) for c in (0.1, 0.2)])
    report = an.analyze(stamp, tmp_path, out_dir=tmp_path / "out")["report"]
    lines = report.splitlines()
    top = lines[3:6]
    assert [ln.split()[0] for ln in top] == ["QUALITY", "RESULT", "AMOUNT"]
    assert top[2].split()[1:3] == ["TOO", "FEW"] and "usable of 8 (need 12+ each for a threshold fit)" in top[2]
    assert [report.index(h) for h in SECTIONS] == sorted(report.index(h) for h in SECTIONS)
    assert "no return jump in this run" in report and report.count("no return jump in this run") == 2  # legs and return latency still have their section
    assert "WARNING" not in report and "MAIN COMPARISON" not in report and report.count("Suppression index") == 0  # said once, in RESULT
    assert "No answer: 0 (scored as not correct, like a wrong answer; not excluded)" in report
    assert "Excluded as not moving fast yet:" in report and report.encode("cp1252")
    assert all(len(line) < 170 for line in lines)


def test_report_says_when_it_is_a_test_run_and_only_then(tmp_path):
    flashes = [(rel, "correct", 0.1) for rel in (0, 10, 400)]
    stamp = _write_session(tmp_path, ONSETS, flashes)
    assert "TEST run" not in an.analyze(stamp, tmp_path, out_dir=tmp_path / "a")["report"]
    summary = an.analyze(stamp, tmp_path, out_dir=tmp_path / "b")["summary"]
    assert "TEST run" in an.format_report(summary, test_run=True) and "TEST run" not in an.format_report(summary)


def test_quality_line_reads_session_json_and_says_unknown_for_missing_fields():
    from src.experiment.run_quality import quality_line, run_facts

    status, text = quality_line([run_facts(GOOD_META, 0, 30)])
    assert status == "OK" and text == "gaze delay 14 ms · clock offset ok · head still · 0 flashes lost to tracking"
    status, text = quality_line([run_facts({}, None, None)])  # an old or damaged session.json: never a crash
    assert status == "CHECK" and "gaze delay unknown" in text and "clock offset unknown" in text and "head unknown" in text and "tracking loss unknown" in text
    assert quality_line([run_facts("damaged", 0, 30)])[0] == "CHECK"
    bad_clock = {**GOOD_META, "gaze_quality": {**GOOD_META["gaze_quality"], "clock_offset_status": "failed (timeout)"}}
    assert quality_line([run_facts(bad_clock, 0, 30)])[0] == "BAD"
    slow = {**GOOD_META, "gaze_quality": {**GOOD_META["gaze_quality"], "latency_ms_median": 400.0}}
    assert quality_line([run_facts(slow, 0, 30)])[0] == "CHECK"
    flagged = {**GOOD_META, "head_motion": {"status": "ok", "flagged_trials": 2}}
    assert quality_line([run_facts(flagged, 1, 30)])[0] == "CHECK" and "1 flashes lost to tracking" in quality_line([run_facts(flagged, 1, 30)])[1]
    assert quality_line([run_facts(GOOD_META, 10, 30)])[0] == "BAD"  # a third of the flashes lost to tracking


def test_pooled_quality_line_takes_the_worst_run_and_a_delay_range():
    from src.experiment.run_quality import quality_line, run_facts

    slow = {**GOOD_META, "gaze_quality": {**GOOD_META["gaze_quality"], "latency_ms_median": 20.0}}
    status, text = quality_line([run_facts(GOOD_META, 0, 30), run_facts(slow, 2, 30)])
    assert status == "CHECK" and "gaze delay 14-20 ms" in text and "2 flashes lost to tracking" in text


def test_dashboard_lights_and_report_quality_share_the_same_rules():
    from src.dashboard import results
    from src.experiment import run_quality

    assert results.latency_light is run_quality.latency_light and results.LATENCY_RULE is run_quality.LATENCY_RULE


def test_return_latency_flags_a_spread_that_is_off_by_more_than_a_quarter():
    base = {"trials_jumped": 80, "forced_jumps": 0, "no_return_found": 0, "n": 80, "median_ms": 150.0, "sd_ms": 70.0, "robust_sd_ms": 64.0,
            "share_predictive": 0.0, "live_n": 0, "return_flash_spread_sd_ms": 40.0, "return_jump_trigger": "arrival"}
    assert "measured 64 ms vs set 40 ms: consider raising RETURN_FLASH_SPREAD_SD_MS" in an.format_return_latency(base)
    assert "consider lowering" in an.format_return_latency({**base, "robust_sd_ms": 20.0})
    assert "CHECK" not in an.format_return_latency({**base, "robust_sd_ms": 45.0})  # within 25 %
    assert "no return jump in these runs" in an.format_return_latency(None, "these runs")


def test_leg_table_gives_each_legs_flash_count_and_saccade_flashes():
    leg = {"index": None, "ci": None, "method": None, "n_saccade": 0, "n_fixation": 9, "correct_saccade": 0, "correct_fixation": 8}
    text = an.format_leg_suppression({"out": leg, "return": {**leg, "n_saccade": 4, "correct_saccade": 3}, "pooled": {**leg, "n_saccade": 4}}, {"out": 6, "return": 24})
    out_row = next(ln for ln in text.splitlines() if ln.split()[:1] == ["out"])
    assert out_row.split()[1:3] == ["6", "0"] and next(ln for ln in text.splitlines() if ln.split()[:1] == ["pooled"]).split()[1:3] == ["30", "4"]


def test_not_moving_fast_exclusions_are_split_into_before_and_after_onset():
    from src.experiment.flash_gates import EXCLUDED, NOT_MOVING_REASON, GateResult

    def placed(rel):
        return an.Placed(an.Flash(0, 1000.0, 0.06, "above", "correct"), rel, False)

    gates = [GateResult(EXCLUDED, NOT_MOVING_REASON)] * 4 + [GateResult(EXCLUDED, "tracking gap around the flash")]
    assert an.speed_exclusions([placed(-30.0), placed(-5.0), placed(20.0), placed(None), placed(10.0)], gates) == (2, 1, 1)
    text = "\n".join(an.format_excluded({"excluded": [0, 5], "reasons": {"x": 5}}, (2, 1, 0)))
    assert "2 before saccade onset, 1 after" in text and "Excluded: 5 (5 x x)" in text


def test_only_thresholds_that_differ_from_the_defaults_are_printed():
    assert "defaults (analysis_constants.py)" in an.format_thresholds(an.thresholds_record())
    text = an.format_thresholds(an.thresholds_record(min_amplitude=0.2, flash_delay_ms=15.0))
    lines = [ln.strip() for ln in text.splitlines()[1:]]
    assert lines == ["min_amplitude = 0.2  (default 0.12)", "flash_delay_ms = 15  (default 0)"]
    assert an.thresholds_record(min_amplitude=0.2)["min_amplitude"] == 0.2 and "speed_high" in an.thresholds_record()  # the record itself stays complete


def test_geometry_is_one_line_with_a_short_placeholder_note():
    from src.experiment.geometry import Geometry

    text = an.format_geometry(Geometry.default().as_settings())
    assert text.count("\n") == 0 and "Screen 26 cm @ 50 cm (29.1 deg)" in text and text.endswith("[placeholder, not measured]")
    assert "placeholder" not in an.format_geometry(Geometry(19.5, 60).as_settings())
    assert "Screen geometry: not recorded" in an.format_report(an.summarize([], []) | {"gates": {"saccade": [0, 0], "fixation": [0, 0], "excluded": [0, 0], "reasons": {}},
                                                                                       "suppression": an.suppression_index([])}, geometry=None)


def test_a_run_stopped_before_the_first_trial_says_so_instead_of_blaming_the_clock(tmp_path):
    from src.experiment import session_files

    folder = sr.write_run(tmp_path, "EMPTY", 1_790_000_000.0, n_trials=20, seed=5)
    trials = folder / session_files.TRIALS_FILE
    header = trials.read_text(encoding="utf-8").splitlines()[0]
    trials.write_text(header + "\n", encoding="utf-8")  # header only: no trial was run
    ref = session_files.find_session(tmp_path, f"{folder.parent.name}/{folder.name}")
    with pytest.raises(ValueError, match="no saccade-phase flashes in trials.csv"):
        an.analyze_run(ref)


def test_a_run_with_flashes_but_no_gaze_recording_says_so(tmp_path):
    from src.experiment import session_files

    folder = sr.write_run(tmp_path, "NOTRACE", 1_790_000_000.0, n_trials=20, seed=6)
    ref = session_files.find_session(tmp_path, f"{folder.parent.name}/{folder.name}")
    ref.gaze_trace_path.unlink()
    with pytest.raises(ValueError, match="no gaze recording"):
        an.analyze_run(ref)
