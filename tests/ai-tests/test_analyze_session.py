"""The post-hoc saccade analysis (src/experiment/analyze_session.py) on a synthetic session whose saccade
onsets and flash times are known exactly, plus the startup dialog's saccade-contrast settings."""

import csv
import json

import numpy as np
import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.experiment import analyze_session as an
from src.experiment.settings import resolve_saccade_contrasts, saccade_contrast_levels, validate_saccade_contrast_range

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
    """flashes: (time relative to the matching onset in ms, outcome, contrast); onset index i is used in order."""
    t, x = _trace(onsets)
    with (tmp_path / f"results_{stamp}_gaze_trace.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "device_unix_s", "x_px", "worn"])
        for ti, xi in zip(t, x):
            w.writerow([ti, ti - CLOCK_OFFSET_S, xi * NEON_SCENE_WIDTH, 1])  # phone clock = PC clock - offset
    (tmp_path / f"results_{stamp}_meta.json").write_text(json.dumps({"gaze_quality": {"clock_offset_ms": CLOCK_OFFSET_S * 1000}}))
    fields = ["trial_index", "phase", "grating_shown", "contrast", "orientation", "outcome", "grating_shown_at_unix_ms"]
    with (tmp_path / f"results_{stamp}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (rel_ms, outcome, contrast) in enumerate(flashes):
            w.writerow({"trial_index": i, "phase": "saccade", "grating_shown": True, "contrast": contrast, "orientation": "vertical", "outcome": outcome, "grating_shown_at_unix_ms": (onsets[i] + rel_ms / 1000) * 1000})
        w.writerow({"trial_index": 99, "phase": "saccade", "grating_shown": False, "outcome": "correct_rejection"})  # catch trial: ignored
        w.writerow({"trial_index": 100, "phase": "presaccade", "grating_shown": True, "contrast": 0.01, "outcome": "correct", "grating_shown_at_unix_ms": 1})  # other phase: ignored
    return stamp


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


def test_wilson_interval_brackets_the_proportion_and_is_wide_for_small_n():
    lo, hi = an.wilson(7, 12)
    assert lo < 7 / 12 < hi
    assert an.wilson(0, 0) == (0.0, 1.0)
    assert (an.wilson(2, 2)[1] - an.wilson(2, 2)[0]) > (an.wilson(40, 40)[1] - an.wilson(40, 40)[0])


def test_full_analysis_reproduces_a_planted_suppression_dip(tmp_path):
    # Misses planted exactly where flashes fall within +-50 ms of onset; everything else is answered.
    plan = [(-120, "correct", 0.1), (-30, "miss", 0.1), (-10, "incorrect", 0.15), (5, "miss", 0.15), (20, "miss", 0.2), (45, "correct", 0.2), (110, "correct", 0.1), (350, "correct", 0.15)]
    stamp = _write_session(tmp_path, ONSETS, plan)
    result = an.analyze(stamp, data_dir=tmp_path, out_dir=tmp_path / "out")
    s = result["summary"]
    assert s["flashes"] == 8 and s["saccades"] == len(ONSETS)  # catch + presaccade rows excluded
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
    (tmp_path / f"results_{stamp}_meta.json").write_text(json.dumps({"gaze_quality": {}}))
    with pytest.raises(ValueError, match="offset"):
        an.analyze(stamp, data_dir=tmp_path)
    assert an.analyze(stamp, data_dir=tmp_path, clock_offset_ms=250.0, out_dir=tmp_path / "o")["summary"]["flashes"] == 1


# -- the startup dialog's saccade contrast range --------------------------------------------------


def test_contrast_levels_are_evenly_spaced_between_min_and_max():
    assert saccade_contrast_levels(10, 20) == pytest.approx((0.10, 0.15, 0.20))
    assert saccade_contrast_levels(10, 20, count=2) == pytest.approx((0.10, 0.20))
    assert saccade_contrast_levels(15, 15) == (0.15,)


def test_invalid_contrast_ranges_are_rejected_or_fall_back():
    for bad in ((20, 10), (0.0, 10), (10, 150)):
        with pytest.raises(ValueError):
            validate_saccade_contrast_range(*bad)
    assert resolve_saccade_contrasts("5", "30") == pytest.approx((0.05, 0.175, 0.30))
    assert resolve_saccade_contrasts("", "abc") == pytest.approx((0.10, 0.15, 0.20))  # the defaults
    assert resolve_saccade_contrasts("30", "5") == pytest.approx((0.10, 0.15, 0.20))


def test_session_uses_the_dialog_contrasts_instead_of_the_default(fake_gaze, fake_time):
    from test_session import _make_session

    session = _make_session(fake_gaze, fixed_contrasts=(0.07,))
    assert {session._fixed_contrast_bag.draw() for _ in range(4)} == {0.07}
    assert _make_session(fake_gaze, fixed_contrasts=())._fixed_contrast_bag is None  # empty = the staircase


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
    assert "ACCURACY index" in an.format_suppression(r)


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


def test_analyze_writes_the_index_to_report_and_json(tmp_path):
    flashes = [(rel, "correct", c) for rel in (0, 10, -10, 20) for c in (0.1, 0.2)]
    stamp = _write_session(tmp_path, ONSETS, flashes)
    result = an.analyze(stamp, tmp_path)
    assert "suppression index" in result["report"]
    saved = json.loads(result["suppression_json"].read_text())
    assert saved["n_saccade"] == len(flashes) and "notes" in saved


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
