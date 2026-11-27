"""The time-course figure's numbers (timecourse.py), the companion numbers (relative_suppression.py) and their outputs.
Planted data: planted_flashes.py (a known dip; no files). Bootstrap counts go through the conftest stub (300 resamples)."""

import json
import math

import numpy as np
import pytest
from planted_flashes import DEFAULT_DIP, pooled_planted_flashes, true_curve
from synthetic_runs import write_run

from include.experiment.analysis_constants import (
    RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY,
    RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES,
    TIMECOURSE_MIN_EFFECTIVE_N_DRAWN,
    TIMECOURSE_MIN_EFFECTIVE_N_SOLID,
)
from src.experiment import analyze_session as an
from src.experiment.analysis_output import plot_timecourse_figure
from src.experiment.flash_gates import GateResult
from src.experiment.relative_suppression import companion_numbers
from src.experiment.timecourse import LOW, NONE, OK, curve_grid, kernel_estimate, point_status, timecourse_curve


def _arr(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _placed(rel_ms, contrast, correct):
    return an.Placed(an.Flash(0, 0.0, contrast, "above", "correct" if correct else "incorrect"), rel_ms, False)


# -- the kernel ------------------------------------------------------------------------------------------------------------


def test_the_kernel_recovers_a_constant_and_reports_kish_effective_n():
    rel = np.linspace(-200, 250, 91)
    p, n_eff = kernel_estimate(rel, np.full(91, 0.8), curve_grid(), sigma_ms=20.0)
    assert np.allclose(p[n_eff > 1], 0.8)
    one_flash_p, one_flash_n = kernel_estimate(np.array([10.0]), np.array([1.0]), np.array([0.0, 10.0]), sigma_ms=20.0)
    assert list(one_flash_p) == [1.0, 1.0] and list(one_flash_n) == pytest.approx([1.0, 1.0])  # a single flash is one effective flash
    # evenly spaced flashes (every 1 ms): effective n is about 2 * sqrt(pi) * sigma = 71 flashes at sigma 20
    dense = np.arange(-300.0, 300.0)
    assert kernel_estimate(dense, np.ones_like(dense), np.array([0.0]), sigma_ms=20.0)[1][0] == pytest.approx(2 * math.sqrt(math.pi) * 20, rel=0.01)


def test_a_gap_between_two_clusters_is_not_called_well_supported():
    rel = np.concatenate([np.full(30, -140.0), np.full(30, 60.0)])  # nothing in between
    _, n_eff = kernel_estimate(rel, np.ones(60), np.array([-140.0, -40.0, 60.0]), sigma_ms=20.0)
    assert n_eff[0] == pytest.approx(30, rel=0.01) and n_eff[2] == pytest.approx(30, rel=0.01)
    assert n_eff[1] < TIMECOURSE_MIN_EFFECTIVE_N_DRAWN  # Kish alone would say 60


def test_the_curve_is_masked_by_effective_n():
    assert point_status(TIMECOURSE_MIN_EFFECTIVE_N_DRAWN - 0.1) == NONE
    assert point_status(TIMECOURSE_MIN_EFFECTIVE_N_DRAWN) == LOW and point_status(TIMECOURSE_MIN_EFFECTIVE_N_SOLID - 0.1) == LOW
    assert point_status(TIMECOURSE_MIN_EFFECTIVE_N_SOLID) == OK
    placed = [_placed(float(t), 0.06, t % 4 == 0) for t in range(-100, 101, 2)]  # 101 flashes, only between -100 and +100 ms
    tc = timecourse_curve(placed, n_boot=50)
    status = dict(zip(tc["grid_ms"], tc["status"]))
    assert status[0.0] == OK and status[-150.0] == NONE and status[200.0] == NONE and LOW in status.values()
    assert all(v is not None for v, s in zip(tc["p"], tc["status"]) if s != NONE)
    assert tc["lowest"] is not None and tc["lowest"]["status"] == OK


def test_the_band_brackets_the_curve_and_empty_data_gives_no_curve():
    placed, gates = pooled_planted_flashes(2, seed=3, depth=0.8)
    tc = timecourse_curve(placed, gates, n_boot=100)
    p, lo, hi = _arr(tc["p"]), _arr(tc["lo"]), _arr(tc["hi"])
    drawn = np.isin(tc["status"], (LOW, OK))
    assert np.all(lo[drawn] <= p[drawn] + 1e-9) and np.all(p[drawn] <= hi[drawn] + 1e-9)
    empty = timecourse_curve([], [], n_boot=50)
    assert set(empty["status"]) == {NONE} and empty["lowest"] is None and empty["baseline"]["n"] == 0 and empty["n_flashes"] == 0
    json.dumps(tc), json.dumps(empty)  # JSON-ready (no NaN, no numpy scalars)


def test_flashes_with_a_tracking_gap_are_left_out_and_the_baseline_is_the_fixation_gate():
    placed, gates = pooled_planted_flashes(1, seed=1)
    tc = timecourse_curve(placed, gates, n_boot=50)
    fixation = [p for p, g in zip(placed, gates) if g.condition == "fixation"]
    assert tc["baseline"]["n"] == len(fixation) and tc["baseline"]["correct"] == sum(p.flash.correct for p in fixation)
    lo, hi = tc["baseline"]["ci"]
    assert lo < tc["baseline"]["p"] < hi and tc["baseline"]["source"] == "fixation gate"
    gap = GateResult(an.EXCLUDED, "tracking gap around the flash")
    assert timecourse_curve(placed[:1], [gap], n_boot=50)["n_flashes"] == 0


def test_contrast_is_drawn_independently_of_flash_time():
    """The curve pools contrasts, which is only fair if the session's contrast draw does not depend on the flash time."""
    from scipy.stats import kruskal

    from src.experiment import simulation as sim

    rows = []
    for seed in range(3):
        rows += [(round(r.contrast, 3), r.scheduled_delay_ms) for r in sim.simulate_session(max_trials=150, seed=seed, test_mode=False, return_jump=False).results
                 if r.phase == "saccade" and r.contrast and r.scheduled_delay_ms is not None]
    contrasts, delays = np.asarray([c for c, _ in rows]), np.asarray([d for _, d in rows])
    levels = sorted(set(contrasts))
    assert len(levels) == 3 and len(rows) > 300
    assert kruskal(*[delays[contrasts == c] for c in levels]).pvalue > 0.01


# -- planted suppression ---------------------------------------------------------------------------------------------------


def test_a_planted_dip_is_found_by_the_curve_and_by_relative_suppression():
    placed, gates = pooled_planted_flashes(6, seed=0, depth=0.8)
    tc = timecourse_curve(placed, gates)
    assert tc["lowest"]["status"] == OK and DEFAULT_DIP[0] <= tc["lowest"]["ms"] <= DEFAULT_DIP[1]
    assert tc["lowest"]["p"] < tc["baseline"]["ci"][0] - 0.1  # clearly below the fixation band
    grid, p = curve_grid(), _arr(tc["p"])
    ok = np.asarray(tc["status"]) == OK
    assert float(np.sqrt(np.mean((p[ok] - true_curve(grid, 0.8)[ok]) ** 2))) < 0.09  # it follows the planted curve
    suppression = an.suppression_index(placed, gates=gates)
    c = companion_numbers(placed, gates, suppression)
    assert c["relative_suppression"] > 0.3 and c["relative_suppression_ci"][0] > 0  # planted 0.8 (a low draw: S has an SE of ~0.15 here)
    assert c["accuracy_difference_pp"] > 10 and c["accuracy_difference_ci_pp"][0] > 0
    assert c["strata_contrasts_pct"] == [4.0, 6.0, 8.0] and c["strata_dropped"] == []
    ts, tf = suppression["threshold_saccade"], suppression["threshold_fixation"]
    assert c["log10_threshold_ratio"] == pytest.approx(math.log10(ts / tf)) and c["log10_threshold_ratio"] > 0
    assert c["log10_threshold_ratio_ci"][0] < c["log10_threshold_ratio"] < c["log10_threshold_ratio_ci"][1]


def test_without_suppression_there_is_no_spurious_dip():
    placed, gates = pooled_planted_flashes(6, seed=0, depth=0.0)
    tc = timecourse_curve(placed, gates)
    ok = np.asarray(tc["status"]) == OK
    assert not np.any(ok & (_arr(tc["hi"]) < tc["baseline"]["ci"][0]))  # no point whose whole band lies below the fixation band
    c = companion_numbers(placed, gates, an.suppression_index(placed, gates=gates))
    assert c["relative_suppression_ci"][0] < 0 < c["relative_suppression_ci"][1]
    assert c["accuracy_difference_ci_pp"][0] < 0 < c["accuracy_difference_ci_pp"][1]


# -- companion numbers: arithmetic and gates -------------------------------------------------------------------------------


def _group(contrast, n, correct, rel_ms):
    return [_placed(rel_ms, contrast, i < correct) for i in range(n)]


def _gates_for(placed):
    return [GateResult("saccade" if p.rel_ms < 100 else "fixation", "") for p in placed]


def test_the_stratified_difference_and_s_follow_their_definitions():
    placed = (_group(0.04, 10, 6, 20) + _group(0.06, 10, 7, 20) + _group(0.08, 10, 8, 20)  # saccade: 70 % on average
              + _group(0.04, 20, 16, 400) + _group(0.06, 20, 18, 400) + _group(0.08, 20, 20, 400))  # fixation: 0.8 / 0.9 / 1.0 = 90 %
    c = companion_numbers(placed, _gates_for(placed), n_boot=200)
    assert c["p_saccade"] == pytest.approx(0.7) and c["p_fixation"] == pytest.approx(0.9)
    assert c["accuracy_difference_pp"] == pytest.approx(20.0)
    assert c["relative_suppression"] == pytest.approx((0.9 - 0.7) / (0.9 - 0.5))  # 0.5
    assert c["log10_threshold_ratio"] is None  # no threshold fit given
    lo, hi = c["relative_suppression_ci"]
    assert lo < 0.5 < hi


def test_s_is_withheld_with_its_reason_when_the_gates_are_not_met():
    few = _group(0.04, 5, 3, 20) + _group(0.06, 5, 3, 20) + _group(0.08, 4, 2, 20) + _group(0.06, 40, 38, 400)
    c = companion_numbers(few, _gates_for(few), n_boot=100)
    assert c["relative_suppression"] is None and f"only 14 saccade flashes (need {RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES})" in c["relative_suppression_note"]
    assert c["accuracy_difference_pp"] is not None  # the plain difference is still given
    assert {d["contrast_pct"]: d["why"] for d in c["strata_dropped"]} == {4.0: "no fixation flashes", 8.0: "fewer than 5 saccade flashes"}
    poor = _group(0.04, 10, 5, 20) + _group(0.06, 10, 5, 20) + _group(0.04, 30, 17, 400) + _group(0.06, 30, 19, 400)  # fixation 60 %
    c = companion_numbers(poor, _gates_for(poor), n_boot=100)
    assert c["relative_suppression"] is None and f"need {100 * RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY:.0f}%" in c["relative_suppression_note"]
    nothing = companion_numbers([], [], n_boot=50)
    assert nothing["accuracy_difference_pp"] is None and nothing["relative_suppression_note"]


def test_a_stratum_without_fixation_flashes_is_listed_not_used():
    placed = _group(0.04, 8, 4, 20) + _group(0.06, 8, 4, 20) + _group(0.04, 30, 24, 400)
    c = companion_numbers(placed, _gates_for(placed), n_boot=50)
    assert c["strata_contrasts_pct"] == [4.0] and [(d["contrast_pct"], d["why"]) for d in c["strata_dropped"]] == [(6.0, "no fixation flashes")]


# -- outputs -----------------------------------------------------------------------------------------------------------------


def test_the_analysis_writes_the_companion_numbers_and_the_time_course(tmp_path):
    write_run(tmp_path, "S01", 1790000000, n_trials=90, seed=2)
    result = an.analyze("1790000000", data_dir=tmp_path, out_dir=tmp_path / "out")
    saved = json.loads(result["suppression_json"].read_text())
    assert {"companion", "timecourse"} <= set(saved) and saved["index"] == result["summary"]["suppression"]["index"]  # the headline is untouched
    assert saved["companion"]["accuracy_difference_pp"] is not None and len(saved["timecourse"]["grid_ms"]) == len(curve_grid())
    report = result["report"]
    assert "COMPANION NUMBERS" in report and "percentage points" in report and "Threshold ratio saccade / fixation: log10" in report
    assert report.index("INDEX FIT") < report.index("COMPANION NUMBERS") < report.index("ACCURACY BY FLASH TIME")
    assert report.splitlines()[3].startswith("QUALITY") and report.splitlines()[4].startswith("RESULT")  # the top block is unchanged
    secondary = saved["secondary_onset_window"]
    assert secondary["conditions_from"] == "onset window" and secondary["n_fixation"] == saved["n_fixation"]
    assert report.index("RESULT") < report.index("SECONDARY INDEX") < report.index("DATA QUALITY")
    assert f"{secondary['n_saccade']} onset-window flashes" in report


def test_the_new_constants_are_in_the_thresholds_record():
    from src.experiment.analysis_report import thresholds_record

    record = thresholds_record()
    assert record["timecourse_sigma_ms"] == 20.0 and record["timecourse_step_ms"] == 5.0 and record["timecourse_range_ms"] == [-150.0, 200.0]
    assert record["timecourse_min_effective_n_drawn"] == 10.0 and record["timecourse_min_effective_n_solid"] == 20.0
    assert record["relative_suppression_min_saccade_flashes"] == 15 and record["stratum_min_saccade_flashes"] == 5
    assert record["onset_window_ms"] == 50.0


def test_timecourse_figure_draws_run_pooled_and_sparse_data(tmp_path):
    placed, gates = pooled_planted_flashes(4, seed=0, depth=0.8)
    suppression = an.suppression_index(placed, gates=gates)
    assert suppression["method"] == "psychometric"  # so the side panel draws the Weibull curves
    run_png = plot_timecourse_figure(placed, gates, None, suppression, 50.0, tmp_path / "run.png", title="Session test")
    pooled_png = plot_timecourse_figure(placed, gates, None, suppression, 50.0, tmp_path / "pooled.png", title="Participant X", pooled=True)
    assert run_png.stat().st_size > 20_000 and pooled_png.stat().st_size > 20_000
    # no gates (window rule), a handful of flashes, and nothing at all must draw too
    plot_timecourse_figure(placed[:12], None, None, None, None, tmp_path / "few.png")
    assert (tmp_path / "few.png").stat().st_size > 10_000
    plot_timecourse_figure([], [], None, None, None, tmp_path / "none.png")
    assert (tmp_path / "none.png").stat().st_size > 10_000


def test_the_figure_has_the_literature_layout_and_says_which_plot_to_read(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda fig=None: None)  # keep the figure to look at it
    placed, gates = pooled_planted_flashes(2, seed=0, depth=0.8)
    plot_timecourse_figure(placed, gates, None, an.suppression_index(placed, gates=gates), 50.0, tmp_path / "a.png")
    run_fig = plt.gcf()
    main, hist, side = run_fig.axes[:3]
    assert main.get_ylabel() == "correct (%)" and "onset (ms)" in hist.get_xlabel() and side.get_xlabel() == "line contrast (%)"
    assert [t.get_text() for t in side.get_xticklabels()] == ["4", "6", "8"]
    assert main.get_xlim() == (-150.0, 200.0) and hist.get_xlim() == (-150.0, 200.0)
    legend_text = " ".join(t.get_text() for t in run_fig.legends[0].get_texts())
    assert "chance (50%)" in legend_text and "fixation flashes" in legend_text and "typical saccade" in legend_text
    assert "indication only" in run_fig._suptitle.get_text()
    plot_timecourse_figure(placed, gates, None, None, 50.0, tmp_path / "b.png", pooled=True)
    assert "pooled per-person plot is the one to read" in plt.gcf()._suptitle.get_text()
    plt.close("all")


@pytest.mark.parametrize("return_jump", [False, True])
def test_a_planted_dip_survives_the_whole_file_pipeline(tmp_path, return_jump):
    """Raw gaze trace -> saccade detection -> placement (own leg in return-jump runs) -> gates -> curve and companion numbers:
    the in-saccade flashes (20-60 ms after their own onset) are planted hard to see, so the curve's lowest point lies there."""
    write_run(tmp_path, "S01", 1790000000, n_trials=200, seed=4, threshold_saccade=0.16, threshold_fixation=0.04, saccade_fraction=0.5, return_jump=return_jump)
    result = an.analyze("1790000000", data_dir=tmp_path, out_dir=tmp_path / "out")
    saved = json.loads(result["suppression_json"].read_text())
    low = saved["timecourse"]["lowest"]
    assert low["status"] == OK and -20 <= low["ms"] <= 100  # planted at 20-60 ms; the synthetic run has no flash between -140 and +20, so the kernel is flat out to there
    assert low["p"] < saved["timecourse"]["baseline"]["ci"][0] - 0.1
    assert saved["companion"]["relative_suppression"] > 0.4 and saved["companion"]["relative_suppression_ci"][0] > 0
    assert saved["index"] is not None and saved["index"] > 0.3


# -- gaze_timeline.png -------------------------------------------------------------------------------------------------------


def test_the_gaze_trace_is_averaged_in_bins_and_rows_cover_the_run():
    from include.experiment.analysis_constants import GAZE_TIMELINE_BIN_MS, GAZE_TIMELINE_ROW_S
    from src.experiment.gaze_timeline import binned_trace, contrast_colours, row_count

    assert GAZE_TIMELINE_BIN_MS == 10.0
    t = 100.0025 + 0.005 * np.arange(10)  # 200 Hz, two samples per 10 ms bin (away from the bin edges)
    x = np.array([0.0, 0.2, 0.4, 0.4, 1.0, 1.0, 0.1, 0.3, 0.5, 0.5])
    bt, bx = binned_trace(t, x, 100.0, 101.0)
    assert list(bx) == pytest.approx([0.1, 0.4, 1.0, 0.2, 0.5]) and list(bt) == pytest.approx([100.005 + 0.01 * i for i in range(5)])
    assert len(binned_trace(t, x, 200.0, 201.0)[0]) == 0  # nothing in the window
    assert row_count(0.0, 0.0) == 1 and row_count(0.0, 400.0) == 21  # 20 s rows with a 1 s lead-in and a 1.5 s tail
    light, mid, dark = (np.mean(c[:3]) for c in contrast_colours([0.08, 0.04, 0.06]).values() if True)  # keyed by contrast: 0.04 / 0.06 / 0.08
    assert light > mid > dark  # light = low contrast
    assert GAZE_TIMELINE_ROW_S == 20.0


def test_gaze_timeline_is_written_with_one_row_per_span_of_a_long_run(tmp_path):
    from include.experiment.analysis_constants import GAZE_TIMELINE_ROW_S
    from src.experiment.gaze_timeline import DPI, ROW_HEIGHT_IN

    write_run(tmp_path, "S01", 1790000000, n_trials=50, seed=3, threshold_saccade=0.12)  # 2.5 s per trial = 125 s of trials
    result = an.analyze("1790000000", data_dir=tmp_path, out_dir=tmp_path / "out")
    png = result["gaze_png"]
    assert png == tmp_path / "out" / "gaze_timeline.png" and png.exists()
    data = png.read_bytes()
    height = int.from_bytes(data[20:24], "big")  # PNG IHDR: width at 16, height at 20
    rows = round((height / DPI - 1.8) / ROW_HEIGHT_IN)
    assert rows >= math.ceil(125 / GAZE_TIMELINE_ROW_S) and rows > 1  # one row per 20 s of the run, stacked in one tall file
