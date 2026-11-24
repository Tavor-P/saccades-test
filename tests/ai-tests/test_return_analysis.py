"""Post-hoc analysis of return-jump runs (analyze_session per-leg gating, return latency, analyze_participant pooling), on synthetic
run folders (synthetic_runs.write_run(return_jump=True)) and on simulated sessions written as run folders (sim_runs) - all in pytest
tmp folders, never under data/."""

import csv
import json

import numpy as np
import pytest
import sim_runs
import synthetic_runs as sr

from src.experiment import analyze_participant as ap
from src.experiment import analyze_session as an
from src.experiment import flash_gates as fg
from src.experiment import session_files
from src.experiment import simulation as sim
from src.experiment.logger import read_trial_rows

START = 1_790_000_000
DAY = 86_400


def _analyse(data, pid="SYN", **kw):
    folder = sr.write_run(data, pid, kw.pop("start", START), return_jump=True, **kw)
    return folder, an.analyze_run(session_files.find_session(data, f"{folder.parent.name}/{folder.name}"))


def _planted(folder) -> dict[int, tuple[str, bool]]:
    """trial -> (flash_leg, flash really mid-saccade) as written by the generator."""
    return {int(r["trial_index"]): (r["flash_leg"], r["flash_during_saccade"] == "True") for r in read_trial_rows(folder / session_files.TRIALS_FILE)}


# -- synthetic return-jump runs ------------------------------------------------------------------------------------------


def test_each_flash_is_gated_against_its_own_legs_saccade(tmp_path):
    folder, run = _analyse(tmp_path, n_trials=120, seed=3)
    assert run.return_jump and len(run.saccades) == 240
    assert all(tr.out is not None and tr.back is not None for tr in run.trials)
    assert all(tr.out.x1 > tr.out.x0 and tr.back.x1 < tr.back.x0 for tr in run.trials)  # out LO -> HI, return HI -> LO
    planted = _planted(folder)
    legs = {leg for leg, _ in planted.values()}
    assert legs == {"out", "return"}
    for p, g in zip(run.placed, run.gates):
        leg, during = planted[p.flash.trial]
        assert p.flash.leg == leg
        assert g.condition == (fg.SACCADE if during else fg.FIXATION), (p.flash.trial, leg, g.reason)
        if g.condition == fg.SACCADE:
            own = run.trials[run.placed.index(p)].of(leg)
            assert p.inside and 0 < p.rel_ms < 80 and own.onset < p.flash.time < own.offset
            assert f"({leg} saccade)" in g.reason
            # latency is from the leg's own anchor: the go cue (200 ms planted) or the jump (150 ms planted)
            assert g.latency_ms == pytest.approx(200 if leg == "out" else 150, abs=12)
    # the per-leg indices split the saccade flashes by leg and share every fixation flash
    n_fix = run.summary["gates"][fg.FIXATION][1]
    for leg in an.LEGS:
        s = run.by_leg[leg]
        assert s["n_fixation"] == n_fix
        assert s["n_saccade"] == sum(g.condition == fg.SACCADE and p.flash.leg == leg for p, g in zip(run.placed, run.gates))
    assert run.by_leg["pooled"] is run.suppression
    assert run.by_leg["out"]["n_saccade"] + run.by_leg["return"]["n_saccade"] == run.by_leg["pooled"]["n_saccade"]


def test_return_latency_report_matches_the_planted_latencies(tmp_path):
    _, run = _analyse(tmp_path, n_trials=120, seed=4, return_latency_ms=150, return_latency_sd_ms=30)
    r = run.return_latency
    assert (r["trials_jumped"], r["forced_jumps"], r["no_return_found"], r["n"]) == (120, 0, 0, 120)
    assert r["median_ms"] == pytest.approx(r["live_median_ms"], abs=8)  # trace onset vs the planted (logged) latency
    assert 22 < r["sd_ms"] < 38 and 20 < r["robust_sd_ms"] < 40
    assert r["share_predictive"] == 0.0
    assert r["return_flash_spread_sd_ms"] == 40.0 and r["return_jump_trigger"] == "arrival"


def test_predictive_returns_are_reported_and_still_gated_as_return_saccades(tmp_path):
    # The jump comes 150 ms after the out arrival, so returns planned 40 ms BEFORE it still start after the eye has landed.
    folder, run = _analyse(tmp_path, n_trials=120, seed=5, return_latency_ms=-40, return_latency_sd_ms=30, jump_delay_ms=150)
    r = run.return_latency
    assert r["n"] == 120 and -55 < r["median_ms"] < -25 and r["share_predictive"] > 0.75
    assert "predictive" in an.format_return_latency(r)
    planted = _planted(folder)
    return_sacc = [g for p, g in zip(run.placed, run.gates) if planted[p.flash.trial] == ("return", True)]
    assert return_sacc and all(g.condition == fg.SACCADE and g.latency_ms < 50 for g in return_sacc)  # no lower latency bound
    assert any(g.latency_ms < 0 for g in return_sacc)


def test_an_overshoot_and_its_correction_are_not_taken_for_the_return(tmp_path):
    _, plain = _analyse(tmp_path / "plain", n_trials=60, seed=6)
    _, over = _analyse(tmp_path / "over", n_trials=60, seed=6, overshoot=True)
    assert len(over.saccades) == 3 * 60  # out, correction (> MIN_AMPLITUDE) and return all detected
    for tr in over.trials:
        assert tr.back.x0 - tr.back.x1 > an.RETURN_MIN_AMPLITUDE_REL * tr.out.amplitude  # the big move back, not the correction
        assert tr.back.onset - tr.out.offset > sr.CORRECTION_DELAY_S + sr.CORRECTION_S
    assert over.return_latency["median_ms"] == pytest.approx(plain.return_latency["median_ms"], abs=2)
    assert [g.condition for g in over.gates] == [g.condition for g in plain.gates]


def test_forced_jump_and_wrong_leg_and_slow_return_are_excluded():
    t = np.arange(0.0, 3.0, 0.005)
    out = an.Saccade(1.0, 1.08, 0.4, 0.7, 6.0)
    back = an.Saccade(1.5, 1.58, 0.7, 0.4, 6.0)
    u = lambda a: np.clip((t - a) / 0.08, 0, 1)  # noqa: E731
    x = 0.4 + 0.3 * (3 * u(1.0) ** 2 - 2 * u(1.0) ** 3) - 0.3 * (3 * u(1.5) ** 2 - 2 * u(1.5) ** 3)
    sacc = [out, back]
    mid_back, mid_out = 1.54, 1.04
    ok = fg.classify_leg_flash(mid_back, sacc, t, x, "return", back, anchor=1.35)
    assert ok.condition == fg.SACCADE and ok.latency_ms == pytest.approx(150)
    forced = fg.classify_leg_flash(mid_back, sacc, t, x, "return", back, anchor=1.35, jump_forced=True)
    assert forced.condition == fg.EXCLUDED and "forced jump" in forced.reason
    # a return-leg flash that fell during the OUT saccade is not a return-saccade flash
    wrong = fg.classify_leg_flash(mid_out, sacc, t, x, "return", back, anchor=1.35)
    assert wrong.condition == fg.EXCLUDED and "other than this trial's return saccade" in wrong.reason
    missing = fg.classify_leg_flash(mid_back, sacc, t, x, "return", None, anchor=1.35)
    assert missing.condition == fg.EXCLUDED and "no return saccade was found" in missing.reason
    slow = fg.classify_leg_flash(mid_back, sacc, t, x, "return", back, anchor=0.9)
    assert slow.condition == fg.EXCLUDED and "over 500" in slow.reason
    early = fg.classify_leg_flash(mid_back, sacc, t, x, "return", back, anchor=1.6)  # predictive: onset 100 ms before the jump
    assert early.condition == fg.SACCADE and early.latency_ms == pytest.approx(-100)
    # fixation: >= 100 ms from BOTH saccades, whatever the leg
    assert fg.classify_leg_flash(1.3, sacc, t, x, "return", back, anchor=1.35).condition == fg.FIXATION
    gap = fg.classify_leg_flash(1.42, sacc, t, x, "return", back, anchor=1.35)  # landing-to-return gap, within 100 ms of the return
    assert gap.condition == fg.EXCLUDED


def test_runs_without_a_logged_jump_take_the_old_path_unchanged(tmp_path):
    folder = sr.write_run(tmp_path, "OLD", START, n_trials=60, seed=2)
    ref = session_files.find_session(tmp_path, f"{folder.parent.name}/{folder.name}")
    run = an.analyze_run(ref)
    assert not run.return_jump and run.by_leg is None and run.return_latency is None and "by_leg" not in run.summary
    placed = an.place_flashes(an.load_flashes(ref.trials_path), run.saccades)
    gates = fg.classify_flashes(placed, run.saccades, run.t, run.x, 0.0, [p.flash.go_cue for p in placed])
    assert run.placed == placed and run.gates == gates
    # a jump-off run of the new version writes flash_leg 'out' and no jump: still the old path, same result
    trials = folder / session_files.TRIALS_FILE
    rows = list(csv.DictReader(trials.open(newline="")))
    with trials.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, "flash_leg": "out"} for r in rows)
    again = an.analyze_run(ref)
    assert not again.return_jump and again.gates == run.gates and [p.rel_ms for p in again.placed] == [p.rel_ms for p in run.placed]
    result = an.analyze(f"{folder.parent.name}/{folder.name}", tmp_path)
    assert "Return latency" not in result["report"] and "by_leg" not in json.loads(result["suppression_json"].read_text())
    assert {r["flash_leg"] for r in csv.DictReader(result["csv"].open())} == {""}


def test_analyze_writes_the_leg_report_and_json(tmp_path):
    folder = sr.write_run(tmp_path, "SYN", START, n_trials=80, seed=7, return_jump=True)
    result = an.analyze(f"{folder.parent.name}/{folder.name}", tmp_path)
    report = result["report"]
    assert report.startswith("Return latency") and "Suppression index by leg" in report
    assert report.index("Return latency") < report.index("Gated analysis") < report.index("Suppression index by leg")
    saved = json.loads(result["suppression_json"].read_text())
    assert set(saved["by_leg"]) == {"out", "return"} and saved["return_latency"]["n"] == 80
    assert saved["index"] == result["summary"]["suppression"]["index"]
    rows = list(csv.DictReader(result["csv"].open()))
    assert {r["flash_leg"] for r in rows} == {"out", "return"}
    assert all(r["saccade_latency_ms"] for r in rows if r["condition"] == fg.SACCADE)


# -- participant pooling -------------------------------------------------------------------------------------------------


def test_return_jump_runs_pool_together_but_never_with_single_runs_or_other_triggers(tmp_path):
    for i in range(3):
        sr.write_run(tmp_path, "P", START + i * DAY, n_trials=60, seed=10 + i, return_jump=True)
    sr.write_run(tmp_path, "P", START + 3 * DAY, n_trials=60, seed=13)  # a single-saccade run
    sr.write_run(tmp_path, "P", START + 4 * DAY, n_trials=60, seed=14, return_jump=True, return_trigger="onset")
    s = ap.analyze_participant("P", tmp_path, write=False, n_boot=100).summary
    assert s["run_kind"] == "return_jump/arrival" and len(s["runs_used"]) == 3
    assert {r["run_kind"] for r in s["runs_used"]} == {"return_jump/arrival"}
    reasons = sorted(r["reason"] for r in s["runs_excluded"])
    assert len(reasons) == 2 and all("never pooled" in why for why in reasons)
    assert any("run kind single differs" in why for why in reasons) and any("run kind return_jump/onset differs" in why for why in reasons)
    assert "run kind" in s["analysis"]["pooling_rule"]
    assert set(s["by_leg"]) == {"out", "return", "pooled"}
    assert s["by_leg"]["out"]["n_saccade"] + s["by_leg"]["return"]["n_saccade"] == s["n_flashes"]["saccade"]
    assert all(s["by_leg"][leg]["n_fixation"] == s["n_flashes"]["fixation"] for leg in an.LEGS)
    assert s["by_leg"]["pooled"]["index"] == s["index"]
    rl = s["return_latency"]
    assert rl["n"] == 180 and rl["share_predictive"] == 0.0 and len(rl["return_flash_spread_sd_ms_by_run"]) == 3
    json.dumps(s)  # summary.json must stay serialisable


def test_participant_report_and_summary_file_carry_the_leg_results(tmp_path):
    for i in range(2):
        sr.write_run(tmp_path, "P", START + i * DAY, n_trials=60, seed=20 + i, return_jump=True)
    result = ap.analyze_participant("P", tmp_path, n_boot=100)
    assert "Return-jump runs (return_jump/arrival):" in result.report and "Suppression index by leg" in result.report
    saved = json.loads(result.paths["summary"].read_text())
    assert saved["schema_version"] == ap.SCHEMA_VERSION == 3 and list(saved) == list(ap.SUMMARY_KEYS)
    assert saved["return_latency"]["n"] == 120


# -- simulator ground truth ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["reactive", "learned"])
def simulated(request, tmp_path_factory):
    profile = sim.ParticipantProfile(return_mode=request.param)
    result, ref = sim_runs.simulate_run(tmp_path_factory.mktemp(f"sim_{request.param}"), profile, seed=1, n_trials=150)
    return request.param, result, an.analyze_run(ref)


def test_gates_agree_with_the_simulators_truth_on_both_legs(simulated):
    mode, result, run = simulated
    assert run.return_jump
    agree = sim_runs.agreement(result, run)
    for leg in an.LEGS:
        a = agree[leg]
        assert a["n_saccade"] >= 3 and a["n_fixation"] >= 5, (mode, leg, a)
        assert a["saccade_true"] >= 0.95 and a["fixation_true"] >= 0.95, (mode, leg, a)  # gate labels match the truth
        assert a["as_fixation"] == 0
    # in-flight flashes lost to the pre-specified gates (mostly the first/last ~10 ms of flight, below MIN_FLASH_SPEED); pooled,
    # since the out leg holds only ~10 truly in-flight flashes
    missed, n_true = (sum(agree[leg][k] for leg in an.LEGS) for k in ("missed", "n_true"))
    assert missed <= 0.35 * n_true, (mode, agree)
    assert agree["return"]["n_saccade"] > agree["out"]["n_saccade"]  # most flashes are timed for the return leg


def test_trace_return_latency_matches_the_simulated_onsets(simulated):
    mode, result, run = simulated
    pairs = np.asarray(sim_runs.true_return_latencies(result, run))
    assert len(pairs) >= 0.95 * run.return_latency["n"]
    err = pairs[:, 1] - pairs[:, 0]
    assert 0 <= np.median(err) < 10 and np.abs(err).max() < 20  # detection sits a few ms after the true onset
    r = run.return_latency
    assert r["no_return_found"] == 0 and r["median_ms"] == pytest.approx(np.median(pairs[:, 0]), abs=10)
    if mode == "reactive":
        assert r["share_predictive"] == 0.0
    else:
        assert r["median_ms"] < 100 and r["share_predictive"] > 0  # learned returns start sooner, some before the jump
