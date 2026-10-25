"""The simulated participant's optional reaction-time realism (ParticipantProfile.reaction_* /
anticipation_* / lapse_* fields, the `realistic` preset) and its recorded ground truth."""

import random
import statistics

import pytest
import sim_tools
from invariants import check_session

from src.experiment.simulation import (
    REALISTIC_RT,
    ParticipantProfile,
    SimulatedGazeSource,
    SimulatedParticipant,
    simulate_session,
)


def _participant(profile: ParticipantProfile, seed: int = 0) -> SimulatedParticipant:
    return SimulatedParticipant(None, SimulatedGazeSource(), profile, random.Random(seed), [])  # type: ignore[arg-type]


def _draws(profile: ParticipantProfile, n: int, seed: int = 0, rightward: bool = True):
    """n consecutive latencies from _draw_reaction_ms, with cues recorded as the session would."""
    p = _participant(profile, seed)
    out = []
    for _ in range(n):
        latency, true_mean, kind = p._draw_reaction_ms(rightward)
        p.cues.append(type("C", (), {})())  # only len(cues) is read
        out.append((latency, true_mean, kind))
    return out


def test_defaults_reproduce_the_old_plain_gaussian_exactly():
    profile = ParticipantProfile()
    legacy = random.Random(7)
    ours = _participant(profile, 7)
    for _ in range(50):
        expected = max(0.12, legacy.gauss(profile.reaction_ms_mean, profile.reaction_ms_sd)) / 1000
        latency, true_mean, kind = ours._draw_reaction_ms(True)
        ours.cues.append(None)  # type: ignore[arg-type]
        assert latency / 1000 == expected and true_mean == profile.reaction_ms_mean and kind == "normal"


def test_default_session_is_unchanged_and_records_ground_truth():
    a, b = simulate_session(8, seed=3), simulate_session(8, seed=3)
    assert [r.reaction_latency_ms for r in a.results] == [r.reaction_latency_ms for r in b.results]
    assert a.cues and all(c.kind == "normal" and c.true_mean_ms == 230.0 for c in a.cues)
    row = a.saccade_results[0]
    assert a.trial_cues[(row.phase, row.index)] in a.cues


def test_deterministic_per_seed_and_seed_matters():
    prof = ParticipantProfile.realistic()
    lat = lambda seed: [c.latency_ms for c in simulate_session(12, prof, seed=seed).cues]  # noqa: E731
    assert lat(2) == lat(2)
    assert lat(2) != lat(3)


def test_tail_keeps_the_mean_and_skews_right():
    xs = [d[0] for d in _draws(ParticipantProfile(reaction_ms_sd=30, reaction_ms_tau=60), 4000)]
    assert statistics.fmean(xs) == pytest.approx(230, abs=5)
    assert statistics.stdev(xs) == pytest.approx((30**2 + 60**2) ** 0.5, rel=0.08)
    assert statistics.fmean(xs) > statistics.median(xs) + 8  # right skew


def test_drift_moves_the_true_mean_and_the_latencies():
    draws = _draws(ParticipantProfile(reaction_drift_ms_per_trial=0.5, reaction_ms_sd=20), 400)
    assert draws[100][1] == pytest.approx(230 + 50) and draws[0][1] == 230
    assert statistics.fmean(d[0] for d in draws[300:]) - statistics.fmean(d[0] for d in draws[:100]) > 100


def test_wave_is_periodic():
    draws = _draws(ParticipantProfile(reaction_wave_ms=30, reaction_wave_period_trials=40), 41)
    assert draws[10][1] == pytest.approx(260) and draws[30][1] == pytest.approx(200) and draws[40][1] == pytest.approx(230)


def _lag1(profile: ParticipantProfile) -> float:
    dev = [d[0] - d[1] for d in _draws(profile, 3000)]
    m = statistics.fmean(dev)
    return sum((a - m) * (b - m) for a, b in zip(dev, dev[1:])) / sum((a - m) ** 2 for a in dev)


def test_autocorrelation_sign_and_size():
    base = dict(reaction_ms_sd=30, reaction_ms_mean=300)  # far above the 120 ms floor, so no clipping
    assert _lag1(ParticipantProfile(**base)) == pytest.approx(0, abs=0.06)
    assert _lag1(ParticipantProfile(**base, reaction_autocorr=0.6)) == pytest.approx(0.6, abs=0.06)
    assert _lag1(ParticipantProfile(**base, reaction_autocorr=-0.5)) == pytest.approx(-0.5, abs=0.06)


def test_anticipation_and_lapse_rates_and_ranges():
    draws = _draws(ParticipantProfile(anticipation_rate=0.1, lapse_rate=0.05, reaction_ms_sd=20), 5000)
    kinds = [d[2] for d in draws]
    assert kinds.count("anticipation") / 5000 == pytest.approx(0.1, abs=0.02)
    assert kinds.count("lapse") / 5000 == pytest.approx(0.05, abs=0.015)
    assert all(80 <= d[0] <= 130 for d in draws if d[2] == "anticipation")
    assert all(d[0] >= 230 - 100 + 300 for d in draws if d[2] == "lapse")  # normal part (>= ~mean-5sd) + 300..600


def test_direction_offset_splits_around_the_mean():
    right = _draws(ParticipantProfile(reaction_dir_offset_ms=20), 1, rightward=True)[0][1]
    left = _draws(ParticipantProfile(reaction_dir_offset_ms=20), 1, rightward=False)[0][1]
    assert (right, left) == (240, 220)


def test_each_default_is_off_and_preset_values():
    p = ParticipantProfile()
    assert not any((p.reaction_ms_tau, p.reaction_drift_ms_per_trial, p.reaction_wave_ms, p.reaction_autocorr,
                    p.anticipation_rate, p.lapse_rate, p.reaction_dir_offset_ms))
    r = ParticipantProfile.realistic(lapse_rate=0.0)
    assert r.reaction_ms_mean == 220 and r.reaction_ms_tau == 50 and r.anticipation_rate == 0.03 and r.lapse_rate == 0
    assert sim_tools.make_profile("realistic") == ParticipantProfile(**REALISTIC_RT)


def test_realistic_preset_runs_end_to_end_and_passes_invariants():
    sim = sim_tools.simulate("realistic", trials=150, seed=1, test_mode=False)
    assert len(sim.saccade_results) == 150
    assert not check_session(sim, max_trials_cap=200)
    summary = sim_tools.summarize(sim)
    assert summary["avg_rt_abs_err_ms"] is not None and summary["true_rt_mean_ms"] > 220  # drift pushed the true mean up
    assert {c.kind for c in sim.cues} >= {"normal"}
