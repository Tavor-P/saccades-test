import subprocess
import sys
import time

import pytest

from src import timebase
from src.experiment.simulation import ParticipantProfile, simulate_session, simulate_trial


def test_simulation_never_imports_a_graphics_stack():
    code = "import sys, src.experiment.simulation as s; s.simulate_trial(seed=1); print('psychopy' in sys.modules, 'pyglet' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_one_trial_completes_without_real_time_and_restores_the_global_random_state(monkeypatch):
    import random

    random.seed(123)
    expected = random.random()
    random.seed(123)

    def boom():
        raise AssertionError("simulation read the real clock")

    monkeypatch.setattr(timebase, "_clock", boom)
    monkeypatch.setattr(timebase, "_wall_clock", boom)
    monkeypatch.setattr(time, "monotonic", boom)
    monkeypatch.setattr(time, "perf_counter", boom)
    sim = simulate_trial(seed=0)
    assert random.random() == expected
    assert len(sim.saccade_results) == 1
    result = sim.saccade_results[0]
    assert result.outcome in {"correct", "incorrect", "miss", "timeout"}
    assert result.reaction_latency_ms is not None  # classifier saw the saccade
    assert any("go-cue" in text for _, text in sim.events)


def test_seeds_are_deterministic_and_profile_reaction_time_and_dropouts_show_up():
    a = simulate_session(max_trials=6, seed=5)
    b = simulate_session(max_trials=6, seed=5)
    c = simulate_session(max_trials=6, seed=6)
    def key(s):
        return [(r.outcome, r.contrast, r.reaction_latency_ms) for r in s.saccade_results]
    assert key(a) == key(b)
    assert key(a) != key(c)

    fast = simulate_session(max_trials=8, seed=1, profile=ParticipantProfile(reaction_ms_mean=180, reaction_ms_sd=5))
    slow = simulate_session(max_trials=8, seed=1, profile=ParticipantProfile(reaction_ms_mean=320, reaction_ms_sd=5))
    def mean(s):
        return sum(r.reaction_latency_ms for r in s.saccade_results if r.reaction_latency_ms) / len(s.saccade_results)
    assert mean(slow) - mean(fast) == pytest.approx(140, abs=30)

    sim = simulate_session(max_trials=25, seed=3, profile=ParticipantProfile(dropout_rate_per_s=4.0))
    assert any(r.tracking_ok is False for r in sim.saccade_results)
    clean = simulate_session(max_trials=25, seed=3)
    assert all(r.tracking_ok is not False for r in clean.saccade_results)


def test_no_catch_trials_every_simulated_trial_flashes():
    for seed in range(6):
        sim = simulate_session(max_trials=12, seed=seed, skip_practice_trials=False)
        rows = [r for r in sim.saccade_results if r.phase == "saccade"]
        assert len(rows) >= 12, seed
        assert all(r.flash_shown for r in rows), seed
        assert not any(r.outcome in ("false_alarm", "correct_rejection") for r in rows), seed
