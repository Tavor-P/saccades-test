import subprocess
import sys
import time

import pytest

from src.experiment.simulation import ParticipantProfile, simulate_session, simulate_trial


def test_simulation_never_imports_a_graphics_stack():
    code = "import sys, src.experiment.simulation as s; s.simulate_trial(seed=1); print('psychopy' in sys.modules, 'pyglet' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_simulation_does_not_depend_on_real_time(monkeypatch):
    def boom():
        raise AssertionError("simulation read the real monotonic clock")

    monkeypatch.setattr(time, "monotonic", boom)
    assert simulate_trial(seed=2).saccade_results


def test_one_trial_completes_with_a_logged_result():
    sim = simulate_trial(seed=0)
    assert len(sim.saccade_results) == 1
    result = sim.saccade_results[0]
    assert result.outcome in {"correct", "incorrect", "miss", "false_alarm", "correct_rejection", "timeout"}
    assert result.reaction_latency_ms is not None  # classifier saw the saccade
    assert any("go-cue" in text for _, text in sim.events)


def test_same_seed_is_deterministic_and_different_seeds_differ():
    a = simulate_session(max_trials=6, seed=5)
    b = simulate_session(max_trials=6, seed=5)
    c = simulate_session(max_trials=6, seed=6)
    key = lambda s: [(r.outcome, r.contrast, r.reaction_latency_ms) for r in s.saccade_results]
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_global_random_state_is_restored():
    import random

    random.seed(123)
    expected = random.random()
    random.seed(123)
    simulate_trial(seed=9)
    assert random.random() == expected


def test_reaction_latency_tracks_the_participants_reaction_time():
    fast = simulate_session(max_trials=8, seed=1, profile=ParticipantProfile(reaction_ms_mean=180, reaction_ms_sd=5))
    slow = simulate_session(max_trials=8, seed=1, profile=ParticipantProfile(reaction_ms_mean=320, reaction_ms_sd=5))
    mean = lambda s: sum(r.reaction_latency_ms for r in s.saccade_results if r.reaction_latency_ms) / len(s.saccade_results)
    assert mean(slow) - mean(fast) == pytest.approx(140, abs=30)


def test_tracking_dropouts_get_flagged_as_not_ok():
    sim = simulate_session(max_trials=25, seed=3, profile=ParticipantProfile(dropout_rate_per_s=4.0))
    assert any(r.tracking_ok is False for r in sim.saccade_results)
    clean = simulate_session(max_trials=25, seed=3)
    assert all(r.tracking_ok is not False for r in clean.saccade_results)
