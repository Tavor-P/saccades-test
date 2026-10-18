"""Fixed-seed campaign: the real session, hostile and random participants, all invariants.
A bigger random campaign: `python tests/ai-tests/find_bugs.py --sessions 300`."""

import random

import pytest
from find_bugs import EXTREME_PROFILES, random_profile
from invariants import check_session

from src.experiment import simulation
from src.experiment.simulation import ParticipantProfile, simulate_session


@pytest.mark.parametrize("name", sorted(EXTREME_PROFILES))
def test_hostile_participants_terminate_cleanly_with_valid_rows(name):
    sim = simulate_session(max_trials=10, profile=ParticipantProfile(**EXTREME_PROFILES[name]), seed=1)
    assert check_session(sim) == []


@pytest.mark.parametrize("seed", range(12))
def test_random_participants_produce_valid_rows(seed):
    rng = random.Random(1000 + seed)
    sim = simulate_session(max_trials=10, profile=ParticipantProfile(**random_profile(rng)), seed=seed)
    assert check_session(sim) == []


def test_a_completely_dead_tracker_waits_for_signal_instead_of_crashing(monkeypatch):
    """No gaze at all: calibration can't fill its sample window, so the session just waits for a
    usable signal. The simulator must end that with its explicit 'did not finish' error (here
    after 5 simulated seconds) - not an unrelated exception, and not an endless loop."""
    monkeypatch.setattr(simulation, "_MAX_SIM_SECONDS", 5.0)
    profile = ParticipantProfile(dropout_rate_per_s=1e6, dropout_ms=1e6)
    with pytest.raises(RuntimeError, match="did not finish"):
        simulate_session(max_trials=3, profile=profile, seed=0)


def test_full_sessions_reach_completion_and_respect_the_trial_cap():
    sim = simulate_session(max_trials=10_000, seed=3)
    assert sim.session.render_state()["hud"]["phase"] == "COMPLETE"
    assert check_session(sim) == []
