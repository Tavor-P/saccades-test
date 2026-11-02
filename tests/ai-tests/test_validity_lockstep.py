"""The live ZEST update must feed the staircase exactly the trials the shared validity gate (scoring.is_valid_for_saccadic_analysis)
keeps. Bug class: a live gate that forgets a condition (e.g. tracking_ok) the shared rule has."""

import pytest

from src.experiment.scoring import is_valid_for_saccadic_analysis
from src.experiment.simulation import ParticipantProfile, simulate_session
from src.experiment.zest import ZestStaircase

pytestmark = pytest.mark.usefixtures("staircase_no_jitter")  # these tests drive the live staircase

PROFILES = [
    ParticipantProfile(),
    ParticipantProfile(dropout_rate_per_s=2.0),  # produces tracking_ok=False rows
    ParticipantProfile(tracker_latency_ms=0.0, reaction_ms_sd=10.0),  # produces more flash_during_saccade=True rows
]


@pytest.mark.parametrize("profile", PROFILES, ids=["default", "dropouts", "low_latency"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_zest_feed_matches_the_validity_gate(monkeypatch, profile, seed):
    fed: list[float] = []
    original_update = ZestStaircase.update

    def spy(self, contrast, detected):
        fed.append(contrast)
        return original_update(self, contrast, detected)

    monkeypatch.setattr(ZestStaircase, "update", spy)
    sim = simulate_session(max_trials=25, profile=profile, seed=seed)
    rows = sim.saccade_results

    # What the live session should have fed ZEST: flashed, answered-or-missed, valid trials.
    expected = [
        r.contrast
        for r in rows
        if r.grating_shown
        and r.outcome in ("correct", "incorrect", "miss")
        and is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok)
    ]
    assert fed == expected


def test_the_scenarios_above_actually_exercise_both_valid_and_invalid_flashes():
    """Guard against the lockstep test going vacuous: we need both kinds of rows to exist somewhere."""
    rows = [r for p in PROFILES[1:] for s in range(6) for r in simulate_session(max_trials=25, profile=p, seed=s).saccade_results if r.grating_shown]
    valid = [r for r in rows if is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok)]
    assert valid and len(valid) < len(rows)
    assert any(r.tracking_ok is False for r in rows)
