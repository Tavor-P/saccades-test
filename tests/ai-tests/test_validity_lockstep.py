"""The live ZEST update and the end-of-run analysis graph must apply the SAME validity gate
(CLAUDE.md: 'the two intentionally stay in lockstep'). Bug class: changing one side's rule
(e.g. adding tracking_ok) without the other, so the in-session estimate and the graph disagree."""

import pytest

from src.experiment.results_graph import _exclude_flashes_not_during_saccade
from src.experiment.scoring import is_valid_for_saccadic_analysis
from src.experiment.simulation import ParticipantProfile, simulate_session
from src.experiment.zest import ZestStaircase

PROFILES = [
    ParticipantProfile(),
    ParticipantProfile(dropout_rate_per_s=2.0),  # produces tracking_ok=False rows
    ParticipantProfile(tracker_latency_ms=0.0, reaction_ms_sd=10.0),  # produces more flash_during_saccade=True rows
]


@pytest.mark.parametrize("profile", PROFILES, ids=["default", "dropouts", "low_latency"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_zest_feed_matches_the_graph_filter(monkeypatch, profile, seed):
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

    # What the graph keeps among flashed trials is exactly the valid ones; catch trials untouched.
    kept = _exclude_flashes_not_during_saccade(rows)
    kept_flashed = [r for r in kept if r.grating_shown]
    assert kept_flashed == [r for r in rows if r.grating_shown and is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok)]
    assert [r for r in kept if not r.grating_shown] == [r for r in rows if not r.grating_shown]


def test_the_scenarios_above_actually_exercise_both_valid_and_invalid_flashes():
    """Guard against the lockstep test going vacuous: we need both kinds of rows to exist somewhere."""
    rows = [r for p in PROFILES[1:] for s in range(2) for r in simulate_session(max_trials=25, profile=p, seed=s).saccade_results if r.grating_shown]
    valid = [r for r in rows if is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok)]
    assert valid and len(valid) < len(rows)
    assert any(r.tracking_ok is False for r in rows)
