"""Does the bug-finding machinery actually find bugs? Each test re-introduces a plausible
defect (a 'mutation') into the real code and asserts the invariant checker or the
simulation notices. If one of these starts passing silently, the checker got weaker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from invariants import check_session  # noqa: E402
from src.experiment import session as session_module  # noqa: E402
from src.experiment.simulation import simulate_session  # noqa: E402
from src.experiment.zest import ZestStaircase  # noqa: E402


def _violations(trials=14, seed=2):
    return check_session(simulate_session(max_trials=trials, seed=seed))


def test_baseline_is_clean():
    assert _violations() == []


def test_catches_scoring_that_always_says_correct(monkeypatch):
    monkeypatch.setattr(session_module, "score_outcome", lambda *a, **k: "correct")
    assert any("correct" in v for v in _violations())


def test_catches_a_staircase_that_returns_impossible_contrast(monkeypatch):
    monkeypatch.setattr(ZestStaircase, "next_contrast", lambda self: 7.0)
    assert any("contrast out of range" in v for v in _violations())


def test_catches_flash_validity_being_recorded_for_catch_trials(monkeypatch):
    monkeypatch.setattr(session_module.ExperimentSession, "_compute_flash_during_saccade", lambda self: True)
    assert any("no grating shown but flash_during_saccade" in v for v in _violations(trials=20))


def test_catches_a_response_time_measured_from_the_wrong_origin(monkeypatch):
    original = session_module.ExperimentSession.on_response_key

    def buggy(self, orientation):
        original(self, orientation)
        if self._responded and self._response_time_ms is not None:
            self._response_time_ms = -self._response_time_ms - 5  # e.g. subtracting the wrong way round

    monkeypatch.setattr(session_module.ExperimentSession, "on_response_key", buggy)
    assert any("response_time_ms" in v for v in _violations(trials=20))


def test_catches_a_session_that_never_stops(monkeypatch):
    monkeypatch.setattr(session_module.ExperimentSession, "_should_stop_main_block", lambda self: False)
    sim = simulate_session(max_trials=50, seed=1)
    assert any("exceed the safety cap" in v for v in check_session(sim))


def test_catches_onset_detection_reporting_before_the_debounce(monkeypatch):
    from include.experiment import constants

    monkeypatch.setattr("src.experiment.trial_mechanics.SACCADE_ONSET_STABILITY_MS", 400)  # detector waits 400 ms...
    # ...while the invariant still believes in the configured debounce window.
    assert any("onset_detection_lag_ms" in v for v in _violations(trials=10))
    assert constants.SACCADE_ONSET_STABILITY_MS < 400


# ---- Past bugs: re-introduce each one and confirm a regression test goes red -------------------------
# (If one of these stops raising, the regression test it points at has become toothless.)
# Not covered on purpose: the GazeMapper deque 'race' - in CPython it could not be made to fail even
# with the lock removed (0/8 stress attempts), so there is no honest red/green test for it.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ - where the regression tests live


def test_would_have_caught_the_stale_stream_bug(monkeypatch):
    """Bug: when the Neon stream died, latest_sample() kept returning the last gaze as valid."""
    import test_neon_source
    from src.eye_tracking import neon_source

    monkeypatch.setattr(neon_source, "NEON_STALE_TIMEOUT_S", 1e9)  # = no staleness detection
    with pytest.raises(AssertionError):
        test_neon_source.test_dead_stream_reports_no_gaze_instead_of_repeating_last_sample(monkeypatch)


def test_would_have_caught_the_clock_offset_sign_bug(monkeypatch):
    """Bug: latency came out ~400 ms because the clock offset was applied with the wrong sign."""
    import test_neon_source
    from src.eye_tracking.neon_source import NeonGazeSource

    original = NeonGazeSource._estimate_clock_offset

    def wrong_sign(self):
        original(self)
        if self._clock_offset_ms is not None:
            self._clock_offset_ms = -self._clock_offset_ms

    monkeypatch.setattr(NeonGazeSource, "_estimate_clock_offset", wrong_sign)
    with pytest.raises(AssertionError):
        test_neon_source.test_latency_uses_clock_offset_with_correct_sign(monkeypatch)


def test_would_have_caught_the_graph_ignoring_tracking_ok(monkeypatch):
    """Bug class: the live ZEST gate and the analysis graph drifting apart."""
    import test_validity_lockstep
    from src.experiment import results_graph
    from src.experiment.simulation import ParticipantProfile

    monkeypatch.setattr(results_graph, "is_valid_for_saccadic_analysis", lambda flash, tracking_ok=None: flash is True)
    with pytest.raises(AssertionError):
        test_validity_lockstep.test_live_zest_feed_matches_the_graph_filter(
            monkeypatch, ParticipantProfile(dropout_rate_per_s=2.0), 1
        )


def test_would_have_caught_a_live_gate_that_forgets_tracking_ok(monkeypatch):
    import test_validity_lockstep
    from src.experiment.simulation import ParticipantProfile

    monkeypatch.setattr(session_module, "is_valid_for_saccadic_analysis", lambda flash, tracking_ok=None: flash is True)
    with pytest.raises(AssertionError):
        test_validity_lockstep.test_live_zest_feed_matches_the_graph_filter(
            monkeypatch, ParticipantProfile(dropout_rate_per_s=2.0), 1
        )
