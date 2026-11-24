"""Does the bug-finding machinery actually find bugs? Each test re-introduces a plausible
defect (a 'mutation') into the real code and asserts the invariant checker or the
simulation notices. If one of these starts passing silently, the checker got weaker."""

import pytest
from invariants import check_session

from src.experiment import session as session_module
from src.experiment.simulation import simulate_session


def _violations(trials=14, seed=2, **kwargs):
    return check_session(simulate_session(max_trials=trials, seed=seed, **kwargs))


@pytest.mark.parametrize("return_jump", [True, False])
def test_baseline_is_clean(return_jump):
    assert _violations(return_jump=return_jump) == []


def test_catches_scoring_that_always_says_correct(monkeypatch):
    monkeypatch.setattr(session_module, "score_outcome", lambda *a, **k: "correct")
    assert any("correct" in v for v in _violations(trials=30))


def test_catches_a_flash_at_a_contrast_outside_the_fixed_set(monkeypatch):
    original = session_module.ShuffledBag.draw

    def buggy(self):
        value = original(self)
        return 7.0 if isinstance(value, float) else value  # contrasts are floats, timing offsets ints

    monkeypatch.setattr(session_module.ShuffledBag, "draw", buggy)
    assert any("contrast out of range" in v for v in _violations())


def test_catches_a_response_time_measured_from_the_wrong_origin(monkeypatch):
    original = session_module.ExperimentSession.on_response_key

    def buggy(self, line_side):
        original(self, line_side)
        if self._responded and self._response_time_ms is not None:
            self._response_time_ms = -self._response_time_ms - 5  # e.g. subtracting the wrong way round

    monkeypatch.setattr(session_module.ExperimentSession, "on_response_key", buggy)
    assert any("response_time_ms" in v for v in _violations(trials=20))


def test_catches_a_session_that_never_stops(monkeypatch):
    monkeypatch.setattr(session_module.ExperimentSession, "_should_stop_main_block", lambda self: False)
    sim = simulate_session(max_trials=50, seed=1)
    assert any("exceed the safety cap" in v for v in check_session(sim))


def test_catches_onset_detection_reporting_before_the_debounce(monkeypatch):
    monkeypatch.setattr("src.experiment.trial_mechanics.SACCADE_ONSET_STABILITY_MS", 400)  # detector waits 400 ms...
    # ...while the invariant still believes in the configured debounce window. Return jump off: with it on, a 400 ms
    # debounce outlasts the landing debounce, so the jump back ends out-leg onset detection before it could report.
    assert any("onset_detection_lag_ms" in v for v in _violations(trials=10, return_jump=False))


# ---- Past bugs: re-introduce each one and confirm a regression test goes red -------------------------
# (If one of these stops raising, the regression test it points at has become toothless.)
# Not covered on purpose: the GazeMapper deque 'race' - in CPython it could not be made to fail even
# with the lock removed (0/8 stress attempts), so there is no honest red/green test for it.

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
        test_neon_source.test_samples_are_stamped_with_capture_time_using_the_offset_with_the_right_sign(monkeypatch)
