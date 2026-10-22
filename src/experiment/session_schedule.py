"""Session-level scheduling decisions, PsychoPy-free: which phase runs first (counterbalanced by participant ID), when a rest break is due, and when
the wall-clock session cap is hit. Time comes in through an injected `now` callable (the runner passes time.monotonic) so tests control it."""

import zlib
from collections.abc import Callable

PHASE_BASELINE = "baseline"
PHASE_SACCADE = "saccade"

# Candidates to move into include/experiment/constants.py (kept here so this change touches no shared file).
REST_BREAK_EVERY_S = 6 * 60  # active wall time between scheduled rest breaks (real runs only)
MIN_REST_BREAK_S = 20  # SPACE is ignored this long into a break, so a stray keypress can't skip it
MAX_SESSION_S = 60 * 60  # wall-clock cap for the whole run, rest breaks and pauses included (real runs only)


def phase_order(participant_id: str, test_mode: bool) -> tuple[str, str]:
    """Test runs always go saccade-first (the phase under development). Real runs alternate by participant ID so that order effects (practice,
    fatigue) are balanced across participants: digit-only IDs alternate by parity (consecutive IDs get opposite orders), other IDs by a stable
    hash - never Python's salted hash(), which would differ between runs for the same ID."""
    if test_mode:
        return PHASE_SACCADE, PHASE_BASELINE
    pid = participant_id.strip()
    bucket = int(pid) % 2 if pid.isdigit() else zlib.crc32(pid.lower().encode("utf8")) % 2
    return (PHASE_BASELINE, PHASE_SACCADE) if bucket == 0 else (PHASE_SACCADE, PHASE_BASELINE)


class SessionSchedule:
    """Wall-clock bookkeeping for rest breaks and the session cap. Pass None for an interval/cap to disable it (test runs). The runner decides
    *when* it is safe to act (only on entry to a main trial's foreperiod), this only says whether a break/stop is due."""

    def __init__(
        self,
        now: Callable[[], float],
        rest_every_s: float | None = REST_BREAK_EVERY_S,
        min_rest_s: float = MIN_REST_BREAK_S,
        max_session_s: float | None = MAX_SESSION_S,
    ) -> None:
        self._now = now
        self._rest_every_s = rest_every_s
        self._min_rest_s = min_rest_s
        self._max_session_s = max_session_s
        self._started_at = now()
        self._active_since = self._started_at  # start of the current stretch of work (session start or end of the last break)
        self._rest_started_at: float | None = None
        self.breaks_taken = 0

    def cap_reached(self) -> bool:
        return self._max_session_s is not None and self._now() - self._started_at >= self._max_session_s

    def rest_due(self) -> bool:
        if self._rest_every_s is None or self._rest_started_at is not None:
            return False
        return self._now() - self._active_since >= self._rest_every_s

    def begin_rest(self) -> None:
        self._rest_started_at = self._now()
        self.breaks_taken += 1

    @property
    def resting(self) -> bool:
        return self._rest_started_at is not None

    def rest_may_end(self) -> bool:
        return self._rest_started_at is not None and self._now() - self._rest_started_at >= self._min_rest_s

    def end_rest(self) -> None:
        self._rest_started_at = None
        self._active_since = self._now()
