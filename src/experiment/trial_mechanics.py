"""Small shared building blocks for session logic: zone_for, OnsetDetector (THE saccade-onset debounce - reuse it, don't hand-roll), ShuffledBag, average_ms."""

import random
import statistics
from collections import deque
from typing import Generic, Sequence, TypeVar

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    ONSET_BASELINE_NEAR_SOURCE,
    ONSET_BASELINE_WINDOW_MS,
    RT_MIN_PLAUSIBLE_MS,
    RT_OUTLIER_MAD_FLOOR_MS,
    RT_OUTLIER_MAD_K,
    SACCADE_ONSET_DISPLACEMENT,
    SACCADE_ONSET_STABILITY_MS,
)
from include.experiment.types import Orientation, Target


# GazeSample.position scale: 0 = the dot (left target), 1 = the cross (right target) - see GazeMapper.
LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION = 0.0, 0.5, 1.0


def zone_for(target: Target) -> GazeZone:
    """Which GazeZone a fixation on `target` counts as - the dot is always
    the left-hand target, the cross always the right, in both the real
    saccade trials and the tutorial's gaze-practice/dress-rehearsal stages."""
    return GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT


def random_orientation() -> Orientation:
    return random.choice([Orientation.VERTICAL, Orientation.HORIZONTAL])


def average_ms(samples: Sequence[float], default: float) -> float:
    """Mean of `samples`, or `default` if empty - shared by the reaction-time
    test's initial average (ExperimentSession and TutorialSession both fall
    back to DEFAULT_REACTION_TIME_MS if every attempt timed out) and
    ExperimentSession's in-session rolling-average recompute (falls back to
    the current average unchanged if the whole window was timeouts), so
    "empty means fall back to X" can't drift between call sites duplicating
    it independently."""
    return sum(samples) / len(samples) if samples else default


def robust_average_ms(samples: Sequence[float], default: float) -> float:
    """Reaction-time average that one bad measurement can't move: drops
    anticipations (< RT_MIN_PLAUSIBLE_MS), then slow outliers (more than
    RT_OUTLIER_MAD_K scaled MADs above the median, MAD floored at
    RT_OUTLIER_MAD_FLOOR_MS), and returns the median of the rest - or `default`
    if nothing plausible is left. Used for avg_reaction_time_ms, which
    schedules every flash."""
    plausible = [s for s in samples if s >= RT_MIN_PLAUSIBLE_MS]
    if not plausible:
        return default
    median = statistics.median(plausible)
    mad = statistics.median(abs(s - median) for s in plausible) * 1.4826
    limit = median + RT_OUTLIER_MAD_K * max(mad, RT_OUTLIER_MAD_FLOOR_MS)
    return statistics.median(s for s in plausible if s <= limit)


class OnsetDetector:
    """Detects "gaze left the source symbol" as a displacement of the calibrated
    gaze position: fires once `sample.position` has moved away from the
    fixation baseline at the source by more than SACCADE_ONSET_DISPLACEMENT
    (position units: 0 = dot, 1 = cross) and stayed beyond it for
    SACCADE_ONSET_STABILITY_MS of sample capture time. `since` is the first
    sample that exceeded the threshold, i.e. the onset estimate, so the
    debounce delays confirmation but not the reported onset.

    This replaces a zone-crossing rule ("outside the source zone, CENTER
    included") that only noticed the movement after ~40% of the travel
    (GAZE_DEAD_ZONE) - a real saccade lasts 30-35 ms in total. The baseline is
    the median of the last ONSET_BASELINE_WINDOW_MS of samples that were still
    near the source (within ONSET_BASELINE_NEAR_SOURCE of it and not yet past
    the threshold), so a fixation offset or slow drift is subtracted but the
    movement itself never drags the baseline along. The first near-source
    sample seeds it at once (the participant is already fixating the source
    when the detector is created); with none yet, the nominal source position
    (0 or 1) is used. A sample without a face/position resets the detector
    (a blink is not an onset, and the baseline is stale afterwards).

    Used in four places (ExperimentSession's RT-test and main trials,
    TutorialSession's scoped-down mirrors of both) - one tested implementation
    here instead of four hand-copied ones. `avoid_zone` LEFT/RIGHT picks the
    source side; any other value means "away from the middle" in either direction.

    `since` is set the first tick gaze is beyond the threshold and reset to None
    on any tick it isn't - readable at any time, not just once confirmed.
    `confirmed` is the flag. Once `confirmed` is True, the caller is expected to
    stop calling `update()` (matching every call site's `if not <confirmed>:`
    gate) so `since` freezes at the sample that actually triggered confirmation.
    """

    def __init__(self, avoid_zone: GazeZone) -> None:
        self._source_position = {GazeZone.LEFT: LEFT_POSITION, GazeZone.RIGHT: RIGHT_POSITION}.get(avoid_zone, CENTER_POSITION)
        # +1: moving right (increasing position) leaves the source; -1: moving left does; 0: either way.
        self._direction = {GazeZone.LEFT: 1, GazeZone.RIGHT: -1}.get(avoid_zone, 0)
        self._baseline: deque[tuple[float, float]] = deque()  # (time, position) of recent near-source samples
        self.since: float | None = None
        self.confirmed = False

    def _baseline_position(self) -> float:
        if not self._baseline:
            return self._source_position
        return statistics.median(p for _, p in self._baseline)

    def _displacement(self, position: float, baseline: float) -> float:
        delta = position - baseline
        return abs(delta) if self._direction == 0 else delta * self._direction

    def update(self, sample, now: float) -> bool:
        """Call once per tick while not yet confirmed. Returns True on the
        exact tick confirmation fires. `now` should be the sample's own
        capture time on the session clock (see ExperimentSession._sample_time),
        not the render tick's - then `since` and the debounce resolve at the
        tracker's sample rate; re-reading the same sample is harmless."""
        position = sample.position
        if not sample.face_found or position is None:
            self.since = None
            self._baseline.clear()
            return False
        if not self._baseline and abs(position - self._source_position) <= ONSET_BASELINE_NEAR_SOURCE:
            # First near-source sample: it IS the fixation (whatever calibration offset it has), nothing to compare yet.
            self._baseline.append((now, position))
            return False
        baseline = self._baseline_position()
        if self._displacement(position, baseline) > SACCADE_ONSET_DISPLACEMENT:
            if self.since is None:
                self.since = now
            if now - self.since >= SACCADE_ONSET_STABILITY_MS / 1000 - 1e-9:  # epsilon: 0.21 - 0.20 < 0.01 in floats
                self.confirmed = True
                return True
            return False
        self.since = None
        if abs(position - self._source_position) <= ONSET_BASELINE_NEAR_SOURCE:
            self._baseline.append((now, position))
        cutoff = now - ONSET_BASELINE_WINDOW_MS / 1000
        while self._baseline and self._baseline[0][0] < cutoff:
            self._baseline.popleft()
        return False


_T = TypeVar("_T")


class ShuffledBag(Generic[_T]):
    """Hands out every item from `pool` in a random order before any repeats,
    reshuffling a fresh copy of the pool once exhausted - roughly even
    coverage of the pool over many draws (e.g. TIMING_OFFSETS_MS) without a
    predictable fixed cycle a participant could anticipate. A draw at the
    seam between two shuffles can legitimately repeat the previous one - this
    only guarantees no repeats *within* one pass through the pool, not across
    reshuffles."""

    def __init__(self, pool: Sequence[_T]) -> None:
        self._pool = list(pool)
        self._remaining: deque[_T] = deque()

    def draw(self) -> _T:
        if not self._remaining:
            shuffled = list(self._pool)
            random.shuffle(shuffled)
            self._remaining = deque(shuffled)
        return self._remaining.popleft()
