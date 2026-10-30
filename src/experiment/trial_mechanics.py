"""Small shared building blocks for session logic: zone_for, OnsetDetector (THE saccade-onset debounce - reuse it, don't hand-roll), ShuffledBag, RollingReactionTime."""

import random
import statistics
from collections import deque
from typing import Generic, Sequence, TypeVar

from include.eye_tracking.constants import CENTER_POSITION, LEFT_POSITION, RIGHT_POSITION
from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    LINE_OFFSET_HEIGHT_UNITS,
    ONSET_BASELINE_NEAR_SOURCE,
    ONSET_BASELINE_WINDOW_MS,
    RT_AVERAGE_ROLLING_WINDOW,
    RT_MIN_PLAUSIBLE_MS,
    RT_MIN_SAMPLES_FOR_AVERAGE,
    RT_OUTLIER_MAD_FLOOR_MS,
    RT_OUTLIER_MAD_K,
    SACCADE_ONSET_DISPLACEMENT,
    SACCADE_ONSET_STABILITY_MS,
)
from include.experiment.types import Side, Target



def zone_for(target: Target) -> GazeZone:
    """Which GazeZone a fixation on `target` counts as - the dot is always
    the left-hand target, the cross always the right, in both the real
    saccade trials and the tutorial's practice trials."""
    return GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT


def random_orientation() -> Side:
    """A random line side (the trial field is still called `orientation`, see Side)."""
    return random.choice([Side.ABOVE, Side.BELOW])


def line_position(reference: tuple[float, float], side: Side) -> tuple[float, float]:
    """(x, y) ratio position of the flashed line's centre: `reference` (the saccade target's, or the
    fixation point's, (xRatio, yRatio)) shifted LINE_OFFSET_HEIGHT_UNITS up (Side.ABOVE) or down
    (Side.BELOW). y is a fraction of the screen height measured from the TOP, so above = smaller y.
    The renderer draws the line the full screen width; x only records which target it belongs to."""
    x, y = reference
    return x, y - LINE_OFFSET_HEIGHT_UNITS if side is Side.ABOVE else y + LINE_OFFSET_HEIGHT_UNITS


def robust_kept_ms(samples: Sequence[float]) -> list[float]:
    """The samples a reaction-time average keeps: anticipations (< RT_MIN_PLAUSIBLE_MS) are dropped,
    then slow outliers (more than RT_OUTLIER_MAD_K scaled MADs above the median, MAD floored at
    RT_OUTLIER_MAD_FLOOR_MS)."""
    plausible = [s for s in samples if s >= RT_MIN_PLAUSIBLE_MS]
    if not plausible:
        return []
    median = statistics.median(plausible)
    mad = statistics.median(abs(s - median) for s in plausible) * 1.4826
    limit = median + RT_OUTLIER_MAD_K * max(mad, RT_OUTLIER_MAD_FLOOR_MS)
    return [s for s in plausible if s <= limit]


def robust_average_ms(samples: Sequence[float], default: float) -> float:
    """Reaction-time average that one bad measurement can't move: the median of `robust_kept_ms`,
    or `default` if nothing plausible is left."""
    kept = robust_kept_ms(samples)
    return statistics.median(kept) if kept else default


class RollingReactionTime:
    """The live reaction-time average that schedules every flash: the robust average
    (`robust_average_ms`) of the last RT_AVERAGE_ROLLING_WINDOW saccade-onset latencies, recomputed on
    every `add`. One instance serves the tutorial's practice trials and the saccade phase (seed the
    latter with the former's `samples`), so the average carries over seamlessly. With fewer than
    RT_MIN_SAMPLES_FOR_AVERAGE plausible (>= RT_MIN_PLAUSIBLE_MS) samples in the window the average
    keeps its previous value (initially `default`), so one or two latencies never decide the schedule."""

    def __init__(self, default: float, seed: Sequence[float] = (), window: int = RT_AVERAGE_ROLLING_WINDOW) -> None:
        self._window: deque[float] = deque(maxlen=window)
        self._average = default
        self.total_added = 0
        for sample in seed:
            self.add(sample)

    def add(self, sample_ms: float) -> float:
        """Adds one detected latency and returns the updated average."""
        self._window.append(sample_ms)
        self.total_added += 1
        if sum(1 for s in self._window if s >= RT_MIN_PLAUSIBLE_MS) >= RT_MIN_SAMPLES_FOR_AVERAGE:
            self._average = robust_average_ms(self._window, default=self._average)
        return self._average

    @property
    def average(self) -> float:
        return self._average

    @property
    def samples(self) -> list[float]:
        """The latencies currently in the window (oldest first)."""
        return list(self._window)

    @property
    def n_dropped(self) -> int:
        """How many samples in the window the robust rule currently leaves out of the average."""
        return len(self._window) - len(robust_kept_ms(self._window))


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


def sample_capture_time(sample, now: float, latency_s: float, from_perf_counter) -> float:
    """When the eye was where `sample` says, on the session clock: the sample's own timestamp (converted by
    `from_perf_counter`) minus the gaze latency. Falls back to the tick time `now` for a missing/non-positive timestamp or a
    stale one (>1 s, or in the future)."""
    if sample.timestamp is None or sample.timestamp <= 0:
        return now - latency_s
    t = from_perf_counter(sample.timestamp)
    if t > now or now - t > 1.0:
        t = now
    return t - latency_s
