import random
from collections import deque
from typing import Generic, Sequence, TypeVar

from include.eye_tracking.types import GazeZone
from include.experiment.constants import SACCADE_ONSET_STABILITY_MS
from include.experiment.types import Orientation, Target


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


class OnsetDetector:
    """Debounces the real-time gaze classifier's per-tick zone reading into a
    confirmed "gaze left `avoid_zone`" event: requires SACCADE_ONSET_STABILITY_MS
    of continuous residence outside both `avoid_zone` and GazeZone.UNKNOWN
    before confirming, so a single misclassified/jittery frame (blink,
    lighting flicker) doesn't read as a real saccade onset. This exact
    debounce is needed in four places (ExperimentSession's RT-test and main
    trials, TutorialSession's scoped-down mirrors of both) - one tested
    implementation here instead of four hand-copied ones that could drift
    out of sync with each other.

    `since` mirrors the field this replaces in each caller (e.g. the old
    `_away_from_source_since`): set optimistically the first tick gaze is
    seen outside `avoid_zone`, reset to None on any tick it isn't - readable
    at any time, not just once confirmed. `confirmed` mirrors the old
    `_gaze_left_source`-style flag. Once `confirmed` is True, the caller is
    expected to stop calling `update()` (matching every call site's existing
    `if not <confirmed>:` gate) so `since` freezes at the timestamp that
    actually triggered confirmation.
    """

    def __init__(self, avoid_zone: GazeZone) -> None:
        self._avoid_zone = avoid_zone
        self.since: float | None = None
        self.confirmed = False

    def update(self, sample, now: float) -> bool:
        """Call once per tick while not yet confirmed. Returns True on the
        exact tick confirmation fires."""
        if sample.face_found and sample.zone not in (self._avoid_zone, GazeZone.UNKNOWN):
            if self.since is None:
                self.since = now
            elif now - self.since >= SACCADE_ONSET_STABILITY_MS / 1000:
                self.confirmed = True
                return True
        else:
            self.since = None
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
