import random
from collections import deque
from typing import Generic, Sequence, TypeVar

from include.eye_tracking.types import GazeZone
from include.experiment.types import Orientation, Target


def zone_for(target: Target) -> GazeZone:
    """Which GazeZone a fixation on `target` counts as - the dot is always
    the left-hand target, the cross always the right, in both the real
    saccade trials and the tutorial's gaze-practice/dress-rehearsal stages."""
    return GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT


def random_orientation() -> Orientation:
    return random.choice([Orientation.VERTICAL, Orientation.HORIZONTAL])


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
