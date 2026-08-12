import random

from include.eye_tracking.types import GazeZone
from include.experiment.types import Orientation, Target


def zone_for(target: Target) -> GazeZone:
    """Which GazeZone a fixation on `target` counts as - the dot is always
    the left-hand target, the cross always the right, in both the real
    saccade trials and the tutorial's gaze-practice/dress-rehearsal stages."""
    return GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT


def random_orientation() -> Orientation:
    return random.choice([Orientation.VERTICAL, Orientation.HORIZONTAL])
