import random

from include.experiment.constants import CONTRAST_LEVELS, NUM_TRIALS, SQUARE_FLASH_PROBABILITY
from include.experiment.types import Target, TrialSpec


def build_trial_sequence(count: int = NUM_TRIALS) -> list[TrialSpec]:
    """Alternating dot<->cross saccade trials. Each trial independently flashes
    a square mid-saccade with probability SQUARE_FLASH_PROBABILITY, at a random
    contrast level, to test perisaccadic detection."""
    trials = []
    source = Target.DOT
    for index in range(count):
        target = Target.CROSS if source is Target.DOT else Target.DOT
        square_shown = random.random() < SQUARE_FLASH_PROBABILITY
        contrast = random.choice(CONTRAST_LEVELS) if square_shown else None
        trials.append(
            TrialSpec(index=index, source=source, target=target, square_shown=square_shown, contrast=contrast)
        )
        source = target
    return trials
