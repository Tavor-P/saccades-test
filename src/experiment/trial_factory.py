import random

from include.experiment.constants import CONTRAST_LEVELS, REPEATS_PER_LEVEL, TRIALS_PER_PHASE
from include.experiment.types import FlashTrialSpec, Target, TrialSpec


def _build_flash_schedule() -> list[tuple[bool, float | None]]:
    """REPEATS_PER_LEVEL trials at each of CONTRAST_LEVELS, padded with
    zero-contrast catch trials up to TRIALS_PER_PHASE, shuffled into a random
    order (no fixed pattern - a level can repeat back-to-back or not)."""
    schedule: list[tuple[bool, float | None]] = []
    for level in CONTRAST_LEVELS:
        schedule.extend([(True, level)] * REPEATS_PER_LEVEL)

    catch_count = TRIALS_PER_PHASE - len(schedule)
    schedule.extend([(False, None)] * catch_count)

    random.shuffle(schedule)
    return schedule


def build_presaccade_sequence() -> list[FlashTrialSpec]:
    """Phase 1: fixate center only, no saccade - just detect the flash (or not)."""
    schedule = _build_flash_schedule()
    return [
        FlashTrialSpec(index=index, square_shown=square_shown, contrast=contrast)
        for index, (square_shown, contrast) in enumerate(schedule)
    ]


def build_saccade_sequence() -> list[TrialSpec]:
    """Phase 2: same contrast/catch schedule, but each trial is also an
    alternating dot<->cross saccade."""
    schedule = _build_flash_schedule()
    trials = []
    source = Target.DOT
    for index, (square_shown, contrast) in enumerate(schedule):
        target = Target.CROSS if source is Target.DOT else Target.DOT
        trials.append(
            TrialSpec(index=index, source=source, target=target, square_shown=square_shown, contrast=contrast)
        )
        source = target
    return trials
