import random

from include.experiment.constants import CATCH_TRIAL_COUNT, NUM_TRIALS_PER_PHASE
from include.experiment.types import FlashTrialSpec, Target, TrialSpec


def _build_shown_schedule() -> list[bool]:
    """NUM_TRIALS_PER_PHASE trials, CATCH_TRIAL_COUNT of which show no grating,
    shuffled into a random order (no fixed pattern - catch trials can land
    back-to-back or not). Actual contrast for the rest is decided live by each
    session's ZEST staircase, not precomputed here."""
    schedule = [True] * (NUM_TRIALS_PER_PHASE - CATCH_TRIAL_COUNT) + [False] * CATCH_TRIAL_COUNT
    random.shuffle(schedule)
    return schedule


def build_presaccade_sequence() -> list[FlashTrialSpec]:
    """Phase 1: fixate center only, no saccade - just detect the flash (or not)."""
    schedule = _build_shown_schedule()
    return [FlashTrialSpec(index=index, grating_shown=shown) for index, shown in enumerate(schedule)]


def build_saccade_sequence() -> list[TrialSpec]:
    """Phase 2: same shown/catch schedule, but each trial is also an
    alternating dot<->cross saccade."""
    schedule = _build_shown_schedule()
    trials = []
    source = Target.DOT
    for index, shown in enumerate(schedule):
        target = Target.CROSS if source is Target.DOT else Target.DOT
        trials.append(TrialSpec(index=index, source=source, target=target, grating_shown=shown))
        source = target
    return trials
