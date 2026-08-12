import random

from include.experiment.constants import CATCH_TRIAL_FRACTION
from include.experiment.types import FlashTrialSpec, Target, TrialSpec
from src.experiment.trial_mechanics import random_orientation as _random_orientation


def _build_shown_schedule(num_trials: int) -> list[bool]:
    """num_trials trials, CATCH_TRIAL_FRACTION of which show no grating,
    shuffled into a random order (no fixed pattern - catch trials can land
    back-to-back or not). Actual contrast for the rest is decided live by each
    session's ZEST staircase, not precomputed here."""
    catch_count = round(num_trials * CATCH_TRIAL_FRACTION)
    schedule = [True] * (num_trials - catch_count) + [False] * catch_count
    random.shuffle(schedule)
    return schedule


def _build_practice_schedule(num_practice: int) -> list[bool]:
    """A handful of throwaway trials before the real block, alternating
    shown/catch so practice also covers "correctly withhold a response", not
    just "there's always something to see"."""
    return [i % 2 == 0 for i in range(num_practice)]


def build_presaccade_sequence(num_trials: int, num_practice: int) -> list[FlashTrialSpec]:
    """Phase 1: fixate center only, no saccade - just detect the flash (or not).
    Practice trials come first, then the real (ZEST-staircased, logged) block."""
    practice = [
        FlashTrialSpec(index=index, grating_shown=shown, orientation=_random_orientation() if shown else None, practice=True)
        for index, shown in enumerate(_build_practice_schedule(num_practice))
    ]
    real = [
        FlashTrialSpec(index=index, grating_shown=shown, orientation=_random_orientation() if shown else None)
        for index, shown in enumerate(_build_shown_schedule(num_trials))
    ]
    return practice + real


def _build_saccade_trials(schedule: list[bool], practice: bool, start_source: Target) -> tuple[list[TrialSpec], Target]:
    trials = []
    source = start_source
    for index, shown in enumerate(schedule):
        target = Target.CROSS if source is Target.DOT else Target.DOT
        trials.append(
            TrialSpec(
                index=index,
                source=source,
                target=target,
                grating_shown=shown,
                orientation=_random_orientation() if shown else None,
                practice=practice,
            )
        )
        source = target
    return trials, source


def build_saccade_sequence(num_trials: int, num_practice: int) -> list[TrialSpec]:
    """Phase 2: same shown/catch schedule as the presaccade phase, but each
    trial is also an alternating dot<->cross saccade. Practice trials come
    first, continuing the same dot/cross alternation into the real block."""
    practice_trials, source = _build_saccade_trials(
        _build_practice_schedule(num_practice), practice=True, start_source=Target.DOT
    )
    real_trials, _ = _build_saccade_trials(_build_shown_schedule(num_trials), practice=False, start_source=source)
    return practice_trials + real_trials
