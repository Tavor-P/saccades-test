import random

from include.experiment.constants import CATCH_TRIAL_FRACTION
from include.experiment.types import FlashTrialSpec, Target, TrialSpec
from src.experiment.trial_mechanics import ShuffledBag
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


def build_saccade_sequence(num_practice: int) -> list[TrialSpec]:
    """Phase 2's practice block: alternating shown/catch (see
    _build_practice_schedule), each trial also an alternating dot<->cross
    saccade starting from Target.DOT. The real block is no longer built
    ahead of time here - contrast stays ZEST-adaptive rather than a fixed
    trial count, so ExperimentSession generates it one trial at a time
    instead (see generate_next_saccade_trial), continuing the same
    dot<->cross alternation from wherever this practice block leaves off."""
    practice_trials, _ = _build_saccade_trials(
        _build_practice_schedule(num_practice), practice=True, start_source=Target.DOT
    )
    return practice_trials


def generate_next_saccade_trial(
    index: int, source: Target, offset_bag: ShuffledBag[int]
) -> tuple[TrialSpec, Target]:
    """One real-block trial at a time, for ExperimentSession's open-ended
    main loop (see include.experiment.constants' stopping-criterion
    constants) - total trial count isn't known upfront the way it is for
    build_saccade_sequence's fixed-length schedule, so grating_shown is an
    independent Bernoulli draw at CATCH_TRIAL_FRACTION each call rather than
    an exact-ratio pre-shuffled schedule (converges to the same rate over
    many trials, just without a guaranteed exact count). Source/target
    dot<->cross alternation stays deterministic, same as
    _build_saccade_trials. Returns (trial, next source) - the caller threads
    `source` through consecutive calls the same way _build_saccade_trials
    does internally."""
    target = Target.CROSS if source is Target.DOT else Target.DOT
    shown = random.random() >= CATCH_TRIAL_FRACTION
    trial = TrialSpec(
        index=index,
        source=source,
        target=target,
        grating_shown=shown,
        orientation=_random_orientation() if shown else None,
        practice=False,
        timing_offset_ms=offset_bag.draw(),
    )
    return trial, target
