"""Builds trial sequences/specs: alternating dot/cross sources, practice blocks, and the next dynamically generated trial. Every trial shows a flash."""

from include.experiment.types import Target, TrialSpec
from src.experiment.trial_mechanics import ShuffledBag
from src.experiment.trial_mechanics import random_orientation as _random_orientation


def _build_saccade_trials(count: int, practice: bool, start_source: Target) -> tuple[list[TrialSpec], Target]:
    trials = []
    source = start_source
    for index in range(count):
        target = Target.CROSS if source is Target.DOT else Target.DOT
        trials.append(
            TrialSpec(
                index=index,
                source=source,
                target=target,
                orientation=_random_orientation(),
                practice=practice,
            )
        )
        source = target
    return trials, source


def build_saccade_sequence(num_practice: int) -> list[TrialSpec]:
    """The practice block: each trial an alternating dot<->cross
    saccade starting from Target.DOT. The real block is no longer built
    ahead of time here - contrast stays ZEST-adaptive rather than a fixed
    trial count, so ExperimentSession generates it one trial at a time
    instead (see generate_next_saccade_trial), continuing the same
    dot<->cross alternation from wherever this practice block leaves off."""
    practice_trials, _ = _build_saccade_trials(num_practice, practice=True, start_source=Target.DOT)
    return practice_trials


def generate_next_saccade_trial(
    index: int, source: Target, offset_bag: ShuffledBag[int]
) -> tuple[TrialSpec, Target]:
    """One real-block trial at a time, for ExperimentSession's open-ended
    main loop (see include.experiment.constants' stopping-criterion
    constants) - total trial count isn't known upfront the way it is for
    build_saccade_sequence's fixed-length schedule. Source/target
    dot<->cross alternation stays deterministic, same as
    _build_saccade_trials. Returns (trial, next source) - the caller threads
    `source` through consecutive calls the same way _build_saccade_trials
    does internally."""
    target = Target.CROSS if source is Target.DOT else Target.DOT
    trial = TrialSpec(
        index=index,
        source=source,
        target=target,
        orientation=_random_orientation(),
        practice=False,
        timing_offset_ms=offset_bag.draw(),
    )
    return trial, target
