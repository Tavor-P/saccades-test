"""Builds trial sequences/specs: alternating dot/cross sources, practice blocks, and the next dynamically generated trial. Every trial shows a flash."""

from include.experiment.types import Target, TrialSpec
from src.experiment.trial_mechanics import ShuffledBag
from src.experiment.trial_mechanics import random_orientation as _random_orientation


def _build_saccade_trials(count: int, practice: bool, start_source: Target, alternate: bool = True) -> tuple[list[TrialSpec], Target]:
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
        if alternate:
            source = target
    return trials, source


def build_saccade_sequence(num_practice: int, alternate: bool = True) -> list[TrialSpec]:
    """The practice block: each trial an alternating dot<->cross
    saccade starting from Target.DOT. The real block is no longer built
    ahead of time here - ExperimentSession generates it one trial at a time
    (drawing each contrast from its fixed-contrast bag) (see generate_next_saccade_trial), continuing the same
    dot<->cross alternation from wherever this practice block leaves off.
    alternate=False (return-jump trials): every trial starts at the dot, because the target jumps back to it."""
    practice_trials, _ = _build_saccade_trials(num_practice, practice=True, start_source=Target.DOT, alternate=alternate)
    return practice_trials


def generate_next_saccade_trial(
    index: int, source: Target, offset_bag: ShuffledBag[int], alternate: bool = True
) -> tuple[TrialSpec, Target]:
    """One real-block trial at a time, generated trial by trial; ExperimentSession
    stops at `SACCADE_FIXED_TRIALS_*` (include.experiment.constants). Unlike
    build_saccade_sequence, which builds its whole fixed-length schedule at once. Source/target
    dot<->cross alternation stays deterministic, same as
    _build_saccade_trials. Returns (trial, next source) - the caller threads
    `source` through consecutive calls the same way _build_saccade_trials
    does internally. alternate=False (return-jump trials): the next source is this
    trial's source again, since the target jumps back to it."""
    target = Target.CROSS if source is Target.DOT else Target.DOT
    trial = TrialSpec(
        index=index,
        source=source,
        target=target,
        orientation=_random_orientation(),
        practice=False,
        timing_offset_ms=offset_bag.draw(),
    )
    return trial, (target if alternate else source)
