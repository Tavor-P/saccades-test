from include.experiment.constants import (
    NUM_PRACTICE_TRIALS_TEST,
    TIMING_OFFSETS_MS,
)
from include.experiment.types import Side, Target
from src.experiment.trial_factory import build_saccade_sequence, generate_next_saccade_trial
from src.experiment.trial_mechanics import ShuffledBag


def test_practice_sequence_has_the_count_is_practice_only_and_alternates_dot_and_cross():
    # ExperimentSession's real block is generated trial by trial (generate_next_saccade_trial);
    # build_saccade_sequence only builds the fixed-size practice block.
    for count in (NUM_PRACTICE_TRIALS_TEST, 4):
        trials = build_saccade_sequence(num_practice=count)
        assert len(trials) == count and all(t.practice for t in trials)
        assert trials[0].source is Target.DOT
        for current, following in zip(trials, trials[1:]):
            assert current.target == following.source
            assert current.source != current.target


def test_generated_trials_alternate_draw_offsets_from_the_bag_and_always_flash_above_or_below():
    trial, next_source = generate_next_saccade_trial(0, Target.DOT, ShuffledBag(TIMING_OFFSETS_MS))
    assert (trial.source, trial.target, next_source) == (Target.DOT, Target.CROSS, Target.CROSS)
    assert trial.practice is False and trial.index == 0
    assert trial.timing_offset_ms in TIMING_OFFSETS_MS
    trial2, next_source2 = generate_next_saccade_trial(1, next_source, ShuffledBag(TIMING_OFFSETS_MS))
    assert (trial2.source, trial2.target, next_source2) == (Target.CROSS, Target.DOT, Target.DOT)

    # Catch trials (no flash, false alarms) were removed: every trial has a side, and both sides occur.
    seen = set()
    for _ in range(20):
        assert all(t.orientation in (Side.ABOVE, Side.BELOW) for t in build_saccade_sequence(num_practice=NUM_PRACTICE_TRIALS_TEST))
    bag, source = ShuffledBag(TIMING_OFFSETS_MS), Target.DOT
    for i in range(200):
        trial, source = generate_next_saccade_trial(i, source, bag)
        assert trial.orientation in (Side.ABOVE, Side.BELOW)
        seen.add(trial.orientation)
    assert seen == {Side.ABOVE, Side.BELOW}


def test_shuffled_bag_covers_every_item_before_any_repeat_and_refills():
    pool = [1, 2, 3, 4, 5]
    bag = ShuffledBag(pool)
    assert sorted(bag.draw() for _ in range(len(pool))) == pool
    pool = ["a", "b", "c"]
    bag = ShuffledBag(pool)
    drawn = [bag.draw() for _ in range(len(pool) * 10)]
    assert len(drawn) == 30 and set(drawn) == set(pool)
    for k in range(10):
        assert sorted(drawn[3 * k:3 * k + 3]) == sorted(pool)
