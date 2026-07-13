from include.experiment.constants import CATCH_TRIAL_COUNT, NUM_PRACTICE_TRIALS, NUM_TRIALS_PER_PHASE
from include.experiment.types import Target
from src.experiment.trial_factory import build_presaccade_sequence, build_saccade_sequence


def test_presaccade_sequence_counts():
    trials = build_presaccade_sequence()
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert len(practice) == NUM_PRACTICE_TRIALS
    assert len(real) == NUM_TRIALS_PER_PHASE
    assert sum(1 for t in real if not t.grating_shown) == CATCH_TRIAL_COUNT


def test_presaccade_indices_are_sequential_within_each_block():
    trials = build_presaccade_sequence()
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert [t.index for t in practice] == list(range(len(practice)))
    assert [t.index for t in real] == list(range(len(real)))


def test_practice_schedule_alternates_shown_and_catch():
    trials = build_presaccade_sequence()
    practice_shown = [t.grating_shown for t in trials if t.practice]
    assert practice_shown == [i % 2 == 0 for i in range(len(practice_shown))]


def test_saccade_sequence_counts():
    trials = build_saccade_sequence()
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert len(practice) == NUM_PRACTICE_TRIALS
    assert len(real) == NUM_TRIALS_PER_PHASE
    assert sum(1 for t in real if not t.grating_shown) == CATCH_TRIAL_COUNT


def test_saccade_sequence_alternates_dot_and_cross_continuously():
    trials = build_saccade_sequence()
    assert trials[0].source is Target.DOT
    for current, following in zip(trials, trials[1:]):
        assert current.target == following.source
        assert current.source != current.target


def test_saccade_schedule_is_shuffled_not_fixed():
    # With NUM_TRIALS_PER_PHASE trials and CATCH_TRIAL_COUNT catches, the
    # number of distinct orderings is large enough that independent builds
    # matching every time would indicate the shuffle isn't happening at all.
    schedules = {tuple(t.grating_shown for t in build_saccade_sequence() if not t.practice) for _ in range(20)}
    assert len(schedules) > 1
