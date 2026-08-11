from include.experiment.constants import CATCH_TRIAL_FRACTION, NUM_PRACTICE_TRIALS_TEST, NUM_TRIALS_PER_PHASE_TEST
from include.experiment.types import Target
from src.experiment.trial_factory import build_presaccade_sequence, build_saccade_sequence

_CATCH_TRIAL_COUNT = round(NUM_TRIALS_PER_PHASE_TEST * CATCH_TRIAL_FRACTION)


def _presaccade_sequence():
    return build_presaccade_sequence(num_trials=NUM_TRIALS_PER_PHASE_TEST, num_practice=NUM_PRACTICE_TRIALS_TEST)


def _saccade_sequence():
    return build_saccade_sequence(num_trials=NUM_TRIALS_PER_PHASE_TEST, num_practice=NUM_PRACTICE_TRIALS_TEST)


def test_presaccade_sequence_counts():
    trials = _presaccade_sequence()
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert len(practice) == NUM_PRACTICE_TRIALS_TEST
    assert len(real) == NUM_TRIALS_PER_PHASE_TEST
    assert sum(1 for t in real if not t.grating_shown) == _CATCH_TRIAL_COUNT


def test_presaccade_indices_are_sequential_within_each_block():
    trials = _presaccade_sequence()
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert [t.index for t in practice] == list(range(len(practice)))
    assert [t.index for t in real] == list(range(len(real)))


def test_practice_schedule_alternates_shown_and_catch():
    trials = _presaccade_sequence()
    practice_shown = [t.grating_shown for t in trials if t.practice]
    assert practice_shown == [i % 2 == 0 for i in range(len(practice_shown))]


def test_saccade_sequence_counts():
    trials = _saccade_sequence()
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert len(practice) == NUM_PRACTICE_TRIALS_TEST
    assert len(real) == NUM_TRIALS_PER_PHASE_TEST
    assert sum(1 for t in real if not t.grating_shown) == _CATCH_TRIAL_COUNT


def test_saccade_sequence_alternates_dot_and_cross_continuously():
    trials = _saccade_sequence()
    assert trials[0].source is Target.DOT
    for current, following in zip(trials, trials[1:]):
        assert current.target == following.source
        assert current.source != current.target


def test_saccade_schedule_is_shuffled_not_fixed():
    # With NUM_TRIALS_PER_PHASE_TEST trials and _CATCH_TRIAL_COUNT catches, the
    # number of distinct orderings is large enough that independent builds
    # matching every time would indicate the shuffle isn't happening at all.
    schedules = {tuple(t.grating_shown for t in _saccade_sequence() if not t.practice) for _ in range(20)}
    assert len(schedules) > 1


def test_orientation_is_set_only_when_grating_is_shown():
    for trials in (_presaccade_sequence(), _saccade_sequence()):
        for trial in trials:
            if trial.grating_shown:
                assert trial.orientation is not None
            else:
                assert trial.orientation is None


def test_orientation_is_randomized_not_fixed():
    # A single build has both a large trial count and a random per-trial
    # choice, so a run that comes back all-one-orientation would indicate the
    # randomization isn't happening at all.
    orientations = {t.orientation for t in _saccade_sequence() if t.grating_shown}
    assert len(orientations) > 1


def test_build_sequences_respect_the_given_trial_counts():
    # Regression test: num_trials/num_practice are now caller-supplied
    # (test_mode is resolved at runtime from the startup dialog, not a
    # hardcoded constant), so the factory must actually honor whatever it's
    # given rather than silently falling back to a fixed count.
    presaccade = build_presaccade_sequence(num_trials=10, num_practice=3)
    saccade = build_saccade_sequence(num_trials=10, num_practice=3)
    for trials in (presaccade, saccade):
        practice = [t for t in trials if t.practice]
        real = [t for t in trials if not t.practice]
        assert len(practice) == 3
        assert len(real) == 10
