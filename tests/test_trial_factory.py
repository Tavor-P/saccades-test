from include.experiment.constants import (
    CATCH_TRIAL_FRACTION,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_TRIALS_PER_PHASE_TEST,
    TIMING_OFFSETS_MS,
)
from include.experiment.types import Target
from src.experiment.trial_factory import build_presaccade_sequence, build_saccade_sequence, generate_next_saccade_trial
from src.experiment.trial_mechanics import ShuffledBag

_CATCH_TRIAL_COUNT = round(NUM_TRIALS_PER_PHASE_TEST * CATCH_TRIAL_FRACTION)


def _presaccade_sequence():
    return build_presaccade_sequence(num_trials=NUM_TRIALS_PER_PHASE_TEST, num_practice=NUM_PRACTICE_TRIALS_TEST)


def _saccade_practice_sequence():
    return build_saccade_sequence(num_practice=NUM_PRACTICE_TRIALS_TEST)


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


def test_saccade_practice_sequence_counts_and_alternates_shown_and_catch():
    trials = _saccade_practice_sequence()
    assert len(trials) == NUM_PRACTICE_TRIALS_TEST
    assert all(t.practice for t in trials)
    assert [t.grating_shown for t in trials] == [i % 2 == 0 for i in range(len(trials))]


def test_saccade_practice_sequence_alternates_dot_and_cross_continuously():
    trials = _saccade_practice_sequence()
    assert trials[0].source is Target.DOT
    for current, following in zip(trials, trials[1:]):
        assert current.target == following.source
        assert current.source != current.target


def test_orientation_is_set_only_when_grating_is_shown():
    for trials in (_presaccade_sequence(), _saccade_practice_sequence()):
        for trial in trials:
            if trial.grating_shown:
                assert trial.orientation is not None
            else:
                assert trial.orientation is None


def test_presaccade_orientation_is_randomized_not_fixed():
    # A single build has both a large trial count and a random per-trial
    # choice, so a run that comes back all-one-orientation would indicate the
    # randomization isn't happening at all. (The saccade phase's real block
    # is generated dynamically now - see
    # test_generate_next_saccade_trial_shown_rate_is_not_fixed_exact_ratio
    # below for its equivalent coverage.)
    orientations = {t.orientation for t in _presaccade_sequence() if t.grating_shown}
    assert len(orientations) > 1


def test_build_presaccade_sequence_respects_the_given_trial_counts():
    # Regression test: num_trials/num_practice are caller-supplied
    # (test_mode is resolved at runtime from the startup dialog, not a
    # hardcoded constant), so the factory must actually honor whatever it's
    # given rather than silently falling back to a fixed count.
    trials = build_presaccade_sequence(num_trials=10, num_practice=3)
    practice = [t for t in trials if t.practice]
    real = [t for t in trials if not t.practice]
    assert len(practice) == 3
    assert len(real) == 10


def test_build_saccade_sequence_produces_only_practice_trials():
    # ExperimentSession's real block is dynamically generated instead (see
    # generate_next_saccade_trial below) - build_saccade_sequence only ever
    # builds the fixed-size practice block now.
    trials = build_saccade_sequence(num_practice=4)
    assert len(trials) == 4
    assert all(t.practice for t in trials)


def test_generate_next_saccade_trial_alternates_source_and_target():
    trial, next_source = generate_next_saccade_trial(0, Target.DOT, ShuffledBag(TIMING_OFFSETS_MS))
    assert trial.source is Target.DOT
    assert trial.target is Target.CROSS
    assert next_source is Target.CROSS
    assert trial.practice is False
    assert trial.index == 0

    trial2, next_source2 = generate_next_saccade_trial(1, next_source, ShuffledBag(TIMING_OFFSETS_MS))
    assert trial2.source is Target.CROSS
    assert trial2.target is Target.DOT
    assert next_source2 is Target.DOT


def test_generate_next_saccade_trial_draws_a_timing_offset_from_the_bag():
    bag = ShuffledBag(TIMING_OFFSETS_MS)
    trial, _ = generate_next_saccade_trial(0, Target.DOT, bag)
    assert trial.timing_offset_ms in TIMING_OFFSETS_MS


def test_generate_next_saccade_trial_shown_rate_is_not_fixed_exact_ratio():
    # Unlike build_saccade_sequence's pre-shuffled exact-ratio schedule, each
    # call draws grating_shown independently - over enough draws that should
    # produce both outcomes, not a suspiciously exact CATCH_TRIAL_FRACTION
    # split every time.
    bag = ShuffledBag(TIMING_OFFSETS_MS)
    source = Target.DOT
    shown_flags = []
    for i in range(200):
        trial, source = generate_next_saccade_trial(i, source, bag)
        shown_flags.append(trial.grating_shown)
    assert True in shown_flags and False in shown_flags


def test_shuffled_bag_covers_every_item_before_any_repeat():
    pool = [1, 2, 3, 4, 5]
    bag = ShuffledBag(pool)
    drawn = [bag.draw() for _ in range(len(pool))]
    assert sorted(drawn) == pool


def test_shuffled_bag_refills_after_exhaustion():
    pool = ["a", "b", "c"]
    bag = ShuffledBag(pool)
    drawn = [bag.draw() for _ in range(len(pool) * 10)]
    assert len(drawn) == 30
    assert set(drawn) == set(pool)
