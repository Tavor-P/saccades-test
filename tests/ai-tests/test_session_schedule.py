import pytest

from src.experiment.session_schedule import PHASE_BASELINE, PHASE_SACCADE, SessionSchedule, phase_order


def test_test_mode_is_always_saccade_first():
    assert phase_order("", True) == (PHASE_SACCADE, PHASE_BASELINE)
    assert phase_order("7", True) == (PHASE_SACCADE, PHASE_BASELINE)


def test_numeric_ids_alternate_order():
    orders = [phase_order(str(i), False) for i in range(1, 7)]
    assert orders[0] != orders[1]
    assert orders[0] == orders[2] == orders[4]
    assert orders[1] == orders[3] == orders[5]


def test_order_is_deterministic_and_ignores_surrounding_whitespace():
    assert phase_order("alice", False) == phase_order("  alice ", False) == phase_order("ALICE", False)


def test_non_numeric_ids_reach_both_orders():
    orders = {phase_order(f"p{c}", False) for c in "abcdefghij"}
    assert orders == {(PHASE_BASELINE, PHASE_SACCADE), (PHASE_SACCADE, PHASE_BASELINE)}


@pytest.fixture
def clock():
    t = {"now": 1000.0}

    def now():
        return t["now"]

    now.advance = lambda s: t.__setitem__("now", t["now"] + s)
    return now


def test_rest_is_due_after_the_interval_and_not_before(clock):
    s = SessionSchedule(clock, rest_every_s=100, min_rest_s=10, max_session_s=None)
    clock.advance(99)
    assert not s.rest_due()
    clock.advance(1)
    assert s.rest_due()


def test_rest_cannot_end_before_the_minimum_then_restarts_the_interval(clock):
    s = SessionSchedule(clock, rest_every_s=100, min_rest_s=10, max_session_s=None)
    clock.advance(100)
    s.begin_rest()
    assert s.resting and not s.rest_due()
    clock.advance(9)
    assert not s.rest_may_end()
    clock.advance(1)
    assert s.rest_may_end()
    s.end_rest()
    assert not s.resting and not s.rest_due()
    clock.advance(100)
    assert s.rest_due()
    assert s.breaks_taken == 1


def test_cap_counts_rest_time_and_can_be_disabled(clock):
    s = SessionSchedule(clock, rest_every_s=None, max_session_s=50)
    clock.advance(49)
    assert not s.cap_reached()
    clock.advance(1)
    assert s.cap_reached()
    off = SessionSchedule(clock, rest_every_s=None, max_session_s=None)
    clock.advance(10**6)
    assert not off.cap_reached() and not off.rest_due()
