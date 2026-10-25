from src.experiment.session_schedule import PHASE_BASELINE, PHASE_SACCADE, phase_order


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
