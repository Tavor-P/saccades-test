from include.experiment.types import Side
from src.experiment.scoring import score_outcome


def test_correct_when_response_orientation_matches():
    assert (
        score_outcome(
            responded=True,
            response_orientation=Side.ABOVE,
            actual_orientation=Side.ABOVE,
        )
        == "correct"
    )


def test_incorrect_when_response_orientation_does_not_match():
    assert (
        score_outcome(
            responded=True,
            response_orientation=Side.BELOW,
            actual_orientation=Side.ABOVE,
        )
        == "incorrect"
    )


def test_miss_when_grating_shown_but_no_response():
    assert (
        score_outcome(
            responded=False, response_orientation=None, actual_orientation=Side.ABOVE
        )
        == "miss"
    )


def test_every_outcome_is_a_flash_outcome():
    # No catch trials any more: scoring only ever yields correct / incorrect / miss.
    outcomes = {
        score_outcome(responded, answer, Side.ABOVE)
        for responded in (True, False)
        for answer in (None, Side.ABOVE, Side.BELOW)
    }
    assert outcomes == {"correct", "incorrect", "miss"}
