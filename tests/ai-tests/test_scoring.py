from include.experiment.types import Orientation
from src.experiment.scoring import score_outcome


def test_correct_when_response_orientation_matches():
    assert (
        score_outcome(
            responded=True,
            response_orientation=Orientation.VERTICAL,
            actual_orientation=Orientation.VERTICAL,
        )
        == "correct"
    )


def test_incorrect_when_response_orientation_does_not_match():
    assert (
        score_outcome(
            responded=True,
            response_orientation=Orientation.HORIZONTAL,
            actual_orientation=Orientation.VERTICAL,
        )
        == "incorrect"
    )


def test_miss_when_grating_shown_but_no_response():
    assert (
        score_outcome(
            responded=False, response_orientation=None, actual_orientation=Orientation.VERTICAL
        )
        == "miss"
    )


def test_every_outcome_is_a_flash_outcome():
    # No catch trials any more: scoring only ever yields correct / incorrect / miss.
    outcomes = {
        score_outcome(responded, answer, Orientation.VERTICAL)
        for responded in (True, False)
        for answer in (None, Orientation.VERTICAL, Orientation.HORIZONTAL)
    }
    assert outcomes == {"correct", "incorrect", "miss"}
