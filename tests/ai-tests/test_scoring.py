from include.experiment.types import Side
from src.experiment.scoring import score_outcome


def test_correct_when_response_orientation_matches():
    assert (
        score_outcome(
            grating_shown=True,
            responded=True,
            response_orientation=Side.ABOVE,
            actual_orientation=Side.ABOVE,
        )
        == "correct"
    )


def test_incorrect_when_response_orientation_does_not_match():
    assert (
        score_outcome(
            grating_shown=True,
            responded=True,
            response_orientation=Side.BELOW,
            actual_orientation=Side.ABOVE,
        )
        == "incorrect"
    )


def test_miss_when_grating_shown_but_no_response():
    assert (
        score_outcome(
            grating_shown=True, responded=False, response_orientation=None, actual_orientation=Side.ABOVE
        )
        == "miss"
    )


def test_false_alarm():
    assert (
        score_outcome(grating_shown=False, responded=True, response_orientation=Side.BELOW, actual_orientation=None)
        == "false_alarm"
    )


def test_correct_rejection():
    assert (
        score_outcome(grating_shown=False, responded=False, response_orientation=None, actual_orientation=None)
        == "correct_rejection"
    )
