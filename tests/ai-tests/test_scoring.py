from include.experiment.types import Side
from src.experiment.scoring import score_outcome


def test_scoring_yields_only_flash_outcomes():
    assert score_outcome(responded=True, response_orientation=Side.ABOVE, actual_orientation=Side.ABOVE) == "correct"
    assert score_outcome(responded=True, response_orientation=Side.BELOW, actual_orientation=Side.ABOVE) == "incorrect"
    assert score_outcome(responded=False, response_orientation=None, actual_orientation=Side.ABOVE) == "miss"
    # No catch trials any more: scoring only ever yields correct / incorrect / miss.
    outcomes = {
        score_outcome(responded, answer, Side.ABOVE)
        for responded in (True, False)
        for answer in (None, Side.ABOVE, Side.BELOW)
    }
    assert outcomes == {"correct", "incorrect", "miss"}
