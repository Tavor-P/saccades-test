from src.experiment.scoring import score_outcome


def test_hit():
    assert score_outcome(grating_shown=True, responded=True) == "hit"


def test_miss():
    assert score_outcome(grating_shown=True, responded=False) == "miss"


def test_false_alarm():
    assert score_outcome(grating_shown=False, responded=True) == "false_alarm"


def test_correct_rejection():
    assert score_outcome(grating_shown=False, responded=False) == "correct_rejection"
