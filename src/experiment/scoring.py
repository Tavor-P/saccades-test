from include.experiment.types import Orientation


def score_outcome(
    grating_shown: bool,
    responded: bool,
    response_orientation: Orientation | None,
    actual_orientation: Orientation | None,
) -> str:
    """Shared scoring used by both phases (outside of the saccade phase's
    "timeout" case, which its own state machine handles before falling back
    to this). Grating-shown trials are scored as a 2-alternative orientation
    discrimination (correct/incorrect/miss) rather than plain yes/no
    detection - correctly reporting *which* orientation it was rules out a
    lucky guess in a way merely detecting *that* something flashed doesn't."""
    if not grating_shown:
        return "false_alarm" if responded else "correct_rejection"
    if not responded:
        return "miss"
    return "correct" if response_orientation == actual_orientation else "incorrect"
