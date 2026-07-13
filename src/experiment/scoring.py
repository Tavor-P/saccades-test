def score_outcome(grating_shown: bool, responded: bool) -> str:
    """Shared yes/no signal-detection scoring used by both phases (outside of
    the saccade phase's "timeout" case, which its own state machine handles
    before falling back to this)."""
    if grating_shown and responded:
        return "hit"
    if grating_shown and not responded:
        return "miss"
    if not grating_shown and responded:
        return "false_alarm"
    return "correct_rejection"
