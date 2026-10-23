"""score_outcome (correct/incorrect/miss/false_alarm/correct_rejection) and is_valid_for_saccadic_analysis - the single validity gate for saccade-phase
flashes (flash_during_saccade AND tracking_ok), shared by live ZEST updates and the end-of-run graph."""

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


def is_valid_for_saccadic_analysis(flash_during_saccade: bool | None, tracking_ok: bool | None = None) -> bool:
    """The single validity gate for a saccade-phase, grating-shown trial's
    flash_during_saccade - shared by both the live in-session ZEST update
    (ExperimentSession._finish_trial) and the end-of-run analysis
    (results_graph._exclude_flashes_not_during_saccade), so the two can't
    silently diverge if this rule ever changes. Only True counts - False
    (the flash landed outside the real-time-detected saccade window) and
    None (landing was never confirmed, so validity is genuinely
    undeterminable, not merely invalid) are equally unusable for either
    purpose.

    `tracking_ok` is the second gate: an explicit False (gaze tracking
    dropped out around the flash) invalidates the trial even if the
    reconstructed timing says the flash landed in the saccade, since a
    dropout corrupts the onset/landing timestamps that check relies on. None
    (unknown - older CSVs, or a row that never had the check) doesn't
    invalidate anything, so existing data keeps its previous validity."""
    return flash_during_saccade is True and tracking_ok is not False
