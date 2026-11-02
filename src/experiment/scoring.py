"""score_outcome (correct/incorrect/miss) and is_valid_for_saccadic_analysis - the single validity gate for saccade-phase
flashes (flash_during_saccade AND tracking_ok), used by the live ZEST update."""

from include.experiment.types import Side


def score_outcome(
    responded: bool,
    response_orientation: Side | None,
    actual_orientation: Side | None,
) -> str:
    """Shared scoring (outside of the saccade phase's "timeout" case, which its
    own state machine handles before falling back to this). Every trial flashes a grating and is scored as a 2-alternative
    orientation discrimination (correct/incorrect/miss) rather than plain yes/no
    detection - correctly reporting *which* orientation it was rules out a
    lucky guess in a way merely detecting *that* something flashed doesn't.
    (Old sessions also hold "false_alarm"/"correct_rejection" outcomes from the
    removed catch trials; they are only ever read, never produced.)"""
    if not responded:
        return "miss"
    return "correct" if response_orientation == actual_orientation else "incorrect"


def is_valid_for_saccadic_analysis(flash_during_saccade: bool | None, tracking_ok: bool | None = None) -> bool:
    """The single validity gate for a saccade-phase, grating-shown trial's
    flash_during_saccade - applied by the live in-session ZEST update
    (ExperimentSession._finish_trial). Only True counts - False
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
