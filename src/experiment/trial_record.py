"""The trials.csv row of a finished main trial: `saccade_trial_result()` maps the session's per-trial values and its
TrialLegs onto a TrialResult. Pure: scoring, the tracking/head gate and logging stay in ExperimentSession._finish_trial."""

from include.experiment.types import Side, TrialResult, TrialSpec
from src.experiment.trial_legs import TrialLegs
from src.timebase import ms_to_s, s_to_ms


def saccade_trial_result(
    trial: TrialSpec,
    legs: TrialLegs,
    *,
    outcome: str,
    flash_shown: bool,
    flash_leg: str | None,
    flash_scheduled_at: float,
    flash_unix_ms: float | None,
    contrast: float | None,
    flash_during_saccade: bool | None,
    tracking_ok: bool | None,
    responded: bool,
    answer_side: Side | None,
    response_time_ms: float | None,
    reaction_latency_ms: float | None,
    onset_detection_lag_ms: float | None,
    avg_rt_used_ms: float | None,
    scheduled_delay_ms: float | None,
    gaze_latency_ms: float | None,
) -> TrialResult:
    """`flash_scheduled_at` is on the capture-time axis (as the legs' onset/landing times); flash_relative_to_onset_ms is
    measured against the onset on the flash's own leg. Fields that describe the flash are None when it never showed."""
    flash_onset_at, _ = legs.onset_landing(flash_leg)
    return TrialResult(
        index=trial.index,
        phase="saccade",
        source=trial.source,
        target=trial.target,
        saccade_duration_ms=legs.saccade_duration_ms(),
        flash_shown=flash_shown,
        contrast=contrast,
        line_side=trial.line_side if flash_shown else None,
        responded=responded,
        answer_side=answer_side,
        response_time_ms=response_time_ms,
        outcome=outcome,
        reaction_latency_ms=reaction_latency_ms,
        onset_detection_lag_ms=onset_detection_lag_ms,
        flash_during_saccade=flash_during_saccade,
        timing_offset_ms=trial.timing_offset_ms,
        flash_unix_ms=flash_unix_ms,
        tracking_ok=tracking_ok,
        flash_relative_to_onset_ms=(
            s_to_ms(flash_scheduled_at - flash_onset_at) if flash_shown and flash_onset_at is not None else None
        ),
        avg_rt_used_ms=avg_rt_used_ms,
        scheduled_delay_ms=scheduled_delay_ms,
        gaze_latency_ms=gaze_latency_ms,
        flash_leg=flash_leg if flash_shown else None,
        jump_back_ms_after_go_cue=legs.ms_after_go_cue(legs.jump_at),
        return_reaction_ms=legs.return_reaction_ms,
        jump_back_forced=legs.jump_forced,
        out_arrival_ms_after_go_cue=legs.ms_after_go_cue(legs.out_arrived_at),
        left_target_before_jump=legs.left_target_before_jump(),
    )
