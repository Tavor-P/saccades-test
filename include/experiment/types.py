"""Dataclasses/enums for experiment data: Target, Side, trial specs, and TrialResult (one logged CSV row per trial).
Adding a TrialResult field: see CLAUDE.md "Recipes" (this file, logger.TRIAL_COLUMNS or DROPPED_FROM_CSV, trial_from_row, session.py, tests)."""

from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


class Side(str, Enum):
    """Where the flashed line sits relative to the saccade target on a given trial - reported back with the UP / DOWN arrow key, turning detection
    into a 2-alternative forced-choice task (Braun, Schutz & Gegenfurtner 2021). None in the
    rt_test rows of the log, where no line is shown. The trial/result fields that hold a Side are
    `line_side` / `answer_side` (the trials.csv columns of the same names); old CSVs held the
    grating's 'vertical' / 'horizontal' in columns called orientation / response_orientation
    (logger reads them as ABOVE / BELOW), new ones 'above' / 'below'."""

    ABOVE = "above"
    BELOW = "below"


@dataclass(frozen=True)
class TrialSpec:
    index: int
    source: Target
    target: Target
    line_side: Side
    practice: bool = False
    # Which of TIMING_OFFSETS_MS this trial's flash is scheduled at, relative
    # to the avg_reaction_time_ms in effect when the trial started - None for
    # trials predating this field, or for anything that doesn't use open-loop
    # scheduling. See ExperimentSession._tick_trial_active.
    timing_offset_ms: int | None = None
    # 'scheduled' (the open-loop delay above) or 'onset': flashed when the live onset detector fires on the flash's
    # leg (ONSET_TRIGGER_FRACTION; such a trial has no timing_offset_ms).
    flash_trigger: str = "scheduled"


@dataclass
class TrialResult:
    index: int
    phase: str  # "saccade" | "rt_test" (old sessions also: "presaccade", the removed baseline phase)
    source: Target | None
    target: Target | None
    saccade_duration_ms: float | None
    flash_shown: bool
    contrast: float | None
    line_side: Side | None
    responded: bool
    answer_side: Side | None
    response_time_ms: float | None
    outcome: str  # "correct" | "incorrect" | "miss" | "timeout"  (old sessions also: "false_alarm" | "correct_rejection")
    # Saccade phase only (None in old presaccade rows, which have no saccade to
    # time): go-cue to the gaze classifier's first sample beyond the
    # displacement threshold, and that sample to the debounced onset (confirmation
    # lag) - together they let SACCADE_ONSET_STABILITY_MS be tuned against
    # measured latency instead of guesswork. Also populated (onset only) on
    # "rt_test" rows, where it's literally the thing being measured.
    reaction_latency_ms: float | None = None
    onset_detection_lag_ms: float | None = None
    # Saccade phase, flash-shown trials only (None otherwise - old presaccade
    # rows or rt_test rows, which never flash): whether
    # this trial's scheduled flash time actually fell within
    # [onset, landing] - the real-time-detected saccade window. False if
    # onset was never detected (no saccade evidence at flash time at all);
    # None if landing was never confirmed before the trial ended (the window
    # is open-ended, so validity is genuinely undeterminable, not merely
    # false). See ExperimentSession._finish_trial.
    flash_during_saccade: bool | None = None
    # Which of TIMING_OFFSETS_MS this trial's flash was scheduled at - see
    # TrialSpec.timing_offset_ms. None for rows that don't use open-loop
    # scheduling (old presaccade, rt_test, or CSVs logged before this existed).
    timing_offset_ms: int | None = None
    # Wall-clock (`timebase.unix_ms()`: milliseconds since the Unix epoch)
    # timestamp of the moment the line actually flashed - unlike every other
    # timing field here, which is on the `src/timebase.py` clock (`now()` =
    # perf_counter) and only meaningful within one trial/session. This is what lets the flash
    # be located in absolute time against something recorded independently
    # (e.g. a synced video/frame log). None for old presaccade
    # rows and rt_test rows, which never flash.
    flash_unix_ms: float | None = None
    # Saccade phase, flash-shown trials only (None otherwise, and for CSVs
    # logged before this existed): False if the gaze source lost tracking
    # within TRACKING_LOSS_WINDOW_MS of the scheduled flash. See
    # scoring.is_valid_for_saccadic_analysis.
    tracking_ok: bool | None = None
    # Saccade phase, flash-shown trials only: the scheduled flash time minus
    # the detected onset (ms; positive = flash after onset, None if onset was
    # never detected) - the quantity the open-loop timing is trying to land
    # inside the saccade. avg_rt_used_ms / scheduled_delay_ms: the reaction-
    # time average in effect when the trial was scheduled and the resulting
    # go-cue-to-flash delay, drawn from the bell curve around the average RT
    # (`timing_offset_ms` is the slice label), for every saccade trial.
    # gaze_latency_ms: measured latency; subtracted only for sources that stamp
    # arrival time (not Neon).
    flash_relative_to_onset_ms: float | None = None
    avg_rt_used_ms: float | None = None
    scheduled_delay_ms: float | None = None
    gaze_latency_ms: float | None = None
    # Return-jump trials (run setting "Return jump"; None in rt_test rows and CSVs logged before it existed):
    # flash_leg: 'out' (flash during the go-cue saccade) or 'return' (during the saccade back after the
    # target jumped back to the start symbol); set on every flash-shown saccade-phase trial ('out' when the
    # setting is off). jump_back_ms_after_go_cue: the frame the target jumped back, ms after the go cue
    # (same timeline as scheduled_delay_ms, which stays flash minus go cue for both legs).
    # return_reaction_ms: jump-back frame to the return saccade's detected onset, SIGNED (capture time; < 0 =
    # the eye turned back before the jump was drawn; None if no onset). jump_back_forced: True if the jump
    # trigger (RETURN_JUMP_TRIGGER) never happened and the jump came at RETURN_JUMP_LANDING_TIMEOUT_MS.
    # out_arrival_ms_after_go_cue: first sample inside the target zone, ms after the go cue (with the jump on
    # this is the out landing used for saccade_duration_ms / flash_during_saccade; lets the analysis re-anchor
    # return latency to the arrival). left_target_before_jump: the last sample captured before the jump frame
    # was already outside the target zone (after arriving). These are None when the setting is off or no jump happened.
    # For 'return' rows flash_during_saccade / flash_relative_to_onset_ms / avg_rt_used_ms refer to the
    # return leg; reaction_latency_ms / onset_detection_lag_ms / saccade_duration_ms always to the out leg.
    flash_leg: str | None = None
    jump_back_ms_after_go_cue: float | None = None
    return_reaction_ms: float | None = None
    jump_back_forced: bool | None = None
    out_arrival_ms_after_go_cue: float | None = None
    left_target_before_jump: bool | None = None
    # Saccade phase, every row (None in rt_test rows and CSVs logged before it existed): what the trial's flash was timed
    # by, 'scheduled' (open-loop delay) or 'onset' (the live onset on the flash's leg, run setting "Onset-triggered
    # flashes"). An 'onset' row without a flash had no detected onset to trigger on. For shown 'onset' flashes
    # scheduled_delay_ms is still flash minus go cue; timing_offset_ms / avg_rt_used_ms are None (nothing was scheduled).
    flash_trigger: str | None = None
    # Return-jump trials with the "arrival" trigger: this trial's drawn dwell (ms), the continuous stay on the target the
    # jump back waited for (RETURN_JUMP_MAIN_DWELL_MIN_MS..MAX_MS; a forced jump may have come without it). None with the
    # setting off, with the "onset" trigger, in rt_test rows and in CSVs logged before it existed.
    jump_back_dwell_ms: float | None = None
