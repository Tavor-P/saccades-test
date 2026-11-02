"""Dataclasses/enums for experiment data: Target, Side, trial specs, and TrialResult (one logged CSV row per trial).
Adding a TrialResult field means touching: this file, logger.FIELDNAMES + ResultLogger.log, dashboard data_access parsing, and tests."""

from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


class Side(str, Enum):
    """Where the flashed line sits relative to the saccade target on a given trial - reported back with the UP / DOWN arrow key, turning detection
    into a 2-alternative forced-choice task (Braun, Schutz & Gegenfurtner 2021). None in the
    rt_test rows of the log, where no line is shown. The trial/result fields that hold a Side are
    still called `orientation` / `response_orientation` (and so is the CSV column) so existing
    CSVs, the dashboard and the schema checks keep working; old CSVs hold 'vertical' / 'horizontal'
    there, new ones 'above' / 'below'."""

    ABOVE = "above"
    BELOW = "below"


@dataclass(frozen=True)
class TrialSpec:
    index: int
    source: Target
    target: Target
    orientation: Side
    practice: bool = False
    # Which of TIMING_OFFSETS_MS this trial's flash is scheduled at, relative
    # to the avg_reaction_time_ms in effect when the trial started - None for
    # trials predating this field, or for anything that doesn't use open-loop
    # scheduling. See ExperimentSession._tick_trial_active.
    timing_offset_ms: int | None = None


@dataclass
class TrialResult:
    index: int
    phase: str  # "saccade" | "rt_test" (old sessions also: "presaccade", the removed baseline phase)
    source: Target | None
    target: Target | None
    saccade_duration_ms: float | None
    grating_shown: bool
    contrast: float | None
    orientation: Side | None
    responded: bool
    response_orientation: Side | None
    response_time_ms: float | None
    outcome: str  # "correct" | "incorrect" | "miss" | "timeout"  (old sessions also: "false_alarm" | "correct_rejection")
    # Saccade phase only (None in old presaccade rows, which have no saccade to
    # time): go-cue to the gaze classifier's first sample outside the source
    # zone, and that sample to the debounced onset that actually fired the
    # flash - together they let SACCADE_ONSET_STABILITY_MS be tuned against
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
    # Wall-clock (time.time(), milliseconds since the Unix epoch) timestamp
    # of the moment the grating actually flashed - unlike every other timing
    # field here, which is relative to PausableClock's monotonic clock and
    # only meaningful within one trial/session. This is what lets the flash
    # be located in absolute time against something recorded independently
    # (e.g. a synced video/frame log). None for old presaccade
    # rows and rt_test rows, which never flash.
    grating_shown_at_unix_ms: float | None = None
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
    # go-cue-to-flash delay (avg + timing offset), for every saccade trial.
    # gaze_latency_ms: the eye-to-sample latency subtracted from onset/landing
    # timestamps for this trial (0 if the source reports none).
    flash_relative_to_onset_ms: float | None = None
    avg_rt_used_ms: float | None = None
    scheduled_delay_ms: float | None = None
    gaze_latency_ms: float | None = None
