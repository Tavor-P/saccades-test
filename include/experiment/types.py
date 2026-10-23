"""Dataclasses/enums for experiment data: Target, Orientation, trial specs, and TrialResult (one logged CSV row per trial).
Adding a TrialResult field means touching: this file, logger.FIELDNAMES + ResultLogger.log, dashboard data_access parsing, and tests."""

from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


class Orientation(str, Enum):
    """Grating stripe orientation, reported with up/down (vertical) or left/right (horizontal) keys: a 2AFC task. None on catch trials."""

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@dataclass(frozen=True)
class FlashTrialSpec:
    """A presaccade-phase trial: fixate center, maybe a flash. Contrast comes from the ZEST staircase when the flash happens (PRACTICE_CONTRAST for
    practice trials, which skip the staircase and logging)."""

    index: int
    grating_shown: bool
    orientation: Orientation | None = None
    practice: bool = False


@dataclass(frozen=True)
class TrialSpec:
    index: int
    source: Target
    target: Target
    grating_shown: bool
    orientation: Orientation | None = None
    practice: bool = False
    # Which of TIMING_OFFSETS_MS this trial's flash is scheduled at (added to avg_reaction_time_ms); None if not open-loop scheduled.
    timing_offset_ms: int | None = None


@dataclass
class TrialResult:
    index: int
    phase: str  # "presaccade" | "saccade" | "rt_test"
    source: Target | None
    target: Target | None
    saccade_duration_ms: float | None
    grating_shown: bool
    contrast: float | None
    orientation: Orientation | None
    responded: bool
    response_orientation: Orientation | None
    response_time_ms: float | None
    outcome: str  # "correct" | "incorrect" | "miss" | "false_alarm" | "correct_rejection" | "timeout"
    # Saccade and rt_test rows (None in presaccade rows): go-cue to detected onset (reaction_latency_ms), and first sample beyond the onset rule to
    # its debounced confirmation (onset_detection_lag_ms).
    reaction_latency_ms: float | None = None
    onset_detection_lag_ms: float | None = None
    # Flash-shown saccade trials only: whether the scheduled flash time fell within [onset, landing]. False if onset was never detected; None if
    # landing was never confirmed (validity undeterminable). See ExperimentSession._finish_trial.
    flash_during_saccade: bool | None = None
    # Which of TIMING_OFFSETS_MS this flash was scheduled at; None for rows without open-loop scheduling (presaccade, rt_test, old CSVs).
    timing_offset_ms: int | None = None
    # Wall-clock time (ms since the Unix epoch) the grating flashed - every other timing field is relative to PausableClock - so the flash can be
    # located against independent recordings such as the video frame log. None unless a grating was shown in a saccade trial.
    grating_shown_at_unix_ms: float | None = None
    # Flash-shown saccade trials only: False if gaze tracking was lost within TRACKING_LOSS_WINDOW_MS of the flash (scoring.is_valid_for_saccadic_analysis).
    tracking_ok: bool | None = None
    # flash_relative_to_onset_ms: flash time minus detected onset (ms; positive = after onset; flash trials, None if no onset) - what open-loop
    # timing tries to land inside the saccade. avg_rt_used_ms / scheduled_delay_ms: the reaction-time average used and the resulting go-cue-to-flash
    # delay. gaze_latency_ms: the source's measured gaze latency for this trial (0 if none).
    flash_relative_to_onset_ms: float | None = None
    avg_rt_used_ms: float | None = None
    scheduled_delay_ms: float | None = None
    gaze_latency_ms: float | None = None
