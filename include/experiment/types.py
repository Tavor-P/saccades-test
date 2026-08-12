from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


class Orientation(str, Enum):
    """The grating's stripe orientation on a given trial - reported back via
    up/down (vertical) or left/right (horizontal) arrow keys, turning
    detection into a 2-alternative forced-choice discrimination task. None
    on catch trials, where no grating (and so no orientation) is shown."""

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@dataclass(frozen=True)
class FlashTrialSpec:
    """A presaccade-phase trial: just fixate center, maybe a flash, no saccade.
    Contrast isn't known ahead of time - it's picked from the ZEST staircase
    right when the trial actually flashes (or fixed at PRACTICE_CONTRAST for
    practice trials, which also skip the staircase update and logging)."""

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
    # Which of TIMING_OFFSETS_MS this trial's flash is scheduled at, relative
    # to the avg_reaction_time_ms in effect when the trial started - None for
    # trials predating this field, or for anything that doesn't use open-loop
    # scheduling. See ExperimentSession._tick_trial_active.
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
    # Saccade phase only (None in presaccade rows, which have no saccade to
    # time): go-cue to the gaze classifier's first sample outside the source
    # zone, and that sample to the debounced onset that actually fired the
    # flash - together they let SACCADE_ONSET_STABILITY_MS be tuned against
    # measured latency instead of guesswork. Also populated (onset only) on
    # "rt_test" rows, where it's literally the thing being measured.
    reaction_latency_ms: float | None = None
    onset_detection_lag_ms: float | None = None
    # Saccade phase, flash-shown trials only (None otherwise - presaccade
    # rows, catch trials, or rt_test rows, none of which ever flash): whether
    # this trial's scheduled flash time actually fell within
    # [onset, landing] - the real-time-detected saccade window. False if
    # onset was never detected (no saccade evidence at flash time at all);
    # None if landing was never confirmed before the trial ended (the window
    # is open-ended, so validity is genuinely undeterminable, not merely
    # false). See ExperimentSession._finish_trial.
    flash_during_saccade: bool | None = None
    # Which of TIMING_OFFSETS_MS this trial's flash was scheduled at - see
    # TrialSpec.timing_offset_ms. None for rows that don't use open-loop
    # scheduling (presaccade, rt_test, or CSVs logged before this existed).
    timing_offset_ms: int | None = None
