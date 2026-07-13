from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


@dataclass(frozen=True)
class FlashTrialSpec:
    """A presaccade-phase trial: just fixate center, maybe a flash, no saccade.
    Contrast isn't known ahead of time - it's picked from the ZEST staircase
    right when the trial actually flashes."""

    index: int
    grating_shown: bool


@dataclass(frozen=True)
class TrialSpec:
    index: int
    source: Target
    target: Target
    grating_shown: bool


@dataclass
class TrialResult:
    index: int
    phase: str  # "presaccade" | "saccade"
    source: Target | None
    target: Target | None
    saccade_duration_ms: float | None
    grating_shown: bool
    contrast: float | None
    responded: bool
    response_time_ms: float | None
    outcome: str  # "hit" | "miss" | "false_alarm" | "correct_rejection" | "timeout"
