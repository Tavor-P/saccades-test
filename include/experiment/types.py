from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


@dataclass(frozen=True)
class FlashTrialSpec:
    """A presaccade-phase trial: just fixate center, maybe a flash, no saccade."""

    index: int
    square_shown: bool
    contrast: float | None


@dataclass(frozen=True)
class TrialSpec:
    index: int
    source: Target
    target: Target
    square_shown: bool
    contrast: float | None


@dataclass
class TrialResult:
    index: int
    phase: str  # "presaccade" | "saccade"
    source: Target | None
    target: Target | None
    saccade_duration_ms: float | None
    square_shown: bool
    contrast: float | None
    responded: bool
    response_time_ms: float | None
    outcome: str  # "hit" | "miss" | "false_alarm" | "correct_rejection" | "timeout"
