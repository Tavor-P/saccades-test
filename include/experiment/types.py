from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    DOT = "dot"
    CROSS = "cross"


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
    source: Target
    target: Target
    saccade_duration_ms: float | None
    square_shown: bool
    contrast: float | None
    responded: bool
    response_time_ms: float | None
    outcome: str  # "hit" | "miss" | "false_alarm" | "correct_rejection" | "timeout"
