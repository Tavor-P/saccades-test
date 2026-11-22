"""Persistent settings (data/settings.json): the startup dialog's remembered values, plus the saccade-contrast range helpers it validates against."""

import json
from pathlib import Path

from include.experiment.constants import (
    HARDWARE_CONTRAST_FLOOR,
    SACCADE_CONTRAST_LEVELS,
    SACCADE_FIXED_CONTRASTS,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _settings_path() -> Path:
    return DATA_DIR / "settings.json"


# -- saccade-phase probe contrasts (the startup dialog's min/max fields) ---------------------

MAX_SACCADE_CONTRAST_PERCENT = 100.0
DEFAULT_SACCADE_CONTRAST_MIN_PERCENT = min(SACCADE_FIXED_CONTRASTS) * 100
DEFAULT_SACCADE_CONTRAST_MAX_PERCENT = max(SACCADE_FIXED_CONTRASTS) * 100


def validate_saccade_contrast_range(min_percent: float, max_percent: float) -> tuple[float, float]:
    """Both ends must be renderable (>= the 8-bit hardware floor), at most 100 %, and min <= max."""
    low = HARDWARE_CONTRAST_FLOOR * 100
    if not (low <= min_percent <= max_percent <= MAX_SACCADE_CONTRAST_PERCENT):
        raise ValueError(
            f"saccade contrast range must satisfy {low:.2f}% <= min <= max <= {MAX_SACCADE_CONTRAST_PERCENT:.0f}% "
            f"(got {min_percent} .. {max_percent})"
        )
    return min_percent, max_percent


def saccade_contrast_levels(min_percent: float, max_percent: float, count: int = SACCADE_CONTRAST_LEVELS) -> tuple[float, ...]:
    """`count` evenly spaced contrasts (as 0-1 fractions) from min to max, inclusive; a single value
    if count <= 1 or min == max."""
    validate_saccade_contrast_range(min_percent, max_percent)
    if count <= 1 or min_percent == max_percent:
        return (round(min_percent / 100, 6),)
    step = (max_percent - min_percent) / (count - 1)
    return tuple(round((min_percent + i * step) / 100, 6) for i in range(count))


# -- the startup dialog's remembered values (key `last_startup` in settings.json) -------------


def load_last_startup() -> dict:
    """The startup dialog's last values (a plain dict; {} if there is no settings file, it is unreadable or predates this key)."""
    path = _settings_path()
    try:
        value = json.loads(path.read_text()).get("last_startup") if path.exists() else None
    except (OSError, ValueError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_last_startup(values: dict) -> None:
    """Remember the startup dialog's values, leaving the other settings alone."""
    DATA_DIR.mkdir(exist_ok=True)
    path = _settings_path()
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["last_startup"] = values
    path.write_text(json.dumps(data, indent=2))
