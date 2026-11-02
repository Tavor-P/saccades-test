"""Persistent dashboard-editable settings (data/settings.json), currently the default contrast floor, with validation against ZEST's renderable range."""

import json
from pathlib import Path

from include.experiment.constants import (
    HARDWARE_CONTRAST_FLOOR,
    SACCADE_CONTRAST_LEVELS,
    SACCADE_FIXED_CONTRASTS,
    ZEST_LOG_CONTRAST_MAX,
    ZEST_LOG_CONTRAST_MIN,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Bounds a configured floor can't cross: below MIN, trials would flash below
# what an 8-bit display can even render (see HARDWARE_CONTRAST_FLOOR in
# constants.py) - exactly the bug ZEST_LOG_CONTRAST_MIN itself was fixed to
# avoid, so a user-editable floor can't be allowed to reintroduce it. At or
# above MAX it collides with ZEST's own contrast ceiling.
MIN_CONTRAST_FLOOR_PERCENT = HARDWARE_CONTRAST_FLOOR * 100
MAX_CONTRAST_FLOOR_PERCENT = (10**ZEST_LOG_CONTRAST_MAX) * 100
DEFAULT_CONTRAST_FLOOR_PERCENT = (10**ZEST_LOG_CONTRAST_MIN) * 100


def _settings_path() -> Path:
    return DATA_DIR / "settings.json"


def validate_contrast_floor_percent(value: float) -> float:
    if not (MIN_CONTRAST_FLOOR_PERCENT <= value < MAX_CONTRAST_FLOOR_PERCENT):
        raise ValueError(
            f"contrast floor must be between {MIN_CONTRAST_FLOOR_PERCENT:.2f}% and "
            f"{MAX_CONTRAST_FLOOR_PERCENT:.0f}%"
        )
    return value


def load_contrast_floor_percent() -> float:
    """The dashboard-configurable default contrast floor (%), read fresh each
    call so a change saved there takes effect the next time the experiment
    launches without anything needing to be restarted. Falls back to ZEST's
    own built-in floor if settings.json doesn't exist yet or predates this
    setting."""
    path = _settings_path()
    if path.exists():
        value = json.loads(path.read_text()).get("contrast_floor_percent")
        if value is not None:
            return float(value)
    return DEFAULT_CONTRAST_FLOOR_PERCENT


def save_contrast_floor_percent(value: float) -> None:
    validate_contrast_floor_percent(value)
    DATA_DIR.mkdir(exist_ok=True)
    path = _settings_path()
    data = json.loads(path.read_text()) if path.exists() else {}
    data["contrast_floor_percent"] = value
    path.write_text(json.dumps(data, indent=2))


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


def resolve_saccade_contrasts(raw_min: str, raw_max: str) -> tuple[float, ...]:
    """The levels for the dialog's two text fields; a blank, non-numeric or out-of-range entry falls
    back to the defaults rather than crashing the start of a session."""
    try:
        return saccade_contrast_levels(float(raw_min), float(raw_max))
    except (TypeError, ValueError):
        return saccade_contrast_levels(DEFAULT_SACCADE_CONTRAST_MIN_PERCENT, DEFAULT_SACCADE_CONTRAST_MAX_PERCENT)


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
