import json
from pathlib import Path

from include.experiment.constants import HARDWARE_CONTRAST_FLOOR, ZEST_LOG_CONTRAST_MAX, ZEST_LOG_CONTRAST_MIN

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
