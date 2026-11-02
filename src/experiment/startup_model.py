"""Qt-free model of the startup dialog (src/experiment/startup_dialog.py): the settings, their validation, the remembered defaults and the help texts.
Everything here is unit-tested without a display."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from include.experiment.constants import HARDWARE_CONTRAST_FLOOR, SCREEN_HEIGHT_CM, VIEWING_DISTANCE_CM
from src.experiment import session_files
from src.experiment.geometry import Geometry
from src.experiment.settings import (
    DATA_DIR,
    DEFAULT_SACCADE_CONTRAST_MAX_PERCENT,
    DEFAULT_SACCADE_CONTRAST_MIN_PERCENT,
    MAX_SACCADE_CONTRAST_PERCENT,
    load_last_startup,
)

MODE_REAL = "Real"
MODE_TEST = "Test"

SCREEN_HEIGHT_RANGE_CM = (5.0, 100.0)
VIEWING_DISTANCE_RANGE_CM = (20.0, 300.0)

PLACEHOLDER_NOTE = "Placeholder values: please measure your screen."

# Field key -> label text, in dialog order.
FIELD_LABELS = {
    "participant_id": "Participant ID",
    "mode": "Mode",
    "run_tutorial": "Run tutorial",
    "screen_height_cm": "Screen height (cm)",
    "viewing_distance_cm": "Viewing distance (cm)",
    "contrast_min_percent": "Saccade contrast min (%)",
    "contrast_max_percent": "Saccade contrast max (%)",
    "show_gaze_indicator": "Show gaze indicator",
}
_CONTRAST_HELP = (
    "Faintest and brightest line contrast in percent (default 4 and 8); three levels in between are shown. "
    "Too faint = guessing; too bright = everyone gets it right."
)
# Field key -> help bubble text. One place for the dialog, the README and the tests.
HELP_TEXTS = {
    "participant_id": "Who is doing the test. Use a study code (for example P01), not a name. Letters, digits, - and _. Results go to data/<ID>/ and a new person gets the tutorial.",
    "mode": "Real = full session. Test = short trial run with debug text, saved as a _test run.",
    "run_tutorial": "Explains the task step by step and calibrates the eye tracker. Recommended for anyone new.",
    "screen_height_cm": "Height of the visible picture on the monitor in cm (measure it). Used to make the line the right size in degrees.",
    "viewing_distance_cm": "From the participant's eyes to the screen in cm (chin rest distance).",
    "contrast_min_percent": _CONTRAST_HELP,
    "contrast_max_percent": _CONTRAST_HELP,
    "show_gaze_indicator": "Debug only: shows where the tracker thinks you look.",
}


@dataclass(frozen=True)
class StartupSettings:
    """What the startup dialog hands back (validated)."""

    participant_id: str = ""
    test_mode: bool = False
    run_tutorial: bool = True
    screen_height_cm: float = SCREEN_HEIGHT_CM
    viewing_distance_cm: float = VIEWING_DISTANCE_CM
    contrast_min_percent: float = DEFAULT_SACCADE_CONTRAST_MIN_PERCENT
    contrast_max_percent: float = DEFAULT_SACCADE_CONTRAST_MAX_PERCENT
    show_gaze_indicator: bool = False

    @property
    def geometry(self) -> Geometry:
        return Geometry(self.screen_height_cm, self.viewing_distance_cm)

    @property
    def folder_id(self) -> str:
        """The folder name the participant ID is saved under (`anonymous` if blank)."""
        return session_files.sanitize_participant_id(self.participant_id)


def parse_number(text: str) -> float:
    """A decimal number typed by a person: a comma may be the decimal separator ('26,5'). Raises ValueError."""
    return float(text.strip().replace(",", "."))


def _number_in(text: str, name: str, low: float, high: float, unit: str) -> float:
    try:
        value = parse_number(text)
    except ValueError:
        raise ValueError(f"{name}: enter a number") from None
    if not (low <= value <= high):  # also rejects nan
        raise ValueError(f"{name} must be between {low:g} and {high:g} {unit}")
    return value


def validate(
    participant_id: str,
    mode: str,
    run_tutorial: bool,
    screen_height_text: str,
    viewing_distance_text: str,
    contrast_min_text: str,
    contrast_max_text: str,
    show_gaze_indicator: bool,
) -> StartupSettings:
    """The dialog's raw field values -> StartupSettings, or ValueError with a message for the status line."""
    if mode == MODE_REAL and not participant_id.strip():
        raise ValueError("Real mode needs a Participant ID")
    height = _number_in(screen_height_text, "Screen height", *SCREEN_HEIGHT_RANGE_CM, "cm")
    distance = _number_in(viewing_distance_text, "Viewing distance", *VIEWING_DISTANCE_RANGE_CM, "cm")
    low = round(HARDWARE_CONTRAST_FLOOR * 100, 2)
    cmin = _number_in(contrast_min_text, "Contrast min", low, MAX_SACCADE_CONTRAST_PERCENT, "%")
    cmax = _number_in(contrast_max_text, "Contrast max", low, MAX_SACCADE_CONTRAST_PERCENT, "%")
    if not cmin < cmax:
        raise ValueError("Contrast min must be smaller than max")
    return StartupSettings(
        participant_id=participant_id.strip(),
        test_mode=mode == MODE_TEST,
        run_tutorial=bool(run_tutorial),
        screen_height_cm=height,
        viewing_distance_cm=distance,
        contrast_min_percent=cmin,
        contrast_max_percent=cmax,
        show_gaze_indicator=bool(show_gaze_indicator),
    )


def is_placeholder_geometry(height_text: str, distance_text: str) -> bool:
    """True when the two screen fields still hold the shipped placeholder values."""
    try:
        return (parse_number(height_text), parse_number(distance_text)) == (SCREEN_HEIGHT_CM, VIEWING_DISTANCE_CM)
    except ValueError:
        return False


def suggest_tutorial(participant_id: str, data_dir: Path | None = None) -> bool:
    """Whether to default 'Run tutorial' to Yes: a participant with no earlier run folder under data/ is new."""
    folder = (data_dir or DATA_DIR) / session_files.sanitize_participant_id(participant_id)
    try:
        return not (folder.is_dir() and any(folder.iterdir()))
    except OSError:
        return True


def defaults_from_disk(data_dir: Path | None = None) -> StartupSettings:
    """The last used values (data/settings.json, key `last_startup`; missing, damaged or old files -> shipped defaults), with 'Run tutorial'
    derived from whether the last participant already has runs."""
    stored = load_last_startup()
    shipped = StartupSettings()
    values = {}
    for field in fields(StartupSettings):
        if field.name == "run_tutorial" or field.name not in stored:
            continue
        value, default = stored[field.name], getattr(shipped, field.name)
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            value = float(value) if ok else value
        else:
            ok = isinstance(value, type(default))
        if ok:
            values[field.name] = value
    base = StartupSettings(**values)
    return StartupSettings(**{**asdict(base), "run_tutorial": suggest_tutorial(base.participant_id, data_dir)})
