"""Screen geometry for the visual-angle sizes: `Geometry` turns the screen height and viewing distance (entered in the startup dialog) into
the line's offset and thickness in screen-height units. The constants in include/experiment/constants.py are only the defaults (placeholders)."""

import math
from dataclasses import dataclass

from include.experiment.constants import (
    LINE_OFFSET_DEG,
    LINE_SD_DEG,
    SCREEN_HEIGHT_CM,
    VIEWING_DISTANCE_CM,
)


@dataclass(frozen=True)
class Geometry:
    """Immutable (screen height, viewing distance) pair; every angular size is derived from it. Built once per run and passed to what draws or
    places the line, so nothing depends on import-time globals."""

    screen_height_cm: float = SCREEN_HEIGHT_CM
    viewing_distance_cm: float = VIEWING_DISTANCE_CM

    def __post_init__(self) -> None:
        if not (self.screen_height_cm > 0 and self.viewing_distance_cm > 0):
            raise ValueError("screen height and viewing distance must be positive")

    @classmethod
    def default(cls) -> "Geometry":
        """The shipped placeholder values (read from the constants at call time, so tests may override them)."""
        return cls(SCREEN_HEIGHT_CM, VIEWING_DISTANCE_CM)

    @property
    def height_deg(self) -> float:
        return 2 * math.degrees(math.atan((self.screen_height_cm / 2) / self.viewing_distance_cm))

    @property
    def is_placeholder(self) -> bool:
        return (self.screen_height_cm, self.viewing_distance_cm) == (SCREEN_HEIGHT_CM, VIEWING_DISTANCE_CM)

    def deg_to_height_units(self, degrees: float) -> float:
        return degrees / self.height_deg

    @property
    def line_offset_height_units(self) -> float:
        return self.deg_to_height_units(LINE_OFFSET_DEG)

    @property
    def line_sd_height_units(self) -> float:
        return self.deg_to_height_units(LINE_SD_DEG)

    @property
    def line_height_height_units(self) -> float:
        """PsychoPy's 'gauss' mask has SD = size/6, so a mask this tall has SD = LINE_SD_DEG."""
        return self.line_sd_height_units * 6

    def as_settings(self) -> dict:
        """What session.json `settings` records about the geometry."""
        return {
            "screen_height_cm": self.screen_height_cm,
            "viewing_distance_cm": self.viewing_distance_cm,
            "screen_height_deg": round(self.height_deg, 3),
            "line_offset_deg": LINE_OFFSET_DEG,
            "line_sd_deg": LINE_SD_DEG,
            "line_offset_height_units": round(self.line_offset_height_units, 5),
            "line_sd_height_units": round(self.line_sd_height_units, 6),
            "geometry_is_placeholder": self.is_placeholder,
        }
