import csv
import json
import time
from datetime import datetime
from pathlib import Path

from include.experiment.constants import (
    GRATING_ENVELOPE_SIGMA_DEG,
    GRATING_SPATIAL_FREQUENCY_CPD,
    SCREEN_HEIGHT_CM,
    TEST_MODE,
    VIEWING_DISTANCE_CM,
)
from include.experiment.types import TrialResult

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FIELDNAMES = [
    "trial_index",
    "phase",
    "source",
    "target",
    "saccade_duration_ms",
    "grating_shown",
    "contrast",
    "responded",
    "response_time_ms",
    "outcome",
]


class ResultLogger:
    """Appends one CSV row per completed trial to data/results_<timestamp>.csv,
    plus a data/results_<timestamp>_meta.json sidecar with session-level
    context that doesn't fit a per-trial row (who ran it, when, under what
    assumptions/calibration) - useful provenance once you're comparing CSVs
    across sessions or participants.

    Shared across both phases of a run (presaccade + saccade), so the whole
    session lands in one CSV distinguished by the `phase` column - that's
    what the end-of-run comparison graph reads from.
    """

    def __init__(self, participant_id: str, camera_label: str | None = None) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        timestamp = int(time.time())
        csv_path = DATA_DIR / f"results_{timestamp}.csv"
        self._meta_path = DATA_DIR / f"results_{timestamp}_meta.json"

        self._file = csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

        self._metadata = {
            "participant_id": participant_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "test_mode": TEST_MODE,
            "camera": camera_label,
            "viewing_distance_cm": VIEWING_DISTANCE_CM,
            "screen_height_cm": SCREEN_HEIGHT_CM,
            "grating_spatial_frequency_cpd": GRATING_SPATIAL_FREQUENCY_CPD,
            "grating_envelope_sigma_deg": GRATING_ENVELOPE_SIGMA_DEG,
            "calibration": None,
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        self._meta_path.write_text(json.dumps(self._metadata, indent=2))

    def set_calibration(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._metadata["calibration"] = {
            "left_ratio": left_ratio,
            "center_ratio": center_ratio,
            "right_ratio": right_ratio,
        }
        self._write_metadata()

    def log(self, result: TrialResult) -> None:
        self._writer.writerow(
            {
                "trial_index": result.index,
                "phase": result.phase,
                "source": result.source.value if result.source is not None else "",
                "target": result.target.value if result.target is not None else "",
                "saccade_duration_ms": result.saccade_duration_ms,
                "grating_shown": result.grating_shown,
                "contrast": result.contrast,
                "responded": result.responded,
                "response_time_ms": result.response_time_ms,
                "outcome": result.outcome,
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
