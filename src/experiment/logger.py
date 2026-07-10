import csv
import time
from pathlib import Path

from include.experiment.types import TrialResult

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FIELDNAMES = [
    "trial_index",
    "source",
    "target",
    "saccade_duration_ms",
    "square_shown",
    "contrast",
    "responded",
    "response_time_ms",
    "outcome",
]


class ResultLogger:
    """Appends one CSV row per completed trial to data/results_<timestamp>.csv."""

    def __init__(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        path = DATA_DIR / f"results_{int(time.time())}.csv"
        self._file = path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

    def log(self, result: TrialResult) -> None:
        self._writer.writerow(
            {
                "trial_index": result.index,
                "source": result.source.value,
                "target": result.target.value,
                "saccade_duration_ms": result.saccade_duration_ms,
                "square_shown": result.square_shown,
                "contrast": result.contrast,
                "responded": result.responded,
                "response_time_ms": result.response_time_ms,
                "outcome": result.outcome,
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
