import csv
import time
from pathlib import Path

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
    """Appends one CSV row per completed trial to data/results_<timestamp>.csv.

    Shared across both phases of a run (presaccade + saccade), so the whole
    session lands in one file distinguished by the `phase` column - that's
    what the end-of-run comparison graph reads from.
    """

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
