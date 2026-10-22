"""ResultLogger: appends one CSV row per trial to data/results_<ts>.csv plus a _meta.json sidecar (participant, calibration, gaze_quality) and
provides the paths for the optional video and raw gaze-trace files of the same session."""

import csv
import functools
import importlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

from include.experiment.constants import (
    GRATING_ENVELOPE_SIGMA_DEG,
    GRATING_SPATIAL_FREQUENCY_CPD,
    SCREEN_HEIGHT_CM,
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
    "orientation",
    "responded",
    "response_orientation",
    "response_time_ms",
    "outcome",
    "reaction_latency_ms",
    "onset_detection_lag_ms",
    "flash_during_saccade",
    "timing_offset_ms",
    "grating_shown_at_unix_ms",
    "tracking_ok",
    "flash_relative_to_onset_ms",
    "avg_rt_used_ms",
    "scheduled_delay_ms",
    "gaze_latency_ms",
]


@functools.lru_cache(maxsize=1)
def _git_state() -> dict:
    """Commit SHA + whether the tree had uncommitted changes, so a results file
    can be tied back to the exact code that produced it. Best effort."""
    root = Path(__file__).resolve().parents[2]
    try:
        run = lambda *args: subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"sha": None, "dirty": None}


def _constants_snapshot() -> dict:
    """Every plain-valued UPPER_CASE constant the experiment and gaze tracking
    were configured with, so tuning at the time of a run is reproducible."""
    snapshot = {}
    for module_name in ("include.experiment.constants", "include.eye_tracking.constants"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, value in vars(module).items():
            if not name.isupper() or name.startswith("_"):
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                snapshot[name] = value
            elif isinstance(value, (tuple, list)) and all(isinstance(v, (bool, int, float, str)) for v in value):
                snapshot[name] = list(value)
    return snapshot


def _psychopy_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("psychopy")
    except Exception:
        return None


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

    def __init__(
        self,
        participant_id: str,
        camera_label: str | None = None,
        contrast_floor_percent: float | None = None,
        test_mode: bool | None = None,
    ) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        self._timestamp = int(time.time())
        csv_path = DATA_DIR / f"results_{self._timestamp}.csv"
        self._meta_path = DATA_DIR / f"results_{self._timestamp}_meta.json"

        self._file = csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

        self._metadata = {
            "participant_id": participant_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "test_mode": test_mode,
            "camera": camera_label,
            "viewing_distance_cm": VIEWING_DISTANCE_CM,
            "screen_height_cm": SCREEN_HEIGHT_CM,
            "grating_spatial_frequency_cpd": GRATING_SPATIAL_FREQUENCY_CPD,
            "grating_envelope_sigma_deg": GRATING_ENVELOPE_SIGMA_DEG,
            "contrast_floor_percent": contrast_floor_percent,
            "calibration": None,
            "git": _git_state(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "psychopy_version": _psychopy_version(),
            "display_refresh_hz": None,
            "constants": _constants_snapshot(),
            "phase_summaries": {},
        }
        self._write_metadata()

    @property
    def video_path(self) -> Path:
        """Where an optional session video recording (see
        src/eye_tracking/video_recorder.py) belongs, named to match this
        session's own results_<timestamp>.csv/_meta.json pair."""
        return DATA_DIR / f"results_{self._timestamp}_video.mp4"

    @property
    def gaze_trace_path(self) -> Path:
        """Raw gaze recording for this session (see src/eye_tracking/gaze_trace.py)."""
        return DATA_DIR / f"results_{self._timestamp}_gaze_trace.csv"

    @property
    def video_timestamps_path(self) -> Path:
        return DATA_DIR / f"results_{self._timestamp}_video_timestamps.csv"

    def _write_metadata(self) -> None:
        self._meta_path.write_text(json.dumps(self._metadata, indent=2))

    def set_calibration(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._metadata["calibration"] = {
            "left_ratio": left_ratio,
            "center_ratio": center_ratio,
            "right_ratio": right_ratio,
        }
        self._write_metadata()

    def set_display_refresh_hz(self, refresh_hz: float | None) -> None:
        """Measured monitor refresh rate (e.g. win.getActualFrameRate()), if the
        runner has one - frame-counted flash durations depend on it."""
        self._metadata["display_refresh_hz"] = refresh_hz
        self._write_metadata()

    def set_phase_summary(self, phase: str, summary: dict) -> None:
        """Final numbers for one phase ("saccade" / "presaccade"): threshold
        estimate, credible interval, trial counts, whether the estimate sits at
        a ZEST grid bound."""
        self._metadata.setdefault("phase_summaries", {})[phase] = summary
        self._write_metadata()

    def set_gaze_quality(self, summary: dict) -> None:
        """Session-level gaze tracking quality (sample rate, gaps, latency,
        clock offset) from a GazeSource that reports it - see
        NeonGazeSource.quality_summary()."""
        self._metadata["gaze_quality"] = summary
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
                "orientation": result.orientation.value if result.orientation is not None else "",
                "responded": result.responded,
                "response_orientation": result.response_orientation.value if result.response_orientation is not None else "",
                "response_time_ms": result.response_time_ms,
                "outcome": result.outcome,
                "reaction_latency_ms": result.reaction_latency_ms,
                "onset_detection_lag_ms": result.onset_detection_lag_ms,
                "flash_during_saccade": result.flash_during_saccade,
                "timing_offset_ms": result.timing_offset_ms,
                "grating_shown_at_unix_ms": result.grating_shown_at_unix_ms,
                "tracking_ok": result.tracking_ok,
                "flash_relative_to_onset_ms": result.flash_relative_to_onset_ms,
                "avg_rt_used_ms": result.avg_rt_used_ms,
                "scheduled_delay_ms": result.scheduled_delay_ms,
                "gaze_latency_ms": result.gaze_latency_ms,
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
