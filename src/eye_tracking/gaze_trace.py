"""Record raw Neon gaze to a CSV, and replay it later as a GazeSource.

A recorded trace holds real tracker noise, real saccades and real stalls, so
tests (and offline experiments with detector settings like
SACCADE_ONSET_STABILITY_MS) can run against genuine gaze without the glasses.

Rows are the raw readings, before any calibration/mapping - replay pushes them
through the same GazeMapper NeonGazeSource uses, so calibration, zone
classification and smoothing behave identically to the live stream.
"""

import csv
import math
import time
from bisect import bisect_right
from pathlib import Path
from typing import Callable, NamedTuple

from include.eye_tracking.constants import NEON_SCENE_WIDTH, NEON_STALE_TIMEOUT_S
from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.eye_tracking.gaze_mapping import GazeMapper

TRACE_FIELDS = ["t_s", "device_unix_s", "x_px", "worn"]


class TraceRow(NamedTuple):
    t_s: float  # PC arrival time (monotonic), seconds
    device_unix_s: float | None  # the phone's capture timestamp, if known
    x_px: float | None  # scene-camera gaze x in pixels; None/NaN when unusable
    worn: bool


class GazeTraceWriter:
    """Appends raw gaze readings to a CSV. Called from the Neon receive thread
    at ~200 Hz, so it only does a buffered csv write - flushed on close()."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(TRACE_FIELDS)

    def write(self, t_s: float, device_unix_s: float | None, x_px: float | None, worn: bool) -> None:
        self._writer.writerow(
            [
                f"{t_s:.6f}",
                "" if device_unix_s is None else f"{device_unix_s:.6f}",
                "" if x_px is None else f"{x_px:.3f}",
                int(worn),
            ]
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def load_trace(path: Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                TraceRow(
                    t_s=float(r["t_s"]),
                    device_unix_s=float(r["device_unix_s"]) if r["device_unix_s"] else None,
                    x_px=float(r["x_px"]) if r["x_px"] else None,
                    worn=r["worn"] == "1",
                )
            )
    rows.sort(key=lambda row: row.t_s)
    return rows


class ReplayGazeSource(GazeSource):
    """Plays a recorded trace back in real time (or fake time, via `time_fn`).

    latest_sample() feeds every row that has "arrived" since the previous call
    through the mapper, in order, so the mapper's calibration history and
    smoothing evolve exactly as they did on the live stream, no matter how
    often the caller polls. After the trace ends the source reports a lost
    track (as the live source does once its stream goes stale)."""

    def __init__(self, trace: list[TraceRow], time_fn: Callable[[], float] = time.monotonic) -> None:
        if not trace:
            raise ValueError("empty gaze trace")
        self._trace = trace
        self._times = [row.t_s for row in trace]
        self._time_fn = time_fn
        self._mapper = GazeMapper()
        self._t0: float | None = None
        self._next_row = 0
        self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=0.0)

    @property
    def is_available(self) -> bool:
        return True

    def start(self) -> None:
        self._t0 = self._time_fn()

    def stop(self) -> None:
        pass

    @property
    def elapsed_s(self) -> float:
        return (self._time_fn() - self._t0) if self._t0 is not None else 0.0

    @property
    def finished(self) -> bool:
        return self.elapsed_s >= self._times[-1] - self._times[0]

    def latest_sample(self) -> GazeSample:
        if self._t0 is None:
            return self._latest
        trace_time = self._times[0] + self.elapsed_s
        stop = bisect_right(self._times, trace_time)
        for row in self._trace[self._next_row : stop]:
            self._latest = self._sample_for(row)
        self._next_row = max(self._next_row, stop)
        if self._next_row > 0 and trace_time - self._times[self._next_row - 1] > NEON_STALE_TIMEOUT_S:
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=self._time_fn())
        return self._latest

    def _sample_for(self, row: TraceRow) -> GazeSample:
        if not row.worn or row.x_px is None or math.isnan(row.x_px):
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=row.t_s)
        return self._mapper.map(row.x_px / NEON_SCENE_WIDTH, row.t_s)

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._mapper.calibrate(left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return self._mapper.average_recent_ratio()

    def begin_calibration_sample(self) -> None:
        self._mapper.reset_ratio_history()
