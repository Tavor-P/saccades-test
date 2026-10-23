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
import statistics
from bisect import bisect_right
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple

from include.eye_tracking.constants import (
    NEON_HISTORY_SAMPLES,
    NEON_MAX_PLAUSIBLE_LATENCY_MS,
    NEON_REPLAY_LATENCY_MS,
    NEON_SCENE_WIDTH,
    NEON_STALE_TIMEOUT_S,
)
from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.eye_tracking.gaze_mapping import GazeMapper
from src import timebase

TRACE_FIELDS = ["t_s", "device_unix_s", "x_px", "worn"]


class TraceRow(NamedTuple):
    t_s: float  # PC arrival time (timebase.now(), perf_counter), seconds
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

    Rows "arrive" at their recorded arrival times; each is fed through the
    mapper in order, so the mapper's calibration history and smoothing evolve
    exactly as they did on the live stream, no matter how often the caller
    polls. After the trace ends the source reports a lost track (as the live
    source does once its stream goes stale).

    Like the live Neon sources, samples are stamped with their capture time on
    the replay clock, not their arrival time. A trace has no clock offset, so
    the capture times are rebuilt from the phone's own device timestamps (5 ms
    spacing, free of Wi-Fi jitter and of the coarse arrival clock older traces
    were recorded with), placed so the median arrival-minus-capture latency is
    `assumed_latency_ms` (default: the Neon's measured median). Rows without a
    device timestamp are stamped with their arrival time."""

    def __init__(
        self,
        trace: list[TraceRow],
        time_fn: Callable[[], float] = timebase.now,
        assumed_latency_ms: float = NEON_REPLAY_LATENCY_MS,
    ) -> None:
        if not trace:
            raise ValueError("empty gaze trace")
        self._trace = trace
        self._times = [row.t_s for row in trace]
        self._time_fn = time_fn
        self._mapper = GazeMapper()
        self._t0: float | None = None
        self._next_row = 0
        self._history: deque[GazeSample] = deque(maxlen=NEON_HISTORY_SAMPLES)
        self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=0.0)
        self._has_device_time = any(row.device_unix_s is not None for row in trace)
        self._assumed_latency_ms = assumed_latency_ms
        self._capture_rel = self._capture_offsets(assumed_latency_ms / 1000)

    def _capture_offsets(self, latency_s: float) -> list[float]:
        """Per row: capture time in seconds after the first row's arrival (see class docstring)."""
        t0 = self._times[0]
        arrival_rel = [t - t0 for t in self._times]
        with_device = [(i, r.device_unix_s) for i, r in enumerate(self._trace) if r.device_unix_s is not None]
        if not with_device:
            return arrival_rel
        dev0 = with_device[0][1]
        median_d = statistics.median(arrival_rel[i] - (dev - dev0) for i, dev in with_device)
        out: list[float] = []
        last = -math.inf
        for i, row in enumerate(self._trace):
            capture = arrival_rel[i]
            if row.device_unix_s is not None:
                raw = (row.device_unix_s - dev0) + median_d - latency_s
                if 0.0 <= arrival_rel[i] - raw <= NEON_MAX_PLAUSIBLE_LATENCY_MS / 1000:
                    capture = raw
            capture = max(capture, last + 1e-6)
            out.append(capture)
            last = capture
        return out

    @property
    def timestamps_are_capture_time(self) -> bool:
        return self._has_device_time

    @property
    def gaze_latency_ms(self) -> float | None:
        return self._assumed_latency_ms if self._has_device_time else None

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

    def _advance(self) -> float:
        """Feeds every row that has arrived by now through the mapper; returns the current trace time."""
        trace_time = self._times[0] + self.elapsed_s
        stop = bisect_right(self._times, trace_time)
        for i in range(self._next_row, stop):
            self._latest = self._sample_for(self._trace[i], self._t0 + self._capture_rel[i])
            self._history.append(self._latest)
        self._next_row = max(self._next_row, stop)
        return trace_time

    def latest_sample(self) -> GazeSample:
        if self._t0 is None:
            return self._latest
        trace_time = self._advance()
        if self._next_row > 0 and trace_time - self._times[self._next_row - 1] > NEON_STALE_TIMEOUT_S:
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=self._time_fn())
        return self._latest

    def samples_since(self, timestamp: float | None) -> list[GazeSample]:
        """Replayed samples stamped after `timestamp` (all buffered ones for None), oldest first."""
        if self._t0 is not None:
            self._advance()
        return [s for s in self._history if timestamp is None or s.timestamp > timestamp]

    def _sample_for(self, row: TraceRow, stamp: float) -> GazeSample:
        if not row.worn or row.x_px is None or math.isnan(row.x_px):
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=stamp)
        return self._mapper.map(row.x_px / NEON_SCENE_WIDTH, stamp)

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._mapper.calibrate(left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return self._mapper.average_recent_ratio()

    def begin_calibration_sample(self) -> None:
        self._mapper.reset_ratio_history()
