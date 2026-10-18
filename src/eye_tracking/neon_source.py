import math
import statistics
import threading
import time
from collections import deque

from include.eye_tracking.constants import (
    NEON_DEVICE_ADDRESS,
    NEON_DEVICE_PORT,
    NEON_GAP_THRESHOLD_MS,
    NEON_DISCOVERY_TIMEOUT_S,
    NEON_SCENE_WIDTH,
)
from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.eye_tracking.gaze_mapping import GazeMapper

_MAX_LATENCY_SAMPLES = 100_000  # ~8 min at 200 Hz; older samples drop off, bounding memory


def _connect():
    """Imported lazily so the rest of the project (and the test suite) works
    without pupil-labs-realtime-api installed."""
    from pupil_labs.realtime_api.simple import Device, discover_one_device

    if NEON_DEVICE_ADDRESS:
        return Device(NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT)
    return discover_one_device(max_search_duration_seconds=NEON_DISCOVERY_TIMEOUT_S)


class NeonGazeSource(GazeSource):
    """Gaze from Pupil Labs Neon glasses via the Real-Time API, polled on a
    background thread. Uses the scene-camera gaze x coordinate (normalised by
    the scene width) as the "ratio", so the 3-point calibration works exactly
    as it does for the webcam. Because the scene camera is head-mounted, the
    participant should keep their head still after calibrating (e.g. chin
    rest) - head movement shifts gaze x without any eye movement.

    `device` is injectable for tests; otherwise the Neon is auto-discovered
    (or reached at NEON_DEVICE_ADDRESS) in start()."""

    def __init__(self, device=None) -> None:
        self._device = device
        self._mapper = GazeMapper()
        self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=time.monotonic())
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        # Session quality bookkeeping (see quality_summary()) - all guarded by _lock.
        self._clock_offset_ms: float | None = None
        self._clock_offset_std_ms: float | None = None
        self._roundtrip_ms: float | None = None
        self._latencies_ms: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)
        self._samples = 0
        self._lost_samples = 0  # not worn / NaN gaze
        self._gaps = 0
        self._longest_gap_ms = 0.0
        self._first_arrival: float | None = None
        self._last_arrival: float | None = None

    @property
    def is_available(self) -> bool:
        return self._device is not None

    def start(self) -> None:
        if self._device is None:
            self._device = _connect()
        if self._device is None:
            raise RuntimeError(
                "No Pupil Labs Neon found on the network - check the companion app is running and the PC "
                "is on the same Wi-Fi/hotspot (or set NEON_DEVICE_ADDRESS in include/eye_tracking/constants.py)."
            )
        self._estimate_clock_offset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                datum = self._device.receive_gaze_datum(timeout_seconds=0.5)
            except Exception:
                datum = None
            if datum is None:
                time.sleep(0.005)
                continue
            self._ingest(datum.x, getattr(datum, "worn", True), getattr(datum, "timestamp_unix_seconds", None))

    def _estimate_clock_offset(self) -> None:
        """Phone-vs-PC clock offset (Pupil's Time Echo protocol), needed to turn
        a gaze datum's device timestamp into real end-to-end latency. Best
        effort: a failure just leaves latency unreported."""
        try:
            est = self._device.estimate_time_offset()
            self._clock_offset_ms = float(est.time_offset_ms.mean)
            self._clock_offset_std_ms = float(est.time_offset_ms.std)
            self._roundtrip_ms = float(est.roundtrip_duration_ms.mean)
        except Exception:
            pass

    def _ingest(self, x: float | None, worn: bool = True, device_timestamp_s: float | None = None) -> None:
        """One gaze reading (scene-camera pixel x). Not worn / NaN counts as
        'no face' so session logic treats it like a lost track.
        `device_timestamp_s` (the phone's unix timestamp for this sample) is
        only used for latency statistics."""
        now = time.monotonic()
        if not worn or x is None or math.isnan(x):
            sample = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=now)
        else:
            sample = self._mapper.map(x / NEON_SCENE_WIDTH, now)
        with self._lock:
            self._latest = sample
            self._record_quality(now, sample.face_found, device_timestamp_s)

    def _record_quality(self, now: float, valid: bool, device_timestamp_s: float | None) -> None:
        # Caller holds self._lock.
        self._samples += 1
        if not valid:
            self._lost_samples += 1
        if self._last_arrival is not None:
            gap_ms = (now - self._last_arrival) * 1000
            if gap_ms > NEON_GAP_THRESHOLD_MS:
                self._gaps += 1
            self._longest_gap_ms = max(self._longest_gap_ms, gap_ms)
        else:
            self._first_arrival = now
        self._last_arrival = now
        # Latency = PC receive time minus the sample's capture time, with the
        # phone clock converted to PC time (local = device + offset).
        if self._clock_offset_ms is not None and device_timestamp_s is not None:
            local_capture_s = device_timestamp_s + self._clock_offset_ms / 1000
            self._latencies_ms.append((time.time() - local_capture_s) * 1000)

    def quality_summary(self) -> dict:
        """Session-level tracking quality, for the results _meta.json: how many
        samples arrived and at what rate, how many were unusable (not worn /
        NaN), stalls in the stream, and end-to-end gaze latency (phone capture
        -> PC receive). Latency fields are None if the clock offset
        estimate failed."""
        with self._lock:
            duration_s = (
                self._last_arrival - self._first_arrival
                if self._first_arrival is not None and self._last_arrival is not None
                else 0.0
            )
            latencies = sorted(self._latencies_ms)
            return {
                "source": "pupil_labs_neon",
                "samples": self._samples,
                "lost_samples": self._lost_samples,
                "sample_rate_hz": round(self._samples / duration_s, 1) if duration_s > 0 else None,
                "gaps_over_threshold": self._gaps,
                "gap_threshold_ms": NEON_GAP_THRESHOLD_MS,
                "longest_gap_ms": round(self._longest_gap_ms, 1),
                "clock_offset_ms": self._clock_offset_ms,
                "clock_offset_std_ms": self._clock_offset_std_ms,
                "clock_roundtrip_ms": self._roundtrip_ms,
                "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
                "latency_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
                "latency_ms_max": round(latencies[-1], 1) if latencies else None,
            }

    def latest_sample(self) -> GazeSample:
        with self._lock:
            return self._latest

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._mapper.calibrate(left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return self._mapper.average_recent_ratio()

    def begin_calibration_sample(self) -> None:
        self._mapper.reset_ratio_history()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
