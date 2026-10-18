import math
import threading
import time

from include.eye_tracking.constants import (
    NEON_DEVICE_ADDRESS,
    NEON_DEVICE_PORT,
    NEON_DISCOVERY_TIMEOUT_S,
    NEON_SCENE_WIDTH,
)
from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.eye_tracking.gaze_mapping import GazeMapper


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
            self._ingest(datum.x, getattr(datum, "worn", True))

    def _ingest(self, x: float | None, worn: bool = True) -> None:
        """One gaze reading (scene-camera pixel x). Not worn / NaN counts as
        'no face' so session logic treats it like a lost track."""
        now = time.monotonic()
        if not worn or x is None or math.isnan(x):
            sample = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=now)
        else:
            sample = self._mapper.map(x / NEON_SCENE_WIDTH, now)
        with self._lock:
            self._latest = sample

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
