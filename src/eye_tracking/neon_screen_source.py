import math
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

from include.eye_tracking.constants import (
    NEON_HEAD_SPEED_LIMIT_DPS,
    NEON_HEAD_SPEED_WINDOW_S,
    NEON_SCENE_WIDTH,
    SCREEN_MARKER_PX,
)
from include.eye_tracking.types import GazeSample, GazeZone
from src.eye_tracking.neon_source import NeonGazeSource, _connect


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    return round(sorted_values[int(fraction * (len(sorted_values) - 1))], 1)


class NeonScreenGazeSource(NeonGazeSource):
    """Neon gaze mapped onto the *screen* via corner AprilTag markers, so the
    reported position doesn't move when the participant's head does (the plain
    NeonGazeSource reports scene-camera coordinates, which do).

    Three background threads: gaze (~200 Hz, each sample mapped with the newest
    screen location), scene video (~30 Hz, finds the markers) and the IMU
    (head rotation speed from the gyroscope, attached to every sample and
    summarised per session - the IMU is used to *flag* head movement, not to
    steer the mapping). A sample whose screen can't be located (markers hidden
    longer than the short hold) is reported as lost rather than silently
    falling back to head-dependent scene coordinates.

    `ratio` is screen x as a 0-1 fraction of the width, so the experiment's
    existing 3-point calibration works unchanged; `ratio_y` is the vertical
    fraction. `device`/`screen_mapper` are injectable for tests.

    `trace_path` records the gaze for replay (see gaze_trace.py). What is saved
    is the screen-mapped ratio x NEON_SCENE_WIDTH in the trace's `x_px` column
    (not the raw scene-camera pixel): replay divides by the scene width again, so
    it reproduces exactly the ratios - and so zones and calibration - this
    source produced, with head movement already compensated."""

    def __init__(
        self, screen_size: tuple[int, int], device=None, screen_mapper=None, trace_path: Path | None = None
    ) -> None:
        super().__init__(device=device, trace_path=trace_path)
        self._screen_size = screen_size
        self._screen_mapper = screen_mapper
        self._extra_threads: list[threading.Thread] = []
        self._speeds: deque[tuple[float, float]] = deque()  # (monotonic time, gyro magnitude deg/s)
        self._head_speeds_dps: deque[float] = deque(maxlen=100_000)
        self._mapping_ages_ms: deque[float] = deque(maxlen=100_000)
        self._unmapped_samples = 0  # worn, valid gaze, but screen not located
        self._head_moving_samples = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._device is None:
            self._device = _connect()
        if self._device is None:
            raise RuntimeError(
                "No Pupil Labs Neon found on the network - check the companion app is running and the PC "
                "is on the same Wi-Fi/hotspot (or set NEON_DEVICE_ADDRESS in include/eye_tracking/constants.py)."
            )
        self._estimate_clock_offset()
        if self._screen_mapper is None:
            from src.eye_tracking.screen_gaze_mapper import ScreenGazeMapper

            self._screen_mapper = ScreenGazeMapper(self._device.get_calibration(), self._screen_size)
        self._running = True
        self._thread = threading.Thread(target=self._gaze_loop, daemon=True)
        self._extra_threads = [
            threading.Thread(target=self._scene_loop, daemon=True),
            threading.Thread(target=self._imu_loop, daemon=True),
        ]
        self._thread.start()
        for thread in self._extra_threads:
            thread.start()

    def stop(self) -> None:
        self._running = False
        for thread in self._extra_threads:
            thread.join(timeout=1)
        super().stop()

    # -- background loops --------------------------------------------------

    def _receive(self, fn):
        try:
            return fn(timeout_seconds=0.5)
        except Exception as exc:
            self._record_error(getattr(fn, "__name__", "receive"), exc)
            time.sleep(0.005)
            return None

    def _guarded(self, where: str, fn, *args) -> None:
        """Run a per-item handler so one bad frame/datum is counted and logged
        instead of killing the receive thread."""
        try:
            fn(*args)
        except Exception as exc:
            self._record_error(where, exc)
            time.sleep(0.005)

    def _gaze_loop(self) -> None:
        while self._running:
            datum = self._receive(self._device.receive_gaze_datum)
            if datum is None:
                time.sleep(0.005)
                continue
            self._guarded("ingest_screen", self._ingest_screen, datum)

    def _scene_loop(self) -> None:
        while self._running:
            frame = self._receive(self._device.receive_scene_video_frame)
            if frame is not None:
                self._guarded("process_scene", self._screen_mapper.process_scene, frame)

    def _imu_loop(self) -> None:
        while self._running:
            datum = self._receive(self._device.receive_imu_datum)
            if datum is not None:
                self._guarded("ingest_imu", self._ingest_imu, datum)

    # -- ingest ------------------------------------------------------------

    def _ingest_imu(self, datum) -> None:
        gyro = datum.gyro_data
        speed = math.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2)
        with self._lock:
            self._speeds.append((time.monotonic(), speed))

    def _head_speed(self, now: float) -> float | None:
        """Peak gyro magnitude over the last NEON_HEAD_SPEED_WINDOW_S (caller holds the lock). None if no IMU data."""
        while self._speeds and now - self._speeds[0][0] > NEON_HEAD_SPEED_WINDOW_S:
            self._speeds.popleft()
        return max(speed for _, speed in self._speeds) if self._speeds else None

    def _ingest_screen(self, datum) -> None:
        now = time.monotonic()
        x = getattr(datum, "x", None)
        worn = getattr(datum, "worn", True)
        usable = bool(worn) and x is not None and not math.isnan(x)
        screen = self._screen_mapper.map_gaze(datum) if usable else None

        with self._lock:
            head_speed = self._head_speed(now)
        if screen is None:
            sample = GazeSample(
                zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=now, head_speed_dps=head_speed
            )
        else:
            width, height = self._screen_size
            sample = replace(
                self._mapper.map(screen.x / width, now),
                ratio_y=screen.y / height,
                head_speed_dps=head_speed,
                mapping_age_ms=screen.mapping_age_s * 1000,
            )
        with self._lock:
            self._latest = sample
            self._record_quality(now, sample.face_found, getattr(datum, "timestamp_unix_seconds", None))
            if usable and screen is None:
                self._unmapped_samples += 1
            if screen is not None:
                self._mapping_ages_ms.append(sample.mapping_age_ms)
            if head_speed is not None:
                self._head_speeds_dps.append(head_speed)
                if head_speed > NEON_HEAD_SPEED_LIMIT_DPS:
                    self._head_moving_samples += 1
        # Outside the lock, like NeonGazeSource._ingest: a disk write must never stall latest_sample().
        if self._trace is not None:
            x_px = sample.ratio * NEON_SCENE_WIDTH if sample.ratio is not None else None
            self._trace.write(now, getattr(datum, "timestamp_unix_seconds", None), x_px, sample.face_found)

    # -- status for the experiment's pre-flight gate and logs ---------------

    @property
    def markers_detected(self) -> int:
        return self._screen_mapper.markers_detected if self._screen_mapper is not None else 0

    @property
    def screen_located(self) -> bool:
        return self._screen_mapper is not None and self._screen_mapper.located

    def quality_summary(self) -> dict:
        summary = super().quality_summary()
        with self._lock:
            speeds = sorted(self._head_speeds_dps)
            ages = sorted(self._mapping_ages_ms)
            samples = max(self._samples, 1)
            summary.update(
                {
                    "source": "pupil_labs_neon_screen_markers",
                    "marker_px": SCREEN_MARKER_PX,
                    "screen_unlocated_samples": self._unmapped_samples,
                    "screen_unlocated_fraction": round(self._unmapped_samples / samples, 4),
                    "mapping_age_ms_median": _percentile(ages, 0.5),
                    "mapping_age_ms_p95": _percentile(ages, 0.95),
                    "head_speed_dps_median": _percentile(speeds, 0.5),
                    "head_speed_dps_p95": _percentile(speeds, 0.95),
                    "head_speed_limit_dps": NEON_HEAD_SPEED_LIMIT_DPS,
                    "head_moving_fraction": round(self._head_moving_samples / max(len(speeds), 1), 4) if speeds else None,
                }
            )
        return summary
