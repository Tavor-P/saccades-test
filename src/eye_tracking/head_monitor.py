"""Head-movement MONITOR from the Neon IMU. It detects and flags head movement; it does NOT correct gaze (not validated, needs glasses).

What the installed Real-Time API (pupil_labs.realtime_api 1.9.0, `simple.Device`) provides, read from the package source:
- `Device.receive_imu_datum(timeout_seconds=None)` returns an `IMUData` or None on timeout (None = wait forever, 0 = do not wait).
  The first call starts the IMU stream. The device keeps ONLY the newest datum per sensor (deque maxlen=1), so polling slower than
  the sensor (~110 Hz) subsamples instead of building a backlog; each datum can be received once.
- `IMUData` fields: `gyro_data` (x, y, z) in deg/s; `accel_data` (x, y, z) in m/s^2; `quaternion` (x, y, z, w; x first, w LAST)
  = the device rotation as a unit quaternion; `timestamp_unix_seconds` = seconds since the Unix epoch on the PHONE clock
  (the same clock as gaze datums' `timestamp_unix_seconds`, so the gaze clock offset + anchor in NeonGazeSource convert it too).
Not verified without glasses: the real rate, the gyro axis conventions (irrelevant here: only the vector norm is used), whether the
quaternion is gyro-only or magnetometer-fused (a fused yaw can drift slowly by itself), and the clock alignment with the gaze stamps.

Everything here is pure (no threads, no device): feed `add()` from whatever polls the IMU and query it from the session.
"""

import math
import threading
from collections import deque
from typing import NamedTuple

from include.eye_tracking.constants import HEAD_BUFFER_S, HEAD_MAX_LOGGED_SAMPLES

Quat = tuple[float, float, float, float]  # x, y, z, w (the Real-Time API's order)


class HeadSample(NamedTuple):
    t: float  # on the session's timebase.now() timeline (seconds)
    speed_dps: float  # |gyro|, deg/s
    quat: Quat | None  # unit quaternion, None when the orientation reading was unusable


def _unit(q) -> Quat | None:
    try:
        x, y, z, w = (float(v) for v in q)
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return (x / norm, y / norm, z / norm, w / norm)


def angle_between_deg(a: Quat, b: Quat) -> float:
    """Total rotation angle (deg) taking orientation a to b; q and -q are the same rotation."""
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return math.degrees(2 * math.acos(min(1.0, dot)))


class HeadMotionTracker:
    """Short ring buffer of IMU samples plus session-long summaries. Thread-safe (one writer thread, the session reading)."""

    def __init__(self, buffer_s: float = HEAD_BUFFER_S) -> None:
        self._buffer_s = buffer_s
        self._lock = threading.Lock()
        self._ring: deque[HeadSample] = deque()
        self._reference: Quat | None = None
        self._latest_quat: Quat | None = None
        self._latest_speed: float | None = None
        self._max_drift_deg = 0.0
        self._max_speed_dps = 0.0
        self._count = 0
        self._first_t: float | None = None
        self._last_t: float | None = None
        self._log: list[tuple[float, float | None, float, float | None]] = []  # t, phone unix s, speed, drift (head_motion.csv)

    # -- feeding ------------------------------------------------------------------------------

    def add(self, t: float, gyro_dps, quaternion, unix_s: float | None = None) -> None:
        """One IMU sample: `gyro_dps` = (x, y, z) in deg/s, `quaternion` = (x, y, z, w). Unusable values are tolerated (speed -> sample
        skipped, orientation -> None). Samples must come in time order."""
        try:
            gx, gy, gz = (float(v) for v in gyro_dps)
            speed = math.sqrt(gx * gx + gy * gy + gz * gz)
        except (TypeError, ValueError):
            return
        if not math.isfinite(speed):
            return
        quat = _unit(quaternion)
        with self._lock:
            self._ring.append(HeadSample(t, speed, quat))
            while self._ring and self._ring[0].t < t - self._buffer_s:
                self._ring.popleft()
            self._latest_speed = speed
            if quat is not None:
                self._latest_quat = quat
            self._max_speed_dps = max(self._max_speed_dps, speed)
            drift = self._drift_locked()
            if drift is not None:
                self._max_drift_deg = max(self._max_drift_deg, drift)
            self._count += 1
            if self._first_t is None:
                self._first_t = t
            self._last_t = t
            if len(self._log) < HEAD_MAX_LOGGED_SAMPLES:
                self._log.append((t, unix_s, speed, drift))

    def set_reference(self) -> bool:
        """Takes the latest orientation as "the head where it was at calibration"; drift is measured from it and the running maximum
        restarts. False if no orientation has been seen yet (drift then stays None)."""
        with self._lock:
            if self._latest_quat is None:
                return False
            self._reference = self._latest_quat
            self._max_drift_deg = 0.0
            return True

    # -- queries ------------------------------------------------------------------------------

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._count > 0

    @property
    def speed_dps(self) -> float | None:
        """Most recent head angular speed (deg/s), None before any sample."""
        with self._lock:
            return self._latest_speed

    @property
    def drift_deg(self) -> float | None:
        """Head rotation (total angle, deg) from the reference orientation; None until a reference and an orientation exist."""
        with self._lock:
            return self._drift_locked()

    @property
    def max_drift_deg(self) -> float:
        with self._lock:
            return self._max_drift_deg

    def _drift_locked(self) -> float | None:
        if self._reference is None or self._latest_quat is None:
            return None
        return angle_between_deg(self._reference, self._latest_quat)

    def max_speed_between(self, t0: float, t1: float) -> float | None:
        """Highest head angular speed (deg/s) among buffered samples with t0 <= t <= t1; None if there are none."""
        with self._lock:
            speeds = [s.speed_dps for s in self._ring if t0 <= s.t <= t1]
        return max(speeds) if speeds else None

    def speed_spike_between(self, t0: float, t1: float, threshold_dps: float) -> bool:
        peak = self.max_speed_between(t0, t1)
        return peak is not None and peak > threshold_dps

    def moved_since(self, t0: float, t1: float, threshold_deg: float) -> bool:
        """True if, within [t0, t1], the head orientation differs by more than `threshold_deg` from where it was at t0 (the last sample at or
        before t0, else the first one in the window). False when there is too little data to say."""
        with self._lock:
            base = None
            inside: list[Quat] = []
            for s in self._ring:
                if s.quat is None:
                    continue
                if s.t <= t0:
                    base = s.quat
                elif s.t <= t1:
                    inside.append(s.quat)
        if base is None and inside:
            base = inside[0]
        return base is not None and any(angle_between_deg(base, q) > threshold_deg for q in inside)

    def summary(self) -> dict:
        with self._lock:
            duration = (self._last_t - self._first_t) if self._first_t is not None and self._last_t is not None else 0.0
            return {
                "samples": self._count,
                "sample_rate_hz": round(self._count / duration, 1) if duration > 0 else None,
                "max_speed_dps": round(self._max_speed_dps, 2),
                "max_drift_deg": round(self._max_drift_deg, 2),
                "reference_set": self._reference is not None,
            }

    def csv_rows(self) -> list[tuple[float, float | None, float, float | None]]:
        """(t_s on the timebase.now() timeline, phone unix s or None, speed deg/s, drift deg or None) for every stored sample."""
        with self._lock:
            return list(self._log)
