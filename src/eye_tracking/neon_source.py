import csv
import math
import statistics
import sys
import threading
import time
from collections import deque

from include.eye_tracking.constants import (
    NEON_DEVICE_ADDRESS,
    NEON_DEVICE_PORT,
    NEON_GAP_THRESHOLD_MS,
    NEON_STALE_TIMEOUT_S,
    NEON_MAX_PLAUSIBLE_LATENCY_MS,
    NEON_HISTORY_SAMPLES,
    NEON_LATENCY_WINDOW_SAMPLES,
    NEON_CLOCK_REESTIMATE_S,
    NEON_CLOCK_REESTIMATE_MAX_STD_MS,
    NEON_DISCOVERY_TIMEOUT_S,
    NEON_SCENE_WIDTH,
    HEAD_POLL_HZ,
    HEAD_NO_DATA_TIMEOUT_S,
    HEAD_MAX_CONSECUTIVE_ERRORS,
)
from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from pathlib import Path

from src.eye_tracking.gaze_mapping import GazeMapper
from src.eye_tracking.gaze_trace import GazeTraceWriter
from src.eye_tracking.head_monitor import HeadMotionTracker
from src import timebase

_MAX_LATENCY_SAMPLES = 100_000  # ~8 min at 200 Hz; older samples drop off, bounding memory
_MAX_LOGGED_ERROR_KINDS = 50  # distinct error signatures remembered for log de-duplication


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

    def __init__(self, device=None, trace_path: Path | None = None) -> None:
        self._device = device
        # Optional raw-gaze recording (see gaze_trace.py) for later replay.
        self._trace = GazeTraceWriter(trace_path) if trace_path is not None else None
        self._mapper = GazeMapper()
        self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=timebase.now())
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        # Session quality bookkeeping (see quality_summary()) - all guarded by _lock.
        self._clock_offset_ms: float | None = None
        self._offset_history: list[dict] = []  # every offset in force, with the phone-clock time it applies from (see _note_offset)
        self._anchor = timebase.measure_anchor()  # pairs with _clock_offset_ms; refreshed together (phone/PC clocks drift)
        self._offset_stop = threading.Event()
        self._offset_thread: threading.Thread | None = None
        self._recording_id: str | None = None  # the phone-side recording started by start_recording()
        self._clock_offset_std_ms: float | None = None
        self._roundtrip_ms: float | None = None
        self._latencies_ms: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)
        self._history: deque[GazeSample] = deque(maxlen=NEON_HISTORY_SAMPLES)  # for samples_since()
        self._recent_latencies_ms: deque[float] = deque(maxlen=NEON_LATENCY_WINDOW_SAMPLES)
        self._latency_median_ms: float | None = None
        self._last_stamp = 0.0
        self._samples = 0
        self._lost_samples = 0  # not worn / NaN gaze
        self._gaps = 0
        self._longest_gap_ms = 0.0
        self._first_arrival: float | None = None
        self._last_arrival: float | None = None
        # Receive-loop error bookkeeping - guarded by _lock.
        self._error_count = 0
        self._last_error: str | None = None
        self._seen_errors: set[tuple[str, str]] = set()
        # Head-movement monitor (head_monitor.py): None until start() finds a working IMU, and again if it later switches itself off.
        self.head_motion: HeadMotionTracker | None = None
        self._head_status = "not started"  # "ok" or "unavailable (reason)", for session.json
        self._head_tracker: HeadMotionTracker | None = None  # kept after a switch-off so the summary still has what was recorded
        self._head_stop = threading.Event()
        self._head_thread: threading.Thread | None = None

    def _record_error(self, where: str, exc: Exception) -> None:
        """Count a receive-loop error and print the first occurrence of each
        distinct (type, message) to stderr; repeats are only counted, so a
        persistent failure at ~200 Hz can't flood the console."""
        text = f"{type(exc).__name__}: {exc}"
        key = (type(exc).__name__, str(exc))
        with self._lock:
            self._error_count += 1
            self._last_error = f"{where}: {text}"
            is_new = key not in self._seen_errors and len(self._seen_errors) < _MAX_LOGGED_ERROR_KINDS
            if is_new:
                self._seen_errors.add(key)
        if is_new:
            print(f"[neon] {where} error (further repeats counted, not logged): {text}", file=sys.stderr)

    @property
    def is_available(self) -> bool:
        return self._device is not None and not self._stream_stalled()

    def _stream_stalled(self) -> bool:
        """True once samples have been flowing but have now stopped for
        NEON_STALE_TIMEOUT_S (phone asleep, Wi-Fi dropped, glasses unplugged)."""
        with self._lock:
            return self._last_arrival is not None and timebase.now() - self._last_arrival > NEON_STALE_TIMEOUT_S

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
        self._start_offset_thread()
        self._start_head_monitor()

    def _loop(self) -> None:
        while self._running:
            try:
                datum = self._device.receive_gaze_datum(timeout_seconds=0.5)
            except Exception as exc:
                self._record_error("receive_gaze_datum", exc)
                datum = None
            if datum is None:
                time.sleep(0.005)
                continue
            try:
                self._ingest(datum.x, getattr(datum, "worn", True), getattr(datum, "timestamp_unix_seconds", None))
            except Exception as exc:
                self._record_error("ingest", exc)
                time.sleep(0.005)

    # -- head-movement monitor (IMU) ----------------------------------------------------------------
    # One separate daemon thread that wakes HEAD_POLL_HZ times a second, does a non-blocking receive (timeout 0) and a handful of
    # float operations, then sleeps: it never touches the gaze receive loop. Failure-proof by design: no IMU, an exception or no data
    # switches the monitor off with one warning; nothing here raises into the experiment.

    def _start_head_monitor(self) -> None:
        if not hasattr(self._device, "receive_imu_datum"):
            self._disable_head_monitor("this device object has no IMU")
            return
        self._head_tracker = self.head_motion = HeadMotionTracker()
        self._head_status = "ok"
        self._head_stop.clear()
        self._head_thread = threading.Thread(target=self._head_loop, args=(self.head_motion,), daemon=True)
        self._head_thread.start()

    def _disable_head_monitor(self, reason: str) -> None:
        self.head_motion = None
        self._head_status = f"unavailable ({reason})"
        print(f"WARNING: head-movement monitor off ({reason}); head movement will not be flagged this session.", file=sys.stderr)

    def _imu_time(self, datum) -> tuple[float, float | None]:
        """(time on the perf_counter timeline, phone unix stamp) of an IMU datum: phone capture time + the gaze clock offset when that gives a
        plausible latency, else arrival time. Reads the offset/anchor without the gaze lock (plain attribute reads; a rare one-poll skew is harmless)."""
        arrival = timebase.now()
        stamp = getattr(datum, "timestamp_unix_seconds", None)
        offset_ms = self._clock_offset_ms
        if offset_ms is not None and stamp is not None and not math.isnan(stamp):
            capture = self._anchor.to_perf(stamp + offset_ms / 1000)
            if 0.0 <= arrival - capture <= NEON_MAX_PLAUSIBLE_LATENCY_MS / 1000:
                return capture, stamp
        return arrival, stamp

    def _head_loop(self, tracker: HeadMotionTracker) -> None:
        started = timebase.now()
        errors = 0
        while not self._head_stop.wait(1.0 / HEAD_POLL_HZ):
            try:
                datum = self._device.receive_imu_datum(timeout_seconds=0)
            except Exception as exc:  # noqa: BLE001 - no IMU, firmware too old, network: all mean "monitor off", never an experiment error
                self._record_error("receive_imu_datum", exc)
                errors += 1
                if errors >= HEAD_MAX_CONSECUTIVE_ERRORS:
                    self._disable_head_monitor(f"IMU receive failed {errors} times in a row: {type(exc).__name__}: {exc}")
                    return
                continue
            errors = 0
            if datum is None:
                if not tracker.has_data and timebase.now() - started > HEAD_NO_DATA_TIMEOUT_S:
                    self._disable_head_monitor(f"no IMU data within {HEAD_NO_DATA_TIMEOUT_S:g} s")
                    return
                continue
            try:
                t, stamp = self._imu_time(datum)
                tracker.add(t, tuple(datum.gyro_data), tuple(datum.quaternion), stamp)
            except Exception as exc:  # noqa: BLE001
                self._record_error("imu_ingest", exc)

    def head_motion_summary(self, flagged_trials: int | None = None) -> dict:
        """For session.json "head_motion": {"status": "ok" | "unavailable (reason)" | "not started", sample count/rate, max speed and drift,
        flagged_trials (trials the session excluded for head speed)}."""
        summary: dict = {"status": self._head_status}
        if self._head_tracker is not None:
            summary.update(self._head_tracker.summary())
        if flagged_trials is not None:
            summary["flagged_trials"] = flagged_trials
        return summary

    def write_head_motion_csv(self, path: Path) -> bool:
        """head_motion.csv (t_s on the gaze trace's arrival-time axis, phone_unix_s, speed_dps, drift_deg); nothing is written without data. Best effort."""
        if self._head_tracker is None:
            return False
        rows = self._head_tracker.csv_rows()
        if not rows:
            return False
        try:
            with Path(path).open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t_s", "phone_unix_s", "speed_dps", "drift_deg"])
                for t, unix_s, speed, drift in rows:
                    writer.writerow([f"{t:.6f}", "" if unix_s is None else f"{unix_s:.6f}", f"{speed:.3f}", "" if drift is None else f"{drift:.3f}"])
        except OSError as exc:
            print(f"WARNING: could not write {path}: {exc}", file=sys.stderr)
            return False
        return True

    def _estimate_clock_offset(self) -> None:
        """Phone-vs-PC clock offset (Pupil's Time Echo protocol), needed to turn
        a gaze datum's device timestamp into real end-to-end latency. Best
        effort: a failure just leaves latency unreported."""
        try:
            est = self._device.estimate_time_offset()
            offset_ms = float(est.time_offset_ms.mean)
            std_ms = float(est.time_offset_ms.std)
            roundtrip_ms = float(est.roundtrip_duration_ms.mean)
            anchor = timebase.measure_anchor()  # as soon after the estimate as possible: offset is in system-clock terms
        except Exception:
            return
        with self._lock:
            self._clock_offset_ms = offset_ms
            self._clock_offset_std_ms = std_ms
            self._roundtrip_ms = roundtrip_ms
            self._anchor = anchor
            self._note_offset()

    def _note_offset(self) -> None:
        """Remembers the offset now in force and the phone-clock time (PC unix time minus the offset) it was measured at: the two clocks drift
        apart (~1 ms/min), so the post-hoc analysis interpolates over this history instead of applying the final offset to the whole trace.
        Caller holds self._lock (or is single-threaded)."""
        self._offset_history.append({"phone_unix_s": round(timebase.unix_now() - self._clock_offset_ms / 1000, 3), "offset_ms": round(self._clock_offset_ms, 3)})

    def _start_offset_thread(self) -> None:
        if NEON_CLOCK_REESTIMATE_S and self._clock_offset_ms is not None:
            self._offset_stop.clear()
            self._offset_thread = threading.Thread(target=self._offset_loop, daemon=True)
            self._offset_thread.start()

    def _offset_loop(self) -> None:
        """Re-estimates the phone-vs-PC clock offset every NEON_CLOCK_REESTIMATE_S: the two clocks drift apart (tens of ppm,
        plus time-service slews), and capture stamps depend on it. Runs on its own thread, over the device's separate
        time-echo connection; a noisy or failed estimate is ignored and the previous offset kept."""
        while not self._offset_stop.wait(NEON_CLOCK_REESTIMATE_S):
            try:
                est = self._device.estimate_time_offset()
                offset_ms = float(est.time_offset_ms.mean)
                std_ms = float(est.time_offset_ms.std)
                roundtrip_ms = float(est.roundtrip_duration_ms.mean)
                anchor = timebase.measure_anchor()
            except Exception as exc:
                self._record_error("estimate_time_offset", exc)
                continue
            if not (std_ms <= NEON_CLOCK_REESTIMATE_MAX_STD_MS):
                continue
            with self._lock:
                previous = self._clock_offset_ms
                # The anchor moves too, so express the old offset in the new anchor's terms is unnecessary: both
                # are system-clock values and the blend only smooths estimator noise.
                self._clock_offset_ms = offset_ms if previous is None else 0.5 * previous + 0.5 * offset_ms
                self._clock_offset_std_ms = std_ms
                self._roundtrip_ms = roundtrip_ms
                self._anchor = anchor
                self._note_offset()

    @property
    def timestamps_are_capture_time(self) -> bool:
        """True when samples are stamped with their capture time (phone clock offset known), so the session must not
        subtract `gaze_latency_ms` from the stamps again."""
        return self._clock_offset_ms is not None

    @property
    def gaze_latency_ms(self) -> float | None:
        """Rolling median of recent arrival-minus-capture latency (cheap: a cached value refreshed every few samples),
        None until measured. For logging and for sessions whose stamps are arrival times."""
        return self._latency_median_ms

    def samples_since(self, timestamp: float | None) -> list[GazeSample]:
        """Buffered samples (about the last NEON_HISTORY_SAMPLES) stamped after `timestamp`, oldest first; all of
        them for None. Stamps are strictly increasing, so a caller can use the last one as its cursor."""
        with self._lock:
            if timestamp is None:
                return list(self._history)
            newer: list[GazeSample] = []
            for sample in reversed(self._history):
                if sample.timestamp <= timestamp:
                    break
                newer.append(sample)
        newer.reverse()
        return newer

    def _stamp(self, arrival: float, device_timestamp_s: float | None) -> float:
        """Capture time of a sample on the perf_counter timeline: the phone's timestamp + clock offset, moved through the
        clock anchor - the eye was there ~16 ms before the sample arrived, and arrival time is Wi-Fi jitter. Falls back to
        `arrival` when the device timestamp or offset is missing, the result is in the future, or it implies an absurd
        latency (> NEON_MAX_PLAUSIBLE_LATENCY_MS: a bad offset). Always strictly increasing across calls, so
        samples_since() cursors never skip a sample. Caller holds self._lock."""
        capture = arrival
        if self._clock_offset_ms is not None and device_timestamp_s is not None and not math.isnan(device_timestamp_s):
            raw = self._anchor.to_perf(device_timestamp_s + self._clock_offset_ms / 1000)
            latency_ms = (arrival - raw) * 1000
            self._recent_latencies_ms.append(latency_ms)
            self._latencies_ms.append(latency_ms)
            if self._samples % 20 == 0 or self._latency_median_ms is None:
                self._latency_median_ms = statistics.median(self._recent_latencies_ms)
            if 0.0 <= latency_ms <= NEON_MAX_PLAUSIBLE_LATENCY_MS:
                capture = raw
            elif latency_ms < 0.0:
                capture = arrival  # offset error put the capture in the future
        capture = max(capture, self._last_stamp + 1e-6)
        self._last_stamp = capture
        return capture

    def _ingest(self, x: float | None, worn: bool = True, device_timestamp_s: float | None = None) -> None:
        """One gaze reading (scene-camera pixel x). Not worn / NaN counts as
        'no face' so session logic treats it like a lost track.
        `device_timestamp_s` (the phone's unix timestamp for this sample) gives
        the sample's capture time - see _stamp()."""
        arrival = timebase.now()
        usable = worn and x is not None and not math.isnan(x)
        with self._lock:
            stamp = self._stamp(arrival, device_timestamp_s)
            if usable:
                sample = self._mapper.map(x / NEON_SCENE_WIDTH, stamp)
            else:
                sample = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=stamp)
            self._publish(sample, arrival)
        # Outside the lock: latest_sample() takes it every rendered frame, and a
        # disk write (the buffer fills ~every 0.5 s) must never stall that. Safe
        # because only this receive thread writes and stop() joins it before close().
        if self._trace is not None:
            self._trace.write(arrival, device_timestamp_s, x, bool(worn))

    def _publish(self, sample: GazeSample, arrival: float) -> None:
        # Caller holds self._lock.
        self._latest = sample
        self._history.append(sample)
        self._record_quality(arrival, sample.face_found)

    def _record_quality(self, now: float, valid: bool) -> None:
        # Caller holds self._lock. `now` is the ARRIVAL time (gaps/staleness are about the stream, not the eye).
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
                "clock_offset_history": list(self._offset_history),
                "clock_roundtrip_ms": self._roundtrip_ms,
                "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
                "latency_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
                "latency_ms_max": round(latencies[-1], 1) if latencies else None,
                "receive_errors": self._error_count,
                "last_receive_error": self._last_error,
            }

    def latest_sample(self) -> GazeSample:
        with self._lock:
            if self._last_arrival is not None and timebase.now() - self._last_arrival > NEON_STALE_TIMEOUT_S:
                return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=timebase.now())
            return self._latest

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._mapper.calibrate(left_ratio, center_ratio, right_ratio)
        tracker = self.head_motion
        if tracker is not None:
            tracker.set_reference()  # head drift is measured from the orientation at calibration

    def average_recent_ratio(self) -> float | None:
        return self._mapper.average_recent_ratio()

    def begin_calibration_sample(self) -> None:
        self._mapper.reset_ratio_history()

    # -- the phone's own recording of the glasses (scene video) -------------------------------------
    # Best effort by design: a failure here must never cost the participant their session, so none of these raise.

    def start_recording(self) -> dict:
        """Starts a recording on the phone (Real-Time API `Device.recording_start()`). Returns the fields for session.json's
        glasses_video section: recording = "started" (+ id and PC start time) or "failed (reason)"."""
        try:
            recording_id = self._device.recording_start()
        except Exception as exc:  # noqa: BLE001 - DeviceError (low battery, already recording, no wearer selected...) or a network error
            reason = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: could not start the phone recording ({reason}); the session runs without glasses video.", file=sys.stderr)
            return {"recording": f"failed ({reason})"}
        self._recording_id = recording_id
        return {"recording": "started", "recording_id": str(recording_id), "recording_started_pc_unix_ms": round(timebase.unix_ms(), 1)}

    def stop_recording(self) -> dict:
        """Stops and saves the phone recording (`Device.recording_stop_and_save()`); a no-op if none was started."""
        if self._recording_id is None:
            return {}
        recording_id, self._recording_id = self._recording_id, None
        try:
            self._device.recording_stop_and_save()
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: could not stop/save the phone recording ({reason}); check the Companion app.", file=sys.stderr)
            return {"recording": f"failed to stop/save ({reason})", "recording_id": str(recording_id)}
        return {"recording": "saved", "recording_id": str(recording_id), "recording_stopped_pc_unix_ms": round(timebase.unix_ms(), 1)}

    def send_event(self, name: str) -> bool:
        """Marks `name` in the phone's recording (`Device.send_event`; stamped with the phone's clock on arrival). True if sent."""
        try:
            self._device.send_event(name)
            return True
        except Exception as exc:  # noqa: BLE001
            self._record_error("send_event", exc)
            return False

    def stop(self) -> None:
        if self._recording_id is not None:  # safety net: an aborted session must not leave the phone recording
            self.stop_recording()
        self._running = False
        self._offset_stop.set()
        self._head_stop.set()
        if self._head_thread is not None:
            self._head_thread.join(timeout=1)
        if self._offset_thread is not None:
            self._offset_thread.join(timeout=1)
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._trace is not None:
            with self._lock:
                self._trace.close()
                self._trace = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
