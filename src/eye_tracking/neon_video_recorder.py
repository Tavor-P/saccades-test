"""Recording the Neon's camera streams to disk with per-frame metadata (used by latency_test.py).

No PsychoPy import on purpose: SceneRecorderProcess runs this in a separate
process, so the (CPU-heavy) video decode + encode can't slow the main process's
gaze reading and display - in one process the scene stream fell steadily behind
real time (delay 0.4 s -> 4 s over 25 s) while gaze degraded too.
"""
import csv
import multiprocessing
import threading
import time
from pathlib import Path

import cv2
from src import timebase

SCENE_VIDEO_SCALE = 0.4  # scene video is encoded at 640x480 (from 1600x1200) - larger sizes did not keep up


def retime_video(path: Path, n_frames: int, duration_s: float) -> None:
    """Rewrites an mp4 at its true average frame rate (frames / real seconds).
    The cameras deliver frames at whatever rate they manage, not the nominal
    rate the file was opened with, so without this the video plays back too
    fast or slow and ends 'too early'. Frames stay 1:1 with the CSV rows."""
    if n_frames < 2 or duration_s <= 0 or not path.exists():
        return
    source = cv2.VideoCapture(str(path))
    temp = path.with_suffix(".retimed.mp4")
    ok, frame = source.read()
    if not ok:
        source.release()
        return
    height, width = frame.shape[:2]
    exact = n_frames / duration_s
    writer = None
    # MPEG-4 can't encode some exact rates (timebase denominator limit), so fall back to coarser ones.
    for fps in (exact, round(exact, 1), float(max(1, round(exact)))):
        writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:  # couldn't re-encode: leave the original file untouched
        source.release()
        temp.unlink(missing_ok=True)
        return
    while ok:
        writer.write(frame)
        ok, frame = source.read()
    source.release()
    writer.release()
    temp.replace(path)


class NeonVideoRecorder:
    """Saves one Neon camera stream ("scene" ~30 fps or "eyes" ~200 fps) to an
    .mp4 plus a CSV with one row per frame: the phone's capture timestamp, the
    PC arrival time, and the capture time converted to PC time (device clock +
    the clock offset estimated at startup). The mp4's own frame rate is only
    nominal - the per-frame timestamps are the real timing."""

    def __init__(self, device, kind: str, video_path: Path, csv_path: Path, offset_ms, nominal_fps: float, scale: float = 1.0) -> None:
        self._scale = scale
        self._first_pc = self._last_pc = None
        self._receive = device.receive_scene_video_frame if kind == "scene" else device.receive_eyes_video_frame
        self._video_path = video_path
        self._offset_ms = offset_ms  # callable -> current clock offset estimate in ms (PC = device + offset)
        self._nominal_fps = nominal_fps
        self._writer = None
        self._file = csv_path.open("w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["frame_index", "device_unix_s", "arrival_unix_s", "pc_unix_s", "width", "height"])
        self.frames = 0
        self.error: str | None = None
        # A Neon stream only connects on its first receive call, which takes ~5 s -
        # do that now (before the dot starts) and keep the frame it returns.
        self._pending = None
        try:
            self._pending = self._receive(timeout_seconds=15)
        except Exception as exc:
            self.error = repr(exc)
        self._pending_arrival = timebase.unix_now()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            if self._pending is not None:
                frame, arrival, self._pending = self._pending, self._pending_arrival, None
            else:
                try:
                    frame = self._receive(timeout_seconds=0.5)
                except Exception as exc:  # e.g. this stream isn't enabled on the phone
                    self.error = repr(exc)
                    time.sleep(0.2)
                    continue
                arrival = timebase.unix_now()
            if frame is None:
                continue
            image = frame.bgr_pixels
            if self._scale != 1.0:
                image = cv2.resize(image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]
            if self._writer is None:
                self._writer = cv2.VideoWriter(
                    str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), self._nominal_fps, (width, height)
                )
            self._writer.write(image)
            device_s = frame.timestamp_unix_seconds
            self._csv.writerow(
                [self.frames, f"{device_s:.6f}", f"{arrival:.6f}", f"{device_s + self._offset_ms() / 1000:.6f}", width, height]
            )
            self.frames += 1
            pc_s = device_s + self._offset_ms() / 1000
            self._first_pc = pc_s if self._first_pc is None else self._first_pc
            self._last_pc = pc_s

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        if self._writer is not None:
            self._writer.release()
        self._file.close()
        if self._first_pc is not None:
            retime_video(self._video_path, self.frames, self._last_pc - self._first_pc)


def _scene_worker(address, port, discovery_timeout_s, video_path, csv_path, offset_ms, ready, go, stop, result) -> None:
    from pupil_labs.realtime_api.simple import Device, discover_one_device

    device = Device(address, port) if address else discover_one_device(max_search_duration_seconds=discovery_timeout_s)
    recorder = NeonVideoRecorder(device, "scene", video_path, csv_path, lambda: offset_ms, 30.0, scale=SCENE_VIDEO_SCALE)
    ready.set()  # stream connected (the constructor waits for the first frame)
    go.wait()
    recorder.start()
    stop.wait()
    recorder.stop()
    device.close()
    result.put({"frames": recorder.frames, "error": recorder.error})


class SceneRecorderProcess:
    """Records the Neon scene video in a child process with its own device connection."""

    def __init__(self, address, port, discovery_timeout_s, video_path: Path, csv_path: Path, offset_ms: float) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._ready, self._go, self._stop, self._result = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
        self._process = ctx.Process(
            target=_scene_worker,
            args=(address, port, discovery_timeout_s, video_path, csv_path, offset_ms, self._ready, self._go, self._stop, self._result),
            daemon=True,
        )
        self.frames = 0
        self.error: str | None = None

    def prepare(self, timeout_s: float = 90.0) -> bool:
        """Launches the child and waits until its stream is connected."""
        self._process.start()
        return self._ready.wait(timeout_s)

    def start(self) -> None:
        self._go.set()

    def stop(self) -> None:
        self._go.set()  # never leave the child waiting
        self._stop.set()
        try:
            info = self._result.get(timeout=120)  # includes the end-of-run video re-timing
            self.frames, self.error = info["frames"], info["error"]
        except Exception:
            self.error = "scene recorder process did not report back"
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
