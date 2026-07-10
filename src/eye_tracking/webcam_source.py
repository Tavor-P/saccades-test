import threading
import time

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.eye_tracking.camera import Camera
from src.eye_tracking.gaze_tracker import GazeTracker


class WebcamGazeSource(GazeSource):
    """Runs camera capture + gaze estimation on a background thread and exposes
    the latest GazeSample so the experiment loop can poll it cheaply each frame
    without blocking on camera IO or MediaPipe inference."""

    def __init__(self) -> None:
        self._camera = Camera()
        self._tracker = GazeTracker()
        self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=time.monotonic())
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_available(self) -> bool:
        return self._camera.is_open

    def start(self) -> None:
        self._camera.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            sample = self._tracker.process(frame)
            with self._lock:
                self._latest = sample

    def latest_sample(self) -> GazeSample:
        with self._lock:
            return self._latest

    def calibrate(self, left_ratio: float, right_ratio: float) -> None:
        self._tracker.calibrate(left_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return self._tracker.average_recent_ratio()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._camera.stop()
