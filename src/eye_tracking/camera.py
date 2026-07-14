import threading
import time

import cv2

from include.eye_tracking.constants import CAMERA_INDEX, CAMERA_PROBE_LIMIT, FRAME_HEIGHT, FRAME_WIDTH


def _open_capture(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    return capture


def list_available_cameras(max_index: int = CAMERA_PROBE_LIMIT) -> list[int]:
    """Probes indices 0..max_index-1 and returns the ones that actually open
    - used to offer a camera picker when more than one is connected (e.g. a
    laptop's built-in webcam plus a newly added external one), rather than
    requiring CAMERA_INDEX to be hand-edited to switch between them."""
    available = []
    for index in range(max_index):
        capture = _open_capture(index)
        if capture.isOpened():
            available.append(index)
        capture.release()
    return available


class Camera:
    """Continuously reads frames from a webcam on a background thread."""

    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self._index = index
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def start(self) -> None:
        capture = _open_capture(self._index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running and self._capture is not None:
            ok, frame = self._capture.read()
            if ok:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
