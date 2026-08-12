import csv
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2

from include.eye_tracking.constants import FRAME_HEIGHT, FRAME_WIDTH, VIDEO_RECORDING_FPS


class VideoRecorder:
    """Records a session's camera feed to an .mp4 file, alongside a CSV
    logging every written frame's wall-clock (time.time()) timestamp - lets
    a session be replayed frame-accurately, and lined up exactly against the
    results CSV's own grating_shown_at_unix_ms timestamps, without relying on
    the video container's own frame-rate metadata (which drifts from real
    elapsed time over a long recording if the source doesn't deliver frames
    at a perfectly even interval - a per-frame timestamp sidesteps that
    entirely instead of periodically re-anchoring against it).

    `frame_source` is anything exposing `.latest_frame()` -> a raw Mono8
    ndarray or None (WebcamGazeSource already provides this for the debug
    camera preview) - polled on a background thread at VIDEO_RECORDING_FPS,
    independent of the camera's own much faster acquisition rate.
    """

    def __init__(
        self,
        video_path: Path,
        timestamps_path: Path,
        frame_source,
        writer_factory: Callable[[], "cv2.VideoWriter"] | None = None,
    ) -> None:
        self._video_path = video_path
        self._timestamps_path = timestamps_path
        self._frame_source = frame_source
        self._writer_factory = writer_factory or self._default_writer
        self._writer: "cv2.VideoWriter | None" = None
        self._timestamps_file = None
        self._timestamps_writer = None
        self._frame_index = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def _default_writer(self) -> "cv2.VideoWriter":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(self._video_path), fourcc, VIDEO_RECORDING_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    def start(self) -> None:
        self._timestamps_path.parent.mkdir(parents=True, exist_ok=True)
        self._timestamps_file = self._timestamps_path.open("w", newline="")
        self._timestamps_writer = csv.writer(self._timestamps_file)
        self._timestamps_writer.writerow(["frame_index", "unix_time_ms"])
        self._writer = self._writer_factory()
        # cv2.VideoWriter doesn't raise on a codec/container it can't open -
        # .write() calls just silently do nothing, which would mean a whole
        # session records zero video with no error anywhere. Surface that
        # loudly instead: this is something the researcher running the
        # session needs to notice (a missing mp4v codec on this machine's
        # OpenCV build, an unwritable path), not something a participant
        # would ever see, so print rather than raise - failing recording
        # shouldn't crash a session already in progress.
        if not self._writer.isOpened():
            print(
                f"WARNING: video recording failed to start (could not open {self._video_path}) - "
                "this session will have no video, but data collection will proceed normally.",
                file=sys.stderr,
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _write_one_frame(self, frame, now_ms: float) -> None:
        # The camera hands out raw Mono8 (grayscale) - converted to BGR here
        # (not left single-channel) for maximum player/codec compatibility;
        # not free, but this runs at VIDEO_RECORDING_FPS (30), not the
        # camera's full acquisition rate, so it's negligible.
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self._writer.write(bgr)
        self._timestamps_writer.writerow([self._frame_index, f"{now_ms:.3f}"])
        self._frame_index += 1

    def _loop(self) -> None:
        interval_s = 1.0 / VIDEO_RECORDING_FPS
        while self._running:
            loop_start = time.monotonic()
            frame = self._frame_source.latest_frame()
            if frame is not None:
                self._write_one_frame(frame, time.time() * 1000)
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, interval_s - elapsed))

    @property
    def frames_written(self) -> int:
        return self._frame_index

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._timestamps_file is not None:
            self._timestamps_file.close()
            self._timestamps_file = None
