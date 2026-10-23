import csv
import time
from types import SimpleNamespace

import numpy as np

from src.eye_tracking.neon_video_recorder import NeonVideoRecorder


def test_neon_video_recorder_writes_video_and_per_frame_metadata(tmp_path):
    counter = iter(range(1000))

    def receive(timeout_seconds=None):
        i = next(counter)
        time.sleep(0.005)
        return SimpleNamespace(bgr_pixels=np.full((48, 64, 3), i % 255, np.uint8), timestamp_unix_seconds=1000.0 + i * 0.005)

    device = SimpleNamespace(receive_scene_video_frame=receive, receive_eyes_video_frame=receive)
    rec = NeonVideoRecorder(device, "scene", tmp_path / "s.mp4", tmp_path / "s.csv", lambda: 250.0, 30.0)
    rec.start()
    time.sleep(0.2)
    rec.stop()

    rows = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert len(rows) == rec.frames > 5
    assert float(rows[0]["pc_unix_s"]) == 1000.0 + 0.25  # device time + offset
    assert (rows[0]["width"], rows[0]["height"]) == ("64", "48")
    assert (tmp_path / "s.mp4").stat().st_size > 0


def test_retime_makes_video_duration_match_real_time(tmp_path):
    import cv2

    from src.eye_tracking.latency_test import _retime_video

    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (64, 48))  # opened at a wrong fps
    for i in range(60):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()

    _retime_video(path, 60, 2.0)  # 60 frames actually spanned 2 real seconds -> 30 fps

    capture = cv2.VideoCapture(str(path))
    assert capture.get(cv2.CAP_PROP_FPS) == 30.0
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
