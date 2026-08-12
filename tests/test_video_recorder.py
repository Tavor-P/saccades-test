import csv
import time

import numpy as np
import pytest

from src.eye_tracking.video_recorder import VideoRecorder


class _FakeWriter:
    def __init__(self, opened: bool = True) -> None:
        self.frames_written = []
        self.released = False
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame) -> None:
        self.frames_written.append(frame)

    def release(self) -> None:
        self.released = True


class _FakeFrameSource:
    def __init__(self, frames) -> None:
        self._frames = iter(frames)

    def latest_frame(self):
        return next(self._frames, None)


@pytest.fixture
def fake_writer():
    return _FakeWriter()


def _make_recorder(tmp_path, fake_writer):
    video_path = tmp_path / "video.mp4"
    timestamps_path = tmp_path / "timestamps.csv"
    recorder = VideoRecorder(
        video_path, timestamps_path, frame_source=_FakeFrameSource([]), writer_factory=lambda: fake_writer
    )
    return recorder, video_path, timestamps_path


def test_write_one_frame_sends_a_bgr_frame_to_the_writer(tmp_path, fake_writer):
    recorder, _, timestamps_path = _make_recorder(tmp_path, fake_writer)
    recorder.start()
    try:
        gray_frame = np.zeros((4, 4), dtype=np.uint8)
        recorder._write_one_frame(gray_frame, now_ms=1234.5)
        assert len(fake_writer.frames_written) == 1
        written = fake_writer.frames_written[0]
        assert written.shape == (4, 4, 3)  # converted to 3-channel BGR
    finally:
        recorder.stop()


def test_write_one_frame_logs_a_timestamp_row(tmp_path, fake_writer):
    recorder, _, timestamps_path = _make_recorder(tmp_path, fake_writer)
    recorder.start()
    try:
        gray_frame = np.zeros((4, 4), dtype=np.uint8)
        recorder._write_one_frame(gray_frame, now_ms=1234.5)
        recorder._write_one_frame(gray_frame, now_ms=1267.8)
    finally:
        recorder.stop()

    with timestamps_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["frame_index"] for r in rows] == ["0", "1"]
    assert rows[0]["unix_time_ms"] == "1234.500"
    assert rows[1]["unix_time_ms"] == "1267.800"


def test_frames_written_counts_written_frames(tmp_path, fake_writer):
    recorder, _, _ = _make_recorder(tmp_path, fake_writer)
    recorder.start()
    try:
        gray_frame = np.zeros((2, 2), dtype=np.uint8)
        assert recorder.frames_written == 0
        recorder._write_one_frame(gray_frame, now_ms=0.0)
        recorder._write_one_frame(gray_frame, now_ms=1.0)
        assert recorder.frames_written == 2
    finally:
        recorder.stop()


def test_start_warns_on_stderr_when_the_writer_fails_to_open(tmp_path, capsys):
    video_path = tmp_path / "video.mp4"
    timestamps_path = tmp_path / "timestamps.csv"
    broken_writer = _FakeWriter(opened=False)
    recorder = VideoRecorder(
        video_path, timestamps_path, frame_source=_FakeFrameSource([]), writer_factory=lambda: broken_writer
    )
    recorder.start()
    try:
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert str(video_path) in captured.err
    finally:
        recorder.stop()


def test_start_does_not_warn_when_the_writer_opens_successfully(tmp_path, fake_writer, capsys):
    recorder, _, _ = _make_recorder(tmp_path, fake_writer)
    recorder.start()
    try:
        captured = capsys.readouterr()
        assert captured.err == ""
    finally:
        recorder.stop()


def test_stop_releases_the_writer_and_closes_the_timestamps_file(tmp_path, fake_writer):
    recorder, _, timestamps_path = _make_recorder(tmp_path, fake_writer)
    recorder.start()
    recorder.stop()
    assert fake_writer.released is True
    # File is closed but its contents (just the header, no frames written) are
    # still readable from disk.
    with timestamps_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_loop_polls_the_frame_source_in_the_background(tmp_path, fake_writer):
    # The background loop polls at VIDEO_RECORDING_FPS (30/s, ~33ms/frame) -
    # 0.3s real time is comfortably enough for the 3 available frames to be
    # consumed (the fake source then just returns None forever, same as a
    # real camera source between frames, and the loop keeps polling until
    # stop() - it doesn't exit on its own).
    video_path = tmp_path / "video.mp4"
    timestamps_path = tmp_path / "timestamps.csv"
    frames = [np.zeros((2, 2), dtype=np.uint8) for _ in range(3)]
    recorder = VideoRecorder(
        video_path, timestamps_path, frame_source=_FakeFrameSource(frames), writer_factory=lambda: fake_writer
    )
    recorder.start()
    time.sleep(0.3)
    recorder.stop()
    assert recorder.frames_written == 3
    assert len(fake_writer.frames_written) == 3
