"""Glasses video: the sync-flash detector, the PC<->video clock offset, the import CLI, and the phone recording's failure handling.
All on synthetic frames / fake devices - no real video or glasses needed."""

import json

import numpy as np
import pytest

from src.eye_tracking.neon_source import NeonGazeSource
from src.experiment import glasses_video as gv

FPS = 30.0
PC_OFFSET_S = 1_700_000_000.123  # PC unix seconds = video seconds + this


def _frames(duration_s=20.0, flash_onsets=(2.0, 2.45, 2.9, 15.0), flash_s=0.15, size=(48, 64), distractors=(), seed=0):
    """Grey noisy frames; pure-green full-frame flashes at flash_onsets; white-ish flashes at `distractors` (must be ignored)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(duration_s * FPS)):
        t = i / FPS
        frame = rng.integers(100, 140, size=(*size, 3), dtype=np.uint8)
        if any(o <= t < o + flash_s for o in flash_onsets):
            frame[...] = (20, 230, 30)  # BGR green
        elif any(o <= t < o + flash_s for o in distractors):
            frame[...] = (250, 250, 250)
        out.append(frame)
    return out


def _pc_ms(video_times, offset=PC_OFFSET_S, jitter_ms=0.0, seed=1):
    rng = np.random.default_rng(seed)
    return [(v + offset) * 1000 + rng.uniform(-jitter_ms, jitter_ms) for v in video_times]


# -- flash pattern --------------------------------------------------------------------------------


@pytest.mark.parametrize("hz, expected", [(60.0, (9, 18)), (120.0, (18, 36)), (144.0, (22, 43)), (None, (9, 18)), (5.0, (9, 18))])
def test_flash_lengths_are_counted_in_frames_from_the_refresh_rate(hz, expected):
    assert gv.sync_flash_frames(hz) == expected


# -- detector -------------------------------------------------------------------------------------


def test_detector_finds_each_flash_onset_within_a_frame():
    found = gv.find_sync_flashes(_frames(), fps=FPS)
    assert len(found) == 4
    for got, want in zip(found, (2.0, 2.45, 2.9, 15.0)):
        assert abs(got - want) <= 1 / FPS


def test_detector_ignores_white_flashes_and_a_dark_room():
    assert gv.find_sync_flashes(_frames(flash_onsets=(), distractors=(5.0, 9.0)), fps=FPS) == []


def test_detector_accepts_timestamped_frames_and_reports_event_details():
    frames = _frames(flash_onsets=(4.0,))
    events = gv.find_sync_flash_events(((i / FPS + 100.0, f) for i, f in enumerate(frames)))
    (event,) = events
    assert event.time_s == pytest.approx(104.0, abs=1 / FPS)
    assert 0.1 <= event.duration_s <= 0.25 and event.frames >= 3


def test_detector_survives_a_green_object_that_is_much_weaker_than_the_flash():
    frames = _frames(flash_onsets=(3.0,))
    for i in range(int(8 * FPS), int(8.2 * FPS)):  # a small green patch (3% of the picture) walking past
        frames[i][:2, :32] = (20, 230, 30)
    assert len(gv.find_sync_flashes(frames, fps=FPS)) == 1


def test_bare_frames_need_a_frame_rate():
    with pytest.raises(ValueError, match="fps"):
        gv.find_sync_flashes(_frames(duration_s=1))


# -- offset ---------------------------------------------------------------------------------------


def test_offset_recovers_the_clock_difference_and_reports_a_small_residual():
    video = [2.0, 2.45, 2.9, 15.0]
    result = gv.compute_offset(video, _pc_ms(video, jitter_ms=15))
    assert result["status"] == "ok" and result["matched"] == 4
    assert result["offset_s"] == pytest.approx(PC_OFFSET_S, abs=0.02)
    assert result["residual_max_ms"] <= 30
    assert gv.video_time_for_pc_unix_ms((2.0 + PC_OFFSET_S) * 1000, result["offset_s"]) == pytest.approx(2.0, abs=0.02)


def test_offset_ignores_extra_video_events_by_matching_the_pattern():
    video_flashes = [2.0, 2.45, 2.9, 15.0]
    video_seen = sorted(video_flashes + [0.7, 9.1, 9.6])  # a green shirt, lamp flicker...
    result = gv.compute_offset(video_seen, _pc_ms(video_flashes))
    assert result["status"] == "ok" and result["matched"] == 4
    assert result["offset_s"] == pytest.approx(PC_OFFSET_S, abs=0.01)


def test_offset_with_a_missed_flash_still_works_but_a_single_match_asks_for_a_check():
    video = [2.0, 2.45, 2.9, 15.0]
    three = gv.compute_offset(video[:3], _pc_ms(video))
    assert three["matched"] == 3 and three["status"] == "ok"
    one = gv.compute_offset([2.45], _pc_ms(video))
    assert one["matched"] == 1 and one["status"] == "check manually" and one["notes"]


def test_offset_flags_a_bad_residual_and_missing_inputs():
    video = [2.0, 2.45, 2.9]
    pc = _pc_ms(video)
    pc[2] += 130  # the third flash is 130 ms late on the PC: still matched, but not trustworthy
    bad = gv.compute_offset(video, pc)
    assert bad["status"] == "check manually" and bad["residual_max_ms"] > 100
    assert gv.compute_offset([], pc)["offset_s"] is None
    assert gv.compute_offset(video, [])["status"] == "check manually"
    assert gv.compute_offset([50.0], pc)["status"] == "check manually"  # a lone unrelated event: no real match


def test_offset_reports_clock_drift_between_the_start_and_end_flashes():
    video = [2.0, 2.45, 2.9, 602.0]
    pc = [(v + PC_OFFSET_S + 0.00005 * v) * 1000 for v in video]  # 50 ppm drift: 30 ms over 10 minutes
    assert gv.compute_offset(video, pc, tolerance_s=0.2)["drift_ms_per_min"] == pytest.approx(3.0, abs=0.5)


# -- the import CLI -------------------------------------------------------------------------------


def _write_video(path, frames):
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write video in this environment")
    for f in frames:
        writer.write(f)
    writer.release()
    if not path.exists() or path.stat().st_size < 1000:
        pytest.skip("OpenCV wrote an empty video in this environment")


def test_import_cli_copies_the_video_syncs_it_and_updates_session_json(tmp_path, monkeypatch):
    from src.experiment import import_glasses_video as imp
    from src.experiment import logger as logger_module
    from src.experiment.logger import ResultLogger

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    log = ResultLogger("p1")
    for i, v in enumerate([2.0, 2.45, 2.9]):
        log.add_sync_flash(f"start {i + 1}", (v + PC_OFFSET_S) * 1000)
    log.close()
    video = tmp_path / "export.mp4"
    _write_video(video, _frames(duration_s=6.0, flash_onsets=(2.0, 2.45, 2.9)))

    assert imp.main([str(log.folder), str(video), "--data-dir", str(tmp_path / "data")]) in (0, 2)
    sync = json.loads((log.glasses_dir / "video_sync.json").read_text())
    assert (log.glasses_dir / "export.mp4").exists()
    assert sync["offset_s"] == pytest.approx(PC_OFFSET_S, abs=0.1)  # mp4v re-encoding: a few frames of slack
    assert sync["meaning"].startswith("PC unix time")
    meta = json.loads((log.folder / "session.json").read_text())
    assert meta["glasses_video"]["video_file"] == "glasses/export.mp4" and meta["glasses_video"]["sync_status"] == sync["status"]


def test_import_cli_refuses_an_old_style_session(tmp_path):
    from src.experiment import import_glasses_video as imp

    with pytest.raises(SystemExit):
        imp.main(["1792967858", str(tmp_path / "x.mp4"), "--data-dir", str(tmp_path)])


# -- phone recording failure handling -------------------------------------------------------------


class _FakeDevice:
    def __init__(self, fail_start=False, fail_stop=False, fail_event=False):
        self.fail_start, self.fail_stop, self.fail_event = fail_start, fail_stop, fail_event
        self.calls = []

    def recording_start(self):
        self.calls.append("start")
        if self.fail_start:
            raise RuntimeError("low battery")
        return "rec-1"

    def recording_stop_and_save(self):
        self.calls.append("stop")
        if self.fail_stop:
            raise RuntimeError("phone went to sleep")

    def send_event(self, name, event_timestamp_unix_ns=None):
        self.calls.append(("event", name))
        if self.fail_event:
            raise RuntimeError("no connection")

    def receive_gaze_datum(self, timeout_seconds=0.5):
        return None

    def close(self):
        self.calls.append("close")


def test_recording_start_and_stop_report_their_status():
    device = _FakeDevice()
    source = NeonGazeSource(device=device)
    started = source.start_recording()
    assert started["recording"] == "started" and started["recording_id"] == "rec-1" and started["recording_started_pc_unix_ms"] > 0
    assert source.send_event("pc_sync_flashes_done") is True
    stopped = source.stop_recording()
    assert stopped["recording"] == "saved" and device.calls == ["start", ("event", "pc_sync_flashes_done"), "stop"]
    assert source.stop_recording() == {}  # nothing left to stop


def test_a_failing_recording_start_never_raises_and_is_recorded(capsys):
    device = _FakeDevice(fail_start=True)
    source = NeonGazeSource(device=device)
    status = source.start_recording()
    assert status["recording"].startswith("failed (") and "low battery" in status["recording"]
    assert source.stop_recording() == {} and "stop" not in device.calls  # never stop what never started
    assert "WARNING" in capsys.readouterr().err


def test_a_failing_stop_or_event_never_raises():
    device = _FakeDevice(fail_stop=True, fail_event=True)
    source = NeonGazeSource(device=device)
    source.start_recording()
    assert source.send_event("x") is False
    assert source.stop_recording()["recording"].startswith("failed to stop/save")
    source.stop()  # the session's cleanup still completes


def test_stopping_the_source_stops_a_recording_left_running():
    device = _FakeDevice()
    source = NeonGazeSource(device=device)
    source.start_recording()
    source.stop()
    assert device.calls == ["start", "stop", "close"]
