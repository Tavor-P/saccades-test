from pathlib import Path
from types import SimpleNamespace

import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH, NEON_STALE_TIMEOUT_S
from include.eye_tracking.types import GazeZone
from include.experiment.constants import SACCADE_ONSET_STABILITY_MS
from src.eye_tracking.gaze_trace import GazeTraceWriter, ReplayGazeSource, TraceRow, load_trace
from src.eye_tracking.neon_source import NeonGazeSource
from src.experiment.trial_mechanics import OnsetDetector

SAMPLE_TRACE = Path(__file__).parent / "fixtures" / "neon_trace_sample.csv"


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _synthetic_saccade_trace() -> list[TraceRow]:
    """200 Hz: 0.5 s fixating center (ratio 0.5), then a jump to the right
    (ratio 0.8) at t=0.5 s, held until 1.0 s."""
    rows = []
    for i in range(200):
        t = i / 200
        ratio = 0.5 if t < 0.5 else 0.8
        rows.append(TraceRow(t, 1000.0 + t, ratio * NEON_SCENE_WIDTH, True))
    return rows



def test_writer_loader_round_trip_and_neon_source_records_a_trace(tmp_path):
    path = tmp_path / "trace.csv"
    w = GazeTraceWriter(path)
    w.write(1.5, 1700000000.25, 800.0, True)
    w.write(1.505, None, None, False)
    w.close()
    rows = load_trace(path)
    assert rows[0] == TraceRow(1.5, 1700000000.25, 800.0, True)
    assert rows[1].x_px is None and rows[1].worn is False and rows[1].device_unix_s is None

    path = tmp_path / "rec.csv"
    source = NeonGazeSource(device=SimpleNamespace(close=lambda: None), trace_path=path)
    source._ingest(800.0, True, 1000.0)
    source._ingest(float("nan"), True, 1000.005)
    source._ingest(900.0, False, 1000.010)
    source.stop()
    rows = load_trace(path)
    assert [r.worn for r in rows] == [True, True, False]
    assert rows[0].x_px == 800.0 and rows[0].device_unix_s == 1000.0
    source.stop()  # idempotent

    with pytest.raises(ValueError):
        ReplayGazeSource([])  # an empty trace is rejected


def test_replay_follows_the_timeline_independent_of_polling_with_calibration_lost_track_and_onset():
    clock = _Clock()
    source = ReplayGazeSource(_synthetic_saccade_trace(), time_fn=clock)
    source.start()
    clock.t += 0.3
    assert source.latest_sample().zone is GazeZone.CENTER
    clock.t += 0.3  # t=0.6 s, after the jump
    assert source.latest_sample().zone is GazeZone.RIGHT
    clock.t += 0.35
    assert source.latest_sample().face_found
    clock.t += NEON_STALE_TIMEOUT_S + 1  # the trace end reports lost track like a stalled live stream
    assert not source.latest_sample().face_found
    assert source.finished

    # Polled rarely vs often, every row is fed through the mapper in order: same final sample.
    results = []
    for step in (0.001, 0.2):
        clock = _Clock()
        source = ReplayGazeSource(_synthetic_saccade_trace(), time_fn=clock)
        source.start()
        for _ in range(int(0.9 / step)):
            clock.t += step
            sample = source.latest_sample()
        results.append((sample.position, sample.smoothed_position))
    assert results[0] == pytest.approx(results[1], abs=1e-6)

    clock = _Clock()
    source = ReplayGazeSource(_synthetic_saccade_trace(), time_fn=clock)
    source.calibrate(0.2, 0.5, 0.8)  # ratio 0.8 is now the right target
    source.start()
    clock.t += 0.9
    assert source.latest_sample().position == pytest.approx(1.0, abs=0.02)

    clock = _Clock()
    source = ReplayGazeSource([TraceRow(0.0, None, 800.0, True), TraceRow(0.1, None, None, False)], time_fn=clock)
    source.start()
    clock.t += 0.15
    assert not source.latest_sample().face_found  # not-worn rows

    clock = _Clock()
    source = ReplayGazeSource(_synthetic_saccade_trace(), time_fn=clock)
    source.start()
    detector = OnsetDetector(GazeZone.CENTER)
    fired_at = None
    for _ in range(900):
        clock.t += 0.001
        if detector.update(source.latest_sample(), clock.t - 100.0):
            fired_at = clock.t - 100.0
            break
    # The jump happens at 0.5 s; confirmation needs >= SACCADE_ONSET_STABILITY_MS after that.
    assert fired_at is not None
    assert 0.5 + SACCADE_ONSET_STABILITY_MS / 1000 <= fired_at < 0.5 + SACCADE_ONSET_STABILITY_MS / 1000 + 0.02


def test_real_recorded_trace_replays_with_smooth_capture_times_or_arrival_without_device_stamps():
    """The recorded trace's arrival stamps are quantised to 15.6 ms (many rows share one t_s); the replayed samples are
    stamped from the phone's device timestamps instead: ~5 ms apart, strictly increasing, never after their arrival."""
    trace = load_trace(SAMPLE_TRACE)
    assert len(trace) > 500
    assert all(r.worn for r in trace)
    clock = _Clock()
    source = ReplayGazeSource(trace, time_fn=clock)
    assert source.timestamps_are_capture_time and source.gaze_latency_ms == 16.0
    source.start()
    clock.t += 3.0  # every row of the first 3 s has arrived
    source.latest_sample()
    stamps = [s.timestamp for s in source.samples_since(None)]
    assert len(stamps) > 100
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    assert gaps[len(gaps) // 2] == pytest.approx(0.005, abs=0.001)  # median spacing: 200 Hz, not 15.6 ms steps
    assert source.samples_since(stamps[-1]) == []
    assert [s.timestamp for s in source.samples_since(stamps[-3])] == stamps[-2:]

    clock = _Clock()
    source = ReplayGazeSource(trace, time_fn=clock)
    source.start()
    duration = trace[-1].t_s - trace[0].t_s
    seen = 0
    while source.elapsed_s < duration - 0.05:
        clock.t += 0.016  # a 60 Hz frame loop
        seen += source.latest_sample().face_found
    assert seen > 0.9 * (duration / 0.016)

    clock = _Clock()
    source = ReplayGazeSource([TraceRow(0.0, None, 800.0, True), TraceRow(0.1, None, 800.0, True)], time_fn=clock)
    assert not source.timestamps_are_capture_time and source.gaze_latency_ms is None
    source.start()
    clock.t += 0.11
    assert [s.timestamp for s in source.samples_since(None)] == pytest.approx([100.0, 100.1])
