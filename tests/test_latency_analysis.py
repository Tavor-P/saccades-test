import numpy as np
import pytest

from src.eye_tracking.latency_analysis import measure_latencies


def _step_signal(times, jump_times, delay_s, lo=0.0, hi=1.0, ramp_s=0.04):
    """Square-ish wave alternating lo/hi at each jump (+delay), linear ramp over ramp_s."""
    values = np.full_like(times, lo)
    for k, jump in enumerate(jump_times):
        start = jump + delay_s
        target = hi if k % 2 == 0 else lo
        prev = lo if k % 2 == 0 else hi
        frac = np.clip((times - start) / ramp_s, 0, 1)
        values = np.where(times >= start, prev + (target - prev) * frac, values)
    return values


def test_recovers_known_delay_between_two_sampled_signals():
    rng = np.random.default_rng(0)
    jumps = [1.0, 2.5, 4.0, 5.5, 7.0]
    neon_t = np.arange(0, 8, 0.005)
    cam_t = np.arange(0, 8, 1 / 60)
    neon_x = _step_signal(neon_t, jumps, 0.200) + rng.normal(0, 0.005, neon_t.size)
    cam_x = _step_signal(cam_t, jumps, 0.150) + rng.normal(0, 0.005, cam_t.size)

    rows = measure_latencies(jumps, neon_t, neon_x, cam_t, cam_x)

    assert len(rows) == 4
    for row in rows:
        assert row["neon_minus_cam_ms"] == pytest.approx(50, abs=12)  # webcam's 16.7ms frame spacing limits precision


def test_flat_signal_is_skipped():
    jumps = [1.0, 2.5, 4.0]
    t = np.arange(0, 5, 0.01)
    flat = np.random.default_rng(1).normal(0, 0.01, t.size)
    assert measure_latencies(jumps, t, flat, t, flat) == []
