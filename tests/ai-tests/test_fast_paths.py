"""The analysis shortcuts that make long recordings and the bootstrap cheap must give what the plain formulas give."""

import numpy as np
import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.experiment import flash_gates as fg
from src.experiment import suppression_stats as stats
from src.experiment.saccade_detection import load_trace


def _trace(level: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / 200
    return t, np.where(t > 1.0, level, 0.4)


def test_speed_trace_is_cached_per_trace_and_recomputed_for_another():
    t1, x1 = _trace(0.7)
    t2, x2 = _trace(0.9)
    plain = lambda t, x, when: float(np.interp(when, t, np.abs(np.gradient(np.convolve(x, np.ones(3) / 3, mode="same"), t))))  # noqa: E731
    for t, x in ((t1, x1), (t2, x2), (t1, x1)):  # back and forth: the cache must follow the arrays it was asked about
        for when in (0.5, 1.0, 1.005, 1.5):
            assert fg.speed_at(t, x, when) == pytest.approx(plain(t, x, when), abs=1e-12)
    assert fg.speed_at(t1, x1, 99.0) is None  # outside the trace


def test_loglik_grid_matches_the_direct_formula_for_changing_counts():
    contrasts = np.asarray([0.04, 0.06, 0.08])
    n = np.asarray([20.0, 20.0, 20.0])
    for k in (np.asarray([11.0, 14.0, 18.0]), np.asarray([10.0, 10.0, 11.0])):  # same contrasts, other answers: one cached table
        s = np.asarray(stats._SLOPES)[:, None, None]
        t = stats._THRESHOLDS[None, :, None]
        p = np.clip(stats._weibull(contrasts[None, None, :], t, s), 1e-9, 1 - 1e-9)
        direct = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(axis=2)
        np.testing.assert_array_equal(stats._loglik_grid(contrasts, n, k), direct)


def test_load_trace_skips_rows_without_a_sample_and_tolerates_short_rows(tmp_path):
    path = tmp_path / "gaze_trace.csv"
    path.write_text("t_s,device_unix_s,x_px,worn\r\n0.0,10.00,800,1\r\n0.005,10.005,,1\r\n\r\n0.01,10.01\r\n0.015,10.015,1600,1\r\n", encoding="utf-8")
    t, x = load_trace(path, 0.5)
    assert t.tolist() == pytest.approx([10.5, 10.515]) and x.tolist() == pytest.approx([800 / NEON_SCENE_WIDTH, 1600 / NEON_SCENE_WIDTH])
    path.write_text("t_s,x\r\n0.0,1\r\n", encoding="utf-8")  # not a gaze trace at all: nothing, as before
    t, x = load_trace(path, 0.5)
    assert len(t) == 0 and len(x) == 0
