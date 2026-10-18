import numpy as np
import pytest

pytest.importorskip("pupil_labs.real_time_screen_gaze")

from src.eye_tracking.neon_gaze_view import apply_affine, fit_affine  # noqa: E402


def test_affine_fit_recovers_a_constant_offset_and_scale():
    targets = [(x, y) for y in (270, 540, 810) for x in (422, 960, 1498)]
    raw = [(0.9 * x + 120, 0.95 * y - 60) for x, y in targets]  # simulated gaze bias
    matrix = fit_affine(raw, targets)
    for r, t in zip(raw, targets):
        assert apply_affine(matrix, *r) == pytest.approx(t, abs=1e-6)
