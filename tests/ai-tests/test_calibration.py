import pytest

from src.experiment.calibration import average_calibration_rounds


def test_averages_rounds_two_and_three_elementwise():
    assert average_calibration_rounds((0.2, 0.5, 0.8), (0.4, 0.5, 0.6)) == pytest.approx((0.3, 0.5, 0.7))
    ratios = (0.1, 0.5, 0.9)
    assert average_calibration_rounds(ratios, ratios) == pytest.approx(ratios)  # identical rounds average to themselves
