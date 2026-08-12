import pytest

from src.experiment.calibration import average_calibration_rounds


def test_averages_rounds_two_and_three_elementwise():
    round2 = (0.2, 0.5, 0.8)
    round3 = (0.4, 0.5, 0.6)
    result = average_calibration_rounds(round2, round3)
    assert result == pytest.approx((0.3, 0.5, 0.7))


def test_identical_rounds_average_to_themselves():
    ratios = (0.1, 0.5, 0.9)
    assert average_calibration_rounds(ratios, ratios) == pytest.approx(ratios)
