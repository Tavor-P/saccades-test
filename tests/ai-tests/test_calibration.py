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


from src.eye_tracking.gaze_mapping import GazeMapper  # noqa: E402
from src.experiment.calibration import validate_calibration  # noqa: E402


@pytest.mark.parametrize("ratios", [(0.3, 0.5, 0.7), (0.7, 0.5, 0.3), (0.35, 0.52, 0.66)])
def test_valid_calibrations_pass_either_direction(ratios):
    check = validate_calibration(ratios)
    assert check.ok and not check.problems
    assert check.residual < 0.1


@pytest.mark.parametrize(
    "ratios",
    [(0.5, 0.5, 0.5), (0.3, 0.7, 0.5), (0.5, 0.3, 0.7), (0.3, 0.3, 0.7), (0.49, 0.5, 0.51),
     (0.3, 0.31, 0.7), (float("nan"), 0.5, 0.7), (0.3, 0.5, 0.5)],
)
def test_bad_calibrations_rejected(ratios):
    check = validate_calibration(ratios)
    assert not check.ok and check.problems


def test_noise_limit_rejects_unsteady_window():
    assert validate_calibration((0.3, 0.5, 0.7), (0.01, 0.01, 0.01)).ok
    check = validate_calibration((0.3, 0.5, 0.7), (0.01, 0.2, 0.01))
    assert not check.ok and check.max_sample_std == 0.2


def test_nonlinear_spacing_rejected_by_residual():
    assert not validate_calibration((0.1, 0.9, 1.0), max_residual=0.05).ok


def test_mapper_calibrate_checked_applies_only_valid():
    mapper = GazeMapper()
    assert mapper.calibrate_checked(0.2, 0.5, 0.8).ok
    good = mapper.map(0.2, 0.0).position
    assert good == pytest.approx(0.0, abs=1e-6)
    assert not mapper.calibrate_checked(0.5, 0.5, 0.5).ok
    assert mapper.map(0.2, 0.1).position == pytest.approx(good)  # old fit kept


def test_recent_ratio_std_needs_full_window():
    mapper = GazeMapper()
    assert mapper.recent_ratio_std() is None
    for i in range(30):
        mapper.map(0.5, float(i))
    assert mapper.recent_ratio_std() == pytest.approx(0.0)
