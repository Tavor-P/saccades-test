from include.eye_tracking.constants import CAMERA_INDEX
from src.experiment.run_experiment import _camera_labels, _resolve_camera_choice, _resolve_contrast_floor_percent


def test_camera_labels_marks_the_configured_default():
    labels = _camera_labels([CAMERA_INDEX, CAMERA_INDEX + 1])
    assert labels[0] == f"Camera {CAMERA_INDEX} (default)"
    assert labels[1] == f"Camera {CAMERA_INDEX + 1}"


def test_camera_labels_with_only_a_non_default_camera():
    other = CAMERA_INDEX + 1
    assert _camera_labels([other]) == [f"Camera {other}"]


def test_resolve_camera_choice_maps_the_selected_label_back_to_its_index_and_label():
    indices = [0, 2, 5]
    labels = _camera_labels(indices)
    assert _resolve_camera_choice(indices, labels, labels[1]) == (2, labels[1])


def test_resolve_contrast_floor_percent_accepts_a_valid_value():
    assert _resolve_contrast_floor_percent("3.5", default_percent=1.0) == 3.5


def test_resolve_contrast_floor_percent_falls_back_on_non_numeric_input():
    assert _resolve_contrast_floor_percent("not a number", default_percent=1.0) == 1.0


def test_resolve_contrast_floor_percent_falls_back_on_blank_input():
    assert _resolve_contrast_floor_percent("", default_percent=1.0) == 1.0


def test_resolve_contrast_floor_percent_falls_back_when_out_of_range():
    assert _resolve_contrast_floor_percent("500", default_percent=1.0) == 1.0
