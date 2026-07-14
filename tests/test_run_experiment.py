from include.eye_tracking.constants import CAMERA_INDEX
from src.experiment.run_experiment import _camera_labels, _resolve_camera_choice


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
