from unittest.mock import MagicMock, Mock, patch

from include.eye_tracking.constants import CAMERA_INDEX
from src.experiment.run_experiment import (
    _camera_labels,
    _resolve_camera_choice,
    _resolve_contrast_floor_percent,
    _resolve_show_gaze_indicator,
    _resolve_test_mode,
    _second_monitor_screen_index,
    camera_preview_enabled,
    narrate_if_changed,
)


def _patch_screens(count: int):
    display = MagicMock()
    display.get_screens.return_value = [MagicMock() for _ in range(count)]
    return patch("src.experiment.run_experiment.pyglet.canvas.get_display", return_value=display)


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


def test_camera_preview_enabled_only_on_trial_zero():
    assert camera_preview_enabled(0) is True
    assert camera_preview_enabled(1) is False
    assert camera_preview_enabled(24) is False


def test_second_monitor_screen_index_picks_the_second_screen_when_present():
    with _patch_screens(2):
        assert _second_monitor_screen_index() == 1


def test_second_monitor_screen_index_falls_back_to_the_primary_with_one_screen():
    with _patch_screens(1):
        assert _second_monitor_screen_index() == 0


def test_resolve_show_gaze_indicator_only_true_for_yes():
    assert _resolve_show_gaze_indicator("Yes") is True
    assert _resolve_show_gaze_indicator("No") is False


def test_narrate_if_changed_speaks_when_text_is_new():
    narrator = Mock()
    result = narrate_if_changed(narrator, "look at the dot", last_spoken=None)
    narrator.speak.assert_called_once_with("look at the dot")
    assert result == "look at the dot"


def test_narrate_if_changed_stays_silent_when_text_is_unchanged():
    narrator = Mock()
    result = narrate_if_changed(narrator, "look at the dot", last_spoken="look at the dot")
    narrator.speak.assert_not_called()
    assert result == "look at the dot"


def test_narrate_if_changed_speaks_again_when_text_changes():
    narrator = Mock()
    result = narrate_if_changed(narrator, "now look at the cross", last_spoken="look at the dot")
    narrator.speak.assert_called_once_with("now look at the cross")
    assert result == "now look at the cross"


def test_resolve_test_mode_true_when_participant_id_is_blank():
    assert _resolve_test_mode("") is True
    assert _resolve_test_mode("   ") is True  # whitespace-only counts as blank


def test_resolve_test_mode_false_when_participant_id_is_given():
    assert _resolve_test_mode("42") is False
    assert _resolve_test_mode("alice") is False
    assert _resolve_test_mode("  p001  ") is False
