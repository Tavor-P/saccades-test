"""Startup dialog: the Qt-free model (validation, remembered defaults, tutorial suggestion, geometry) and the PyQt6 dialog run on the offscreen platform
(no window ever appears). Set STARTUP_DIALOG_PNG_DIR to also write PNG renders of the dialog states there."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before Qt is imported

import pytest

from include.experiment.constants import LINE_OFFSET_DEG, LINE_SD_DEG, SCREEN_HEIGHT_CM, VIEWING_DISTANCE_CM
from src.experiment import settings
from src.experiment.geometry import Geometry
from src.experiment.startup_model import (
    HELP_TEXTS,
    PLACEHOLDER_NOTE,
    StartupSettings,
    defaults_from_disk,
    is_placeholder_geometry,
    parse_number,
    suggest_tutorial,
    validate,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.experiment.startup_model.DATA_DIR", tmp_path)
    return tmp_path


def valid(**over):
    args = dict(
        participant_id="p01",
        mode="Real",
        run_tutorial=True,
        screen_height_text="19,5",
        viewing_distance_text="60",
        contrast_min_text="4",
        contrast_max_text="8",
        show_gaze_indicator=False,
    )
    args.update(over)
    return validate(**args)


# -- model ---------------------------------------------------------------------------------------


def test_validate_accepts_comma_decimals_and_returns_settings():
    s = valid()
    assert (s.screen_height_cm, s.viewing_distance_cm) == (19.5, 60.0)
    assert (s.contrast_min_percent, s.contrast_max_percent) == (4.0, 8.0)
    assert s.test_mode is False and s.run_tutorial is True and s.participant_id == "p01"


def test_validate_real_needs_an_id_but_test_does_not():
    with pytest.raises(ValueError, match="Participant ID"):
        valid(participant_id="  ")
    assert valid(participant_id="", mode="Test").test_mode is True
    assert valid(participant_id="", mode="Test").folder_id == "anonymous"


@pytest.mark.parametrize(
    "field,value",
    [
        ("screen_height_text", "4"), ("screen_height_text", "101"), ("screen_height_text", "abc"), ("screen_height_text", ""),
        ("viewing_distance_text", "19"), ("viewing_distance_text", "301"), ("viewing_distance_text", "nan"),
        ("contrast_min_text", "0.1"), ("contrast_max_text", "101"), ("contrast_max_text", "x"),
    ],
)
def test_validate_rejects_out_of_range_and_non_numbers(field, value):
    with pytest.raises(ValueError):
        valid(**{field: value})


def test_validate_range_edges_are_allowed():
    assert valid(screen_height_text="5", viewing_distance_text="300").viewing_distance_cm == 300
    assert valid(screen_height_text="100", viewing_distance_text="20").screen_height_cm == 100


def test_validate_contrast_min_must_be_below_max():
    with pytest.raises(ValueError, match="smaller"):
        valid(contrast_min_text="8", contrast_max_text="8")
    with pytest.raises(ValueError, match="smaller"):
        valid(contrast_min_text="9", contrast_max_text="8")


def test_parse_number_comma():
    assert parse_number(" 26,5 ") == 26.5


def test_placeholder_detection():
    assert is_placeholder_geometry(f"{SCREEN_HEIGHT_CM:g}", f"{VIEWING_DISTANCE_CM:g}")
    assert is_placeholder_geometry(str(SCREEN_HEIGHT_CM).replace(".", ","), "50")
    assert not is_placeholder_geometry("19.5", "50")
    assert not is_placeholder_geometry("", "")


def test_suggest_tutorial_only_for_a_participant_without_runs(data_dir):
    assert suggest_tutorial("new", data_dir) is True
    (data_dir / "old" / "2026-01-01_10-00-00").mkdir(parents=True)
    assert suggest_tutorial("old", data_dir) is False
    (data_dir / "empty").mkdir()
    assert suggest_tutorial("empty", data_dir) is True
    assert suggest_tutorial("bad/id", data_dir) is True  # sanitised, no such folder


def test_defaults_without_settings_file_are_shipped_values(data_dir):
    d = defaults_from_disk(data_dir)
    assert d == StartupSettings(run_tutorial=True)
    assert d.contrast_min_percent == 4 and d.contrast_max_percent == 8


def test_defaults_remember_last_values_and_derive_tutorial(data_dir):
    (data_dir / "p01" / "2026-01-01_10-00-00").mkdir(parents=True)
    settings.save_contrast_floor_percent(1.5)  # other settings must survive
    settings.save_last_startup(
        {"participant_id": "p01", "test_mode": True, "screen_height_cm": 19, "viewing_distance_cm": 60.0, "run_tutorial": True}
    )
    d = defaults_from_disk(data_dir)
    assert (d.participant_id, d.test_mode, d.screen_height_cm, d.viewing_distance_cm) == ("p01", True, 19.0, 60.0)
    assert d.run_tutorial is False  # existing folder; the stored tutorial flag is ignored
    assert json.loads((data_dir / "settings.json").read_text())["contrast_floor_percent"] == 1.5


def test_defaults_survive_damaged_and_old_settings_files(data_dir):
    (data_dir / "settings.json").write_text("{not json")
    assert defaults_from_disk(data_dir) == StartupSettings()
    (data_dir / "settings.json").write_text(json.dumps({"contrast_floor_percent": 2.0}))
    assert defaults_from_disk(data_dir) == StartupSettings()
    (data_dir / "settings.json").write_text(json.dumps({"last_startup": {"screen_height_cm": "tall", "test_mode": "yes", "x": 1}}))
    assert defaults_from_disk(data_dir) == StartupSettings()


# -- geometry ------------------------------------------------------------------------------------


def test_default_geometry_matches_the_constants():
    import include.experiment.constants as c

    g = Geometry.default()
    assert g.is_placeholder
    assert g.height_deg == c.SCREEN_HEIGHT_DEG
    assert g.line_offset_height_units == c.LINE_OFFSET_HEIGHT_UNITS
    assert g.line_sd_height_units == c.LINE_SD_HEIGHT_UNITS
    assert g.line_height_height_units == c.LINE_HEIGHT_HEIGHT_UNITS


def test_geometry_follows_screen_height_and_distance():
    small = Geometry(screen_height_cm=13.0, viewing_distance_cm=50.0)
    assert not small.is_placeholder
    assert small.height_deg < Geometry.default().height_deg
    assert small.line_offset_height_units == pytest.approx(LINE_OFFSET_DEG / small.height_deg)
    assert small.line_offset_height_units > Geometry.default().line_offset_height_units
    assert small.line_height_height_units == pytest.approx(6 * LINE_SD_DEG / small.height_deg)
    assert Geometry(26, 100).height_deg < Geometry(26, 50).height_deg  # farther away = smaller angle


def test_geometry_rejects_non_positive_values():
    with pytest.raises(ValueError):
        Geometry(0, 50)


def test_geometry_settings_record():
    rec = Geometry(13, 40).as_settings()
    assert rec["screen_height_cm"] == 13 and rec["viewing_distance_cm"] == 40 and rec["geometry_is_placeholder"] is False
    assert rec["line_offset_height_units"] == pytest.approx(LINE_OFFSET_DEG / rec["screen_height_deg"], rel=1e-3)


def test_line_position_uses_the_given_geometry():
    from include.experiment.constants import DOT_POSITION
    from include.experiment.types import Side
    from src.experiment.trial_mechanics import line_position

    g = Geometry(13, 40)
    _, y = line_position(DOT_POSITION, Side.ABOVE, g)
    assert y == pytest.approx(DOT_POSITION[1] - g.line_offset_height_units)
    assert line_position(DOT_POSITION, Side.BELOW) == line_position(DOT_POSITION, Side.BELOW, Geometry.default())


# -- Qt dialog -----------------------------------------------------------------------------------

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import QtCore, QtGui  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from src.experiment import startup_dialog  # noqa: E402
from src.experiment.startup_dialog import StartupDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])


@pytest.fixture
def dialog(qapp, data_dir):
    d = StartupDialog(StartupSettings(participant_id="", screen_height_cm=19.5, viewing_distance_cm=60.0), data_dir)
    d.show()
    qapp.processEvents()
    yield d
    d.close()
    d.deleteLater()


def click_label(qapp, label, point_in_screen):
    QtGui.QCursor.setPos(point_in_screen)
    QTest.mousePress(label, Qt.MouseButton.LeftButton, pos=QtCore.QPoint(3, 3))
    QTest.mouseRelease(label, Qt.MouseButton.LeftButton, pos=QtCore.QPoint(3, 3))
    qapp.processEvents()


def send_click(widget):
    """A left click delivered as the window system does (press, then release, through the application's event filters)."""
    local = QtCore.QPointF(widget.rect().center())
    global_pos = QtCore.QPointF(widget.mapToGlobal(widget.rect().center()))
    for kind, buttons in ((QtCore.QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton), (QtCore.QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
        event = QtGui.QMouseEvent(kind, local, global_pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(widget, event)


def visible_top_levels():
    return [w for w in QtWidgets.QApplication.topLevelWidgets() if w.isVisible()]


def test_field_order_and_labels(dialog, qapp):
    dialog.set_advanced_visible(True)
    qapp.processEvents()
    order = list(dialog.help_labels)
    assert order == [
        "participant_id", "mode", "run_tutorial", "screen_height_cm", "viewing_distance_cm",
        "contrast_min_percent", "contrast_max_percent", "show_gaze_indicator",
    ]
    ys = [dialog.help_labels[k].mapTo(dialog, QtCore.QPoint(0, 0)).y() for k in order[:5]]
    assert ys == sorted(ys)
    assert dialog.help_labels["contrast_min_percent"].text() == "Saccade contrast min (%)"
    assert dialog.contrast_min.mapTo(dialog, QtCore.QPoint(0, 0)).y() < dialog.contrast_max.mapTo(dialog, QtCore.QPoint(0, 0)).y()


def test_advanced_is_closed_by_default_and_toggles(dialog, qapp):
    assert not dialog.advanced_visible
    assert not dialog.contrast_min.isVisible() and not dialog.gaze_indicator.isVisible()
    assert dialog.advanced_toggle.text() == "Advanced ▸"
    closed_height = dialog.height()
    dialog.advanced_toggle.click()
    qapp.processEvents()
    assert dialog.advanced_visible and dialog.contrast_min.isVisible() and dialog.contrast_max.isVisible()
    assert dialog.advanced_toggle.text() == "Advanced ▾"
    assert dialog.height() > closed_height
    dialog.advanced_toggle.click()
    qapp.processEvents()
    assert not dialog.advanced_visible and dialog.height() == closed_height


def test_clicking_a_label_shows_a_bubble_at_the_pointer_and_no_window(dialog, qapp, monkeypatch):
    boxes = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda *a, **k: boxes.append(1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "show", lambda *a, **k: boxes.append(1))
    before = {id(w) for w in visible_top_levels()}
    point = QtCore.QPoint(120, 140)
    click_label(qapp, dialog.help_labels["mode"], point)
    bubble = dialog.bubble
    assert bubble.isVisible() and dialog.help_key == "mode"
    assert bubble.text() == HELP_TEXTS["mode"]
    assert not isinstance(bubble, QtWidgets.QDialog) and boxes == []
    assert bubble.windowFlags() & Qt.WindowType.ToolTip  # tool-tip window: no title bar, no taskbar entry
    assert bubble.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert bubble.windowTitle() == ""
    new_windows = [w for w in visible_top_levels() if id(w) not in before]
    assert new_windows == [bubble]
    assert not [w for w in QtWidgets.QApplication.allWidgets() if w.isVisible() and (isinstance(w, QtWidgets.QMessageBox) or (isinstance(w, QtWidgets.QDialog) and w is not dialog))]
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    assert screen.contains(bubble.geometry())
    assert abs(bubble.x() - point.x()) <= 40 and abs(bubble.y() - point.y()) <= 40 or bubble.geometry().adjusted(-40, -40, 40, 40).contains(point)
    assert bubble.width() <= bubble.fontMetrics().averageCharWidth() * 40 + 40  # about 40 characters wide


def test_question_mark_button_shows_the_same_help_and_second_click_closes(dialog, qapp):
    QtGui.QCursor.setPos(QtCore.QPoint(200, 200))
    dialog.help_buttons["screen_height_cm"].pressed.emit()
    assert dialog.help_key == "screen_height_cm"
    assert dialog.bubble.text() == HELP_TEXTS["screen_height_cm"]
    dialog.help_buttons["screen_height_cm"].pressed.emit()  # same field again closes
    assert not dialog.bubble.isVisible()


def test_click_elsewhere_closes_bubble_without_triggering_the_widget_under_it(dialog, qapp):
    click_label(qapp, dialog.help_labels["participant_id"], QtCore.QPoint(100, 100))
    assert dialog.bubble.isVisible()
    clicked = []
    dialog.start_button.clicked.connect(lambda: clicked.append(1))
    # a real click on Start while the bubble is open: the event filter eats the press and the release
    send_click(dialog.start_button)
    qapp.processEvents()
    assert not dialog.bubble.isVisible()
    assert clicked == [] and dialog.result_settings is None and dialog.isVisible()
    send_click(dialog.start_button)  # next click works normally
    assert clicked == [1]


def test_escape_closes_bubble_but_not_the_dialog(dialog, qapp):
    click_label(qapp, dialog.help_labels["mode"], QtCore.QPoint(100, 100))
    QTest.keyClick(dialog.participant, Qt.Key.Key_Escape)
    assert not dialog.bubble.isVisible() and dialog.isVisible()


def test_only_one_bubble_and_clicking_another_label_switches(dialog, qapp):
    click_label(qapp, dialog.help_labels["mode"], QtCore.QPoint(100, 100))
    click_label(qapp, dialog.help_labels["run_tutorial"], QtCore.QPoint(100, 130))
    assert dialog.help_key == "run_tutorial" and dialog.bubble.text() == HELP_TEXTS["run_tutorial"]
    assert dialog.findChildren(startup_dialog.HelpBubble) == [dialog.bubble]


def test_bubble_stays_inside_the_screen_in_a_corner(dialog, qapp):
    area = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    dialog.toggle_help("contrast_min_percent", QtCore.QPoint(area.right() - 2, area.bottom() - 2))
    assert area.contains(dialog.bubble.geometry())


def test_validation_error_keeps_dialog_open_with_input(dialog, qapp):
    dialog.participant.setText("")
    dialog.start_button.click()
    assert dialog.result_settings is None and dialog.isVisible()
    assert "Participant ID" in dialog.status.text()
    dialog.participant.setText("p9")
    dialog.screen_height.setText("500")
    dialog.start_button.click()
    assert "Screen height" in dialog.status.text() and dialog.screen_height.text() == "500" and dialog.participant.text() == "p9"


def test_start_returns_the_settings_and_cancel_returns_none(dialog, qapp):
    dialog.participant.setText("p9")
    dialog.mode.setCurrentText("Test")
    dialog.screen_height.setText("19,5")
    dialog.start_button.click()
    s = dialog.result_settings
    assert s is not None and (s.participant_id, s.test_mode, s.screen_height_cm, s.viewing_distance_cm) == ("p9", True, 19.5, 60.0)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_cancel_button_rejects(qapp, data_dir):
    d = StartupDialog(StartupSettings(participant_id="x"), data_dir)
    d.show()
    d.cancel_button.click()
    assert d.result_settings is None and d.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_tutorial_default_follows_the_id_until_touched(dialog, data_dir, qapp):
    (data_dir / "old" / "2026-01-01_10-00-00").mkdir(parents=True)
    dialog.participant.setText("old")
    assert dialog.tutorial.currentText() == "No"
    dialog.participant.setText("brand_new")
    assert dialog.tutorial.currentText() == "Yes"
    dialog.tutorial.setCurrentText("No")
    dialog.tutorial.activated.emit(1)  # the user picked it
    dialog.participant.setText("another")
    assert dialog.tutorial.currentText() == "No"


def test_saved_as_hint_and_placeholder_note(dialog, qapp):
    dialog.participant.setText("a/b c")
    assert "a_b_c" in dialog.participant_hint.text()
    dialog.participant.setText("")
    assert "anonymous" in dialog.participant_hint.text()
    assert dialog.status.text() == ""  # 19.5 / 60 are measured values
    dialog.screen_height.setText(f"{SCREEN_HEIGHT_CM:g}")
    dialog.viewing_distance.setText(f"{VIEWING_DISTANCE_CM:g}")
    assert dialog.status.text() == PLACEHOLDER_NOTE


def test_no_camera_gaze_source_or_record_video_fields(dialog):
    texts = {label.text() for label in dialog.findChildren(QtWidgets.QLabel)}
    assert not any(("amera" in t or "Gaze source" in t or "Record video" in t) for t in texts)


def test_render_pngs(dialog, qapp, tmp_path):
    out = os.environ.get("STARTUP_DIALOG_PNG_DIR")
    folder = tmp_path if not out else __import__("pathlib").Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    dialog.participant.setText("participant_01")
    dialog.grab().save(str(folder / "dialog_collapsed.png"))
    dialog.advanced_toggle.click()
    qapp.processEvents()
    dialog.grab().save(str(folder / "dialog_expanded.png"))
    dialog.status.setText("Contrast min must be smaller than max")
    dialog.toggle_help("screen_height_cm", dialog.help_labels["screen_height_cm"].mapToGlobal(QtCore.QPoint(10, 10)))
    qapp.processEvents()
    dialog.bubble.grab().save(str(folder / "help_bubble.png"))
    composed = QtGui.QPixmap(dialog.size())
    composed.fill(Qt.GlobalColor.white)
    painter = QtGui.QPainter(composed)
    painter.drawPixmap(0, 0, dialog.grab())
    painter.drawPixmap(dialog.mapFromGlobal(dialog.bubble.pos()), dialog.bubble.grab())
    painter.end()
    composed.save(str(folder / "dialog_with_help.png"))
    assert (folder / "dialog_with_help.png").exists()


# -- geometry logging ----------------------------------------------------------------------------


def test_analyze_header_line_from_the_recorded_geometry():
    from src.experiment.analyze_session import format_geometry

    assert format_geometry(None) is None and format_geometry({"viewing_distance_cm": 50}) is None  # old sessions
    line = format_geometry(Geometry(19.5, 60).as_settings())
    assert "19.5 cm high at 60 cm" in line and "PLACEHOLDER" not in line
    assert "PLACEHOLDER" in format_geometry(Geometry.default().as_settings())


def test_logger_records_the_geometry_in_session_json(tmp_path, monkeypatch):
    from src.experiment import logger as logger_module

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = logger_module.ResultLogger("geo", test_mode=True)
    try:
        log.set_geometry(Geometry(19.5, 60))
        meta = json.loads((log.folder / "session.json").read_text())["settings"]
    finally:
        log.close()
    assert meta["screen_height_cm"] == 19.5 and meta["viewing_distance_cm"] == 60 and meta["geometry_is_placeholder"] is False
    assert meta["line_offset_height_units"] == pytest.approx(LINE_OFFSET_DEG / meta["screen_height_deg"], rel=1e-3)
