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


def test_validate_accepts_comma_decimals_range_edges_and_test_mode_without_id():
    from include.experiment.constants import FLASH_SPREAD_SD_MS

    s = valid()
    assert (s.screen_height_cm, s.viewing_distance_cm) == (19.5, 60.0)
    assert (s.contrast_min_percent, s.contrast_max_percent) == (4.0, 8.0)
    assert s.test_mode is False and s.run_tutorial is True and s.participant_id == "p01"
    assert parse_number(" 26,5 ") == 26.5
    assert valid(screen_height_text="5", viewing_distance_text="300").viewing_distance_cm == 300
    assert valid(screen_height_text="100", viewing_distance_text="20").screen_height_cm == 100
    assert valid().flash_spread_sd_ms == FLASH_SPREAD_SD_MS == StartupSettings().flash_spread_sd_ms
    assert valid(flash_spread_text="60,5").flash_spread_sd_ms == 60.5
    assert valid(flash_spread_text="10").flash_spread_sd_ms == 10 and valid(flash_spread_text="250").flash_spread_sd_ms == 250
    assert valid(participant_id="", mode="Test").test_mode is True
    assert valid(participant_id="", mode="Test").folder_id == "anonymous"


def test_validate_rejects_a_missing_id_bad_ranges_non_numbers_and_min_not_below_max():
    with pytest.raises(ValueError, match="Participant ID"):
        valid(participant_id="  ")
    for field, value in [
        ("screen_height_text", "4"), ("screen_height_text", "101"), ("screen_height_text", "abc"), ("screen_height_text", ""),
        ("viewing_distance_text", "19"), ("viewing_distance_text", "301"), ("viewing_distance_text", "nan"),
        ("contrast_min_text", "0.1"), ("contrast_max_text", "101"), ("contrast_max_text", "x"),
        ("flash_spread_text", "9"), ("flash_spread_text", "251"), ("flash_spread_text", "wide"),
    ]:
        with pytest.raises(ValueError):
            valid(**{field: value})
            pytest.fail(f"{field}={value!r} was accepted")
    for lo, hi in (("8", "8"), ("9", "8")):
        with pytest.raises(ValueError, match="smaller"):
            valid(contrast_min_text=lo, contrast_max_text=hi)


def test_defaults_from_disk_shipped_remembered_and_damaged_files(data_dir):
    d = defaults_from_disk(data_dir)
    assert d == StartupSettings(run_tutorial=True)
    assert d.contrast_min_percent == 4 and d.contrast_max_percent == 8

    (data_dir / "p01" / "2026-01-01_10-00-00").mkdir(parents=True)
    (data_dir / "settings.json").write_text(json.dumps({"contrast_floor_percent": 1.5}))  # an old file's other keys must survive
    settings.save_last_startup(
        {"participant_id": "p01", "test_mode": True, "screen_height_cm": 19, "viewing_distance_cm": 60.0, "run_tutorial": True,
         "flash_spread_sd_ms": 60}
    )
    d = defaults_from_disk(data_dir)
    assert (d.participant_id, d.test_mode, d.screen_height_cm, d.viewing_distance_cm) == ("p01", True, 19.0, 60.0)
    assert d.flash_spread_sd_ms == 60.0
    assert d.run_tutorial is False  # existing folder; the stored tutorial flag is ignored
    assert json.loads((data_dir / "settings.json").read_text())["contrast_floor_percent"] == 1.5

    for damaged in ("{not json", json.dumps({"contrast_floor_percent": 2.0}),
                    json.dumps({"last_startup": {"screen_height_cm": "tall", "test_mode": "yes", "x": 1}})):
        (data_dir / "settings.json").write_text(damaged)
        assert defaults_from_disk(data_dir) == StartupSettings(), damaged


def test_suggest_tutorial_only_without_runs_and_placeholder_detection(data_dir):
    assert suggest_tutorial("new", data_dir) is True
    (data_dir / "old" / "2026-01-01_10-00-00").mkdir(parents=True)
    assert suggest_tutorial("old", data_dir) is False
    (data_dir / "empty").mkdir()
    assert suggest_tutorial("empty", data_dir) is True
    assert suggest_tutorial("bad/id", data_dir) is True  # sanitised, no such folder

    assert is_placeholder_geometry(f"{SCREEN_HEIGHT_CM:g}", f"{VIEWING_DISTANCE_CM:g}")
    assert is_placeholder_geometry(str(SCREEN_HEIGHT_CM).replace(".", ","), "50")
    assert not is_placeholder_geometry("19.5", "50")
    assert not is_placeholder_geometry("", "")


# -- geometry ------------------------------------------------------------------------------------


def test_geometry_default_scaling_and_settings_record():
    import include.experiment.constants as c

    g = Geometry.default()
    assert g.is_placeholder
    assert (g.height_deg, g.line_offset_height_units, g.line_sd_height_units, g.line_height_height_units) == (
        c.SCREEN_HEIGHT_DEG, c.LINE_OFFSET_HEIGHT_UNITS, c.LINE_SD_HEIGHT_UNITS, c.LINE_HEIGHT_HEIGHT_UNITS)

    small = Geometry(screen_height_cm=13.0, viewing_distance_cm=50.0)
    assert not small.is_placeholder
    assert small.height_deg < g.height_deg
    assert small.line_offset_height_units == pytest.approx(LINE_OFFSET_DEG / small.height_deg)
    assert small.line_offset_height_units > g.line_offset_height_units
    assert small.line_height_height_units == pytest.approx(6 * LINE_SD_DEG / small.height_deg)
    assert Geometry(26, 100).height_deg < Geometry(26, 50).height_deg  # farther away = smaller angle
    with pytest.raises(ValueError):
        Geometry(0, 50)

    rec = Geometry(13, 40).as_settings()
    assert rec["screen_height_cm"] == 13 and rec["viewing_distance_cm"] == 40 and rec["geometry_is_placeholder"] is False
    assert rec["line_offset_height_units"] == pytest.approx(LINE_OFFSET_DEG / rec["screen_height_deg"], rel=1e-3)


def test_analyze_header_line_from_the_recorded_geometry():
    from src.experiment.analyze_session import format_geometry

    assert format_geometry(None) is None and format_geometry({"viewing_distance_cm": 50}) is None  # old sessions
    line = format_geometry(Geometry(19.5, 60).as_settings())
    assert "19.5 cm high at 60 cm" in line and "PLACEHOLDER" not in line
    assert "PLACEHOLDER" in format_geometry(Geometry.default().as_settings())


def test_logger_records_the_geometry_and_flash_spread_in_session_json(tmp_path, monkeypatch):
    from src.experiment import logger as logger_module

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = logger_module.ResultLogger("geo", test_mode=True)
    try:
        log.set_geometry(Geometry(19.5, 60))
        log.set_flash_spread_sd_ms(60)
        meta = json.loads((log.folder / "session.json").read_text())["settings"]
    finally:
        log.close()
    assert meta["screen_height_cm"] == 19.5 and meta["viewing_distance_cm"] == 60 and meta["geometry_is_placeholder"] is False
    assert meta["line_offset_height_units"] == pytest.approx(LINE_OFFSET_DEG / meta["screen_height_deg"], rel=1e-3)
    assert meta["flash_spread_sd_ms"] == 60.0

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



def test_field_order_labels_advanced_section_and_png_renders(dialog, qapp, tmp_path):
    assert not dialog.advanced_visible
    assert not dialog.contrast_min.isVisible() and not dialog.gaze_indicator.isVisible() and not dialog.flash_spread.isVisible()
    assert dialog.advanced_toggle.text() == "Advanced ▸"
    texts = {label.text() for label in dialog.findChildren(QtWidgets.QLabel)}
    assert not any(("amera" in t or "Gaze source" in t or "Record video" in t) for t in texts)  # removed fields
    out = os.environ.get("STARTUP_DIALOG_PNG_DIR")
    folder = tmp_path if not out else __import__("pathlib").Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    dialog.grab().save(str(folder / "dialog_collapsed.png"))

    closed_height = dialog.height()
    dialog.advanced_toggle.click()
    qapp.processEvents()
    assert dialog.advanced_visible and dialog.contrast_min.isVisible() and dialog.contrast_max.isVisible() and dialog.flash_spread.isVisible()
    assert dialog.advanced_toggle.text() == "Advanced ▾"
    assert dialog.height() > closed_height
    order = list(dialog.help_labels)
    assert order == [
        "participant_id", "mode", "run_tutorial", "screen_height_cm", "viewing_distance_cm",
        "contrast_min_percent", "contrast_max_percent", "flash_spread_sd_ms", "show_gaze_indicator",
    ]
    ys = [dialog.help_labels[k].mapTo(dialog, QtCore.QPoint(0, 0)).y() for k in order[:5]]
    assert ys == sorted(ys)
    assert dialog.help_labels["contrast_min_percent"].text() == "Saccade contrast min (%)"
    assert dialog.help_labels["flash_spread_sd_ms"].text() == "Flash timing spread (ms)"
    assert dialog.contrast_min.mapTo(dialog, QtCore.QPoint(0, 0)).y() < dialog.contrast_max.mapTo(dialog, QtCore.QPoint(0, 0)).y()
    dialog.grab().save(str(folder / "dialog_expanded.png"))
    dialog.toggle_help("screen_height_cm", dialog.help_labels["screen_height_cm"].mapToGlobal(QtCore.QPoint(10, 10)))
    qapp.processEvents()
    composed = QtGui.QPixmap(dialog.size())
    composed.fill(Qt.GlobalColor.white)
    painter = QtGui.QPainter(composed)
    painter.drawPixmap(0, 0, dialog.grab())
    painter.drawPixmap(dialog.mapFromGlobal(dialog.bubble.pos()), dialog.bubble.grab())
    painter.end()
    composed.save(str(folder / "dialog_with_help.png"))
    assert (folder / "dialog_with_help.png").exists()
    dialog.toggle_help("screen_height_cm", QtCore.QPoint(10, 10))

    dialog.advanced_toggle.click()
    qapp.processEvents()
    assert not dialog.advanced_visible and dialog.height() == closed_height


def test_help_is_a_frameless_bubble_at_the_pointer_never_a_window(dialog, qapp, monkeypatch):
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
    assert [w for w in visible_top_levels() if id(w) not in before] == [bubble]
    assert not [w for w in QtWidgets.QApplication.allWidgets() if w.isVisible() and (isinstance(w, QtWidgets.QMessageBox) or (isinstance(w, QtWidgets.QDialog) and w is not dialog))]
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    assert screen.contains(bubble.geometry())
    assert abs(bubble.x() - point.x()) <= 40 and abs(bubble.y() - point.y()) <= 40 or bubble.geometry().adjusted(-40, -40, 40, 40).contains(point)
    assert bubble.width() <= bubble.fontMetrics().averageCharWidth() * 40 + 40  # about 40 characters wide

    # Only one bubble; another label switches it.
    click_label(qapp, dialog.help_labels["run_tutorial"], QtCore.QPoint(100, 130))
    assert dialog.help_key == "run_tutorial" and dialog.bubble.text() == HELP_TEXTS["run_tutorial"]
    assert dialog.findChildren(startup_dialog.HelpBubble) == [dialog.bubble]

    # Stays inside the screen in a corner.
    area = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    dialog.toggle_help("contrast_min_percent", QtCore.QPoint(area.right() - 2, area.bottom() - 2))
    assert area.contains(dialog.bubble.geometry())

    # The ? button shows the same help; a second press on the same field closes it.
    QtGui.QCursor.setPos(QtCore.QPoint(200, 200))
    dialog.help_buttons["screen_height_cm"].pressed.emit()
    assert dialog.help_key == "screen_height_cm" and dialog.bubble.text() == HELP_TEXTS["screen_height_cm"]
    dialog.help_buttons["screen_height_cm"].pressed.emit()
    assert not dialog.bubble.isVisible()


def test_bubble_closes_on_escape_or_click_elsewhere_without_triggering_the_widget(dialog, qapp):
    click_label(qapp, dialog.help_labels["mode"], QtCore.QPoint(100, 100))
    QTest.keyClick(dialog.participant, Qt.Key.Key_Escape)
    assert not dialog.bubble.isVisible() and dialog.isVisible()

    click_label(qapp, dialog.help_labels["participant_id"], QtCore.QPoint(100, 100))
    assert dialog.bubble.isVisible()
    clicked = []
    dialog.start_button.clicked.connect(lambda: clicked.append(1))
    send_click(dialog.start_button)  # the event filter eats the press and the release
    qapp.processEvents()
    assert not dialog.bubble.isVisible()
    assert clicked == [] and dialog.result_settings is None and dialog.isVisible()
    send_click(dialog.start_button)  # next click works normally
    assert clicked == [1]


def test_start_validates_keeps_input_and_returns_settings_cancel_rejects(dialog, qapp, data_dir):
    dialog.participant.setText("")
    dialog.start_button.click()
    assert dialog.result_settings is None and dialog.isVisible()
    assert "Participant ID" in dialog.status.text()
    dialog.participant.setText("p9")
    dialog.screen_height.setText("500")
    dialog.start_button.click()
    assert "Screen height" in dialog.status.text() and dialog.screen_height.text() == "500" and dialog.participant.text() == "p9"
    dialog.screen_height.setText("19,5")
    dialog.flash_spread.setText("5")
    dialog.start_button.click()
    assert dialog.result_settings is None and "Flash timing spread" in dialog.status.text()

    dialog.flash_spread.setText("60")
    dialog.mode.setCurrentText("Test")
    dialog.start_button.click()
    s = dialog.result_settings
    assert s is not None and (s.participant_id, s.test_mode, s.screen_height_cm, s.viewing_distance_cm, s.flash_spread_sd_ms) == (
        "p9", True, 19.5, 60.0, 60.0)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

    d = StartupDialog(StartupSettings(participant_id="x"), data_dir)
    d.show()
    d.cancel_button.click()
    assert d.result_settings is None and d.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_tutorial_default_follows_the_id_until_touched_and_hints(dialog, data_dir, qapp):
    (data_dir / "old" / "2026-01-01_10-00-00").mkdir(parents=True)
    dialog.participant.setText("old")
    assert dialog.tutorial.currentText() == "No"
    dialog.participant.setText("brand_new")
    assert dialog.tutorial.currentText() == "Yes"
    dialog.tutorial.setCurrentText("No")
    dialog.tutorial.activated.emit(1)  # the user picked it
    dialog.participant.setText("another")
    assert dialog.tutorial.currentText() == "No"

    dialog.participant.setText("a/b c")
    assert "a_b_c" in dialog.participant_hint.text()
    dialog.participant.setText("")
    assert "anonymous" in dialog.participant_hint.text()
    assert dialog.status.text() == ""  # 19.5 / 60 are measured values
    dialog.screen_height.setText(f"{SCREEN_HEIGHT_CM:g}")
    dialog.viewing_distance.setText(f"{VIEWING_DISTANCE_CM:g}")
    assert dialog.status.text() == PLACEHOLDER_NOTE
