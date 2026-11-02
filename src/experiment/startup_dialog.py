"""The startup dialog (PyQt6; PsychoPy's own gui.Dlg cannot fold sections or show click help). Layout: Participant ID, Mode, Run tutorial, Screen height,
Viewing distance, then a closed-by-default 'Advanced' section (contrast min/max, gaze indicator). Clicking a field's name or its '?' shows a help bubble
next to the mouse - a frameless tool-tip window (no title bar, no taskbar entry, never a dialog) that closes on the next click anywhere, on Esc or when
focus moves to another field. The settings model and validation are Qt-free and live in startup_model.py."""

import sys
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from src.experiment import session_files
from src.experiment.settings import DATA_DIR, save_last_startup
from src.experiment.startup_model import (
    FIELD_LABELS,
    HELP_TEXTS,
    MODE_REAL,
    MODE_TEST,
    PLACEHOLDER_NOTE,
    StartupSettings,
    defaults_from_disk,
    is_placeholder_geometry,
    suggest_tutorial,
    validate,
)

BUBBLE_WIDTH_CHARS = 40  # word-wrap width of the help bubble, in average characters
BUBBLE_CURSOR_OFFSET = (14, 18)  # gap between the pointer and the bubble's corner
_HELP_TRIGGER_PROPERTY = "helpKey"


class HelpBubble(QtWidgets.QLabel):
    """A frameless speech-bubble style label shown at the mouse pointer. Not a dialog: a tool-tip window never gets a title bar or a taskbar entry
    and never takes focus."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setMargin(8)
        self.setStyleSheet(
            "QLabel { background: #fffbe6; color: #202020; border: 1px solid #8a8466; border-radius: 6px; }"
        )
        self.hide()

    def show_text(self, text: str, anchor: QtCore.QPoint) -> None:
        self.setText(text)
        self.setFixedWidth(self.fontMetrics().averageCharWidth() * BUBBLE_WIDTH_CHARS + 2 * self.margin() + 2)
        self.setFixedHeight(self.heightForWidth(self.width()))
        self.move(self._position_for(anchor))
        self.show()
        self.raise_()

    def _position_for(self, anchor: QtCore.QPoint) -> QtCore.QPoint:
        """Just below-right of the pointer, kept inside the screen under it (flipped to the other side if it would not fit)."""
        screen = QtGui.QGuiApplication.screenAt(anchor) or QtGui.QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x = anchor.x() + BUBBLE_CURSOR_OFFSET[0]
        y = anchor.y() + BUBBLE_CURSOR_OFFSET[1]
        if x + self.width() > area.right():
            x = anchor.x() - BUBBLE_CURSOR_OFFSET[0] - self.width()
        if y + self.height() > area.bottom():
            y = anchor.y() - BUBBLE_CURSOR_OFFSET[1] - self.height()
        x = max(area.left(), min(x, area.right() - self.width()))
        y = max(area.top(), min(y, area.bottom() - self.height()))
        return QtCore.QPoint(x, y)


class HelpLabel(QtWidgets.QLabel):
    """A field name that can be clicked for help (pointing-hand cursor, underlined while hovered)."""

    pressed = QtCore.pyqtSignal()

    def __init__(self, key: str, text: str) -> None:
        super().__init__(text)
        self.setProperty(_HELP_TRIGGER_PROPERTY, key)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self._plain = text

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed.emit()

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        self.setText(f"<u>{self._plain}</u>")
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.setText(self._plain)
        super().leaveEvent(event)


class StartupDialog(QtWidgets.QDialog):
    """Collects a StartupSettings. `result_settings` is set when Start was pressed with valid input; invalid input keeps the dialog open."""

    def __init__(self, initial: StartupSettings | None = None, data_dir: Path | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._data_dir = data_dir or DATA_DIR
        self.result_settings: StartupSettings | None = None
        self._initial = initial or StartupSettings()
        self._tutorial_touched = False
        self._swallow_release = False
        self._status_is_error = False
        self.setWindowTitle("Saccade experiment")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self.bubble = HelpBubble(self)
        self._bubble_owner: str | None = None
        self.help_labels: dict[str, HelpLabel] = {}
        self.help_buttons: dict[str, QtWidgets.QToolButton] = {}

        self.participant = QtWidgets.QLineEdit(self._initial.participant_id)
        self.participant_hint = QtWidgets.QLabel()
        self.participant_hint.setStyleSheet("color: #666;")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([MODE_REAL, MODE_TEST])
        self.mode.setCurrentText(MODE_TEST if self._initial.test_mode else MODE_REAL)
        self.tutorial = QtWidgets.QComboBox()
        self.tutorial.addItems(["Yes", "No"])
        self.tutorial.setCurrentText("Yes" if self._initial.run_tutorial else "No")
        self.screen_height = QtWidgets.QLineEdit(f"{self._initial.screen_height_cm:g}")
        self.viewing_distance = QtWidgets.QLineEdit(f"{self._initial.viewing_distance_cm:g}")
        self.contrast_min = QtWidgets.QLineEdit(f"{self._initial.contrast_min_percent:g}")
        self.contrast_max = QtWidgets.QLineEdit(f"{self._initial.contrast_max_percent:g}")
        self.gaze_indicator = QtWidgets.QCheckBox()
        self.gaze_indicator.setChecked(self._initial.show_gaze_indicator)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)  # the dialog follows the Advanced section
        main = QtWidgets.QGridLayout()
        main.setColumnMinimumWidth(0, 190)
        main.setColumnMinimumWidth(1, 200)
        outer.addLayout(main)
        self._add_row(main, 0, "participant_id", self.participant)
        main.addWidget(self.participant_hint, 1, 1)
        self._add_row(main, 2, "mode", self.mode)
        self._add_row(main, 3, "run_tutorial", self.tutorial)
        self._add_row(main, 4, "screen_height_cm", self.screen_height)
        self._add_row(main, 5, "viewing_distance_cm", self.viewing_distance)

        self.advanced_toggle = QtWidgets.QPushButton()
        self.advanced_toggle.setFlat(True)
        self.advanced_toggle.setStyleSheet("text-align: left; padding: 4px 0; font-weight: bold;")
        self.advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.advanced_toggle.setAutoDefault(False)
        self.advanced_toggle.clicked.connect(lambda: self.set_advanced_visible(not self.advanced_visible))
        outer.addWidget(self.advanced_toggle)
        self.advanced_box = QtWidgets.QWidget()
        advanced = QtWidgets.QGridLayout(self.advanced_box)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setColumnMinimumWidth(0, 190)
        advanced.setColumnMinimumWidth(1, 200)
        self._add_row(advanced, 0, "contrast_min_percent", self.contrast_min)
        self._add_row(advanced, 1, "contrast_max_percent", self.contrast_max)
        self._add_row(advanced, 2, "show_gaze_indicator", self.gaze_indicator)
        outer.addWidget(self.advanced_box)
        self.set_advanced_visible(False)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(self.fontMetrics().height() * 2 + 4)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._on_start)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.start_button)
        outer.addLayout(buttons)

        self.participant.textChanged.connect(self._on_participant_changed)
        self.tutorial.activated.connect(self._on_tutorial_touched)
        for edit in (self.participant, self.screen_height, self.viewing_distance, self.contrast_min, self.contrast_max):
            edit.textChanged.connect(self._clear_error)
        for edit in (self.screen_height, self.viewing_distance):
            edit.textChanged.connect(self._refresh_note)
        QtWidgets.QApplication.instance().focusChanged.connect(lambda *_: self.hide_help())
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._update_participant_hint()
        self._refresh_note()
        self.participant.setFocus()

    # -- construction helpers ------------------------------------------------------------------

    def _add_row(self, grid: QtWidgets.QGridLayout, row: int, key: str, widget: QtWidgets.QWidget) -> None:
        label = HelpLabel(key, FIELD_LABELS[key])
        label.pressed.connect(lambda k=key: self.toggle_help(k))
        button = QtWidgets.QToolButton()
        button.setText("?")
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setProperty(_HELP_TRIGGER_PROPERTY, key)
        button.pressed.connect(lambda k=key: self.toggle_help(k))
        grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(widget, row, 1)
        grid.addWidget(button, row, 2)
        self.help_labels[key] = label
        self.help_buttons[key] = button

    # -- Advanced section ------------------------------------------------------------------------

    @property
    def advanced_visible(self) -> bool:
        return not self.advanced_box.isHidden()

    def set_advanced_visible(self, visible: bool) -> None:
        self.advanced_box.setVisible(visible)
        self.advanced_toggle.setText("Advanced ▾" if visible else "Advanced ▸")
        self.layout().activate()
        self.adjustSize()

    # -- help bubble -----------------------------------------------------------------------------

    @property
    def help_key(self) -> str | None:
        """The field whose help is showing, or None."""
        return self._bubble_owner if self.bubble.isVisible() else None

    def toggle_help(self, key: str, anchor: QtCore.QPoint | None = None) -> None:
        """Show `key`'s help next to the pointer; the same field again closes it."""
        if self.help_key == key:
            self.hide_help()
            return
        self._bubble_owner = key
        self.bubble.show_text(HELP_TEXTS[key], anchor if anchor is not None else QtGui.QCursor.pos())

    def hide_help(self) -> None:
        self._bubble_owner = None
        self.bubble.hide()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """While a bubble is open the next click anywhere (or Esc) closes it and is consumed, so it cannot also press the button under it; a click
        on another field's name or '?' closes this bubble and opens that field's instead."""
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseButtonRelease and self._swallow_release:
            self._swallow_release = False
            return True
        if not self.bubble.isVisible():
            return False
        if kind == QtCore.QEvent.Type.MouseButtonPress:
            target = QtWidgets.QApplication.widgetAt(event.globalPosition().toPoint())
            is_other_trigger = (
                target is not None
                and target.property(_HELP_TRIGGER_PROPERTY) not in (None, self._bubble_owner)
            )
            self.hide_help()
            if is_other_trigger:
                return False
            self._swallow_release = True
            return True
        if kind == QtCore.QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.hide_help()
            return True
        if kind == QtCore.QEvent.Type.WindowDeactivate and obj is self:
            self.hide_help()
        return False

    def done(self, result: int) -> None:
        self.hide_help()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        super().done(result)

    # -- behaviour -------------------------------------------------------------------------------

    def _on_tutorial_touched(self, _index: int) -> None:
        self._tutorial_touched = True

    def _update_participant_hint(self) -> None:
        text = self.participant.text()
        folder = session_files.sanitize_participant_id(text)
        self.participant_hint.setText(f"will be saved as {folder}")

    def _on_participant_changed(self, text: str) -> None:
        self._update_participant_hint()
        if not self._tutorial_touched:
            self.tutorial.setCurrentText("Yes" if suggest_tutorial(text, self._data_dir) else "No")

    def _clear_error(self) -> None:
        self._refresh_note()

    def set_status(self, message: str, error: bool = False) -> None:
        self._status_is_error = error
        self.status.setStyleSheet("color: #b00020;" if error else "color: #8a5a00;")
        self.status.setText(message)

    def _refresh_note(self) -> None:
        note = PLACEHOLDER_NOTE if is_placeholder_geometry(self.screen_height.text(), self.viewing_distance.text()) else ""
        self.set_status(note)

    def _on_start(self) -> None:
        self.hide_help()
        try:
            settings = validate(
                self.participant.text(),
                self.mode.currentText(),
                self.tutorial.currentText() == "Yes",
                self.screen_height.text(),
                self.viewing_distance.text(),
                self.contrast_min.text(),
                self.contrast_max.text(),
                self.gaze_indicator.isChecked(),
            )
        except ValueError as error:
            self.set_status(str(error), error=True)
            return
        self.result_settings = settings
        self.accept()


def prompt_startup_settings(data_dir: Path | None = None) -> StartupSettings | None:
    """Show the startup dialog with the remembered values and return the validated choice (also remembered for next time), or None on Cancel."""
    QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)  # run_experiment calls psychopy.gui.ensureQtApp() first, so it is PsychoPy's own
    dialog = StartupDialog(defaults_from_disk(data_dir), data_dir)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or dialog.result_settings is None:
        return None
    chosen = dialog.result_settings
    save_last_startup(
        {
            "participant_id": chosen.participant_id,
            "test_mode": chosen.test_mode,
            "screen_height_cm": chosen.screen_height_cm,
            "viewing_distance_cm": chosen.viewing_distance_cm,
            "contrast_min_percent": chosen.contrast_min_percent,
            "contrast_max_percent": chosen.contrast_max_percent,
            "show_gaze_indicator": chosen.show_gaze_indicator,
        }
    )
    return chosen
