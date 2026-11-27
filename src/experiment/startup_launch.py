"""The start of a run: the startup dialog, shown before PsychoPy is imported. `-m src.experiment.run_experiment` hands over to launch() above its
own imports, because PsychoPy (with the pandas, scipy and matplotlib it pulls in) takes ~8 s to import and the dialog needs only PyQt6.
Nothing imported here may load PsychoPy or pyglet (test_architecture_rules.py checks)."""

import sys

from src.experiment.geometry import Geometry
from src.experiment.settings import saccade_contrast_levels
from src.experiment.startup_dialog import prompt_startup_settings

# (participant_id, show_gaze_indicator, run_tutorial, test_mode, saccade_contrasts, geometry, flash_spread_sd_ms, neon_address_text, return_jump)
SessionInfo = tuple[str, bool, bool, bool, tuple[float, ...], Geometry, float, str, bool]

_qt_app: object = None  # holds the QApplication: PyQt destroys it with its last Python reference


def ensure_qt_app() -> None:
    """The one QApplication, shared by the dialog, the error box and the PsychoPy window created afterwards. Does what psychopy.gui.ensureQtApp()
    does (Fusion style) without importing psychopy.gui (~4 s); psychopy.gui adopts an existing QApplication if it is imported later."""
    global _qt_app
    from PyQt6 import QtWidgets

    _qt_app = QtWidgets.QApplication.instance()
    if _qt_app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")  # as PsychoPy does: avoids a PyQt bug with a greyed-out OK button
        _qt_app = app


def prompt_session_info() -> SessionInfo | None:
    """The startup dialog (src/experiment/startup_dialog.py): Participant ID, Mode, Run tutorial, screen height/distance and, under Advanced, the saccade
    phase's contrast range (min/max, see SACCADE_FIXED_CONTRASTS) and the Show gaze indicator toggle. Invalid input keeps the dialog open with a message.
    Returns the choices (SessionInfo), or None if cancelled."""
    ensure_qt_app()
    chosen = prompt_startup_settings()
    if chosen is None:
        return None
    return (
        chosen.folder_id,
        chosen.show_gaze_indicator,
        chosen.run_tutorial,
        chosen.test_mode,
        saccade_contrast_levels(chosen.contrast_min_percent, chosen.contrast_max_percent),
        chosen.geometry,
        chosen.flash_spread_sd_ms,
        chosen.neon_address,
        chosen.return_jump,
    )


def launch() -> None:
    """Dialog first; only after OK import run_experiment (and with it PsychoPy) and run the experiment."""
    session_info = prompt_session_info()
    if session_info is None:
        return  # cancelled: nothing else is loaded
    print("Loading PsychoPy...", flush=True)
    # Not pre-imported on a thread while the dialog is open: importing pyglet creates a hidden window, its GL context and the event loop
    # on the importing thread, which must be the main one.
    from src.experiment import run_experiment

    run_experiment.main(session_info)
