"""Puts this folder (sim_tools, invariants, find_bugs) and tests/ on sys.path, so the tests here can
import their helpers - and each other's regression tests - by plain module name, whichever test file
pytest happens to import first. (The project root comes from pytest.ini's `pythonpath = .`.)"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


import pytest  # noqa: E402


@pytest.fixture
def scheduled_flash_mode(monkeypatch):
    """The older open-loop mode (reaction-time test, flash at average RT + offset). The default is
    now the gaze-triggered "onset" mode (FLASH_TRIGGER); tests of the scheduled behaviour opt in."""
    monkeypatch.setattr("src.experiment.session.FLASH_TRIGGER", "scheduled")
    # ...with the plain legacy details: no random bounce on the offsets and the live staircase
    # (not the fixed probe contrasts).
    monkeypatch.setattr("src.experiment.session.TIMING_JITTER_MS", 0)
    monkeypatch.setattr("src.experiment.session.SACCADE_FIXED_CONTRASTS", ())


@pytest.fixture
def onset_flash_mode(monkeypatch):
    """The gaze-triggered mode (flash fires when the onset detector confirms a saccade)."""
    monkeypatch.setattr("src.experiment.session.FLASH_TRIGGER", "onset")
    monkeypatch.setattr("src.experiment.session.SACCADE_FIXED_CONTRASTS", ())
