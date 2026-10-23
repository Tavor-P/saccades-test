"""Puts this folder (sim_tools, invariants, find_bugs) and tests/ on sys.path, so the tests here can
import their helpers - and each other's regression tests - by plain module name, whichever test file
pytest happens to import first. (The project root comes from pytest.ini's `pythonpath = .`.)"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
