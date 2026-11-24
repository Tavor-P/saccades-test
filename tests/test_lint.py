"""ruff (lint) and mypy (types) must stay clean. Config: ruff.toml, mypy.ini. Skipped when a tool is not installed
(`pip install -r requirements.txt` has both)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(module: str, *args: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed (pip install -r requirements.txt)")
    result = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"{module} found problems:\n{result.stdout}\n{result.stderr}"


def test_ruff_is_clean():
    _run("ruff", "check", ".")


def test_mypy_is_clean():
    _run("mypy")
