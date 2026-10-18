"""Architecture rules from CLAUDE.md, enforced mechanically. Each rule exists because breaking it
cost real debugging time (or would make the headless simulation/tests impossible)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Session/state-machine modules: pure logic, driven by tick()/render_state().
LOGIC_MODULES = ["session", "presaccade_session", "tutorial_session", "trial_mechanics", "trial_factory", "scoring", "zest", "calibration"]


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf8"))


def _imported_modules(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("module", LOGIC_MODULES)
def test_session_logic_does_not_import_a_graphics_or_hardware_stack(module):
    """Why: the simulator/tests run these without a window or camera."""
    imported = _imported_modules(_tree(ROOT / "src" / "experiment" / f"{module}.py"))
    assert not imported & {"psychopy", "pyglet", "cv2", "PySpin", "mediapipe", "pupil_labs"}, imported


@pytest.mark.parametrize("module", ["session", "presaccade_session", "tutorial_session", "trial_mechanics"])
def test_session_logic_reads_time_only_through_the_injected_clock(module):
    """Why: a direct time.monotonic()/sleep() makes the logic untestable and breaks the simulated
    clock (and pausing). (time.time() for the wall-clock unix timestamp is the one allowed use.)"""
    forbidden = []
    for node in ast.walk(_tree(ROOT / "src" / "experiment" / f"{module}.py")):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "time" and node.func.attr in {"monotonic", "sleep", "perf_counter"}:
                forbidden.append(f"time.{node.func.attr} at line {node.lineno}")
    assert not forbidden, forbidden


def test_include_holds_data_only_no_functions():
    """CLAUDE.md: 'no logic ever goes in include/'."""
    offenders = []
    for path in (ROOT / "include").rglob("*.py"):
        for node in _tree(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                offenders.append(f"{path.relative_to(ROOT)}:{node.name}")
    assert not offenders, offenders


def test_only_run_experiment_needs_psychopy_among_experiment_modules():
    offenders = []
    for path in (ROOT / "src" / "experiment").glob("*.py"):
        if path.name == "run_experiment.py":
            continue
        if {"psychopy", "pyglet"} & _imported_modules(_tree(path)):
            offenders.append(path.name)
    assert not offenders, offenders


@pytest.mark.parametrize(
    "module,forbidden",
    [
        ("src.eye_tracking.gaze_factory", {"PySpin", "mediapipe", "cv2", "pupil_labs"}),
        ("src.eye_tracking.neon_source", {"PySpin", "mediapipe", "psychopy"}),
        ("src.eye_tracking.gaze_trace", {"PySpin", "mediapipe", "psychopy", "cv2"}),
        ("src.experiment.simulation", {"PySpin", "mediapipe", "psychopy", "pyglet", "cv2"}),
    ],
)
def test_backends_are_imported_lazily(module, forbidden):
    """Why: a machine with only one backend installed (Neon-only, no Spinnaker SDK) must still run.
    Checked in a fresh interpreter so other tests' imports can't mask a violation."""
    code = (
        "import sys, importlib; importlib.import_module(%r); "
        "print(sorted(m for m in %r if m in sys.modules))" % (module, sorted(forbidden))
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "[]", f"{module} eagerly imported {out.stdout.strip()}"


def test_test_fixtures_are_not_swallowed_by_gitignore():
    """Bug we hit: `.gitignore` has `data/`, which silently ignores tests/data/ too."""
    fixtures = [p.relative_to(ROOT).as_posix() for p in (ROOT / "tests" / "fixtures").glob("*") if p.is_file()]
    assert fixtures, "no fixtures found"
    try:
        out = subprocess.run(["git", "check-ignore", *fixtures], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git not available")
    assert out.stdout.strip() == "", f"ignored fixtures (would never be committed): {out.stdout}"
