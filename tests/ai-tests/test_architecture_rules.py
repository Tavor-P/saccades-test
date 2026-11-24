"""Architecture rules from CLAUDE.md, enforced mechanically. Each rule exists because breaking it
cost real debugging time (or would make the headless simulation/tests impossible)."""

import ast
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Session/state-machine modules: pure logic, driven by tick()/render_state().
LOGIC_MODULES = ["session", "tutorial_session", "trial_mechanics", "trial_factory", "scoring", "calibration"]


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



def test_session_logic_does_not_import_a_graphics_or_hardware_stack():
    """Why: the simulator/tests run these without a window or camera."""
    for module in LOGIC_MODULES:
        imported = _imported_modules(_tree(ROOT / "src" / "experiment" / f"{module}.py"))
        assert not imported & {"psychopy", "pyglet", "cv2", "PySpin", "mediapipe", "pupil_labs"}, (module, imported)


def test_session_logic_reads_time_only_through_the_injected_clock():
    """Why: a direct time.perf_counter()/sleep() makes the logic untestable and breaks the simulated
    clock (and pausing). Wall-clock stamps come from clock.unix_ms() too."""
    forbidden = []
    for module in ["session", "tutorial_session", "trial_mechanics"]:
        for node in ast.walk(_tree(ROOT / "src" / "experiment" / f"{module}.py")):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "time" and node.func.attr in {"monotonic", "sleep", "perf_counter", "time"}:
                    forbidden.append(f"{module}: time.{node.func.attr} at line {node.lineno}")
    assert not forbidden, forbidden


def test_only_the_timebase_reads_the_coarse_or_raw_clocks():
    """Why: on Windows time.monotonic()/time.time() tick every 15.6 ms (half a saccade). Event timing must
    use src/timebase.py (perf_counter); wall-clock values are derived from it via an anchor."""
    forbidden = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "timebase.py":
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "time" and node.func.attr in {"monotonic", "time", "perf_counter", "monotonic_ns", "time_ns"}:
                    forbidden.append(f"{path.relative_to(ROOT)}:{node.lineno} time.{node.func.attr}")
    assert not forbidden, forbidden


# datetime.now() reads the coarse system clock (15.6 ms ticks on Windows), so it must never time an event. These are the legitimate
# human-readable wall-clock STAMPS (text in a file name or a report); each is a count of `datetime.now()`-style calls in that file.
WALL_CLOCK_STAMPS = {
    "src/experiment/logger.py": (1, "session.json run.started_at: a readable label (the exact start is `unix_timestamp`, from timebase)"),
    "src/experiment/analyze_participant.py": (1, "summary.json generated_at: when the analysis was run, never compared with an event time"),
    "src/experiment/compare.py": (2, "comparison.json generated time and the comparison folder's name: when the report was made"),
    "src/experiment/participants.py": (1, "suffix of the backup copy of a damaged participants.json"),
}
_NOW_METHODS = {"now", "utcnow", "today"}


def _wall_clock_calls(tree: ast.AST) -> list[int]:
    """Line numbers of datetime.now() / datetime.utcnow() / date.today() / datetime.datetime.now() style calls."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _NOW_METHODS:
            owner = node.func.value
            name = owner.id if isinstance(owner, ast.Name) else owner.attr if isinstance(owner, ast.Attribute) else None
            if name in {"datetime", "date"}:
                lines.append(node.lineno)
    return lines


def test_datetime_now_is_only_used_for_the_listed_human_readable_stamps():
    """Why: event timing comes from src/timebase.py. datetime.now()/utcnow() read the coarse system clock, so a slip like
    `datetime.now().timestamp()` as an event time would be off by up to 15.6 ms. Only the stamps in WALL_CLOCK_STAMPS are allowed
    (a new one needs a reason there); a listed file whose count changed fails too, so the list cannot go stale."""
    found = {}
    for path in (ROOT / "src").rglob("*.py"):
        lines = _wall_clock_calls(_tree(path))
        if lines:
            found[path.relative_to(ROOT).as_posix()] = lines
    unexpected = {f: lines for f, lines in found.items() if f not in WALL_CLOCK_STAMPS}
    assert not unexpected, f"datetime.now()/utcnow()/today() outside the allow-list (use timebase for timing): {unexpected}"
    changed = {f: (len(found.get(f, [])), n) for f, (n, _why) in WALL_CLOCK_STAMPS.items() if len(found.get(f, [])) != n}
    assert not changed, f"allow-list out of date (file: (found, listed)): {changed}"


def test_the_datetime_rule_catches_the_calls_it_forbids():
    """The rule's detector itself: every spelling it must flag, and a method call that is not a clock read."""
    for source in ("datetime.now()", "datetime.utcnow()", "date.today()", "datetime.datetime.now()", "datetime.now(tz)"):
        assert _wall_clock_calls(ast.parse(source)) == [1], source
    for source in ("datetime.fromtimestamp(t)", "clock.now()", "timebase.now()", "now()"):
        assert _wall_clock_calls(ast.parse(source)) == [], source


def test_timebase_uses_a_fine_resolution_clock_and_its_anchor_gives_sub_tick_unix_time():
    """The unix time derived from the perf counter must advance smoothly (not in 15.6 ms steps) and agree with the system clock."""
    import time

    from src import timebase

    assert timebase._clock is time.perf_counter
    assert time.get_clock_info("perf_counter").resolution <= 1e-6  # 100 ns on Windows; the others are 15.6 ms there
    stamps = {timebase.unix_now() for _ in range(2000)}
    assert len(stamps) > 1000
    assert abs(timebase.unix_now() - time.time()) < 0.05


def test_include_holds_data_only_no_functions():
    """CLAUDE.md: 'no logic ever goes in include/'."""
    offenders = []
    for path in (ROOT / "include").rglob("*.py"):
        for node in _tree(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                offenders.append(f"{path.relative_to(ROOT)}:{node.name}")
    assert not offenders, offenders


def test_only_run_experiment_and_rendering_need_psychopy_among_experiment_modules():
    """run_experiment.py (entry point, phase loops) and rendering.py (window, stimuli, drawing) are the only PsychoPy modules."""
    offenders = []
    for path in (ROOT / "src" / "experiment").glob("*.py"):
        if path.name in ("run_experiment.py", "rendering.py"):
            continue
        if {"psychopy", "pyglet"} & _imported_modules(_tree(path)):
            offenders.append(path.name)
    assert not offenders, offenders


def test_backends_are_imported_lazily():
    """Why: the device library (pupil_labs) loads only when a gaze source is built, so the simulator, tests and analysis run without it.
    Checked in a fresh interpreter per module so other tests' imports can't mask a violation."""
    cases = [
        ("src.eye_tracking.gaze_factory", {"PySpin", "mediapipe", "cv2", "pupil_labs"}),
        ("src.eye_tracking.neon_source", {"PySpin", "mediapipe", "psychopy"}),
        ("src.eye_tracking.neon_address", {"pupil_labs", "PySpin", "mediapipe", "psychopy", "cv2"}),
        ("src.eye_tracking.gaze_trace", {"PySpin", "mediapipe", "psychopy", "cv2"}),
        ("src.experiment.simulation", {"PySpin", "mediapipe", "psychopy", "pyglet", "cv2"}),
    ]

    def run(case):
        module, forbidden = case
        code = (
            f"import sys, importlib; importlib.import_module({module!r}); "
            f"print(sorted(m for m in {sorted(forbidden)!r} if m in sys.modules))"
        )
        return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)

    with ThreadPoolExecutor(len(cases)) as pool:  # the interpreters are independent: start them together
        outputs = list(pool.map(run, cases))
    for (module, _), out in zip(cases, outputs):
        assert out.returncode == 0, (module, out.stderr[-500:])
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


def test_no_participant_data_is_tracked():
    """Safety net for the README's 'Data and privacy' section: data/ is ignored and nothing under it is tracked."""
    assert "data/" in (ROOT / ".gitignore").read_text(encoding="utf8").split()
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is unavailable")
    assert [p for p in tracked if p.startswith("data/")] == []
