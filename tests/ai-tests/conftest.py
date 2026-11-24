"""Puts this folder (sim_tools, invariants, find_bugs) and tests/ on sys.path, so the tests here can
import their helpers - and each other's regression tests - by plain module name, whichever test file
pytest happens to import first. (The project root comes from pytest.ini's `pythonpath = .`.)"""

import struct
import sys
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def no_flash_jitter(monkeypatch):
    """No random bounce on the control-flash offsets, for tests that need exact flash times."""
    monkeypatch.setattr("src.experiment.session.TIMING_JITTER_MS", 0)


# -- fast PNGs -------------------------------------------------------------------------------------------------------------
# Drawing and encoding an analysis plot costs ~0.4 s and most tests only need "a PNG was written" (they assert its name, its
# magic bytes or a download). Those get a small valid PNG; the tests below keep real rendering, so each plot function is still
# drawn (axes, log scales, NaN handling) at least once.
REAL_PLOT_TESTS = (
    "test_full_analysis_reproduces_a_planted_suppression_dip",  # run plot, size check
    "test_plot_report_handles_all_correct_all_wrong_and_single_flash_bins",  # the odd bin shapes
    "test_timecourse_figure_draws_run_pooled_and_sparse_data",  # the time-course figure: planted dip, pooled note, empty / sparse data
    "test_outputs_are_written_in_the_participant_analysis_folder",  # participant plot, size check
    "test_end_to_end_one_vs_one_analyses_once_reuses_then_refreshes_when_a_run_changes",  # comparison plot
)


def _tiny_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return b"".join([bytes.fromhex("89504e470d0a1a0a"), chunk(b"IHDR", header), chunk(b"IDAT", zlib.compress(bytes([0, 128]))), chunk(b"IEND", b"")])


_TINY_PNG = _tiny_png()


@pytest.fixture(autouse=True)
def _fast_png_plots(request, monkeypatch):
    if request.node.name.split("[")[0] in REAL_PLOT_TESTS:
        return

    def savefig(self, fname, *args, **kwargs):
        Path(fname).write_bytes(_TINY_PNG)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)


# -- fewer bootstrap refits ------------------------------------------------------------------------------------------------
# suppression_index() refits 1000 bootstrap resamples per call, and the analysis tests call it a hundred times. The CI logic is
# the same with fewer (it needs >= BOOT_MIN_IDENTIFIED = 200 identified refits, so 300 is the practical floor). The time-course band
# and the companion numbers (their n_boot defaults are 1000 too) go through the same stub.
TEST_BOOTSTRAP_RESAMPLES = 300


@pytest.fixture(scope="session", autouse=True)
def _fewer_bootstrap_resamples():
    import src.experiment.relative_suppression as companion
    import src.experiment.suppression_stats as stats
    import src.experiment.timecourse as timecourse

    functions = (stats.suppression_index, companion.companion_numbers, timecourse.timecourse_curve)
    originals = [f.__defaults__ for f in functions]
    for f, original in zip(functions, originals):
        assert original is not None and 1000 in original
        f.__defaults__ = tuple(TEST_BOOTSTRAP_RESAMPLES if v == 1000 else v for v in original)
    yield
    for f, original in zip(functions, originals):
        f.__defaults__ = original


# -- parallel runs (pytest-xdist, `--dist loadgroup` in pytest.ini) ----------------------------------------------------------
# A module-scoped fixture is built once per worker that runs one of its tests, and importing run_experiment (PsychoPy, Qt) takes
# seconds. Tests that share such a setup are kept on one worker, so it is paid once.
EXPENSIVE_MODULE_FIXTURES = ("template", "pooled", "study")  # synthetic study folders built once per test module
PSYCHOPY_TESTS = {  # tests that import src.experiment.run_experiment (inside the test: collecting it in every worker costs more)
    "test_run_experiment.py": None,  # every test in it
    "test_participant_text.py": "test_dots_are_named_by_position_and_the_sync_step_line_says_what_is_coming",
    "test_line_stimulus.py": "test_answer_keys_scoring_and_instructions_are_up_down_only",
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.parent != _HERE:
            continue
        if item.path.name in PSYCHOPY_TESTS and PSYCHOPY_TESTS[item.path.name] in (None, item.name):
            item.add_marker(pytest.mark.xdist_group("psychopy"))
            continue
        for fixture in EXPENSIVE_MODULE_FIXTURES:
            if fixture in item.fixturenames:
                item.add_marker(pytest.mark.xdist_group(f"{item.path.stem}-{fixture}"))
                break
