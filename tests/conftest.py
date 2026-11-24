import psychopy  # noqa: F401  (imported first, as run_experiment does, so PsychoPy's user-packages dir is on sys.path)
import pytest

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src import timebase


class FakeGazeSource(GazeSource):
    """Minimal in-memory GazeSource double: tests set `.zone`/`.face_found`/
    `.position`/`.smoothed_position` directly instead of driving a real
    camera/tracker."""

    def __init__(self) -> None:
        self.zone = GazeZone.CENTER
        self.ratio: float | None = None
        self._position: float | None = None  # explicit override; None = follow .zone
        self.smoothed_position: float | None = 0.5  # smoothed version - see GazeSample
        self.face_found = True
        self._available = True
        self.calibrated: tuple[float, float, float] | None = None
        self.calibration_samples_begun = 0
        self.calibration_window_full = True  # tests set False to model a still-filling window

    @property
    def position(self) -> float | None:
        """0=left target, 0.5=center, 1=right target. Follows `.zone` (LEFT 0, CENTER 0.5, RIGHT 1) unless a test sets it
        explicitly - saccade onset is a displacement of the position, so tests that only flip the zone must move it too."""
        if self._position is not None:
            return self._position
        return {GazeZone.LEFT: 0.0, GazeZone.CENTER: 0.5, GazeZone.RIGHT: 1.0}.get(self.zone, 0.5)

    @position.setter
    def position(self, value: float | None) -> None:
        self._position = value

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return self._available

    def latest_sample(self) -> GazeSample:
        return GazeSample(
            zone=self.zone,
            ratio=self.ratio,
            face_found=self.face_found,
            timestamp=0.0,
            position=self.position,
            smoothed_position=self.smoothed_position,
        )

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self.calibrated = (left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return 0.5 if self.calibration_window_full else None

    def begin_calibration_sample(self) -> None:
        self.calibration_samples_begun += 1


@pytest.fixture
def fake_gaze() -> FakeGazeSource:
    return FakeGazeSource()


@pytest.fixture
def fake_time(monkeypatch):
    """Monkeypatches PausableClock's time source to a controllable counter, so
    tests can jump the clock forward deterministically instead of racing real
    wall-clock time through stability/response/timeout windows."""
    state = {"t": 0.0}
    monkeypatch.setattr(timebase, "_clock", lambda: state["t"])

    def advance(seconds: float) -> None:
        state["t"] += seconds

    return advance


# One area per test module, so a session can run only its own area: `pytest -m analysis`. A module missing here fails collection.
AREAS = {
    **dict.fromkeys(
        ("test_session", "test_tutorial_session", "test_calibration", "test_trial_factory", "test_onset_detector",
         "test_timing_core", "test_space_input", "test_scoring", "test_simulation", "test_return_jump", "test_invariants",
         "test_bug_injection", "test_sim_rt_variability", "test_constant_stimuli", "test_line_stimulus",
         "test_tracking_quality", "test_fast_paths", "test_ai_tools", "test_participant_text"), "session"),
    **dict.fromkeys(
        ("test_analyze_session", "test_analyze_participant", "test_compare", "test_flash_gates", "test_return_analysis",
         "test_glasses_video", "test_participant_labels", "test_timecourse"), "analysis"),
    **dict.fromkeys(
        ("test_dashboard_app", "test_dashboard_compare", "test_dashboard_data_access", "test_dashboard_results"), "dashboard"),
    **dict.fromkeys(("test_neon_source", "test_gaze_trace", "test_neon_address", "test_head_monitor"), "gaze"),
    **dict.fromkeys(
        ("test_run_experiment", "test_run_summary", "test_startup_dialog", "test_narrator", "test_logger", "test_file_io",
         "test_session_layout"), "run"),
    **dict.fromkeys(
        ("test_docs", "test_lint", "test_schema_sync", "test_architecture_rules", "test_constants_sanity"), "checks"),
}


def pytest_collection_modifyitems(items):
    for item in items:
        area = AREAS.get(item.path.stem)
        if area is None:
            raise pytest.UsageError(f"add {item.path.name} to AREAS in tests/conftest.py")
        item.add_marker(getattr(pytest.mark, area))
