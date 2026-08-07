import psychopy  # noqa: F401  (importing this first wires PySpin onto sys.path for src.eye_tracking.camera)
import pytest

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.experiment import pausable_clock as pausable_clock_module


class FakeGazeSource(GazeSource):
    """Minimal in-memory GazeSource double: tests set `.zone`/`.face_found`/
    `.position` directly instead of driving a real camera/tracker."""

    def __init__(self) -> None:
        self.zone = GazeZone.CENTER
        self.ratio: float | None = None
        self.position: float | None = 0.5  # 0=left target, 0.5=center, 1=right target
        self.face_found = True
        self._available = True
        self.calibrated: tuple[float, float, float] | None = None
        self.calibration_samples_begun = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return self._available

    def latest_sample(self) -> GazeSample:
        return GazeSample(
            zone=self.zone, ratio=self.ratio, face_found=self.face_found, timestamp=0.0, position=self.position
        )

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self.calibrated = (left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return 0.5

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
    monkeypatch.setattr(pausable_clock_module.time, "monotonic", lambda: state["t"])

    def advance(seconds: float) -> None:
        state["t"] += seconds

    return advance
