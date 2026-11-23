"""GazeSource: the one interface the experiment uses to get gaze. Implemented by NeonGazeSource and the test/simulation doubles (tests/conftest.FakeGazeSource, src/experiment/simulation.SimulatedGazeSource, src/eye_tracking/gaze_trace.ReplayGazeSource)."""

from abc import ABC, abstractmethod

from include.eye_tracking.types import GazeSample


class GazeSource(ABC):
    """Common interface the experiment loop uses to get gaze data, regardless of
    whether it comes from the Neon glasses or a test/simulation double. The
    experiment code only ever talks to this interface."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the underlying device (Neon glasses) is actually connected."""

    @abstractmethod
    def latest_sample(self) -> GazeSample: ...

    @abstractmethod
    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        """Set the left/center/right reference points used to classify gaze zone.

        The Neon source takes these as literal gaze ratios and fits a
        ratio->position mapping across all three (GazeMapper), rather than
        assuming center sits exactly at the left/right midpoint.
        """

    @abstractmethod
    def average_recent_ratio(self) -> float | None:
        """Recent gaze-ratio average, used while the participant holds a fixation
        during calibration. Returns None if no reading is currently available."""

    @abstractmethod
    def begin_calibration_sample(self) -> None:
        """Call this right before asking the participant to fixate a new
        calibration target, so average_recent_ratio() reflects only samples
        gathered after this point - not whatever they were looking at
        previously. A no-op for sources that don't use a rolling sample
        window."""
