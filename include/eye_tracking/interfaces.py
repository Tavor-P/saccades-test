from abc import ABC, abstractmethod

from include.eye_tracking.types import GazeSample


class GazeSource(ABC):
    """Common interface the experiment loop uses to get gaze data, regardless of
    whether it comes from the MediaPipe webcam tracker (today) or a future ioHub
    hardware eye tracker such as Tobii/EyeLink/GazePoint (once real hardware is
    built). The experiment code only ever talks to this interface."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the underlying device (camera / eye tracker) is actually connected."""

    @abstractmethod
    def latest_sample(self) -> GazeSample: ...

    @abstractmethod
    def calibrate(self, left_ratio: float, right_ratio: float) -> None:
        """Set the left/right reference points used to classify gaze zone.

        The webcam source takes these as literal iris-position ratios. A future
        ioHub-backed source would more likely ignore the arguments and run its
        own vendor calibration routine (e.g. `tracker.runSetupProcedure()`)
        instead - this method exists on the interface as the "do a 2-point
        calibration now" trigger, not a strict shared data contract.
        """

    @abstractmethod
    def average_recent_ratio(self) -> float | None:
        """Recent gaze-ratio average, used while the participant holds a fixation
        during calibration. Returns None if no reading is currently available."""
