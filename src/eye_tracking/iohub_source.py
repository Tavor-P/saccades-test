import time

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone


class IOHubGazeSource(GazeSource):
    """GazeSource backed by PsychoPy's ioHub common eye tracker interface, for
    real hardware (Tobii/EyeLink/GazePoint/...) instead of the webcam tracker.

    NOT YET VALIDATED: written against the documented ioHub eye tracker API,
    but there is no hardware to test it against yet. Expect to adjust this
    once a real tracker is wired up - in particular:
      - `device_config` must name the actual ioHub tracker class to load, e.g.
        "eyetracker.hw.tobii.EyeTracker" or "eyetracker.hw.sr_research.eyelink.EyeTracker".
      - Unlike the webcam source, calibration is normally a full built-in
        routine the vendor SDK drives (`runSetupProcedure()`), not a simple
        2-point average - `calibrate()` here just triggers that routine and
        ignores its arguments.
      - `target_positions` must be in the same unit system as the ioHub/vendor
        gaze-position output (usually matches the PsychoPy Window's `units`).
    """

    def __init__(self, device_config: dict, target_positions: dict[GazeZone, tuple[float, float]], hit_radius: float) -> None:
        self._device_config = device_config
        self._target_positions = target_positions
        self._hit_radius = hit_radius
        self._io = None
        self._tracker = None

    @property
    def is_available(self) -> bool:
        return self._tracker is not None

    def start(self) -> None:
        from psychopy.iohub.client import launchHubServer

        self._io = launchHubServer(**self._device_config)
        self._tracker = self._io.devices.tracker
        self._tracker.setRecordingState(True)

    def stop(self) -> None:
        if self._tracker is not None:
            self._tracker.setRecordingState(False)
        if self._io is not None:
            self._io.quit()
        self._tracker = None
        self._io = None

    def latest_sample(self) -> GazeSample:
        timestamp = time.monotonic()
        if self._tracker is None:
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=timestamp)

        position = self._tracker.getPosition()
        if position is None:
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=timestamp)

        zone = self._classify(position)
        return GazeSample(zone=zone, ratio=None, face_found=True, timestamp=timestamp)

    def calibrate(self, left_ratio: float, right_ratio: float) -> None:
        if self._tracker is not None:
            self._tracker.runSetupProcedure()

    def average_recent_ratio(self) -> float | None:
        return None  # ioHub reports real gaze position, not our webcam ratio concept

    def _classify(self, position: tuple[float, float]) -> GazeZone:
        best_zone = GazeZone.CENTER
        best_distance = self._hit_radius
        for zone, target in self._target_positions.items():
            distance = ((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_zone = zone
        return best_zone
