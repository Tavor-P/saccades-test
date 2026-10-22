from src.eye_tracking.neon_screen_source import NeonScreenGazeSource
from src.eye_tracking.neon_source import _connect


class NeonOutlineGazeSource(NeonScreenGazeSource):
    """NeonScreenGazeSource whose screen locator finds the monitor outline in the
    scene video (ScreenOutlineMapper) instead of reading corner AprilTags, so
    nothing extra is drawn on the display. Everything else (threads, IMU, quality
    stats, ratio/zone handling) is inherited. `screen_mapper` is injectable."""

    def start(self) -> None:
        if self._screen_mapper is None:
            if self._device is None:
                self._device = _connect()
            if self._device is not None:  # else super().start() raises the usual "no Neon found" error
                from src.eye_tracking.screen_outline_mapper import ScreenOutlineMapper

                self._screen_mapper = ScreenOutlineMapper(self._device.get_calibration(), self._screen_size)
        super().start()

    def quality_summary(self) -> dict:
        summary = super().quality_summary()
        summary["source"] = "pupil_labs_neon_screen_outline"
        summary.pop("marker_px", None)
        return summary
