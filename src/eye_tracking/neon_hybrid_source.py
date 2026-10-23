from src.eye_tracking.neon_screen_source import NeonScreenGazeSource
from src.eye_tracking.neon_source import _connect


class NeonHybridGazeSource(NeonScreenGazeSource):
    """NeonScreenGazeSource that locates the screen with the corner tags *and* the monitor outline
    (HybridScreenMapper): tags when visible, outline otherwise. `screen_mapper` is injectable."""

    def start(self) -> None:
        if self._screen_mapper is None:
            if self._device is None:
                self._device = _connect()
            if self._device is not None:  # else super().start() raises the usual "no Neon found" error
                from src.eye_tracking.hybrid_screen_mapper import HybridScreenMapper
                from src.eye_tracking.screen_gaze_mapper import ScreenGazeMapper
                from src.eye_tracking.screen_outline_mapper import ScreenOutlineMapper

                calibration = self._device.get_calibration()
                self._screen_mapper = HybridScreenMapper(
                    ScreenGazeMapper(calibration, self._screen_size), ScreenOutlineMapper(calibration, self._screen_size)
                )
        super().start()

    def quality_summary(self) -> dict:
        summary = super().quality_summary()
        summary["source"] = "pupil_labs_neon_screen_hybrid"
        counts = getattr(self._screen_mapper, "sample_counts", None)
        if counts:
            summary["samples_mapped_by_tags"] = counts["tags"]
            summary["samples_mapped_by_outline"] = counts["outline"]
        return summary
