"""Corner-tag mapping with the screen-outline locator as a fallback.

Same interface as ScreenGazeMapper / ScreenOutlineMapper. Both locators see every scene frame; a
gaze sample is mapped by the tags when they currently locate the screen (more precise), else by the
outline (works when the tags are hidden, too small or washed out)."""
import threading

from src.eye_tracking.screen_gaze_mapper import ScreenGaze


class HybridScreenMapper:
    def __init__(self, tag_mapper, outline_mapper) -> None:
        self._tags = tag_mapper
        self._outline = outline_mapper
        self._lock = threading.Lock()
        self._tag_samples = 0
        self._outline_samples = 0

    def process_scene(self, frame) -> None:
        self._tags.process_scene(frame)
        self._outline.process_scene(frame)

    def map_gaze(self, gaze) -> ScreenGaze | None:
        mapped = self._tags.map_gaze(gaze) if self._tags.located else None
        if mapped is not None:
            with self._lock:
                self._tag_samples += 1
            return mapped
        mapped = self._outline.map_gaze(gaze)
        if mapped is not None:
            with self._lock:
                self._outline_samples += 1
        return mapped

    @property
    def markers_detected(self) -> int:
        return max(self._tags.markers_detected, self._outline.markers_detected)

    @property
    def located(self) -> bool:
        return self._tags.located or self._outline.located

    @property
    def sample_counts(self) -> dict[str, int]:
        """Samples mapped by each locator so far."""
        with self._lock:
            return {"tags": self._tag_samples, "outline": self._outline_samples}
