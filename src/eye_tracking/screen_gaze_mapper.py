"""Scene-camera gaze -> screen pixels, via the corner markers.

Wraps pupil_labs.real_time_screen_gaze.GazeMapper with: the shared marker
layout, a full-resolution tag detector (small markers), and a short "hold" so
a missed detection frame or two doesn't drop gaze. The scene thread calls
process_scene() (~30 Hz); the gaze thread calls map_gaze() (~200 Hz) with
whatever screen location is newest - head movement between the two is small.
"""
import threading
from dataclasses import dataclass

from include.eye_tracking.constants import NEON_MARKER_HOLD_S, NEON_TAG_QUAD_DECIMATE
from src.eye_tracking.screen_markers import marker_verts
from src import timebase


@dataclass(frozen=True)
class ScreenGaze:
    x: float  # screen px, top-left origin
    y: float
    mapping_age_s: float  # how old the screen location used for this sample is


class ScreenGazeMapper:
    def __init__(self, calibration, screen_size: tuple[int, int], hold_s: float = NEON_MARKER_HOLD_S) -> None:
        from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper

        self._gm = GazeMapper(calibration)
        self._width, self._height = screen_size
        self._surface = self._gm.add_surface(marker_verts(self._width, self._height), screen_size)
        self._hold_s = hold_s
        self._lock = threading.Lock()
        # The library's GazeMapper isn't thread-safe: the scene thread (process_scene
        # + the hold-restore below) and the gaze thread (process_gaze) share its
        # _surface_locations, and without this the gaze thread can catch the window
        # where a lost location was cleared but not yet restored, dropping a sample
        # the hold was meant to keep.
        self._gm_lock = threading.Lock()
        self._good_locations = None
        self._good_at = 0.0
        self._markers_detected = 0
        self._use_full_resolution_detector()

    def _use_full_resolution_detector(self) -> None:
        # The library hardcodes quad_decimate=2.0, which skips small tags; swap in
        # a detector at our setting. Reaches into its private detector on purpose -
        # guarded so a library change degrades to the default instead of crashing.
        try:
            import pupil_apriltags

            self._gm._detector._detector = pupil_apriltags.Detector(
                families="tag36h11", nthreads=2, quad_decimate=NEON_TAG_QUAD_DECIMATE, decode_sharpening=1.0
            )
        except Exception:
            pass

    def process_scene(self, frame) -> None:
        """Locate the screen in one scene frame (call from the scene thread)."""
        now = timebase.now()
        with self._gm_lock:
            self._gm.process_scene(frame)
            locations = getattr(self._gm, "_surface_locations", {})
            with self._lock:
                self._markers_detected = len(getattr(self._gm, "_detected_markers", []))
                if locations.get(self._surface.uid) is not None:
                    self._good_locations, self._good_at = dict(locations), now
                elif self._good_locations is not None and now - self._good_at <= self._hold_s:
                    self._gm._surface_locations = self._good_locations  # briefly keep using the last good location
                else:
                    self._good_locations = None

    def map_gaze(self, gaze) -> ScreenGaze | None:
        """Gaze (a Real-Time API GazeData) -> screen px, or None if the screen isn't currently located."""
        with self._lock:
            good_at = self._good_at if self._good_locations is not None else None
        if good_at is None:
            return None
        with self._gm_lock:
            result = self._gm.process_gaze(gaze)
        mapped = result.mapped_gaze.get(self._surface.uid) if result else None
        if not mapped:
            return None
        # The mapper returns 0-1 screen coordinates with y measured UP from the bottom edge.
        x, y = mapped[0].x * self._width, (1 - mapped[0].y) * self._height
        if not (-0.5 * self._width <= x <= 1.5 * self._width and -0.5 * self._height <= y <= 1.5 * self._height):
            return None  # a wild extrapolation from a bad marker fit
        return ScreenGaze(x, y, timebase.now() - good_at)

    @property
    def markers_detected(self) -> int:
        """Markers found in the most recent scene frame."""
        with self._lock:
            return self._markers_detected

    @property
    def located(self) -> bool:
        """True while a usable (fresh or briefly held) screen location exists."""
        with self._lock:
            return self._good_locations is not None
