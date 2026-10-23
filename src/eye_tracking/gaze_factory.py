"""build_gaze_source(backend, ...): picks the GazeSource for the startup dialog's choice, importing the backend lazily so a machine without PySpin/
mediapipe (or without pupil-labs-realtime-api) can still run the other backends."""

from include.eye_tracking.interfaces import GazeSource

BACKEND_NEON_SCREEN = "Pupil Labs Neon + screen markers"
BACKEND_NEON_OUTLINE = "Pupil Labs Neon + screen outline (no tags)"
BACKEND_NEON_HYBRID = "Pupil Labs Neon + screen markers and outline"
BACKEND_NEON = "Pupil Labs Neon (no head compensation)"
BACKEND_WEBCAM = "Blackfly S + MediaPipe"
# Startup dialog choices (first = default). The tags-only backend is still buildable by name for scripts/tests,
# but the hybrid covers it, so it is not a separate dialog choice.
BACKENDS = [BACKEND_NEON_HYBRID, BACKEND_NEON_OUTLINE, BACKEND_NEON, BACKEND_WEBCAM]
NEON_BACKENDS = (BACKEND_NEON_SCREEN, BACKEND_NEON_OUTLINE, BACKEND_NEON_HYBRID, BACKEND_NEON)


def build_gaze_source(
    backend: str, camera_index: int = 0, trace_path=None, screen_size: tuple[int, int] | None = None
) -> GazeSource:
    """Backend modules are imported lazily: the webcam path needs PySpin/
    MediaPipe, the Neon path needs pupil-labs-realtime-api, and a machine
    using one shouldn't need the other installed."""
    if backend == BACKEND_WEBCAM:
        from src.eye_tracking.webcam_source import WebcamGazeSource

        return WebcamGazeSource(camera_index)
    if backend == BACKEND_NEON_SCREEN:
        from src.eye_tracking.neon_screen_source import NeonScreenGazeSource

        if screen_size is None:
            raise ValueError("the screen-marker Neon backend needs the window size in pixels")
        return NeonScreenGazeSource(screen_size, trace_path=trace_path)
    if backend == BACKEND_NEON_OUTLINE:
        from src.eye_tracking.neon_outline_source import NeonOutlineGazeSource

        if screen_size is None:
            raise ValueError("the screen-outline Neon backend needs the window size in pixels")
        return NeonOutlineGazeSource(screen_size, trace_path=trace_path)
    if backend == BACKEND_NEON_HYBRID:
        from src.eye_tracking.neon_hybrid_source import NeonHybridGazeSource

        if screen_size is None:
            raise ValueError("the hybrid Neon backend needs the window size in pixels")
        return NeonHybridGazeSource(screen_size, trace_path=trace_path)
    from src.eye_tracking.neon_source import NeonGazeSource

    return NeonGazeSource(trace_path=trace_path)


def list_cameras() -> list[int]:
    """Webcam-backend camera indices; imported lazily because camera.py
    imports PySpin at module load."""
    from src.eye_tracking.camera import list_available_cameras

    return list_available_cameras()
