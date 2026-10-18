from include.eye_tracking.interfaces import GazeSource

BACKEND_NEON = "Pupil Labs Neon"
BACKEND_WEBCAM = "Blackfly S + MediaPipe"
BACKENDS = [BACKEND_NEON, BACKEND_WEBCAM]


def build_gaze_source(backend: str, camera_index: int = 0) -> GazeSource:
    """Backend modules are imported lazily: the webcam path needs PySpin/
    MediaPipe, the Neon path needs pupil-labs-realtime-api, and a machine
    using one shouldn't need the other installed."""
    if backend == BACKEND_WEBCAM:
        from src.eye_tracking.webcam_source import WebcamGazeSource

        return WebcamGazeSource(camera_index)
    from src.eye_tracking.neon_source import NeonGazeSource

    return NeonGazeSource()


def list_cameras() -> list[int]:
    """Webcam-backend camera indices; imported lazily because camera.py
    imports PySpin at module load."""
    from src.eye_tracking.camera import list_available_cameras

    return list_available_cameras()
