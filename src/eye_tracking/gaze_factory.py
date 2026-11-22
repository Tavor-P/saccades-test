"""build_gaze_source(): the Pupil Labs Neon GazeSource, imported lazily so modules that only need the label (or a machine
without pupil-labs-realtime-api, e.g. running the simulator) never load the device library."""

from include.eye_tracking.interfaces import GazeSource

BACKEND_NEON = "Pupil Labs Neon (still head)"  # the gaze_source label written to session.json hardware


def build_gaze_source(trace_path=None) -> GazeSource:
    from src.eye_tracking.neon_source import NeonGazeSource

    return NeonGazeSource(trace_path=trace_path)
