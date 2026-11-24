"""build_gaze_source(): the Pupil Labs Neon GazeSource, imported lazily so modules that only need the label (or a machine
without pupil-labs-realtime-api, e.g. running the simulator) never load the device library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from include.eye_tracking.constants import NEON_DEVICE_PORT

if TYPE_CHECKING:  # type-checking only: the real import stays lazy (below)
    from src.eye_tracking.neon_source import NeonGazeSource

BACKEND_NEON = "Pupil Labs Neon (still head)"  # the gaze_source label written to session.json hardware


def build_gaze_source(trace_path=None, address: str | None = None, port: int | None = None) -> NeonGazeSource:
    """`address`/`port`: where run_experiment's start-up check found the phone (neon_address.resolve_neon_address); None = the
    NEON_DEVICE_ADDRESS/NEON_DEVICE_PORT constants."""
    from src.eye_tracking.neon_source import NeonGazeSource

    if address is None:
        return NeonGazeSource(trace_path=trace_path)
    return NeonGazeSource(trace_path=trace_path, address=address, port=port or NEON_DEVICE_PORT)
