"""Record a raw Neon gaze trace to a CSV for later replay (see gaze_trace.py).

    python -m src.eye_tracking.record_gaze_trace out.csv [seconds]

Look around / make saccades while it records - a mix of fixation and left/right
saccades makes the most useful test trace.
"""

import sys
import time
from pathlib import Path

from src.eye_tracking.neon_source import NeonGazeSource


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.eye_tracking.record_gaze_trace out.csv [seconds]")
    path = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    source = NeonGazeSource(trace_path=path)
    source.start()
    print(f"Recording {seconds:.0f}s to {path} ...", flush=True)
    time.sleep(seconds)
    source.stop()
    print(f"Done: {source.quality_summary()['samples']} samples.")


if __name__ == "__main__":
    main()
