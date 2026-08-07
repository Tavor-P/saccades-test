"""Quick diagnostic for "face: no" during the saccade phase: captures one
frame from a camera, saves it so you can look at it, and runs the exact
same MediaPipe face detector the experiment uses against it - without
having to run the full PsychoPy experiment to hit the problem.

Usage (from the project folder):
  "C:\\Program Files\\PsychoPy\\python.exe" -m src.eye_tracking.diagnose_camera [camera_index]

If no camera_index is given, probes for available cameras and uses the
first one found.
"""

import sys
import time
from pathlib import Path

import cv2
import psychopy  # noqa: F401  (importing this first wires PySpin onto sys.path, same as run_experiment.py)

from src.eye_tracking.camera import Camera, list_available_cameras
from src.eye_tracking.gaze_tracker import GazeTracker

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "camera_diagnostic.png"


def main() -> None:
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        print(f"Using camera index {index} (from command line argument).")
    else:
        available = list_available_cameras()
        print(f"Available camera indices: {available}")
        if not available:
            print("No camera opened at all - check it's plugged in and not in use by another program.")
            return
        index = available[0]
        print(f"No index given, using {index}. Pass a specific index as an argument to test a different one.")

    print(f"Opening camera {index}...")
    camera = Camera(index)
    camera.start()

    frame = None
    for _ in range(50):  # up to ~5s waiting for a first frame
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.1)
    camera.stop()

    if frame is None:
        print("No frame was captured at all in 5s - the camera opened but isn't actually delivering images.")
        return

    print(f"Captured a frame: {frame.shape[1]}x{frame.shape[0]} pixels.")
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), frame)
    print(f"Saved it to {OUTPUT_PATH} - open that file and check it looks like a normal photo of your face")
    print("(not black, not washed-out/grayscale IR, not static/garbled colors).")

    print("\nRunning the experiment's own face detector on that exact frame...")
    tracker = GazeTracker()
    sample = tracker.process(frame)
    print(f"face_found: {sample.face_found} | ratio: {sample.ratio} | zone: {sample.zone.value}")

    if not sample.face_found:
        print(
            "\nMediaPipe didn't find a face in this frame. If the saved image DOES look like a normal, "
            "well-lit photo of your face, try moving closer/further from the camera or improving lighting. "
            "If the saved image looks wrong (black, grayscale/IR-washed, garbled, or not actually you), "
            "the camera itself is the problem, not the detector."
        )


if __name__ == "__main__":
    main()
