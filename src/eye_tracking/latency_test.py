"""Records the computer webcam and the Pupil Labs Neon at the same time, each
sample stamped with this PC's wall clock, while a dot jumps left/right on a
PsychoPy window - so the Neon's gaze estimate can be compared against what the
webcam sees of the actual eye movement (see latency_analysis.py).

Usage: python -m src.eye_tracking.latency_test [camera_index] [num_jumps]

Output: data/latency/<timestamp>/{neon.csv, webcam.csv, webcam.mp4, stimulus.csv, meta.json}
"""
import psychopy  # noqa: F401  (must precede other imports that rely on PsychoPy's user-packages path)

import csv
import json
import math
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from psychopy import core, event, visual

from include.eye_tracking.constants import NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT, NEON_DISCOVERY_TIMEOUT_S

JUMP_INTERVAL_S = (1.2, 2.0)
DOT_X = 0.35  # fraction of window width from center


def overlay_lines(neon: "NeonRecorder", unix_time: float) -> list[str]:
    """Wall-clock time plus the Neon's newest gaze estimate (scene-camera pixels) - shown on screen and burned into the video."""
    lines = [datetime.fromtimestamp(unix_time).strftime("%H:%M:%S.%f")[:-3]]
    latest = neon.latest
    if latest is None or not latest[2] or math.isnan(latest[0]):
        lines.append("Neon gaze: none")
    else:
        lines.append(f"Neon gaze: x={latest[0]:.0f} y={latest[1]:.0f}  ({(unix_time - latest[3]) * 1000:.0f} ms old)")
    return lines


def _burn_in(frame, lines: list[str]) -> None:
    """Draws text lines into the bottom-left of a BGR frame, in place (outlined so it reads on any background)."""
    height = frame.shape[0]
    for i, line in enumerate(lines):
        position = (8, height - 10 - 24 * (len(lines) - 1 - i))
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


class NeonRecorder:
    """Logs every Neon gaze sample: PC arrival time, Neon device timestamp, x/y, worn."""

    def __init__(self, path: Path) -> None:
        from pupil_labs.realtime_api.simple import Device, discover_one_device

        self._device = (
            Device(NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT)
            if NEON_DEVICE_ADDRESS
            else discover_one_device(max_search_duration_seconds=NEON_DISCOVERY_TIMEOUT_S)
        )
        if self._device is None:
            raise RuntimeError("No Neon found")
        self.clock_offsets_ms: list[float] = [self._estimate_offset_ms()]
        # The gaze stream only connects on the first receive call, which can take
        # several seconds - do it now so the first jumps aren't missing Neon data.
        self._device.receive_gaze_datum(timeout_seconds=10)
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["arrival_unix_s", "device_unix_s", "x", "y", "worn"])
        self.latest: tuple[float, float, bool, float] | None = None  # (x, y, worn, PC arrival time) of the newest sample
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _estimate_offset_ms(self) -> float:
        return self._device.estimate_time_offset().time_offset_ms.mean

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            datum = self._device.receive_gaze_datum(timeout_seconds=0.5)
            arrival = time.time()
            if datum is None:
                continue
            self._writer.writerow([f"{arrival:.6f}", f"{datum.timestamp_unix_seconds:.6f}", datum.x, datum.y, int(datum.worn)])
            self.latest = (datum.x, datum.y, bool(datum.worn), arrival)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self.clock_offsets_ms.append(self._estimate_offset_ms())  # second estimate, to expose drift
        self._file.close()
        self._device.close()


class WebcamRecorder:
    """Saves every webcam frame to an .mp4 and its PC timestamp (taken as soon
    as read() returns, so it includes the driver's own buffering delay -
    unavoidable with a plain webcam) to a CSV, one row per video frame."""

    def __init__(self, camera_index: int, video_path: Path, csv_path: Path, overlay=None) -> None:
        self._overlay = overlay  # optional fn(unix_time) -> list of text lines burned into each saved frame
        self._capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        self._capture.set(cv2.CAP_PROP_FPS, 60)
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Camera opened but returned no frame")
        height, width = frame.shape[:2]
        self.requested_fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.requested_fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        self._file = csv_path.open("w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["frame_index", "unix_s"])
        self._index = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            now = time.time()
            if not ok:
                continue
            if self._overlay is not None:
                _burn_in(frame, self._overlay(now))
            self._writer.write(frame)
            self._csv.writerow([self._index, f"{now:.6f}"])
            self._index += 1

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._writer.release()
        self._capture.release()
        self._file.close()


def run(camera_index: int, num_jumps: int) -> Path:
    folder = Path("data") / "latency" / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)

    neon = NeonRecorder(folder / "neon.csv")
    webcam = WebcamRecorder(
        camera_index, folder / "webcam.mp4", folder / "webcam.csv", overlay=lambda t: overlay_lines(neon, t)
    )
    win = visual.Window(fullscr=True, units="height", color=[-0.2] * 3, allowGUI=False)
    aspect = win.size[0] / win.size[1]
    dot = visual.Circle(win, radius=0.025, fillColor="white", lineColor="white")
    text = visual.TextStim(win, text="Look at the dot and follow it as it jumps.\nPress SPACE to start (ESC to quit).", height=0.04)

    clock_text = visual.TextStim(win, text="", height=0.035, color="white", pos=(0, 0.45), alignText="center")

    stimulus_file = (folder / "stimulus.csv").open("w", newline="")
    stimulus_csv = csv.writer(stimulus_file)
    stimulus_csv.writerow(["jump_index", "unix_s", "side"])

    try:
        text.draw()
        win.flip()
        if "escape" in event.waitKeys(keyList=["space", "escape"]):
            return folder
        neon.start()
        webcam.start()
        core.wait(1.0)  # let both streams settle

        side = random.choice([-1, 1])
        for i in range(num_jumps + 1):  # +1: the last jump only bounds the final trial's plateau window
            dot.pos = (side * DOT_X * aspect, 0)
            dot.draw()
            win.flip()
            stimulus_csv.writerow([i, f"{time.time():.6f}", "right" if side > 0 else "left"])
            side = -side
            wait_until = core.getTime() + random.uniform(*JUMP_INTERVAL_S)
            while core.getTime() < wait_until:
                if event.getKeys(keyList=["escape"]):
                    return folder
                clock_text.text = "\n".join(overlay_lines(neon, time.time()))
                clock_text.draw()
                dot.draw()
                win.flip()
        core.wait(0.5)
    finally:
        win.close()
        webcam.stop()
        neon.stop()
        stimulus_file.close()
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "neon_clock_offset_ms": neon.clock_offsets_ms,
                    "webcam_requested_fps": webcam.requested_fps,
                    "camera_index": camera_index,
                    "num_jumps": num_jumps,
                    "refresh_note": "stimulus timestamp is taken right after win.flip() returns",
                },
                indent=2,
            )
        )
    return folder


if __name__ == "__main__":
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_jumps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = run(camera_index, num_jumps)
    print(f"Saved to {out}\nAnalyse with: python -m src.eye_tracking.latency_analysis {out}")
