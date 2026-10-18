"""Records the computer webcam and the Pupil Labs Neon at the same time, each
sample stamped with this PC's wall clock, while a dot jumps left/right on a
PsychoPy window - so the Neon's gaze estimate can be compared against what the
webcam sees of the actual eye movement (see latency_analysis.py).

Usage: python -m src.eye_tracking.latency_test [camera_index] [num_jumps] [--no-scene-stream]

Comes to this PC live: the webcam video, every Neon gaze sample, and the Neon
scene video (half size, 30 fps) - each with per-frame/per-sample metadata.

Stays on the phone: the Neon EYE video. It cannot be streamed live - measured:
this library only manages ~90-97 of the eye camera's 200 frames/s even with
nothing else running, and the delay then grows without bound (0.4 s -> 7.5 s
in 15 s), dragging the gaze stream down with it. The phone records it at full
rate with exact per-frame timestamps (the test starts/stops that recording and
marks every dot jump as an event on its timeline); copy it to this PC
afterwards with the Companion app's export over USB, or via Pupil Cloud - no
download endpoint exists in the Real-Time API.

Output: data/latency/<timestamp>/
  webcam.mp4 + webcam.csv         computer camera video + one metadata row per frame
  neon_scene.mp4 + neon_scene.csv Neon scene video + per-frame metadata
  neon.csv                        every Neon gaze sample (the per-sample metadata)
  stimulus.csv, meta.json
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
from src.eye_tracking.neon_video_recorder import SceneRecorderProcess, retime_video

_retime_video = retime_video  # historical name

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
        self.device = self._device
        self.recording_id: str | None = None
        self.latest: tuple[float, float, bool, float] | None = None  # (x, y, worn, PC arrival time) of the newest sample
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start_phone_recording(self) -> str | None:
        """Starts a recording on the phone (scene + eye video, gaze, IMU at full rate). None if it couldn't start."""
        try:
            self.recording_id = self._device.recording_start()
        except Exception as exc:
            print(f"WARNING: could not start the Neon phone recording: {exc!r}", file=sys.stderr)
            self.recording_id = None
        return self.recording_id

    def send_event(self, name: str, pc_unix_s: float) -> None:
        """Marks `name` on the phone's recording timeline at PC time `pc_unix_s` (converted to the phone's clock).
        Sent from a helper thread so the network round trip can't delay the stimulus loop."""
        if self.recording_id is None:
            return
        device_ns = int((pc_unix_s - self.clock_offsets_ms[0] / 1000) * 1e9)

        def send() -> None:
            try:
                self._device.send_event(name, device_ns)
            except Exception as exc:
                print(f"WARNING: could not send event {name}: {exc!r}", file=sys.stderr)

        threading.Thread(target=send, daemon=True).start()

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
        if self.recording_id is not None:
            try:
                self._device.recording_stop_and_save()
            except Exception as exc:
                print(f"WARNING: could not stop/save the Neon phone recording: {exc!r}", file=sys.stderr)
        self._device.close()


class WebcamRecorder:
    """Saves every webcam frame to an .mp4 and its PC timestamp (taken as soon
    as read() returns, so it includes the driver's own buffering delay -
    unavoidable with a plain webcam) to a CSV, one row per video frame."""

    def __init__(self, camera_index: int, video_path: Path, csv_path: Path, overlay=None, neon=None) -> None:
        self._neon = neon  # optional NeonRecorder: its newest gaze sample is logged on every frame row
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
        self._csv.writerow(["frame_index", "unix_s", "neon_x", "neon_y", "neon_worn", "neon_sample_age_ms"])
        self._index = 0
        self._video_path = video_path
        self._first_t = self._last_t = None
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
            latest = self._neon.latest if self._neon is not None else None
            neon_cols = (
                [latest[0], latest[1], int(latest[2]), f"{(now - latest[3]) * 1000:.1f}"] if latest else ["", "", "", ""]
            )
            self._csv.writerow([self._index, f"{now:.6f}", *neon_cols])
            self._index += 1
            self._first_t = now if self._first_t is None else self._first_t
            self._last_t = now

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._writer.release()
        self._capture.release()
        self._file.close()
        if self._first_t is not None:
            retime_video(self._video_path, self._index, self._last_t - self._first_t)


def run(camera_index: int, num_jumps: int, stream_scene: bool = True) -> Path:
    folder = Path("data") / "latency" / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)

    neon = NeonRecorder(folder / "neon.csv")
    webcam = WebcamRecorder(
        camera_index, folder / "webcam.mp4", folder / "webcam.csv", overlay=lambda t: overlay_lines(neon, t), neon=neon
    )
    scene = None
    if stream_scene:
        # Own process + own device connection (see neon_video_recorder.py); connects now, before the window opens.
        scene = SceneRecorderProcess(
            NEON_DEVICE_ADDRESS,
            NEON_DEVICE_PORT,
            NEON_DISCOVERY_TIMEOUT_S,
            folder / "neon_scene.mp4",
            folder / "neon_scene.csv",
            neon.clock_offsets_ms[0],
        )
        if not scene.prepare():
            print("WARNING: Neon scene stream did not connect - continuing without scene video.", file=sys.stderr)
            scene = None
    win = visual.Window(fullscr=True, units="height", color=[-0.2] * 3, allowGUI=False)
    aspect = win.size[0] / win.size[1]
    dot = visual.Circle(win, radius=0.025, fillColor="white", lineColor="white")
    text = visual.TextStim(win, text="Look at the dot and follow it as it jumps.\nPress SPACE to start (ESC to quit).", height=0.04)

    clock_text = visual.TextStim(win, text="", height=0.035, color="white", pos=(0, 0.45), alignText="center")

    stimulus_file = (folder / "stimulus.csv").open("w", newline="")
    stimulus_csv = csv.writer(stimulus_file)
    stimulus_csv.writerow(["jump_index", "unix_s", "side"])

    jumps_shown = 0
    last_text_update = 0.0
    try:
        text.draw()
        win.flip()
        if "escape" in event.waitKeys(keyList=["space", "escape"]):
            return folder
        neon.start()
        neon.start_phone_recording()
        if scene is not None:
            scene.start()
        webcam.start()
        core.wait(1.0)  # let both streams settle

        side = random.choice([-1, 1])
        for i in range(num_jumps + 1):  # +1: the last jump only bounds the final trial's plateau window
            dot.pos = (side * DOT_X * aspect, 0)
            dot.draw()
            win.flip()
            shown_at = time.time()
            stimulus_csv.writerow([i, f"{shown_at:.6f}", "right" if side > 0 else "left"])
            neon.send_event(f"jump_{i}_{'right' if side > 0 else 'left'}", shown_at)
            jumps_shown = i + 1
            side = -side
            wait_until = core.getTime() + random.uniform(*JUMP_INTERVAL_S)
            while core.getTime() < wait_until:
                if event.getKeys(keyList=["escape"]):
                    return folder
                if time.monotonic() - last_text_update > 0.1:  # re-rendering text every frame is costly
                    clock_text.text = "\n".join(overlay_lines(neon, time.time()))
                    last_text_update = time.monotonic()
                clock_text.draw()
                dot.draw()
                win.flip()
        core.wait(0.5)
    finally:
        win.close()
        webcam.stop()
        if scene is not None:
            scene.stop()
        neon.stop()
        stimulus_file.close()
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "neon_clock_offset_ms": neon.clock_offsets_ms,
                    "neon_phone_recording_id": neon.recording_id,
                    "webcam_requested_fps": webcam.requested_fps,
                    "camera_index": camera_index,
                    "num_jumps_requested": num_jumps,
                    "jumps_shown": jumps_shown,
                    "frames": {"webcam": webcam._index, "neon_scene": scene.frames if scene else None},
                    "stream_errors": {"neon_scene": scene.error if scene else None},
                    "refresh_note": "stimulus timestamp is taken right after win.flip() returns",
                },
                indent=2,
            )
        )
    return folder


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    camera_index = int(args[0]) if len(args) > 0 else 0
    num_jumps = int(args[1]) if len(args) > 1 else 40
    out = run(camera_index, num_jumps, stream_scene="--no-scene-stream" not in sys.argv)
    print(f"Saved to {out}\nAnalyse with: python -m src.eye_tracking.latency_analysis {out}")
    recording_id = json.loads((out / "meta.json").read_text()).get("neon_phone_recording_id")
    if recording_id:
        print(
            f"Neon eye video (and full-rate scene/gaze/IMU) is in the phone recording {recording_id} - export it with the "
            "Companion app (USB) or Pupil Cloud; the jump events are in its timeline."
        )
