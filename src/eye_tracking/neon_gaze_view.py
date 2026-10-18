"""Live gaze dot on the screen from the Pupil Labs Neon - x and y, no calibration.

Four AprilTag markers are drawn in the screen corners; the Neon's scene camera
finds them, which tells the real-time screen-gaze mapper where the screen is,
so gaze is mapped to screen pixels directly. Keep all four markers visible to
the glasses (look at the screen, don't cover the corners).

Usage: python -m src.eye_tracking.neon_gaze_view      (ESC quits)

Optional: look at the centre cross and press SPACE to store a constant offset
(corrects the small fixed gaze bias Neon can have); press C to clear it.
"""
import psychopy  # noqa: F401

import threading
import time

import cv2
import numpy as np
from psychopy import event, visual
from pupil_labs.real_time_screen_gaze import marker_generator
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper
from pupil_labs.realtime_api.simple import Device, discover_one_device

from include.eye_tracking.constants import NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT, NEON_DISCOVERY_TIMEOUT_S

MARKER_PX = 240  # black-bordered AprilTag square, in screen pixels
MARGIN_PX = 50  # white quiet zone around each marker (needed for detection)
STALE_AFTER_S = 0.5
SMOOTHING = 0.35  # EMA weight of each new mapped reading; lower = steadier but laggier  # hide the dot if no mapped gaze arrived this recently


def _marker_image(marker_id: int) -> np.ndarray:
    """White-padded marker as a PsychoPy image array (-1..1, row 0 = bottom)."""
    tag = cv2.resize(marker_generator.generate_marker(marker_id), (MARKER_PX, MARKER_PX), interpolation=cv2.INTER_NEAREST)
    padded = cv2.copyMakeBorder(tag, MARGIN_PX, MARGIN_PX, MARGIN_PX, MARGIN_PX, cv2.BORDER_CONSTANT, value=255)
    return np.flipud(padded.astype(np.float32) / 127.5 - 1.0)


def _marker_corners(origin_x: float, origin_y: float) -> list[tuple[float, float]]:
    """Marker corners (top-left, clockwise) in screen px, top-left origin, for a pad whose corner is (origin_x, origin_y)."""
    x0, y0 = origin_x + MARGIN_PX, origin_y + MARGIN_PX
    return [(x0, y0), (x0 + MARKER_PX, y0), (x0 + MARKER_PX, y0 + MARKER_PX), (x0, y0 + MARKER_PX)]


def main() -> None:
    device = (
        Device(NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT)
        if NEON_DEVICE_ADDRESS
        else discover_one_device(max_search_duration_seconds=NEON_DISCOVERY_TIMEOUT_S)
    )
    if device is None:
        raise RuntimeError("No Neon found")

    win = visual.Window(fullscr=True, units="pix", color=[-0.3] * 3, allowGUI=False)
    width, height = win.size
    pad = MARKER_PX + 2 * MARGIN_PX

    # (id, pad top-left corner in screen px, top-left origin)
    placements = {
        0: (0, 0),
        1: (width - pad, 0),
        2: (width - pad, height - pad),
        3: (0, height - pad),
    }
    gaze_mapper = GazeMapper(device.get_calibration())
    surface = gaze_mapper.add_surface({i: _marker_corners(*p) for i, p in placements.items()}, (width, height))

    stims = [
        visual.ImageStim(win, image=_marker_image(i), size=(pad, pad), pos=(px + pad / 2 - width / 2, height / 2 - py - pad / 2), interpolate=False)
        for i, (px, py) in placements.items()
    ]
    dot = visual.Circle(win, radius=18, fillColor=None, lineColor="red", lineWidth=4)
    cross = visual.TextStim(win, text="+", height=40, color="white", pos=(0, 0))
    hud = visual.TextStim(win, text="", height=22, color="white", pos=(0, -height / 2 + 40))

    latest: dict = {"x": None, "y": None, "t": 0.0, "markers": 0, "error": ""}
    running = True

    def reader() -> None:
        while running:
            try:
                frame, gaze = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1)
            except Exception as exc:  # surface it on screen instead of silently looking like 'not detected'
                latest["error"] = repr(exc)
                continue
            if frame is None:
                continue
            result = gaze_mapper.process_frame(frame, gaze)
            latest["markers"] = len(result.markers) if result else 0
            mapped = result.mapped_gaze.get(surface.uid, []) if result else []
            if mapped:
                # The mapper returns 0-1 screen coordinates with y measured UP from the bottom edge.
                latest.update(x=mapped[0].x * width, y=(1 - mapped[0].y) * height, t=time.monotonic())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        smoothed: list[float] | None = None
        offset = (0.0, 0.0)
        last_t = 0.0
        prev_t = 0.0
        while True:
            keys = event.getKeys(keyList=["escape", "space", "c"])
            if "escape" in keys:
                break
            if "space" in keys and smoothed is not None:
                offset = (width / 2 - smoothed[0], height / 2 - smoothed[1])
            if "c" in keys:
                offset = (0.0, 0.0)
            cross.draw()
            for stim in stims:
                stim.draw()
            if latest["x"] is not None and time.monotonic() - latest["t"] < STALE_AFTER_S:
                if latest["t"] != last_t:  # new mapped reading: fold into the smoothed position
                    last_t = latest["t"]
                    if smoothed is None or time.monotonic() - prev_t > 1.0:
                        smoothed = [latest["x"], latest["y"]]
                    else:
                        smoothed[0] += (latest["x"] - smoothed[0]) * SMOOTHING
                        smoothed[1] += (latest["y"] - smoothed[1]) * SMOOTHING
                    prev_t = time.monotonic()
                gx, gy = smoothed[0] + offset[0], smoothed[1] + offset[1]  # screen px, top-left origin
                dot.pos = (gx - width / 2, height / 2 - gy)
                dot.draw()
                hud.text = f"gaze x={gx:.0f}  y={gy:.0f} px  (screen {width}x{height}, markers {latest['markers']}/4)"
            else:
                hud.text = f"No screen located - markers seen: {latest['markers']}/4 (need at least 3 clearly visible)  {latest['error']}"
            hud.draw()
            win.flip()
    finally:
        running = False
        thread.join(timeout=2)
        win.close()
        device.close()


if __name__ == "__main__":
    main()
