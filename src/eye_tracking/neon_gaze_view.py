"""Live gaze dot on the screen from the Pupil Labs Neon - x and y, no calibration.

Four AprilTag markers are drawn in the screen corners; the Neon's scene camera
finds them, which tells the real-time screen-gaze mapper where the screen is,
so gaze is mapped to screen pixels directly. Keep all four markers visible to
the glasses (look at the screen, don't cover the corners).

Usage: python -m src.eye_tracking.neon_gaze_view      (ESC quits)
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

MARKER_PX = 160  # black-bordered AprilTag square, in screen pixels
MARGIN_PX = 40  # white quiet zone around each marker (needed for detection)
STALE_AFTER_S = 0.3  # hide the dot if no mapped gaze arrived this recently


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
    hud = visual.TextStim(win, text="", height=22, color="white", pos=(0, -height / 2 + 40))

    latest: dict = {"x": None, "y": None, "t": 0.0}
    running = True

    def reader() -> None:
        while running:
            try:
                frame, gaze = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1)
            except Exception:
                continue
            if frame is None:
                continue
            result = gaze_mapper.process_frame(frame, gaze)
            mapped = result.mapped_gaze.get(surface.uid, [])
            if mapped:
                latest.update(x=mapped[0].x, y=mapped[0].y, t=time.monotonic())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while not event.getKeys(keyList=["escape"]):
            for stim in stims:
                stim.draw()
            if latest["x"] is not None and time.monotonic() - latest["t"] < STALE_AFTER_S:
                gx, gy = latest["x"], latest["y"]  # screen px, top-left origin
                dot.pos = (gx - width / 2, height / 2 - gy)
                dot.draw()
                hud.text = f"gaze x={gx:.0f}  y={gy:.0f} px  (screen {width}x{height})"
            else:
                hud.text = "No screen detected - make sure all four corner markers are visible to the glasses"
            hud.draw()
            win.flip()
    finally:
        running = False
        thread.join(timeout=2)
        win.close()
        device.close()


if __name__ == "__main__":
    main()
