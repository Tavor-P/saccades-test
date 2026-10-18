"""Live gaze dot on the screen from the Pupil Labs Neon - x and y, after a 9-point calibration.

Four AprilTag markers are drawn in the screen corners; the Neon's scene camera
finds them, which tells the real-time screen-gaze mapper where the screen is,
so gaze is mapped to screen pixels. Once all four markers are in view, press
SPACE to run a 9-point calibration (look at each dot as it appears); the
result corrects the glasses' small fixed gaze bias. Keep all four corner
markers visible to the glasses throughout.

Usage: python -m src.eye_tracking.neon_gaze_view      (SPACE start/recalibrate-confirm, R recalibrate, ESC quits)
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
STALE_AFTER_S = 0.5  # hide the dot if no mapped gaze arrived this recently
SMOOTHING = 0.35  # EMA weight of each new mapped reading; lower = steadier but laggier

# 9 targets as fractions of the screen; kept clear of the corner marker pads.
CAL_X_FRACTIONS = (0.22, 0.5, 0.78)
CAL_Y_FRACTIONS = (0.25, 0.5, 0.75)
CAL_SETTLE_S = 0.8  # time for the eyes to land before samples count
CAL_SAMPLE_S = 0.8  # then collect this long
CAL_MIN_SAMPLES = 8  # fewer mapped readings than this -> show the target again


def _marker_image(marker_id: int) -> np.ndarray:
    """White-padded marker as a PsychoPy image array (-1..1, row 0 = bottom)."""
    tag = cv2.resize(marker_generator.generate_marker(marker_id), (MARKER_PX, MARKER_PX), interpolation=cv2.INTER_NEAREST)
    padded = cv2.copyMakeBorder(tag, MARGIN_PX, MARGIN_PX, MARGIN_PX, MARGIN_PX, cv2.BORDER_CONSTANT, value=255)
    return np.flipud(padded.astype(np.float32) / 127.5 - 1.0)


def _marker_corners(origin_x: float, origin_y: float) -> list[tuple[float, float]]:
    """Marker corners (top-left, clockwise) in screen px, top-left origin, for a pad whose corner is (origin_x, origin_y)."""
    x0, y0 = origin_x + MARGIN_PX, origin_y + MARGIN_PX
    return [(x0, y0), (x0 + MARKER_PX, y0), (x0 + MARKER_PX, y0 + MARKER_PX), (x0, y0 + MARKER_PX)]


def fit_affine(raw_points, target_points) -> np.ndarray:
    """Least-squares 3x2 matrix M such that [x, y, 1] @ M ~= target. Affine
    (not higher-order) on purpose: it can't bend wildly outside the 9 points."""
    raw = np.asarray(raw_points, float)
    design = np.hstack([raw, np.ones((len(raw), 1))])
    matrix, *_ = np.linalg.lstsq(design, np.asarray(target_points, float), rcond=None)
    return matrix


def apply_affine(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    out = np.array([x, y, 1.0]) @ matrix
    return float(out[0]), float(out[1])


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
        visual.ImageStim(
            win,
            image=_marker_image(i),
            size=(pad, pad),
            pos=(px + pad / 2 - width / 2, height / 2 - py - pad / 2),
            interpolate=False,
        )
        for i, (px, py) in placements.items()
    ]
    dot = visual.Circle(win, radius=18, fillColor=None, lineColor="red", lineWidth=4)
    target = visual.Circle(win, radius=14, fillColor="yellow", lineColor="black", lineWidth=2)
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

    def fresh() -> bool:
        return latest["x"] is not None and time.monotonic() - latest["t"] < STALE_AFTER_S

    def draw_markers() -> None:
        for stim in stims:
            stim.draw()

    def show(text: str) -> None:
        draw_markers()
        hud.text = text
        hud.draw()

    def run_calibration() -> np.ndarray | None:
        """Show each of the 9 targets, collect mapped gaze, fit. None if aborted (ESC)."""
        raw_points, target_points = [], []
        targets = [(fx * width, fy * height) for fy in CAL_Y_FRACTIONS for fx in CAL_X_FRACTIONS]
        index = 0
        while index < len(targets):
            tx, ty = targets[index]
            target.pos = (tx - width / 2, height / 2 - ty)
            start, samples, last_seen = time.monotonic(), [], 0.0
            while (elapsed := time.monotonic() - start) < CAL_SETTLE_S + CAL_SAMPLE_S:
                if event.getKeys(keyList=["escape"]):
                    return None
                if elapsed >= CAL_SETTLE_S and latest["t"] != last_seen and fresh():
                    last_seen = latest["t"]
                    samples.append((latest["x"], latest["y"]))
                show(f"Calibration {index + 1}/9 - look at the yellow dot")
                target.draw()
                win.flip()
            if len(samples) >= CAL_MIN_SAMPLES:
                raw_points.append(tuple(np.median(samples, axis=0)))
                target_points.append((tx, ty))
                index += 1
            # else: not enough mapped readings (markers lost?) - same target again
        matrix = fit_affine(raw_points, target_points)
        errors = [
            np.hypot(*(np.array(apply_affine(matrix, *r)) - np.array(t))) for r, t in zip(raw_points, target_points)
        ]
        before = [np.hypot(r[0] - t[0], r[1] - t[1]) for r, t in zip(raw_points, target_points)]
        print(f"Calibration residual: mean {np.mean(errors):.0f} px, max {np.max(errors):.0f} px (uncalibrated mean {np.mean(before):.0f} px)")
        end = time.monotonic() + 2.5
        while time.monotonic() < end:
            show(f"Calibrated: mean error {np.mean(errors):.0f} px (was {np.mean(before):.0f} px before)")
            win.flip()
        return matrix

    try:
        matrix: np.ndarray | None = None
        smoothed: list[float] | None = None
        last_t, prev_t = 0.0, 0.0
        markers_ok_since: float | None = None
        while True:
            keys = event.getKeys(keyList=["escape", "space", "r"])
            if "escape" in keys:
                break
            if "r" in keys:
                matrix, smoothed = None, None

            seen4 = latest["markers"] >= 4 and fresh()
            markers_ok_since = (markers_ok_since or time.monotonic()) if seen4 else None
            ready = markers_ok_since is not None and time.monotonic() - markers_ok_since > 0.5

            if matrix is None:
                if "space" in keys and ready:
                    matrix = run_calibration()
                    if matrix is None:
                        break
                    continue
                show(
                    f"Markers seen: {latest['markers']}/4 - get all 4 corner markers in view"
                    + (", then press SPACE to start the 9-point calibration" if ready else "")
                    + (f"   {latest['error']}" if latest["error"] else "")
                )
                win.flip()
                continue

            draw_markers()
            if fresh():
                if latest["t"] != last_t:  # new mapped reading: fold into the smoothed, calibrated position
                    last_t = latest["t"]
                    cx, cy = apply_affine(matrix, latest["x"], latest["y"])
                    if smoothed is None or time.monotonic() - prev_t > 1.0:
                        smoothed = [cx, cy]
                    else:
                        smoothed[0] += (cx - smoothed[0]) * SMOOTHING
                        smoothed[1] += (cy - smoothed[1]) * SMOOTHING
                    prev_t = time.monotonic()
                dot.pos = (smoothed[0] - width / 2, height / 2 - smoothed[1])
                dot.draw()
                hud.text = f"gaze x={smoothed[0]:.0f}  y={smoothed[1]:.0f} px  (screen {width}x{height}, markers {latest['markers']}/4)   R = recalibrate"
            else:
                hud.text = f"No screen located - markers seen: {latest['markers']}/4  {latest['error']}"
            hud.draw()
            win.flip()
    finally:
        running = False
        thread.join(timeout=2)
        win.close()
        device.close()


if __name__ == "__main__":
    main()
