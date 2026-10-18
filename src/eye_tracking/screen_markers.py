"""Corner AprilTag markers that let the Neon's scene camera locate the screen.

Shared by the live gaze view and the experiment so both draw (and the mapper
expects) exactly the same layout. Pure numpy/cv2 apart from build_marker_stims,
which needs a PsychoPy window.
"""
import cv2
import numpy as np

from include.eye_tracking.constants import SCREEN_MARKER_MARGIN_PX, SCREEN_MARKER_PAD_LEVEL, SCREEN_MARKER_PX

MARKER_IDS = (0, 1, 2, 3)  # top-left, top-right, bottom-right, bottom-left


def pad_size() -> int:
    return SCREEN_MARKER_PX + 2 * SCREEN_MARKER_MARGIN_PX


def marker_pads(width: int, height: int) -> dict[int, tuple[int, int]]:
    """Top-left corner (screen px, top-left origin) of each marker's padded square."""
    pad = pad_size()
    return {0: (0, 0), 1: (width - pad, 0), 2: (width - pad, height - pad), 3: (0, height - pad)}


def marker_verts(width: int, height: int) -> dict[int, list[tuple[float, float]]]:
    """Each tag's four corners (top-left, clockwise) in screen px - the format GazeMapper.add_surface wants."""
    verts = {}
    for marker_id, (px, py) in marker_pads(width, height).items():
        x0, y0 = px + SCREEN_MARKER_MARGIN_PX, py + SCREEN_MARKER_MARGIN_PX
        s = SCREEN_MARKER_PX
        verts[marker_id] = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]
    return verts


def marker_pad_image(marker_id: int) -> np.ndarray:
    """Padded marker as an ordinary top-down uint8 image."""
    from pupil_labs.real_time_screen_gaze import marker_generator

    tag = cv2.resize(
        marker_generator.generate_marker(marker_id), (SCREEN_MARKER_PX, SCREEN_MARKER_PX), interpolation=cv2.INTER_NEAREST
    )
    m = SCREEN_MARKER_MARGIN_PX
    return cv2.copyMakeBorder(tag, m, m, m, m, cv2.BORDER_CONSTANT, value=SCREEN_MARKER_PAD_LEVEL)


def build_marker_stims(win) -> list:
    """PsychoPy ImageStims for the four markers, in pixel units regardless of the window's own units."""
    from psychopy import visual

    width, height = win.size
    pad = pad_size()
    stims = []
    for marker_id, (px, py) in marker_pads(width, height).items():
        image = np.flipud(marker_pad_image(marker_id).astype(np.float32) / 127.5 - 1.0)  # PsychoPy: row 0 = bottom
        stims.append(
            visual.ImageStim(
                win,
                image=image,
                units="pix",
                size=(pad, pad),
                pos=(px + pad / 2 - width / 2, height / 2 - py - pad / 2),
                interpolate=False,
            )
        )
    return stims
