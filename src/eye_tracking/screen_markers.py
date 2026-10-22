"""Corner AprilTag markers that let the Neon's scene camera locate the screen.

Shared by the live gaze view and the experiment so both draw (and the mapper
expects) exactly the same layout. Pure numpy/cv2 apart from build_marker_stims,
which needs a PsychoPy window.
"""
import cv2
import numpy as np

from include.experiment.constants import BACKGROUND_LUMINANCE
from include.eye_tracking.constants import (
    SCREEN_MARKER_CONTRAST,
    SCREEN_MARKER_MARGIN_CELLS,
    SCREEN_MARKER_PX,
    SCREEN_MARKER_SCREEN_WIDTH_CM,
    SCREEN_MARKER_WIDTH_CM,
)

MARKER_IDS = (0, 1, 2, 3)  # top-left, top-right, bottom-right, bottom-left
TAG_CELLS = 8  # tag36h11 as drawn by marker_generator: 8x8 cells incl. the black border
BACKGROUND_LEVEL = round(BACKGROUND_LUMINANCE * 255)


def marker_px(screen_width: int | None = None) -> int:
    """Tag side in screen px: SCREEN_MARKER_WIDTH_CM wide when the display width is known, always a
    multiple of 8 so every cell gets the same number of pixels."""
    if screen_width is None or not SCREEN_MARKER_SCREEN_WIDTH_CM:
        return SCREEN_MARKER_PX
    px = SCREEN_MARKER_WIDTH_CM / SCREEN_MARKER_SCREEN_WIDTH_CM * screen_width
    return max(TAG_CELLS, int(round(px / TAG_CELLS)) * TAG_CELLS)


def margin_px(screen_width: int | None = None) -> int:
    return SCREEN_MARKER_MARGIN_CELLS * marker_px(screen_width) // TAG_CELLS


def pad_size(screen_width: int | None = None) -> int:
    return marker_px(screen_width) + 2 * margin_px(screen_width)


def marker_pads(width: int, height: int) -> dict[int, tuple[int, int]]:
    """Top-left corner (screen px, top-left origin) of each marker's padded square."""
    pad = pad_size(width)
    return {0: (0, 0), 1: (width - pad, 0), 2: (width - pad, height - pad), 3: (0, height - pad)}


def marker_verts(width: int, height: int) -> dict[int, list[tuple[float, float]]]:
    """Each tag's four corners in screen px, in the order GazeMapper.add_surface wants.

That order is bottom-right, bottom-left, top-left, top-right (screen, y down) - NOT top-left clockwise:
the library labels vertex i as a different corner than a reader would assume, and with plain clockwise
order each tag is mapped rotated 180 deg about its own centre, which skews the whole screen mapping by
roughly (tag size / screen size). Invisible with 100 px tags (~1%), ~3.5% with 216 px ones.
"""
    verts = {}
    m, s = margin_px(width), marker_px(width)
    for marker_id, (px, py) in marker_pads(width, height).items():
        x0, y0 = px + m, py + m
        verts[marker_id] = [(x0 + s, y0 + s), (x0, y0 + s), (x0, y0), (x0 + s, y0)]
    return verts


def marker_levels(marker_id: int) -> tuple[int, int]:
    """(black, white) cell levels for one tag: SCREEN_MARKER_CONTRAST apart, positioned so the tag's mean
    equals the background grey (tags differ in how many cells are white)."""
    from pupil_labs.real_time_screen_gaze import marker_generator

    white_fraction = float((marker_generator.generate_marker(marker_id) > 127).mean())
    black = round(BACKGROUND_LEVEL - white_fraction * SCREEN_MARKER_CONTRAST)
    return black, black + SCREEN_MARKER_CONTRAST


def marker_pad_image(marker_id: int, screen_width: int | None = None) -> np.ndarray:
    """Padded marker as an ordinary top-down uint8 image; the quiet zone is the background grey."""
    from pupil_labs.real_time_screen_gaze import marker_generator

    black, white = marker_levels(marker_id)
    cells = np.where(marker_generator.generate_marker(marker_id) > 127, white, black).astype(np.uint8)
    px = marker_px(screen_width)
    tag = cv2.resize(cells, (px, px), interpolation=cv2.INTER_NEAREST)
    m = margin_px(screen_width)
    return cv2.copyMakeBorder(tag, m, m, m, m, cv2.BORDER_CONSTANT, value=BACKGROUND_LEVEL)


def build_marker_stims(win) -> list:
    """PsychoPy ImageStims for the four markers, in pixel units regardless of the window's own units."""
    from psychopy import visual

    width, height = win.size
    pad = pad_size()
    stims = []
    for marker_id, (px, py) in marker_pads(width, height).items():
        image = np.flipud(marker_pad_image(marker_id, width).astype(np.float32) / 127.5 - 1.0)  # PsychoPy: row 0 = bottom
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
