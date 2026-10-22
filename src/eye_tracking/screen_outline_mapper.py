"""Scene-camera gaze -> screen pixels by finding the monitor itself (no tags).

Same interface as ScreenGazeMapper (process_scene / map_gaze / markers_detected /
located), so NeonScreenGazeSource can use it unchanged. The scene thread calls
process_scene() (~30 Hz): the screen is found as a convex quadrilateral (a
uniform grey rectangle, brighter or darker than its surroundings), its four
corners are line-fitted to sub-pixel accuracy in the *undistorted* image,
smoothed, and turned into a perspective transform. The gaze thread calls
map_gaze() (~200 Hz) with the newest transform.

Jitter matters more than absolute accuracy here: a corner that wobbles by a
pixel moves every mapped gaze sample and can look like a saccade onset. So the
corners are low-pass filtered, with the filter opening up when they really move
(head movement) so it adds no lag then.
"""
import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from include.eye_tracking.constants import (
    NEON_OUTLINE_BORDER_MARGIN_PX,
    NEON_OUTLINE_HOLD_S,
    NEON_OUTLINE_MAX_APPROX_FRACTION,
    NEON_OUTLINE_MAX_SIDE_RATIO,
    NEON_OUTLINE_MIN_AREA_FRACTION,
    NEON_OUTLINE_MIN_CONTRAST,
    NEON_OUTLINE_MIN_FILL,
    NEON_OUTLINE_SMOOTH_JUMP_PX,
    NEON_OUTLINE_SMOOTH_MIN_ALPHA,
    NEON_OUTLINE_WORK_SCALE,
)
from src.eye_tracking.screen_gaze_mapper import ScreenGaze

_OPEN_KERNEL = np.ones((5, 5), np.uint8)


@dataclass(frozen=True)
class ScreenQuad:
    corners: np.ndarray  # (4, 2) float, undistorted scene px, ordered TL, TR, BR, BL
    score: float


class _Undistorter:
    """Scene px <-> ideal pinhole px using the device calibration (None = already ideal)."""

    def __init__(self, calibration) -> None:
        self.k = self.dist = None
        if calibration is not None:
            self.k = np.asarray(calibration["scene_camera_matrix"], np.float64).reshape(3, 3)
            self.dist = np.asarray(calibration["scene_distortion_coefficients"], np.float64).ravel()

    def points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.k is None:
            return pts
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.k, self.dist, P=self.k).reshape(-1, 2)

    def distort(self, pts: np.ndarray) -> np.ndarray:
        """Inverse of points() (for probing / drawing in raw scene pixels)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.k is None:
            return pts
        normalized = (pts - self.k[:2, 2]) / np.array([self.k[0, 0], self.k[1, 1]])
        xyz = np.column_stack([normalized, np.ones(len(normalized))])
        return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.k, self.dist)[0].reshape(-1, 2)


def _order_clockwise(quad: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL (image coords, y down). Assumes the head isn't rolled past 45 degrees."""
    centre = quad.mean(axis=0)
    ordered = quad[np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))]  # clockwise on screen
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def _hull_quad(points: np.ndarray, perimeter: float) -> np.ndarray | None:
    hull = cv2.convexHull(points.astype(np.float32))
    for fraction in (0.01, 0.015, 0.02, NEON_OUTLINE_MAX_APPROX_FRACTION):
        approx = cv2.approxPolyDP(hull, fraction * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            return _order_clockwise(approx.astype(np.float64))
        if len(approx) < 4:
            return None
    return None


def _fit_line(points: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


def _refine_edges(points: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Re-fit each side as a line through the contour points near it and intersect neighbours (sub-pixel, outlier-robust)."""
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        length = float(np.hypot(*edge))
        if length < 1e-6:
            return None
        t = ((points - a) @ edge) / (length * length)
        dist = np.abs((points - a) @ np.array([-edge[1], edge[0]])) / length
        near = points[(t > 0.08) & (t < 0.92) & (dist < max(4.0, 0.03 * length))]
        if len(near) < 8:
            return None
        lines.append(_fit_line(near))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None
        s = np.linalg.solve(matrix, p2 - p1)[0]
        corners.append(p1 + s * d1)
    return np.array(corners)


def _is_plausible(quad: np.ndarray) -> bool:
    cross = [np.cross(quad[(i + 1) % 4] - quad[i], quad[(i + 2) % 4] - quad[(i + 1) % 4]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return False  # not convex
    sides = [float(np.hypot(*(quad[(i + 1) % 4] - quad[i]))) for i in range(4)]
    if min(sides) < 1:
        return False
    return max(sides[0], sides[2]) / min(sides[0], sides[2]) <= NEON_OUTLINE_MAX_SIDE_RATIO and (
        max(sides[1], sides[3]) / min(sides[1], sides[3]) <= NEON_OUTLINE_MAX_SIDE_RATIO
    )


def _contrast(blur: np.ndarray, contour: np.ndarray) -> float:
    mask = np.zeros(blur.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    ring = cv2.subtract(cv2.dilate(mask, np.ones((15, 15), np.uint8)), mask)
    if not ring.any() or not mask.any():
        return 0.0
    return abs(float(blur[mask > 0].mean()) - float(blur[ring > 0].mean()))


def find_screen_quad(gray: np.ndarray, undistorter: _Undistorter | None = None) -> ScreenQuad | None:
    """Best screen-shaped blob in a grey scene frame, or None. Pure function of the frame."""
    undistorter = undistorter or _Undistorter(None)
    scale = NEON_OUTLINE_WORK_SCALE
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    height, width = small.shape
    frame_area = float(width * height)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    margin = max(1, round(NEON_OUTLINE_BORDER_MARGIN_PX * scale))
    best: ScreenQuad | None = None
    seen = set()
    for level in (otsu, otsu - 30, otsu + 30):  # Otsu sits at the edge's half-height; the others are fallbacks only
        if best is not None:
            break
        level = int(min(max(level, 5), 250))
        if level in seen:
            continue
        seen.add(level)
        for bright in (True, False):
            mask = ((blur > level) if bright else (blur <= level)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(contour) < NEON_OUTLINE_MIN_AREA_FRACTION * frame_area:
                    break
                candidate = _evaluate(contour, blur, undistorter, scale, (width, height), margin, frame_area)
                if candidate is not None and (best is None or candidate.score > best.score):
                    best = candidate
    return best


def _evaluate(contour, blur, undistorter, scale, size, margin, frame_area) -> ScreenQuad | None:
    raw = contour.reshape(-1, 2).astype(np.float64)
    if (raw.min(axis=0) <= margin).any() or (raw.max(axis=0) >= np.array(size) - 1 - margin).any():
        return None  # touches the frame edge: screen (partly) out of view
    points = undistorter.points((raw + 0.5) / scale - 0.5)  # mask pixel centre -> full-res pixel centre
    perimeter = float(cv2.arcLength(points.astype(np.float32).reshape(-1, 1, 2), True))
    quad = _hull_quad(points, perimeter)
    if quad is None:
        return None
    quad_area = float(cv2.contourArea(quad.astype(np.float32)))
    fill = float(cv2.contourArea(points.astype(np.float32).reshape(-1, 1, 2))) / quad_area if quad_area > 0 else 0.0
    if fill < NEON_OUTLINE_MIN_FILL:
        return None
    refined = _refine_edges(points, quad)
    if refined is None or not _is_plausible(refined):
        return None
    if _contrast(blur, contour) < NEON_OUTLINE_MIN_CONTRAST:
        return None
    return ScreenQuad(refined, quad_area / (frame_area / scale**2) * fill**3)


class ScreenOutlineMapper:
    def __init__(
        self, calibration, screen_size: tuple[int, int], hold_s: float = NEON_OUTLINE_HOLD_S, clock=time.monotonic
    ) -> None:
        self._undistorter = _Undistorter(calibration)
        self._width, self._height = screen_size
        self._screen_corners = np.array(
            [[0, 0], [self._width, 0], [self._width, self._height], [0, self._height]], np.float32
        )
        self._hold_s = hold_s
        self._clock = clock
        self._lock = threading.Lock()
        self._corners: np.ndarray | None = None  # smoothed, undistorted scene px
        self._homography: np.ndarray | None = None  # undistorted scene px -> screen px
        self._good_at = 0.0
        self._detected = False

    # -- scene thread -------------------------------------------------------

    def process_scene(self, frame) -> None:
        """Locate the screen in one scene frame (call from the scene thread)."""
        pixels = getattr(frame, "bgr_pixels", frame)
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
        quad = find_screen_quad(gray, self._undistorter)
        now = self._clock()
        with self._lock:
            self._detected = quad is not None
            if quad is not None:
                self._corners = self._smooth(quad.corners)
                self._homography = cv2.getPerspectiveTransform(self._corners.astype(np.float32), self._screen_corners)
                self._good_at = now
            elif self._homography is not None and now - self._good_at > self._hold_s:
                self._corners = self._homography = None

    def _smooth(self, corners: np.ndarray) -> np.ndarray:
        previous = self._corners
        if previous is None or (self._homography is None):
            return corners
        moved = float(np.linalg.norm(corners - previous, axis=1).mean())
        alpha = min(1.0, NEON_OUTLINE_SMOOTH_MIN_ALPHA + (1 - NEON_OUTLINE_SMOOTH_MIN_ALPHA) * moved / NEON_OUTLINE_SMOOTH_JUMP_PX)
        return previous + alpha * (corners - previous)

    # -- gaze thread --------------------------------------------------------

    def map_gaze(self, gaze) -> ScreenGaze | None:
        """Gaze (a Real-Time API GazeData) -> screen px, or None if the screen isn't currently located."""
        x, y = _gaze_xy(gaze)
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            return None
        with self._lock:
            homography, good_at = self._homography, self._good_at
        if homography is None:
            return None
        px, py = self._undistorter.points([[x, y]])[0]
        sx, sy, w = homography @ np.array([px, py, 1.0])
        if abs(w) < 1e-9:
            return None
        sx, sy = sx / w, sy / w
        if not (-0.5 * self._width <= sx <= 1.5 * self._width and -0.5 * self._height <= sy <= 1.5 * self._height):
            return None  # a wild extrapolation from a bad outline
        return ScreenGaze(float(sx), float(sy), self._clock() - good_at)

    # -- status / diagnostics ----------------------------------------------

    @property
    def markers_detected(self) -> int:
        """4 when the most recent scene frame yielded a screen outline (corner count), else 0."""
        with self._lock:
            return 4 if self._detected else 0

    @property
    def located(self) -> bool:
        """True while a usable (fresh or briefly held) screen location exists."""
        with self._lock:
            return self._homography is not None

    @property
    def corners(self) -> np.ndarray | None:
        """Current smoothed corners (TL, TR, BR, BL) in undistorted scene px, or None."""
        with self._lock:
            return None if self._corners is None else self._corners.copy()

    def screen_to_scene(self, screen_points) -> np.ndarray | None:
        """Raw scene-camera px where the given screen px currently appear (for probes and overlays)."""
        with self._lock:
            homography = self._homography
        if homography is None:
            return None
        pts = np.asarray(screen_points, np.float64).reshape(-1, 1, 2)
        ideal = cv2.perspectiveTransform(pts, np.linalg.inv(homography)).reshape(-1, 2)
        return self._undistorter.distort(ideal)


def _gaze_xy(gaze):
    x, y = getattr(gaze, "x", None), getattr(gaze, "y", None)
    if x is None and isinstance(gaze, (tuple, list)) and len(gaze) >= 2:
        x, y = gaze[0], gaze[1]
    return (None, None) if x is None or y is None else (float(x), float(y))
