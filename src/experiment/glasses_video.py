"""Synchronising the phone's own recording of the glasses' scene camera with the PC clock (no PsychoPy).

The experiment shows a deliberate pattern of full-screen green flashes right after the phone starts recording (and one more at
the end of the saccade phase) and logs the PC wall-clock time of every onset in session.json (glasses_video.sync_flashes).
The scene camera sees the screen, so the same flashes show up in the video. `find_sync_flashes` finds them in the video's
frames, `compute_offset` matches them to the logged PC times (by their spacing, so an extra bright event cannot throw it
off) and returns the offset: PC unix time = video time + offset. NOT verified on real glasses footage yet (see README).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# The flash pattern (shown by run_experiment.run_sync_flashes): n flashes, each ON for SYNC_FLASH_ON_S then dark for SYNC_FLASH_GAP_S.
SYNC_FLASH_COUNT = 3
END_FLASH_COUNT = 1
SYNC_FLASH_ON_S = 0.150
SYNC_FLASH_GAP_S = 0.300
SYNC_FLASH_PERIOD_S = SYNC_FLASH_ON_S + SYNC_FLASH_GAP_S  # onset to onset

GREEN_MARGIN = 40  # a pixel is "strongly green" when G exceeds both R and B by this much (0-255)
BASELINE_WINDOW_S = 2.0  # a frame's greenness is judged against the quiet (25th percentile) level of the frames around it; wide enough that the 3-flash pattern cannot raise it
MIN_FLASH_FRACTION = 0.01  # at least this fraction of the (downsampled) picture must turn green for a frame to count
RELATIVE_THRESHOLD = 0.35  # of the strongest flash: weaker blips (a green shirt walking past) are ignored
MATCH_TOLERANCE_S = 0.15
OK_MIN_MATCHED = 3
OK_MAX_RESIDUAL_MS = 100.0  # about three frames at 30 fps; the video's frame spacing alone is +-17 ms


def sync_flash_frames(refresh_hz: float | None) -> tuple[int, int]:
    """(frames ON, frames dark) of one flash for a display of `refresh_hz`: counted in frames, never timed with the clock,
    so each flash is as long as intended whatever the refresh rate (60 Hz assumed when it is unknown or implausible)."""
    hz = refresh_hz if refresh_hz and 20 <= refresh_hz <= 500 else 60.0
    return max(1, round(SYNC_FLASH_ON_S * hz)), max(1, round(SYNC_FLASH_GAP_S * hz))


# -- finding the flashes in a video ---------------------------------------------------------------


@dataclass(frozen=True)
class FlashEvent:
    time_s: float  # video time of the first green frame
    duration_s: float
    frames: int
    strength: float  # peak extra fraction of green pixels over the surrounding baseline


def green_fraction(frame: np.ndarray, stride: int = 8) -> float:
    """Fraction of (every `stride`-th) pixels that are strongly green. `frame` is HxWx3, channels in any order with G in the
    middle (OpenCV BGR and RGB both qualify)."""
    small = np.asarray(frame)[::stride, ::stride].astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float(np.mean((g - r > GREEN_MARGIN) & (g - b > GREEN_MARGIN)))


def _baseline(scores: np.ndarray, window: int) -> np.ndarray:
    half = max(1, window // 2)
    return np.asarray([np.percentile(scores[max(0, i - half) : i + half + 1], 25) for i in range(len(scores))])


def detect_flash_events(scores: Sequence[float], times: Sequence[float]) -> list[FlashEvent]:
    """Groups consecutive frames whose green score stands out from their neighbourhood into events (a one-frame gap inside
    an event is bridged: compression can dim a frame)."""
    s, t = np.asarray(scores, dtype=float), np.asarray(times, dtype=float)
    if len(s) < 3:
        return []
    frame_s = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    excess = s - _baseline(s, max(3, int(round(BASELINE_WINDOW_S / max(frame_s, 1e-6)))))
    peak = float(np.percentile(excess, 99.5))
    if peak < MIN_FLASH_FRACTION:
        return []
    hot = excess >= max(MIN_FLASH_FRACTION, RELATIVE_THRESHOLD * peak)
    events: list[FlashEvent] = []
    i, n = 0, len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (hot[j + 1] or (j + 2 < n and hot[j + 2])):
            j += 1
        events.append(FlashEvent(time_s=float(t[i]), duration_s=float(t[j] - t[i] + frame_s), frames=j - i + 1, strength=float(excess[i : j + 1].max())))
        i = j + 1
    return events


def read_video_scores(path: Path) -> tuple[list[float], list[float]]:
    """(green scores, frame timestamps in s) of a video file via OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    scores, times, index = [], [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            times.append(msec / 1000 if msec and msec > 0 else index / fps)  # CAP_PROP_POS_MSEC is 0 on some containers
            scores.append(green_fraction(frame))
            index += 1
    finally:
        capture.release()
    if len(times) > 2 and times[-1] <= times[0]:  # unusable stamps: fall back to the nominal frame rate
        times = [i / fps for i in range(len(times))]
    return scores, times


def find_sync_flashes(frames_or_path, fps: float | None = None) -> list[float]:
    """Video times (s) of the green sync flashes. Give a video file path, or an iterable of frames (HxWx3 arrays, `fps` needed)
    or of (time_s, frame) pairs."""
    return [e.time_s for e in find_sync_flash_events(frames_or_path, fps)]


def find_sync_flash_events(frames_or_path, fps: float | None = None) -> list[FlashEvent]:
    if isinstance(frames_or_path, (str, Path)):
        scores, times = read_video_scores(Path(frames_or_path))
    else:
        scores, times = [], []
        for index, item in enumerate(frames_or_path):
            if isinstance(item, tuple):
                times.append(float(item[0]))
                scores.append(green_fraction(item[1]))
            else:
                if not fps:
                    raise ValueError("fps is required when giving bare frames")
                times.append(index / fps)
                scores.append(green_fraction(item))
    return detect_flash_events(scores, times)


# -- matching them to the PC's log ----------------------------------------------------------------


def compute_offset(video_flash_times_s: Iterable[float], pc_flash_unix_ms: Iterable[float], tolerance_s: float = MATCH_TOLERANCE_S) -> dict:
    """The clock offset such that PC unix seconds = video seconds + offset_s, from video flash times and the logged PC onset
    times (same flashes, any extra video events ignored). Matches by pattern: every (video, PC) pairing is tried as the
    anchor and the one that explains the most PC flashes within `tolerance_s` wins. Status "ok" needs >= 3 matched flashes and
    a worst residual under 100 ms (the video's frame spacing alone is ~33 ms), else "check manually"."""
    video = sorted(float(v) for v in video_flash_times_s)
    pc = sorted(float(p) / 1000 for p in pc_flash_unix_ms)
    result = {"offset_s": None, "matched": 0, "pc_flashes": len(pc), "video_events": len(video), "pairs": [], "residual_rms_ms": None, "residual_max_ms": None, "drift_ms_per_min": None, "status": "check manually", "notes": []}
    if not pc:
        result["notes"].append("session.json has no logged sync flashes (recording or flash step did not run)")
        return result
    if not video:
        result["notes"].append("no green flashes found in the video")
        return result

    def match(offset: float) -> list[tuple[float, float]]:
        pairs, used = [], set()
        for p in pc:
            candidates = [(abs(p - offset - v), j) for j, v in enumerate(video) if j not in used and abs(p - offset - v) <= tolerance_s]
            if candidates:
                _, j = min(candidates)
                used.add(j)
                pairs.append((video[j], p))
        return pairs

    best: list[tuple[float, float]] = []
    best_cost = float("inf")
    for p in pc:
        for v in video:
            pairs = match(p - v)
            cost = float(np.sqrt(np.mean([(pp - vv - (p - v)) ** 2 for vv, pp in pairs]))) if pairs else float("inf")
            if (len(pairs), -cost) > (len(best), -best_cost):
                best, best_cost = pairs, cost
    if not best:
        result["notes"].append("no video flash lines up with the logged PC flashes")
        return result

    offset = float(np.median([p - v for v, p in best]))  # refined over all matches
    residuals = np.asarray([(p - v - offset) * 1000 for v, p in best])
    result.update(
        offset_s=offset,
        matched=len(best),
        pairs=[{"video_s": round(v, 4), "pc_unix_ms": round(p * 1000, 1)} for v, p in best],
        residual_rms_ms=round(float(np.sqrt(np.mean(residuals**2))), 1),
        residual_max_ms=round(float(np.abs(residuals).max()), 1),
    )
    span_s = best[-1][0] - best[0][0]
    if len(best) >= 4 and span_s > 30:  # start flashes and the end flash: how far the two clocks drifted apart
        slope = float(np.polyfit([v for v, _ in best], [p - v for v, p in best], 1)[0])
        result["drift_ms_per_min"] = round(slope * 60 * 1000, 3)
    if len(best) < OK_MIN_MATCHED:
        result["notes"].append(f"only {len(best)} of {len(pc)} logged flashes found in the video")
    if result["residual_max_ms"] > OK_MAX_RESIDUAL_MS:
        result["notes"].append(f"worst residual {result['residual_max_ms']} ms is above {OK_MAX_RESIDUAL_MS:.0f} ms")
    if not result["notes"]:
        result["status"] = "ok"
    return result


def video_time_for_pc_unix_ms(pc_unix_ms: float, offset_s: float) -> float:
    """Where, in the video (seconds from its start), a PC wall-clock time falls."""
    return pc_unix_ms / 1000 - offset_s
