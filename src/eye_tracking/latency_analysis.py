"""Offline analysis for latency_test.py recordings: how much later does the
Neon report each saccade than the computer webcam sees it?

Usage: python -m src.eye_tracking.latency_analysis data/latency/<session folder>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ONSET_FRACTION = 0.2  # saccade "onset" = signal has covered this fraction of its baseline->final move
BASELINE_S = 0.25
SEARCH_S = 0.6
MIN_AMPLITUDE_SNRS = 6.0  # move must exceed this many baseline-noise MADs, else the trial is skipped


def crossing_time(times: np.ndarray, values: np.ndarray, t_jump: float, t_next: float) -> float | None:
    """Time (interpolated between samples) at which `values` first covers
    ONSET_FRACTION of the move from its pre-jump baseline to its pre-next-jump
    plateau, searching from t_jump. None if the signal didn't move clearly."""
    base_mask = (times >= t_jump - BASELINE_S) & (times < t_jump)
    plateau_mask = (times >= t_next - BASELINE_S) & (times < t_next)
    if base_mask.sum() < 3 or plateau_mask.sum() < 3:
        return None
    base_vals = values[base_mask]
    baseline = float(np.median(base_vals))
    plateau = float(np.median(values[plateau_mask]))
    noise = float(np.median(np.abs(base_vals - baseline))) * 1.4826 + 1e-9
    amplitude = plateau - baseline
    if abs(amplitude) < MIN_AMPLITUDE_SNRS * noise:
        return None
    level = baseline + ONSET_FRACTION * amplitude
    search = (times >= t_jump) & (times <= min(t_jump + SEARCH_S, t_next))
    t, v = times[search], values[search]
    crossed = (v >= level) if amplitude > 0 else (v <= level)
    if not crossed.any():
        return None
    i = int(np.argmax(crossed))
    if i == 0:
        return float(t[0])
    frac = (level - v[i - 1]) / (v[i] - v[i - 1])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def measure_latencies(jump_times, neon_t, neon_x, cam_t, cam_x) -> list[dict]:
    """Per target jump: when the target changed, when each device first
    registered the saccade (all on the PC clock, seconds), and
    neon_minus_cam_ms - positive means the Neon reported it later."""
    jump_times = list(jump_times)
    neon_t, neon_x = np.asarray(neon_t, float), np.asarray(neon_x, float)
    cam_t, cam_x = np.asarray(cam_t, float), np.asarray(cam_x, float)
    rows = []
    for t_jump, t_next in zip(jump_times, jump_times[1:]):
        neon_on = crossing_time(neon_t, neon_x, t_jump, t_next)
        cam_on = crossing_time(cam_t, cam_x, t_jump, t_next)
        if neon_on is None or cam_on is None:
            continue
        rows.append(
            {
                "jump_t": t_jump,
                "neon_onset_t": neon_on,
                "cam_onset_t": cam_on,
                "neon_from_stim_ms": (neon_on - t_jump) * 1000,
                "cam_from_stim_ms": (cam_on - t_jump) * 1000,
                "neon_minus_cam_ms": (neon_on - cam_on) * 1000,
            }
        )
    return rows


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in rows[0]}


def _webcam_iris_ratios(video_path: Path) -> np.ndarray:
    import cv2

    from src.eye_tracking.gaze_tracker import GazeTracker

    tracker = GazeTracker()
    capture = cv2.VideoCapture(str(video_path))
    ratios = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        sample = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        ratios.append(sample.ratio if sample.ratio is not None else np.nan)
    capture.release()
    return np.array(ratios)


def main(folder: Path) -> None:
    meta = json.loads((folder / "meta.json").read_text())
    neon = _read_csv(folder / "neon.csv")
    cam = _read_csv(folder / "webcam.csv")
    stim = _read_csv(folder / "stimulus.csv")

    # Device time -> PC time: PC = device + offset (Pupil Labs docs); average the start/end estimates.
    offset_s = float(np.mean(meta["neon_clock_offset_ms"])) / 1000
    neon_t = neon["device_unix_s"] + offset_s
    good = (neon["worn"] > 0) & ~np.isnan(neon["x"])

    ratios = _webcam_iris_ratios(folder / "webcam.mp4")
    n = min(len(ratios), len(cam["unix_s"]))
    cam_t, ratios = cam["unix_s"][:n], ratios[:n]
    ok = ~np.isnan(ratios)

    rows = measure_latencies(stim["unix_s"], neon_t[good], neon["x"][good], cam_t[ok], ratios[ok])
    out = folder / "latency_results.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["jump_t"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} of {len(stim['unix_s']) - 1} jumps measurable (written to {out})")
    if not rows:
        return
    for name in ("neon_from_stim_ms", "cam_from_stim_ms", "neon_minus_cam_ms"):
        v = np.array([r[name] for r in rows])
        print(f"{name:20s} median {np.median(v):7.1f}  mean {v.mean():7.1f}  sd {v.std():6.1f}  (ms)")
    cam_dt = np.diff(cam_t)
    print(f"webcam frame interval: median {np.median(cam_dt) * 1000:.1f} ms -> timing resolution limit of the camera side")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
