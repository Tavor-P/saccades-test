"""Offline check of the marker-free screen detector on a recorded scene video (or a folder of frames).

    .\\scripts\\py.ps1 -m src.eye_tracking.screen_detect_compare scene.mp4
    .\\scripts\\py.ps1 -m src.eye_tracking.screen_detect_compare frames_dir --apriltags --screen 1920x1080
    .\\scripts\\py.ps1 -m src.eye_tracking.screen_detect_compare scene.mp4 --save-debug out_dir --every 15

Record with the experiment's screen showing its normal grey background (and, for --apriltags, the
corner tags drawn: e.g. run neon_gaze_view in another window, or a tag-backend session). Export the
scene video from the phone / Pupil Cloud, or use any video/frames of the monitor from the Neon.

Reports per run: how often the outline was found, corner jitter (frame-to-frame step of the smoothed
mapping of fixed scene probes, in SCREEN pixels; trust it for a still head - with head motion it also
contains the real motion), and with --apriltags the screen-pixel disagreement between the two mappers
on frames where both locate the screen. Use --calibration (a .json with scene_camera_matrix and
scene_distortion_coefficients, e.g. dumped from device.get_calibration()) to undistort; without it a
pinhole model of the Neon scene camera (103 deg horizontal FOV) is assumed.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.eye_tracking.screen_outline_mapper import ScreenOutlineMapper

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
_PROBES = [(fx, fy) for fy in (0.1, 0.5, 0.9) for fx in (0.1, 0.5, 0.9)]  # screen fractions


def iter_frames(source: Path):
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES):
            image = cv2.imread(str(path))
            if image is not None:
                yield path.name, image
        return
    capture = cv2.VideoCapture(str(source))
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        yield f"frame{index:06d}", image
        index += 1
    capture.release()


def default_calibration(width: int, height: int) -> dict:
    focal = (width / 2) / math.tan(math.radians(103 / 2))
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    return {"scene_camera_matrix": k.reshape(1, 3, 3), "scene_distortion_coefficients": np.zeros((1, 8))}


def load_calibration(path: str | None, width: int, height: int) -> dict:
    if path is None:
        return default_calibration(width, height)
    raw = json.loads(Path(path).read_text())
    return {key: np.asarray(raw[key], float) for key in ("scene_camera_matrix", "scene_distortion_coefficients")}


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return f"median {ordered[len(ordered) // 2]:.2f}  p95 {p95:.2f}  max {ordered[-1]:.2f}"


def run(source: Path, screen: tuple[int, int], calibration_path: str | None, apriltags: bool, save_debug: Path | None, every: int) -> dict:
    frames = iter_frames(source)
    first = next(frames, None)
    if first is None:
        raise SystemExit(f"no frames found in {source}")
    height, width = first[1].shape[:2]
    calibration = load_calibration(calibration_path, width, height)
    outline = ScreenOutlineMapper(calibration, screen, hold_s=0.0)  # no hold: report raw per-frame detection
    tags = None
    if apriltags:
        from src.eye_tracking.screen_gaze_mapper import ScreenGazeMapper

        tags = ScreenGazeMapper(calibration, screen, hold_s=0.0)
    if save_debug:
        save_debug.mkdir(parents=True, exist_ok=True)

    total = found = tag_found = both = 0
    probes_scene = None  # fixed raw-scene probe pixels, chosen from the first detection
    prev: dict[str, np.ndarray] = {}
    steps: list[float] = []
    disagreement: list[float] = []
    corner_steps: list[float] = []
    prev_corners = None

    def handle(name, image):
        nonlocal total, found, tag_found, both, probes_scene, prev_corners
        total += 1
        outline.process_scene(image)
        if tags is not None:
            tags.process_scene(SimpleNamespace(bgr_pixels=image))
        located = outline.located
        found += located
        if located:
            if probes_scene is None:
                probes_scene = outline.screen_to_scene([(fx * screen[0], fy * screen[1]) for fx, fy in _PROBES])
            corners = outline.corners
            if prev_corners is not None:
                corner_steps.append(float(np.linalg.norm(corners - prev_corners, axis=1).max()))
            prev_corners = corners
            mapped = [outline.map_gaze(SimpleNamespace(x=px, y=py)) for px, py in probes_scene]
            mapped = np.array([[m.x, m.y] for m in mapped]) if all(mapped) else None
            if mapped is not None and "outline" in prev:
                steps.extend(np.linalg.norm(mapped - prev["outline"], axis=1).tolist())
            if mapped is not None:
                prev["outline"] = mapped
            if tags is not None and tags.located:
                tag_found += 1
                tag_mapped = [tags.map_gaze(SimpleNamespace(x=px, y=py)) for px, py in probes_scene]
                if mapped is not None and all(tag_mapped):
                    both += 1
                    disagreement.extend(np.linalg.norm(mapped - np.array([[m.x, m.y] for m in tag_mapped]), axis=1).tolist())
        elif tags is not None and tags.located:
            tag_found += 1
        else:
            prev.pop("outline", None)
            prev_corners = None
        if save_debug and total % every == 0:
            canvas = image.copy()
            if outline.located:
                corners = outline._undistorter.distort(outline.corners)
                cv2.polylines(canvas, [np.int32(corners).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
                for label, corner in zip("0123", corners):
                    cv2.putText(canvas, label, tuple(int(v) for v in corner), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                cv2.putText(canvas, "NOT FOUND", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.imwrite(str(save_debug / f"{name}.jpg"), canvas)

    handle(*first)
    for name, image in frames:
        handle(name, image)

    report = {
        "frames": total,
        "outline_found": found,
        "outline_found_fraction": found / total,
        "corner_step_px": corner_steps,
        "probe_step_screen_px": steps,
    }
    print(f"frames: {total}   scene {width}x{height}   screen {screen[0]}x{screen[1]}")
    print(f"outline found: {found}/{total} ({100 * found / total:.1f}%)")
    print(f"corner step between consecutive found frames, scene px: {_stats(corner_steps)}")
    print(f"probe step between consecutive frames, screen px: {_stats(steps)}")
    if tags is not None:
        print(f"apriltags found: {tag_found}/{total} ({100 * tag_found / total:.1f}%)   both: {both}")
        print(f"outline vs apriltags, screen px: {_stats(disagreement)}")
        report["apriltag_found"], report["both"], report["disagreement_screen_px"] = tag_found, both, disagreement
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="scene video file, or a folder of images")
    parser.add_argument("--screen", default="1920x1080", help="screen size in px, WxH")
    parser.add_argument("--calibration", help="json with scene_camera_matrix + scene_distortion_coefficients")
    parser.add_argument("--apriltags", action="store_true", help="also run the AprilTag mapper and compare")
    parser.add_argument("--save-debug", type=Path, help="write annotated frames (detected outline) to this folder")
    parser.add_argument("--every", type=int, default=30, help="with --save-debug: save every Nth frame")
    parser.add_argument("--json", type=Path, help="write the full report to this file")
    args = parser.parse_args()
    width, height = (int(v) for v in args.screen.lower().split("x"))
    report = run(args.source, (width, height), args.calibration, args.apriltags, args.save_debug, args.every)
    if args.json:
        args.json.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
