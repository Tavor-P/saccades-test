"""Attach the phone's exported glasses recording to a session and synchronise it with the PC clock.

    .\\scripts\\py.ps1 -m src.experiment.import_glasses_video <session-folder> <path-to-exported-video.mp4>

Copies the video into <session>/glasses/, finds the green sync flashes in it (see glasses_video.py), matches them to the PC
times logged in session.json, writes glasses/video_sync.json (offset, residual, 'ok' / 'check manually') and
glasses/flash_video_times.csv (where every flash trial is in the video), and prints how to find a PC time in the video.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from src.experiment import session_files
from src.experiment.glasses_video import compute_offset, find_sync_flash_events, video_time_for_pc_unix_ms
from src.experiment.logger import read_trial_rows

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _resolve_folder(identifier: str, data_dir: Path) -> Path:
    ref = session_files.find_session(data_dir, identifier)
    if ref is None or ref.folder is None:
        raise SystemExit(f"{identifier!r} is not a session folder in the new layout (data/<participant>/<run>/ with a trials.csv)")
    return ref.folder


def import_video(folder: Path, video: Path) -> dict:
    """Does the whole import for one run folder; returns the video_sync.json content."""
    folder, video = Path(folder), Path(video)
    glasses = folder / session_files.GLASSES_DIR
    glasses.mkdir(exist_ok=True)
    target = glasses / video.name
    if video.resolve() != target.resolve():
        shutil.copy2(video, target)

    meta_path = folder / session_files.SESSION_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    logged = (meta.get("glasses_video") or {}).get("sync_flashes") or []
    events = find_sync_flash_events(target)
    sync = compute_offset([e.time_s for e in events], [f["pc_unix_ms"] for f in logged])
    sync = {
        "video_file": target.name,
        **sync,
        "video_flash_events": [{"time_s": round(e.time_s, 4), "duration_s": round(e.duration_s, 3), "strength": round(e.strength, 3)} for e in events],
        "meaning": "PC unix time (s) = video time (s) + offset_s",
    }
    (glasses / "video_sync.json").write_text(json.dumps(sync, indent=2))

    if sync["offset_s"] is not None:
        with (glasses / "flash_video_times.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "flash_unix_ms", "video_s"])
            for row in read_trial_rows(folder / session_files.TRIALS_FILE):
                if row.get("phase") == "saccade" and row.get("grating_shown") == "True" and row.get("grating_shown_at_unix_ms"):
                    unix_ms = float(row["grating_shown_at_unix_ms"])
                    writer.writerow([row["trial_index"], unix_ms, round(video_time_for_pc_unix_ms(unix_ms, sync["offset_s"]), 3)])
    if meta:
        meta.setdefault("glasses_video", {}).update(video_file=f"{session_files.GLASSES_DIR}/{target.name}", sync_status=sync["status"])
        meta_path.write_text(json.dumps(meta, indent=2))
    return sync


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy the phone's exported glasses video into a session and sync it with the PC clock.")
    parser.add_argument("session", help="run folder (data/<participant>/<run>), its name, or its unix timestamp")
    parser.add_argument("video", type=Path, help="the exported scene video (.mp4) from the Neon Companion app / Pupil Cloud")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    folder = _resolve_folder(args.session, args.data_dir)
    sync = import_video(folder, args.video)

    print(f"session: {folder}")
    print(f"flashes found in the video: {sync['video_events']}; logged on the PC: {sync['pc_flashes']}; matched: {sync['matched']}")
    for note in sync["notes"]:
        print(f"  note: {note}")
    if sync["offset_s"] is None:
        print("sync: check manually - no offset could be computed (look for the green screen flashes at the start of the video).")
        return 1
    print(f"sync: {sync['status']} (offset {sync['offset_s']:.3f} s, residual rms {sync['residual_rms_ms']} ms, worst {sync['residual_max_ms']} ms)")
    print("\nFinding a PC time in the video:  video_seconds = pc_unix_ms / 1000 - offset_s   (offset_s above, also in glasses/video_sync.json)")
    print("Every flash trial's video time is listed in glasses/flash_video_times.csv.")
    return 0 if sync["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
