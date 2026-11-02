"""Plain-language summary of one finished run (no PsychoPy): `.\\scripts\\py.ps1 -m src.experiment.run_summary [session]`.

`build_run_summary(run_folder)` reads the run's trials.csv and session.json (either may be missing or damaged: the text then says so
instead of raising), `write_run_summary` saves it as RUN_SUMMARY.txt in the run folder. The experiment calls `finish_run_summary` at
the end of every run, which never raises. The CLI resolves the session like `analyze_session` (newest by default).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from include.experiment.constants import SACCADE_FIXED_TRIALS_REAL, SACCADE_FIXED_TRIALS_TEST
from src.experiment import session_files
from src.experiment.logger import read_trial_rows

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FILE = "RUN_SUMMARY.txt"

MIN_SAMPLE_RATE_HZ = 100.0  # Neon streams ~140-200 Hz; below this the saccade timing gets coarse
MAX_LATENCY_MS = 150.0  # known open problem: ~95 ms in full runs; analysis uses the phone's capture times so only far beyond that matters
MAX_GAP_MS = 150.0
MAX_TRACKING_LOST_SHARE = 0.10
MAX_UNANSWERED_SHARE = 0.20
MIN_TRIAL_SHARE = 0.5  # fewer flashes than this share of the planned count is called "very few"


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _load_rows(path: Path) -> list[dict] | None:
    try:
        return list(read_trial_rows(path))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _is_true(value) -> bool:
    return str(value).strip().lower() in ("true", "yes")


def _float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def trial_counts(rows: list[dict]) -> dict:
    """Counts over the scored rows (everything except the rt_test rows): flashes shown, answered, correct, missing answers, tracking
    lost near the flash, and the mean/median reaction time over ALL rows that have one (rt_test included: they seed the average)."""
    scored = [r for r in rows if r.get("phase") != "rt_test"]
    shown = [r for r in scored if _is_true(r.get("grating_shown"))]
    answered = [r for r in shown if _is_true(r.get("responded"))]
    reactions = [v for v in (_float(r.get("reaction_latency_ms")) for r in rows) if v is not None]
    return {
        "shown": len(shown),
        "answered": len(answered),
        "correct": sum(r.get("outcome") == "correct" for r in shown),
        "missing": len(shown) - len(answered),
        "tracking_lost": sum(r.get("tracking_ok") == "False" for r in shown),
        "median_reaction_ms": round(statistics.median(reactions)) if reactions else None,
    }


def gaze_quality_lines(quality: dict | None) -> tuple[list[str], list[str]]:
    """(report lines, warnings) for session.json's gaze_quality, in words: rate, delay, gaps, each OK or check."""
    if not quality:
        return ["Gaze quality: not recorded (no gaze data was saved)."], ["No gaze quality was saved: the gaze source may never have started."]
    lines, warnings = [], []
    rate, latency, gap, gaps = (quality.get(k) for k in ("sample_rate_hz", "latency_ms_median", "longest_gap_ms", "gaps_over_threshold"))
    if rate is None:
        lines.append("Gaze rate: unknown (check)")
        warnings.append("Gaze sample rate unknown.")
    elif rate < MIN_SAMPLE_RATE_HZ:
        lines.append(f"Gaze rate: {rate:.0f} Hz (check, expected 140+)")
        warnings.append(f"Low gaze sample rate ({rate:.0f} Hz): check the Wi-Fi and that the glasses were worn.")
    else:
        lines.append(f"Gaze rate: {rate:.0f} Hz (OK)")
    if latency is None:
        lines.append("Gaze delay: unknown (check)")
    elif latency > MAX_LATENCY_MS:
        lines.append(f"Gaze delay: {latency:.0f} ms (check, typical 90-100)")
        warnings.append(f"Gaze reached the PC {latency:.0f} ms late; the analysis uses the phone's capture times but check the network.")
    else:
        lines.append(f"Gaze delay: {latency:.0f} ms (OK)")
    if gap is None:
        lines.append("Gaze gaps: unknown (check)")
    elif gap > MAX_GAP_MS:
        lines.append(f"Gaze gaps: longest {gap:.0f} ms, {gaps or 0} over the threshold (check)")
        warnings.append(f"Gaze stream stalled for up to {gap:.0f} ms.")
    else:
        lines.append(f"Gaze gaps: longest {gap:.0f} ms (OK)")
    return lines, warnings


def _ignored_space_total(meta: dict) -> int:
    total = 0
    for counts in (meta.get("ignored_space_presses") or {}).values():
        if isinstance(counts, dict):
            total += sum(v for v in counts.values() if isinstance(v, (int, float)))
    return int(total)


def _recording_line(glasses: dict | None) -> tuple[str, str | None]:
    status = str((glasses or {}).get("recording") or "unknown")
    if status == "saved":
        return "Phone recording: saved on the phone.", None
    if status == "started":
        return "Phone recording: started but not confirmed saved.", "The phone recording was started but not confirmed saved: open the Neon Companion app and check it."
    if status == "not started":
        return "Phone recording: not started.", "No phone recording was made."
    return f"Phone recording: {status}.", f"Phone recording problem: {status}."


def _command_folder(folder: Path) -> str:
    """The run folder as typed in the commands: relative to the repo root when inside it, quoted if it has spaces."""
    try:
        text = str(Path(folder).resolve().relative_to(ROOT))
    except ValueError:
        text = str(Path(folder).resolve())
    return f'"{text}"' if " " in text else text


def build_run_summary(run_folder) -> str:
    """The summary text for the run in `run_folder` (never raises for missing or damaged files)."""
    folder = Path(run_folder)
    meta = _load_json(folder / session_files.SESSION_FILE)
    rows = _load_rows(folder / session_files.TRIALS_FILE)
    warnings: list[str] = []
    out: list[str] = []

    meta = meta or {}
    participant = (meta.get("participant") or {}).get("id") or (folder.parent.name if folder.parent.name else None) or "unknown"
    run = meta.get("run") or {}
    mode = run.get("mode") or ("test" if folder.name.endswith("_test") else "unknown")
    date = str(run.get("started_at") or folder.name.split("_")[0] or "unknown").replace("T", " ")
    out += ["RUN SUMMARY", f"Participant: {participant}   Mode: {mode}   Started: {date}", f"Folder: {folder}", ""]
    if not meta:
        warnings.append("session.json is missing or unreadable: settings and gaze quality could not be read.")

    if rows is None:
        out.append("Flashes: trials.csv is missing or unreadable.")
        warnings.append("trials.csv is missing or unreadable.")
    else:
        counts = trial_counts(rows)
        shown = counts["shown"]
        pct = f" ({100 * counts['correct'] / counts['answered']:.0f} % of answered)" if counts["answered"] else ""
        out.append(f"Flashes shown: {shown}; answered: {counts['answered']}; correct: {counts['correct']}{pct}")
        out.append(f"Missing answers: {counts['missing']}")
        reaction = counts["median_reaction_ms"]
        out.append(f"Average (median) reaction time: {reaction} ms" if reaction is not None else "Average (median) reaction time: not measured")
        planned = SACCADE_FIXED_TRIALS_TEST if mode == "test" else SACCADE_FIXED_TRIALS_REAL
        if shown == 0:
            warnings.append("No flashes were shown: nothing to analyse.")
        elif mode in ("test", "real") and shown < MIN_TRIAL_SHARE * planned:
            warnings.append(f"Very few trials: {shown} flashes shown, a full {mode} run has about {planned}.")
        if shown and counts["tracking_lost"] / shown > MAX_TRACKING_LOST_SHARE:
            warnings.append(f"Tracking was lost near {counts['tracking_lost']} of {shown} flashes ({100 * counts['tracking_lost'] / shown:.0f} %): those flashes are excluded.")
        if shown and counts["missing"] / shown > MAX_UNANSWERED_SHARE:
            warnings.append(f"{counts['missing']} of {shown} flashes got no answer: was the participant pressing UP/DOWN in time?")
    out.append("")

    lines, quality_warnings = gaze_quality_lines(meta.get("gaze_quality"))
    out += lines
    warnings += quality_warnings
    recording, recording_warning = _recording_line(meta.get("glasses_video"))
    out.append(recording)
    if recording_warning:
        warnings.append(recording_warning)
    ignored = _ignored_space_total(meta)
    out.append(f"Ignored SPACE presses: {ignored}" + (" (pressed while nothing could start; harmless)" if ignored else ""))
    out.append("")

    out.append("WARNINGS" if warnings else "No warnings.")
    out += [f"- {w}" for w in warnings]
    path = _command_folder(folder)
    out += [
        "",
        "NEXT STEPS",
        "1. Export the scene video (mp4) of this recording from the phone / Pupil Cloud to the PC.",
        f"2. .\\scripts\\py.ps1 -m src.experiment.import_glasses_video {path} C:\\path\\to\\scene_video.mp4",
        f"3. .\\scripts\\py.ps1 -m src.experiment.analyze_session {path}",
    ]
    return "\n".join(out) + "\n"


def write_run_summary(run_folder) -> str:
    """Builds the summary, saves it as RUN_SUMMARY.txt in the run folder and returns it."""
    text = build_run_summary(run_folder)
    (Path(run_folder) / SUMMARY_FILE).write_text(text, encoding="utf-8")
    return text


def finish_run_summary(run_folder) -> str | None:
    """End-of-run entry point: writes and prints the summary; on any failure prints one line and returns None."""
    try:
        text = write_run_summary(run_folder)
        print("\n" + text)
        return text
    except Exception as exc:  # noqa: BLE001 - the end of a run must never crash because of the summary
        print(f"Could not write the run summary ({exc!r}); the data itself is saved.", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the summary of a run (the newest by default).")
    parser.add_argument("session", nargs="?", help="session timestamp, run folder name, participant/folder or path; default: the newest run")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    ref = session_files.find_session(args.data_dir, args.session) if args.session else session_files.newest_session(args.data_dir)
    if ref is None:
        print(f"No session {args.session!r} found in {args.data_dir}." if args.session else f"No sessions found in {args.data_dir}.", file=sys.stderr)
        return 1
    if ref.folder is None:
        print("This is an old loose-file session (no run folder); summarising its CSV only.\n")
        folder_text = _legacy_summary(ref)
        print(folder_text)
        return 0
    print(build_run_summary(ref.folder))
    return 0


def _legacy_summary(ref: session_files.SessionRef) -> str:
    """Old loose-file layout: a trials-only summary built in a temporary view (no run folder to read from)."""
    rows = _load_rows(ref.trials_path)
    if rows is None:
        return "trials file unreadable."
    counts = trial_counts(rows)
    return f"Flashes shown: {counts['shown']}; answered: {counts['answered']}; correct: {counts['correct']}; missing answers: {counts['missing']}"


if __name__ == "__main__":
    sys.exit(main())
