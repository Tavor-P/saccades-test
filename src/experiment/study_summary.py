"""One row per participant for the whole study (`-m src.experiment.study_summary [--data-dir D] [--include-test]`).

Writes data/study_summary.csv (plain columns, ready for SPSS / R / Excel; see COLUMNS) and data/study_summary.txt (a short
group-by-group overview). Each participant's numbers are their pooled `analyze_participant` result, (re)analysed first
when the saved one is missing or older than their runs (compare.load_person). Groups come from data/participants.json
(src/experiment/participants.py). Every group overview is exploratory at pilot sizes.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from include.eye_tracking.constants import NEON_GAP_THRESHOLD_MS
from include.experiment.constants import TRACKING_LOSS_WINDOW_MS
from src.experiment.analyze_session import DATA_DIR
from src.experiment.compare import EXPLORATORY_BELOW, MIN_FLASHES_PER_CONDITION, load_person
from src.experiment.participants import all_participants, list_labels

CSV_FILE = "study_summary.csv"
TXT_FILE = "study_summary.txt"
COLUMNS = (
    "participant", "group", "note", "status", "runs_used", "runs_excluded", "index", "ci_low", "ci_high", "method",
    "n_saccade", "n_fixation", "n_flashes_total", "correct_saccade", "correct_fixation", "contrasts_pct",
    "screen_height_cm", "viewing_distance_cm", "gaze_delay_ms_median", "longest_gap_ms_max", "gaps_over_threshold_total", "flags",
)
GAZE_DELAY_FLAG_MS = 50.0  # median gaze delay above this: an unusually high delay (live classification only; post-hoc analysis is unaffected)
LONG_GAP_FLAG_MS = TRACKING_LOSS_WINDOW_MS  # a tracking gap at least as long as the window that already excludes nearby flashes


def _flags(person, delays, gaps) -> list[str]:
    flags = []
    if person.included and (person.n_saccade < MIN_FLASHES_PER_CONDITION or person.n_fixation < MIN_FLASHES_PER_CONDITION):
        flags.append(f"fewer than {MIN_FLASHES_PER_CONDITION} flashes in a condition")
    if person.method == "accuracy":
        flags.append("accuracy fallback (not the threshold ratio)")
    if person.included and person.ci is None:
        flags.append("no CI")
    if delays and float(np.median(delays)) > GAZE_DELAY_FLAG_MS:
        flags.append(f"gaze delay > {GAZE_DELAY_FLAG_MS:g} ms (known issue; post-hoc analysis unaffected)")
    if gaps and max(gaps) >= LONG_GAP_FLAG_MS:
        flags.append(f"tracking gap >= {LONG_GAP_FLAG_MS:g} ms")
    if person.runs_excluded:
        flags.append(f"{person.runs_excluded} run(s) not pooled")
    return flags


def _fmt(v, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def build_rows(data_dir: Path = DATA_DIR, include_test: bool = False) -> list[dict]:
    """One dict per participant (data folders plus labelled people without data), keys = COLUMNS."""
    data_dir = Path(data_dir)
    labels = list_labels(data_dir)
    rows = []
    for pid in sorted(set(all_participants(data_dir)) | set(labels)):
        person = load_person(pid, data_dir, include_test)
        label = labels.get(pid) or {}
        records = person.run_records
        delays = [r["gaze_latency_ms_median"] for r in records if r.get("gaze_latency_ms_median") is not None]
        gaps = [r["longest_gap_ms"] for r in records if r.get("longest_gap_ms") is not None]
        n_gaps = [r["gaps_over_threshold"] for r in records if r.get("gaps_over_threshold") is not None]
        geometry = person.geometry or {}
        rows.append({
            "participant": pid, "group": label.get("group") or "", "note": label.get("note") or "",
            "status": "ok" if person.included else person.reason,
            "runs_used": person.runs_used, "runs_excluded": person.runs_excluded,
            "index": _fmt(person.index), "ci_low": _fmt(person.ci[0] if person.ci else None), "ci_high": _fmt(person.ci[1] if person.ci else None),
            "method": person.method or "", "n_saccade": person.n_saccade, "n_fixation": person.n_fixation, "n_flashes_total": person.n_total,
            "correct_saccade": person.correct_saccade, "correct_fixation": person.correct_fixation,
            "contrasts_pct": "/".join(f"{c:g}" for c in person.contrasts_pct or []),
            "screen_height_cm": "" if geometry.get("screen_height_cm") is None else f"{geometry['screen_height_cm']:g}",
            "viewing_distance_cm": "" if geometry.get("viewing_distance_cm") is None else f"{geometry['viewing_distance_cm']:g}",
            "gaze_delay_ms_median": _fmt(float(np.median(delays)) if delays else None, 1),
            "longest_gap_ms_max": _fmt(max(gaps) if gaps else None, 1),
            "gaps_over_threshold_total": sum(n_gaps) if n_gaps else "",
            "flags": "; ".join(_flags(person, delays, gaps)),
        })
    return rows


def format_overview(rows: list[dict]) -> str:
    lines = ["Study summary: one suppression index per participant (pooled over their compatible runs)", ""]
    groups: dict[str, list[dict]] = {}
    names: dict[str, str] = {}  # case-insensitive like participants_in_group ("Control" and "control" are one group), shown as first typed
    for r in rows:
        name = (r["group"] or "").strip() or "(no group)"  # a hand-edited participants.json may hold "group": null
        groups.setdefault(names.setdefault(name.casefold(), name), []).append(r)
    for name, members in sorted(groups.items(), key=lambda item: item[0].casefold()):
        usable = [r for r in members if r["status"] == "ok"]
        lines.append(f"Group {name}: {len(members)} {'person' if len(members) == 1 else 'people'}, {len(usable)} with an index")
        for method in ("psychometric", "accuracy"):
            values = [float(r["index"]) for r in usable if r["method"] == method]
            if values:
                what = "suppression index" if method == "psychometric" else "accuracy-based index (fallback)"
                lines.append(f"  {what}: median {np.median(values):.2f}, range {min(values):.2f} to {max(values):.2f} (n={len(values)})")
        for r in members:
            text = f"{float(r['index']):.2f} ({r['method']})" if r["status"] == "ok" else f"no index: {r['status']}"
            lines.append(f"    {r['participant']:<16} {text}" + (f"  [{r['flags']}]" if r["flags"] else ""))
        lines.append("")
    lines += [
        f"EXPLORATORY: groups this small (fewer than {EXPLORATORY_BELOW} people) are a feasibility pilot, not a group result.",
        "Psychometric indices: 0 = no suppression, 1 = never seen during saccades. Accuracy-based indices are on another scale; never mix them.",
        f"Flags: gaze delay > {GAZE_DELAY_FLAG_MS:g} ms is the known open delay problem (affects only the live classification); "
        f"tracking gap >= {LONG_GAP_FLAG_MS:g} ms; gaps_over_threshold counts gaps > {NEON_GAP_THRESHOLD_MS:g} ms.",
        "To test groups against each other: launchers\\compare.bat (or -m src.experiment.compare group:TBI group:control).",
    ]
    return "\n".join(lines)


def write_study_summary(data_dir: Path = DATA_DIR, include_test: bool = False, out_dir: Path | None = None) -> dict:
    """Builds the rows and writes study_summary.csv / .txt (default into data_dir); returns {"csv", "txt", "rows", "overview"}."""
    rows = build_rows(data_dir, include_test)
    out = Path(out_dir) if out_dir is not None else Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, txt_path = out / CSV_FILE, out / TXT_FILE
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    overview = format_overview(rows)
    txt_path.write_text(overview + "\n", encoding="utf-8")
    return {"csv": csv_path, "txt": txt_path, "rows": rows, "overview": overview}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One row per participant: index, CI, method, flashes, gaze-quality flags.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--include-test", action="store_true", help="also use test-mode runs")
    args = parser.parse_args(argv)
    out = write_study_summary(args.data_dir, args.include_test)
    print(out["overview"])
    print(f"\nspreadsheet: {out['csv']}\noverview: {out['txt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
