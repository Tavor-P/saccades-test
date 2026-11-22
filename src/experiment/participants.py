"""Group labels for choosing comparison sets (`-m src.experiment.participants set P01 TBI` / `list`).

One small file, data/participants.json, maps a participant ID (the sanitized folder name under data/) to
{"group": "<free text, e.g. TBI or control>", "note": "<free text>"}. The labels exist only to pick who is compared with
whom (see compare.py); there are deliberately NO clinical fields here (diagnosis details, severity, medication, ... need
IRB approval first).

Writes are atomic (temp file in the same folder, then os.replace). A missing file means "no labels"; a damaged file is
read as "no labels" with a warning, and the next `set_label` first copies it aside (participants.json.damaged-<stamp>) so
nothing typed by hand is lost. Group names are stored as typed and matched case-insensitively ("tbi" finds "TBI").
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from src.experiment import session_files
from src.experiment.analyze_session import DATA_DIR

LABELS_FILE = "participants.json"


def labels_path(data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / LABELS_FILE


def _clean(entry) -> dict | None:
    if not isinstance(entry, dict):
        return None
    group = entry.get("group")
    note = entry.get("note")
    return {"group": "" if group is None else str(group).strip(), "note": "" if note is None else str(note)}


def _read(data_dir: Path) -> tuple[dict, bool]:
    """(labels, damaged). Never raises."""
    path = labels_path(data_dir)
    if not path.exists():
        return {}, False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, True
    if not isinstance(raw, dict):
        return {}, True
    labels = {str(pid): clean for pid, entry in raw.items() if (clean := _clean(entry)) is not None}
    return labels, len(labels) != len(raw)


def list_labels(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """{participant id: {"group": ..., "note": ...}}; {} when the file is missing or damaged (warns on stderr)."""
    labels, damaged = _read(data_dir)
    if damaged:
        print(f"Warning: {labels_path(data_dir)} is damaged; unreadable entries are ignored.", file=sys.stderr)
    return labels


def get_label(participant_id: str, data_dir: Path = DATA_DIR) -> dict | None:
    """The participant's {"group", "note"}, or None when unlabelled."""
    return list_labels(data_dir).get(session_files.sanitize_participant_id(participant_id))


def _write_atomic(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".participants-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(labels.items())), f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def set_label(participant_id: str, group: str | None, note: str | None = None, data_dir: Path = DATA_DIR) -> dict:
    """Sets (or with group=None and note=None, removes) one participant's label; returns the stored entry ({} if removed).
    note=None keeps the existing note."""
    pid = session_files.sanitize_participant_id(participant_id)
    path = labels_path(data_dir)
    labels, damaged = _read(data_dir)
    if damaged:
        backup = path.with_name(f"{path.name}.damaged-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"Warning: {path} was damaged; a copy is kept as {backup}.", file=sys.stderr)
    if group is None and note is None:
        labels.pop(pid, None)
        _write_atomic(path, labels)
        return {}
    old = labels.get(pid, {"group": "", "note": ""})
    entry = {"group": old["group"] if group is None else str(group).strip(), "note": old["note"] if note is None else str(note)}
    labels[pid] = entry
    _write_atomic(path, labels)
    return entry


def _same_group(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def participants_in_group(group: str, data_dir: Path = DATA_DIR) -> list[str]:
    """IDs labelled with this group (case-insensitive), sorted. Labelled people without data are included (the
    comparison lists them as excluded, with the reason)."""
    return sorted(pid for pid, entry in list_labels(data_dir).items() if _same_group(entry["group"], group))


def all_participants(data_dir: Path = DATA_DIR) -> list[str]:
    """Every participant with at least one run folder under data_dir, sorted (old loose result files have no participant)."""
    return sorted({ref.participant for ref in session_files.list_sessions(data_dir) if ref.folder is not None and ref.participant})


def resolve_set(spec: str | list[str], data_dir: Path = DATA_DIR) -> list[str]:
    """Turns "P01, P02, group:TBI, all" (or a list of such tokens) into participant IDs, de-duplicated, in order."""
    tokens = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for token in (t.strip() for t in tokens):
        if not token:
            continue
        if token.lower().startswith("group:"):
            ids = participants_in_group(token.split(":", 1)[1], data_dir)
            if not ids:
                print(f"Warning: nobody is labelled with group '{token.split(':', 1)[1].strip()}'.", file=sys.stderr)
        elif token.lower() in ("all", "*"):
            ids = all_participants(data_dir)
        else:
            ids = [session_files.sanitize_participant_id(token)]
        out += [i for i in ids if i not in out]
    return out


def _has_all(spec: str | list[str]) -> bool:
    tokens = spec.split(",") if isinstance(spec, str) else list(spec)
    return any(t.strip().lower() in ("all", "*") for t in tokens)


def everyone_else(spec_a: str | list[str], a: list[str], spec_b: str | list[str], b: list[str]) -> tuple[list[str], list[str]]:
    """The one "all" rule shared by the compare CLI and the dashboard: when exactly one side says "all", that side means everyone
    else (the other side's people are dropped from it), so "P01 vs all" compares P01 with the rest. Otherwise the sets are
    returned as they are, and a person picked on both sides stays an error for the caller to report."""
    if _has_all(spec_b) and not _has_all(spec_a):
        return a, [x for x in b if x not in a]
    if _has_all(spec_a) and not _has_all(spec_b):
        return [x for x in a if x not in b], b
    return a, b


# -- CLI -----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group labels used to pick comparison sets (data/participants.json).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    after = argparse.ArgumentParser(add_help=False)  # --data-dir also accepted after the subcommand, like every other CLI here
    after.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    p_set = sub.add_parser("set", parents=[after], help="label a participant: set P01 TBI [--note text]")
    p_set.add_argument("participant")
    p_set.add_argument("group", help='free text, e.g. TBI or control (use quotes for spaces)')
    p_set.add_argument("--note", default=None)
    p_rm = sub.add_parser("remove", parents=[after], help="remove a participant's label")
    p_rm.add_argument("participant")
    sub.add_parser("list", parents=[after], help="show every label and the participants with data but no label")
    args = parser.parse_args(argv)

    if args.command == "set":
        entry = set_label(args.participant, args.group, args.note, args.data_dir)
        print(f"{session_files.sanitize_participant_id(args.participant)}: group '{entry['group']}'" + (f", note '{entry['note']}'" if entry["note"] else ""))
        return 0
    if args.command == "remove":
        set_label(args.participant, None, None, args.data_dir)
        print(f"Label removed for {session_files.sanitize_participant_id(args.participant)}.")
        return 0
    labels = list_labels(args.data_dir)
    with_data = set(all_participants(args.data_dir))
    print(f"Labels in {labels_path(args.data_dir)}:")
    if not labels:
        print("  (none)")
    for pid, entry in labels.items():
        tail = "" if pid in with_data else "   (no run folders)"
        print(f"  {pid:<20} {entry['group'] or '(no group)':<15} {entry['note']}{tail}")
    unlabelled = sorted(with_data - set(labels))
    if unlabelled:
        print("Participants with data but no label: " + ", ".join(unlabelled))
    return 0


if __name__ == "__main__":
    sys.exit(main())
