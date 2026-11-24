"""Where a session's files live on disk (no PsychoPy, no logging).

A run is data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/ holding trials.csv, session.json, gaze_trace.csv, analysis/,
glasses/. A session is identified by its unix start time (`key`, a digit string), stored in
session.json["run"]["unix_timestamp"] (the folder's own timestamp is the fallback). The loose data/results_<ts>.csv
layout of the very first versions is no longer read (none were left on disk when it was removed).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"  # git-ignored runtime output; each module that reads or writes under it imports this name (tests patch it per module)

ANONYMOUS = "anonymous"
TRIALS_FILE = "trials.csv"
SESSION_FILE = "session.json"
GAZE_TRACE_FILE = "gaze_trace.csv"
HEAD_MOTION_FILE = "head_motion.csv"  # Neon IMU head speed/drift, downsampled (see src/eye_tracking/head_monitor.py)
ANALYSIS_DIR = "analysis"
GLASSES_DIR = "glasses"

_FOLDER_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_test)?(_\d+)?$")
_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_MAX_ID_LENGTH = 60


def sanitize_participant_id(participant_id: str | None) -> str:
    """A participant id as a safe Windows folder name: invalid characters and runs of whitespace become '_', trailing
    dots/spaces go, reserved device names get a '_' suffix, blank means `anonymous`."""
    text = _INVALID_CHARS_RE.sub("_", (participant_id or "").strip())
    text = re.sub(r"\s+", "_", text).strip(". _")[:_MAX_ID_LENGTH].rstrip(". _")
    if not text:
        return ANONYMOUS
    if text.split(".")[0].lower() in _RESERVED_NAMES:
        text += "_"
    return text


def session_folder_name(unix_seconds: float, test_mode: bool | None = None) -> str:
    """`YYYY-MM-DD_HH-MM-SS`, plus `_test` for a test-mode run (local time, like the participant's wall clock)."""
    return datetime.fromtimestamp(unix_seconds).strftime("%Y-%m-%d_%H-%M-%S") + ("_test" if test_mode else "")


def create_session_folder(data_dir: Path, participant_id: str | None, unix_seconds: float, test_mode: bool | None = None) -> Path:
    """Makes data/<participant>/<timestamp>[_test]/ (a numeric suffix if one with that name already exists) and its
    analysis/ and glasses/ subfolders; returns it."""
    parent = Path(data_dir) / sanitize_participant_id(participant_id)
    parent.mkdir(parents=True, exist_ok=True)
    base = session_folder_name(unix_seconds, test_mode)
    folder, n = parent / base, 1
    while True:
        try:
            folder.mkdir()
            break
        except FileExistsError:
            n += 1
            folder = parent / f"{base}_{n}"
    (folder / ANALYSIS_DIR).mkdir()
    (folder / GLASSES_DIR).mkdir()
    return folder


@dataclass(frozen=True)
class SessionRef:
    """One run folder on disk."""

    key: str  # unix start time as a digit string
    folder: Path
    trials_path: Path
    meta_path: Path  # session.json (may not exist)
    gaze_trace_path: Path
    analysis_dir: Path  # where analyze_session writes by default
    glasses_dir: Path
    participant: str  # the participant folder name

    @property
    def label(self) -> str:
        return f"{self.folder.parent.name}/{self.folder.name}"


def _ref_from_folder(folder: Path) -> SessionRef | None:
    trials, meta = folder / TRIALS_FILE, folder / SESSION_FILE
    if not trials.exists():
        return None
    key = None
    try:
        key = str(int(json.loads(meta.read_text())["run"]["unix_timestamp"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if key is None:  # session.json lost or damaged: fall back to the folder's own timestamp
        stamp = "_".join(folder.name.split("_")[:2])
        try:
            key = str(int(datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S").timestamp()))
        except ValueError:
            return None
    return SessionRef(
        key=key,
        folder=folder,
        trials_path=trials,
        meta_path=meta,
        gaze_trace_path=folder / GAZE_TRACE_FILE,
        analysis_dir=folder / ANALYSIS_DIR,
        glasses_dir=folder / GLASSES_DIR,
        participant=folder.parent.name,
    )


def list_sessions(data_dir: Path) -> list[SessionRef]:
    """Every run folder under data_dir, newest first."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    refs: list[SessionRef] = []
    for participant in data_dir.iterdir():
        if not participant.is_dir():
            continue
        for folder in participant.iterdir():
            if folder.is_dir() and (folder / TRIALS_FILE).exists():
                ref = _ref_from_folder(folder)
                if ref is not None:
                    refs.append(ref)
    refs.sort(key=lambda r: (int(r.key), r.label), reverse=True)
    return refs


def find_session(data_dir: Path, identifier: str | Path) -> SessionRef | None:
    """Resolves what a user typed: a unix timestamp, a run folder name (`2026-10-25_14-03-11[_test]`),
    `participant/folder`, or a path to a run folder / its trials.csv."""
    data_dir = Path(data_dir)
    text = str(identifier).strip()
    as_path = Path(text)
    if as_path.exists():
        folder = as_path if as_path.is_dir() else as_path.parent
        if (folder / TRIALS_FILE).exists():
            return _ref_from_folder(folder)
    parts = text.replace("\\", "/").strip("/").split("/")
    if len(parts) == 2 and _FOLDER_NAME_RE.match(parts[1]):
        direct = data_dir / parts[0] / parts[1]
        if (direct / TRIALS_FILE).exists():
            return _ref_from_folder(direct)
    for ref in list_sessions(data_dir):
        if ref.key == text or ref.folder.name == text:
            return ref
    return None


def newest_session(data_dir: Path, accept=lambda ref: True) -> SessionRef | None:
    """The newest session (by start time) that `accept` allows."""
    for ref in list_sessions(data_dir):
        if accept(ref):
            return ref
    return None
