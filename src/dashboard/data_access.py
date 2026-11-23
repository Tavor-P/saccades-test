"""Reads/writes the experiment's files under data/ for the dashboard: per-participant run folders (trials.csv + session.json)
and the old loose results_<ts>.csv / _meta.json files, plus settings.json. A session is addressed by its unix start time
in both layouts. Parsers use .get(...) so files written before a column existed still load (missing column -> None)."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.experiment import session_files
from src.experiment.logger import read_trials

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _ref(timestamp: str) -> session_files.SessionRef:
    """The session with this key; a not-found one resolves to its old-layout paths (which don't exist), so callers'
    `.exists()` checks keep working."""
    if not (timestamp.isascii() and timestamp.isdigit()):
        # Only session keys (unix start times) address a session here, never a path: '..', separators or absolute paths
        # would let find_session resolve (and delete_session remove) a folder outside data/.
        return session_files._legacy_ref(DATA_DIR, "0")
    ref = session_files.find_session(DATA_DIR, timestamp)
    if ref is not None and ref.folder is not None and ref.folder.resolve().parent.parent != DATA_DIR.resolve():
        return session_files._legacy_ref(DATA_DIR, "0")
    return ref or session_files._legacy_ref(DATA_DIR, timestamp)


def csv_path_for(timestamp: str) -> Path:
    return _ref(timestamp).trials_path


def meta_path_for(timestamp: str) -> Path:
    return _ref(timestamp).meta_path


def _flatten(raw: dict) -> dict:
    """The dashboard's flat view (participant_id, started_at, test_mode, calibration) of either an old
    _meta.json (already flat) or a new session.json (sectioned)."""
    if "run" not in raw and "participant" not in raw:
        return dict(raw)
    participant, run, settings = raw.get("participant") or {}, raw.get("run") or {}, raw.get("settings") or {}
    mode = run.get("mode")
    return {
        **raw,
        "participant_id": participant.get("id"),
        "started_at": run.get("started_at"),
        "test_mode": None if mode is None else mode == "test",
        "calibration": settings.get("calibration"),
    }


def load_metadata(timestamp: str) -> dict:
    """Sessions logged before the metadata sidecar existed have no meta file
    at all - treat that the same as one with every field blank, rather than
    excluding those older runs from the dashboard."""
    path = meta_path_for(timestamp)
    try:
        meta = _flatten(json.loads(path.read_text())) if path.exists() else {}
    except (OSError, ValueError):  # a damaged file (e.g. an aborted write) must not take the whole listing down
        meta = {}
    meta.setdefault("participant_id", None)
    meta.setdefault("started_at", None)
    meta.setdefault("test_mode", None)
    meta.setdefault("calibration", None)
    return meta


def delete_session(timestamp: str) -> None:
    """Removes a session permanently (no trash/undo): an old session's CSV and metadata sidecar; a
    new-layout session's WHOLE run folder (trials, session.json, gaze trace, analysis and the glasses video). The
    participant's pooled analysis (data/<participant>/analysis/) included that run, so it is removed too (the results
    page re-runs it on the next open), and then the participant folder if nothing else is left in it."""
    ref = _ref(timestamp)
    if ref.is_legacy:
        ref.trials_path.unlink(missing_ok=True)
        ref.meta_path.unlink(missing_ok=True)
    else:
        shutil.rmtree(ref.folder, ignore_errors=True)
        shutil.rmtree(ref.folder.parent / session_files.ANALYSIS_DIR, ignore_errors=True)
        try:
            ref.folder.parent.rmdir()  # only succeeds when empty
        except OSError:
            pass


@dataclass
class SessionSummary:
    timestamp: str
    metadata: dict
    trial_counts: dict[str, int] = field(default_factory=dict)
    participant: str | None = None  # the participant folder name (new layout); None for an old loose-file session


def discover_sessions() -> list[SessionSummary]:
    """All logged sessions (both layouts), newest first."""
    sessions = []
    for ref in session_files.list_sessions(DATA_DIR):
        trial_counts: dict[str, int] = {}
        try:
            for trial in read_trials(ref.trials_path):
                trial_counts[trial.phase] = trial_counts.get(trial.phase, 0) + 1
        except (OSError, ValueError, KeyError):  # a damaged trials file: list the session without counts
            pass
        sessions.append(SessionSummary(timestamp=ref.key, metadata=load_metadata(ref.key), trial_counts=trial_counts,
                                       participant=ref.participant))
    return sessions
