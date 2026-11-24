"""Reads/writes the experiment's files under data/ for the dashboard: per-participant run folders (trials.csv + session.json)
plus settings.json. A session is addressed by its unix start time. Parsers use .get(...) so files written before a column
existed still load (missing column -> None)."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.experiment import session_files
from src.experiment.logger import read_trials
from src.experiment.session_files import DATA_DIR  # patched per module in tests, so always used by this bare name

class DamagedSessionFile(Exception):
    """A session.json exists but cannot be read; the message is plain text for the user."""


_NO_SUCH_SESSION = "no_such_session"  # folder name of the (never existing) paths handed out for an unknown key


def _ref(timestamp: str) -> session_files.SessionRef | None:
    """The session with this key, or None (not found, or not a session key at all)."""
    if not (timestamp.isascii() and timestamp.isdigit()):
        # Only session keys (unix start times) address a session here, never a path: '..', separators or absolute paths
        # would let find_session resolve (and delete_session remove) a folder outside data/.
        return None
    ref = session_files.find_session(DATA_DIR, timestamp)
    if ref is not None and ref.folder.resolve().parent.parent != DATA_DIR.resolve():
        return None
    return ref


def csv_path_for(timestamp: str) -> Path:
    """The session's trials.csv; for an unknown key a path that does not exist, so callers' `.exists()` checks work."""
    ref = _ref(timestamp)
    return ref.trials_path if ref else DATA_DIR / _NO_SUCH_SESSION / session_files.TRIALS_FILE


def meta_path_for(timestamp: str) -> Path:
    ref = _ref(timestamp)
    return ref.meta_path if ref else DATA_DIR / _NO_SUCH_SESSION / session_files.SESSION_FILE


def _flatten(raw: dict) -> dict:
    """The dashboard's flat view (participant_id, started_at, test_mode, calibration, gender, age) of a sectioned session.json."""
    participant, run, settings = raw.get("participant") or {}, raw.get("run") or {}, raw.get("settings") or {}
    mode = run.get("mode")
    return {
        **raw,
        "participant_id": participant.get("id"),
        "gender": participant.get("gender"),
        "age": participant.get("age"),
        "started_at": run.get("started_at"),
        "test_mode": None if mode is None else mode == "test",
        "calibration": settings.get("calibration"),
    }


def load_metadata(timestamp: str) -> dict:
    """Sessions logged before the metadata sidecar existed have no meta file
    at all - treat that the same as one with every field blank, rather than
    excluding those older runs from the dashboard."""
    # A missing or damaged file (e.g. an aborted write) must not take the whole listing down: read_json gives None, which reads as blank.
    meta = _flatten(session_files.read_json(meta_path_for(timestamp)) or {})
    meta.setdefault("participant_id", None)
    meta.setdefault("started_at", None)
    meta.setdefault("test_mode", None)
    meta.setdefault("calibration", None)
    meta.setdefault("gender", None)
    meta.setdefault("age", None)
    return meta


def save_participant_info(timestamp: str, gender: str, age: str) -> None:
    ref = _ref(timestamp)
    if ref is None:
        raise FileNotFoundError(f"no session {timestamp!r}")
    gender = gender.strip() or None
    age_value = int(age) if age.strip() else None
    raw = session_files.read_json(ref.meta_path) if ref.meta_path.exists() else {}
    if raw is None:  # never overwrite a damaged session.json with just these two fields
        raise DamagedSessionFile(f"{ref.label}/{session_files.SESSION_FILE} is damaged, so the participant details were not saved")
    raw.setdefault("participant", {}).update(gender=gender, age=age_value)
    ref.meta_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def delete_session(timestamp: str) -> None:
    """Removes a session permanently (no trash/undo): its WHOLE run folder (trials, session.json, gaze trace, analysis and
    the glasses video). The participant's pooled analysis (data/<participant>/analysis/) included that run, so it is
    removed too (the results page re-runs it on the next open), and then the participant folder if nothing else is left
    in it. An unknown key (or anything that is not a session key) deletes nothing."""
    ref = _ref(timestamp)
    if ref is None:
        return
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
    participant: str | None = None  # the participant folder name


def discover_sessions() -> list[SessionSummary]:
    """All logged sessions, newest first."""
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
