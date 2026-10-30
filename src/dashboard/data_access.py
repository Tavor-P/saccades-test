"""Reads/writes the experiment's files under data/ for the dashboard: per-participant run folders (trials.csv + session.json)
and the old loose results_<ts>.csv / _meta.json files, plus settings.json. A session is addressed by its unix start time
in both layouts. Parsers use .get(...) so files written before a column existed still load (missing column -> None)."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from include.experiment.types import TrialResult
from src.experiment import session_files
from src.experiment.logger import read_trials

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GRAPH_CACHE_DIR = DATA_DIR / "dashboard_cache"


def _ref(timestamp: str) -> session_files.SessionRef:
    """The session with this key; a not-found one resolves to its old-layout paths (which don't exist), so callers'
    `.exists()` checks keep working."""
    return session_files.find_session(DATA_DIR, timestamp) or session_files._legacy_ref(DATA_DIR, timestamp)


def csv_path_for(timestamp: str) -> Path:
    return _ref(timestamp).trials_path


def meta_path_for(timestamp: str) -> Path:
    return _ref(timestamp).meta_path


def graph_cache_path_for(timestamp: str) -> Path:
    return GRAPH_CACHE_DIR / f"{timestamp}.png"


def load_session_trials(timestamp: str) -> list[TrialResult]:
    """Reparses a session's logged CSV back into TrialResult objects, so it
    can be fed straight into results_graph.build_comparison_graph - the exact
    same graph the experiment itself shows at the end of a run. Reads both file layouts."""
    return read_trials(csv_path_for(timestamp))


def _flatten(raw: dict) -> dict:
    """The dashboard's flat view (participant_id, started_at, test_mode, calibration, name, gender, age) of either an old
    _meta.json (already flat) or a new session.json (sectioned)."""
    if "run" not in raw and "participant" not in raw:
        return dict(raw)
    participant, run, settings = raw.get("participant") or {}, raw.get("run") or {}, raw.get("settings") or {}
    mode = run.get("mode")
    return {
        **raw,
        "participant_id": participant.get("id"),
        "name": participant.get("name"),
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
    path = meta_path_for(timestamp)
    meta = _flatten(json.loads(path.read_text())) if path.exists() else {}
    meta.setdefault("participant_id", None)
    meta.setdefault("started_at", None)
    meta.setdefault("test_mode", None)
    meta.setdefault("calibration", None)
    meta.setdefault("name", None)
    meta.setdefault("gender", None)
    meta.setdefault("age", None)
    return meta


def save_participant_info(timestamp: str, name: str, gender: str, age: str) -> None:
    ref = _ref(timestamp)
    name, gender = name.strip() or None, gender.strip() or None
    age_value = int(age) if age.strip() else None
    raw = json.loads(ref.meta_path.read_text()) if ref.meta_path.exists() else {}
    if ref.is_legacy:
        raw.update(name=name, gender=gender, age=age_value)
    else:
        raw.setdefault("participant", {}).update(name=name, gender=gender, age=age_value)
    ref.meta_path.write_text(json.dumps(raw, indent=2))


def delete_session(timestamp: str) -> None:
    """Removes a session permanently (no trash/undo): an old session's CSV and metadata sidecar; a
    new-layout session's WHOLE run folder (trials, session.json, gaze trace, analysis and the glasses video), and its
    participant folder if that is now empty. Also the cached graph."""
    ref = _ref(timestamp)
    if ref.is_legacy:
        ref.trials_path.unlink(missing_ok=True)
        ref.meta_path.unlink(missing_ok=True)
    else:
        shutil.rmtree(ref.folder, ignore_errors=True)
        try:
            ref.folder.parent.rmdir()  # only succeeds when empty
        except OSError:
            pass
    graph_cache_path_for(timestamp).unlink(missing_ok=True)


@dataclass
class SessionSummary:
    timestamp: str
    metadata: dict
    trial_counts: dict[str, int] = field(default_factory=dict)


def discover_sessions() -> list[SessionSummary]:
    """All logged sessions (both layouts), newest first."""
    sessions = []
    for ref in session_files.list_sessions(DATA_DIR):
        trial_counts: dict[str, int] = {}
        for trial in read_trials(ref.trials_path):
            trial_counts[trial.phase] = trial_counts.get(trial.phase, 0) + 1
        sessions.append(SessionSummary(timestamp=ref.key, metadata=load_metadata(ref.key), trial_counts=trial_counts))
    return sessions
