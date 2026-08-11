import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from include.experiment.types import Orientation, Target, TrialResult

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GRAPH_CACHE_DIR = DATA_DIR / "dashboard_cache"

_SESSION_CSV_RE = re.compile(r"^results_(\d+)\.csv$")


def _parse_optional_float(value: str) -> float | None:
    return float(value) if value else None


def _parse_bool(value: str) -> bool:
    return value == "True"


def _parse_optional_bool(value: str) -> bool | None:
    return _parse_bool(value) if value else None


def _parse_target(value: str) -> Target | None:
    return Target(value) if value else None


def _parse_orientation(value: str) -> Orientation | None:
    return Orientation(value) if value else None


def csv_path_for(timestamp: str) -> Path:
    return DATA_DIR / f"results_{timestamp}.csv"


def meta_path_for(timestamp: str) -> Path:
    return DATA_DIR / f"results_{timestamp}_meta.json"


def graph_cache_path_for(timestamp: str) -> Path:
    return GRAPH_CACHE_DIR / f"{timestamp}.png"


def load_session_trials(timestamp: str) -> list[TrialResult]:
    """Reparses a session's logged CSV back into TrialResult objects, so it
    can be fed straight into results_graph.build_comparison_graph - the exact
    same graph the experiment itself shows at the end of a run."""
    trials = []
    with csv_path_for(timestamp).open(newline="") as f:
        for row in csv.DictReader(f):
            trials.append(
                TrialResult(
                    index=int(row["trial_index"]),
                    phase=row["phase"],
                    source=_parse_target(row["source"]),
                    target=_parse_target(row["target"]),
                    saccade_duration_ms=_parse_optional_float(row["saccade_duration_ms"]),
                    grating_shown=_parse_bool(row["grating_shown"]),
                    contrast=_parse_optional_float(row["contrast"]),
                    # .get(): sessions logged before orientation discrimination existed have no
                    # orientation/response_orientation columns at all - treat that as "unknown".
                    orientation=_parse_orientation(row.get("orientation", "")),
                    responded=_parse_bool(row["responded"]),
                    response_orientation=_parse_orientation(row.get("response_orientation", "")),
                    response_time_ms=_parse_optional_float(row["response_time_ms"]),
                    outcome=row["outcome"],
                    # .get(): sessions logged before onset-latency instrumentation existed
                    # have no reaction_latency_ms/onset_detection_lag_ms columns at all.
                    reaction_latency_ms=_parse_optional_float(row.get("reaction_latency_ms", "")),
                    onset_detection_lag_ms=_parse_optional_float(row.get("onset_detection_lag_ms", "")),
                    # .get(): sessions logged before this existed have no flash_during_saccade
                    # column at all - treated the same as "unknown" (None), same as a fresh
                    # row where no flash occurred.
                    flash_during_saccade=_parse_optional_bool(row.get("flash_during_saccade", "")),
                )
            )
    return trials


def load_metadata(timestamp: str) -> dict:
    """Sessions logged before the metadata sidecar existed have no meta.json
    at all - treat that the same as one with every field blank, rather than
    excluding those older runs from the dashboard."""
    path = meta_path_for(timestamp)
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.setdefault("participant_id", None)
    meta.setdefault("started_at", None)
    meta.setdefault("test_mode", None)
    meta.setdefault("calibration", None)
    meta.setdefault("name", None)
    meta.setdefault("gender", None)
    meta.setdefault("age", None)
    return meta


def save_participant_info(timestamp: str, name: str, gender: str, age: str) -> None:
    meta = load_metadata(timestamp)
    meta["name"] = name.strip() or None
    meta["gender"] = gender.strip() or None
    meta["age"] = int(age) if age.strip() else None
    meta_path_for(timestamp).write_text(json.dumps(meta, indent=2))


def delete_session(timestamp: str) -> None:
    """Removes a session's CSV, metadata sidecar, and cached graph, if they
    exist. Permanent - there's no trash/undo, same as deleting the files by
    hand would be."""
    csv_path_for(timestamp).unlink(missing_ok=True)
    meta_path_for(timestamp).unlink(missing_ok=True)
    graph_cache_path_for(timestamp).unlink(missing_ok=True)


@dataclass
class SessionSummary:
    timestamp: str
    metadata: dict
    trial_counts: dict[str, int] = field(default_factory=dict)


def discover_sessions() -> list[SessionSummary]:
    """All logged sessions, newest first."""
    if not DATA_DIR.exists():
        return []

    sessions = []
    for csv_file in DATA_DIR.glob("results_*.csv"):
        match = _SESSION_CSV_RE.match(csv_file.name)
        if not match:
            continue
        timestamp = match.group(1)

        trial_counts: dict[str, int] = {}
        for trial in load_session_trials(timestamp):
            trial_counts[trial.phase] = trial_counts.get(trial.phase, 0) + 1

        sessions.append(SessionSummary(timestamp=timestamp, metadata=load_metadata(timestamp), trial_counts=trial_counts))

    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return sessions
