"""The dashboard's one-click results: runs `analyze_session.analyze` (one run) or `analyze_participant.analyze_participant`
(every run of one person) and turns what they wrote into a results page. Every number comes from those two functions or
from session.json; this module only decides whether a saved analysis is still fresh, reshapes values for the page and
applies the data-quality rules (the traffic lights; each rule is written out on the page as `rule`).

Files are reached only under `data_access.DATA_DIR`, read at call time (tests point it at a tmp folder). Analyses run
one at a time (`_LOCK`): the Flask server is threaded and matplotlib's pyplot is not, and a double click must not give two
writers in one analysis folder. Anything that goes wrong becomes a `FriendlyError` (a plain sentence for the page); the
traceback goes to the dashboard's console only."""

import threading
import traceback
from datetime import datetime
from pathlib import Path

from src import textio
from src.dashboard import data_access
from src.experiment import analyze_participant as ap
from src.experiment import analyze_session as an
from src.experiment import session_files
from src.experiment.analyze_participant import N_BOOT  # read at call time (tests lower it)
from src.experiment.flash_gates import FIXATION, SACCADE
from src.experiment.run_quality import (  # the data-quality rules live there, shared with report.txt's QUALITY line
    AMBER,
    FLASH_RULE,
    GAP_RULE,
    GREEN,
    GREY,
    HEAD_RULE,
    LATENCY_RULE,
    RED,
    flash_light,
    gap_light,
    gap_text,
    head_status,
    latency_light,
    ms_text,
)

_LOCK = threading.Lock()

_SEVERITY = {GREY: -1, GREEN: 0, AMBER: 1, RED: 2}
LIGHT_WORDS = {GREEN: "OK", AMBER: "Check", RED: "Problem", GREY: "Not recorded"}

# kind -> file name in the run's analysis/ folder. "plot" is shown inline, the others are downloads.
RUN_FILES = {"report": "report.txt", "plot": "plot.png", "flashes": "flashes.csv", "suppression": "suppression.json"}
PARTICIPANT_FILES = {"report": "report.txt", "plot": "plot.png", "flashes": "flashes.csv", "summary": "summary.json"}
# kind -> link text of a downloadable file, for the run, participant and comparison pages (comparison.py reuses it)
DOWNLOAD_LABELS = {
    "flashes": "flashes.csv (one row per flash)",
    "suppression": "suppression.json (the index, machine-readable)",
    "summary": "summary.json (the pooled result, machine-readable)",
    "csv": "comparison.csv (one row per person, for Excel / SPSS / R)",
    "json": "comparison.json (every number, machine-readable)",
    "report": "report.txt (the full text report)",
    "plot": "plot.png (save the figure)",
}


class FriendlyError(Exception):
    """A problem to show on the page as a sentence (never a stack trace). `include_test_hint`: offer the test-runs box."""

    def __init__(self, message: str, status: int = 422, include_test_hint: bool = False):
        super().__init__(message)
        self.message = message
        self.status = status
        self.include_test_hint = include_test_hint


def _data_dir() -> Path:
    return Path(data_access.DATA_DIR)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _is_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
    """Every output exists and is newer than every existing input file."""
    out_times = [t for t in (_mtime(p) for p in outputs) if t is not None]
    if not outputs or len(out_times) != len(outputs):
        return False
    in_times = [t for t in (_mtime(p) for p in inputs) if t is not None]
    return not in_times or min(out_times) > max(in_times)


def _stamp(mtime: float | None) -> str | None:
    return None if mtime is None else datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")


def _read_text(path: Path) -> str:
    try:
        return textio.read_text(path)
    except OSError:
        return ""


def _run_inputs(ref: session_files.SessionRef) -> list[Path]:
    return [ref.trials_path, ref.meta_path, ref.gaze_trace_path]


def _unexpected(what: str, exc: Exception) -> FriendlyError:
    traceback.print_exc()
    return FriendlyError(f"The {what} stopped with an unexpected error ({type(exc).__name__}: {exc}). "
                         "The details are in the dashboard's console window.", status=500)


# -- quality rules ---------------------------------------------------------------------------------


def _rule(name: str, light: str, value: str, rule: str) -> dict:
    return {"name": name, "light": light, "word": LIGHT_WORDS[light], "value": value, "rule": rule}


def _worst(lights) -> str:
    lights = [x for x in lights if x != GREY]
    return max(lights, key=lambda light: _SEVERITY[light]) if lights else GREY


METHOD_RULE = ("Green: the index comes from the threshold fit (the main method). Amber: only the accuracy fallback could be computed. "
               "Red: no index could be computed.")
POOL_RULE = ("Green: every analysed run was pooled. Amber: some runs were left out (reasons listed). Red: no run could be pooled. "
             "Test runs are only included when the box is ticked.")


def _method_rule(method, notes) -> dict:
    if method == "psychometric":
        return _rule("Result", GREEN, "threshold fit", METHOD_RULE)
    if method == "accuracy":
        return _rule("Result", AMBER, "accuracy fallback (not a threshold ratio)", METHOD_RULE)
    return _rule("Result", RED, "no index: " + (notes[-1] if notes else "no flashes"), METHOD_RULE)


def run_quality(meta: dict, suppression: dict) -> dict:
    q = meta.get("gaze_quality") or {}
    latency, gap = q.get("latency_ms_median"), q.get("longest_gap_ms")
    head_light, head_text = head_status(meta)
    n_sac, n_fix = suppression.get("n_saccade", 0), suppression.get("n_fixation", 0)
    rules = [
        _rule("Gaze delay", latency_light(latency), ms_text(latency), LATENCY_RULE),
        _rule("Tracking gaps", gap_light(gap), gap_text(gap, q.get("gaps_over_threshold")), GAP_RULE),
        _rule("Head movement", head_light, head_text, HEAD_RULE),
        _rule("Flashes per condition", flash_light(n_sac, n_fix), f"{n_sac} saccade, {n_fix} fixation", FLASH_RULE),
        _method_rule(suppression.get("method"), suppression.get("notes") or []),
    ]
    return {"overall": _worst(r["light"] for r in rules), "rules": rules}


def participant_quality(pid: str, summary: dict) -> dict:
    used, excluded = summary.get("runs_used") or [], summary.get("runs_excluded") or []

    def per_run(light_of, text_of):
        lights = [light_of(r) for r in used]
        text = "; ".join(f"{r['run']}: {text_of(r)}" for r in used) or "no pooled runs"
        return _worst(lights) if used else GREY, text

    lat = per_run(lambda r: latency_light(r.get("gaze_latency_ms_median")), lambda r: ms_text(r.get("gaze_latency_ms_median")))
    gap = per_run(lambda r: gap_light(r.get("longest_gap_ms")), lambda r: gap_text(r.get("longest_gap_ms"), r.get("gaps_over_threshold")))
    heads = {r["run"]: head_status(session_files.read_json(_data_dir() / pid / r["run"] / session_files.SESSION_FILE) or {}) for r in used}
    head = per_run(lambda r: heads[r["run"]][0], lambda r: heads[r["run"]][1])
    n = summary.get("n_flashes") or {}
    n_sac, n_fix = n.get(SACCADE, 0), n.get(FIXATION, 0)
    if not used:
        pool = _rule("Runs pooled", RED, "no run could be pooled" + "".join(f"; {r['run']}: {r.get('reason')}" for r in excluded), POOL_RULE)
    elif excluded:
        pool = _rule("Runs pooled", AMBER, f"{len(used)} pooled, {len(excluded)} left out: " + "; ".join(f"{r['run']}: {r.get('reason')}" for r in excluded), POOL_RULE)
    else:
        pool = _rule("Runs pooled", GREEN, f"all {len(used)} analysed run(s) pooled", POOL_RULE)
    skipped = (summary.get("analysis") or {}).get("test_runs_skipped") or 0
    if skipped:
        pool["value"] += f" ({skipped} test run(s) not included: tick 'Include test runs' to add them)"
    rules = [
        _rule("Gaze delay", lat[0], lat[1], LATENCY_RULE),
        _rule("Tracking gaps", gap[0], gap[1], GAP_RULE),
        _rule("Head movement", head[0], head[1], HEAD_RULE),
        _rule("Flashes per condition", flash_light(n_sac, n_fix), f"{n_sac} saccade, {n_fix} fixation (all pooled runs)", FLASH_RULE),
        _method_rule(summary.get("method"), summary.get("notes") or []),
        pool,
    ]
    return {"overall": _worst(r["light"] for r in rules), "rules": rules}


# -- one run -----------------------------------------------------------------------------------------


def run_ref(timestamp: str) -> session_files.SessionRef:
    ref = data_access._ref(timestamp)
    if ref is None or not ref.trials_path.exists():
        raise FriendlyError(f"There is no run '{timestamp}' (it may have been deleted).", status=404)
    return ref


def run_file(ref: session_files.SessionRef, kind: str) -> Path:
    return ref.analysis_dir / RUN_FILES[kind]


def run_is_fresh(ref: session_files.SessionRef) -> bool:
    return _is_fresh([run_file(ref, k) for k in RUN_FILES], _run_inputs(ref))


def analyse_run(timestamp: str) -> None:
    """Runs analyze_session.analyze for this run (writes its analysis files). Raises FriendlyError."""
    ref = run_ref(timestamp)
    if not ref.gaze_trace_path.exists():
        raise FriendlyError("This run has no gaze recording (gaze_trace.csv), so its saccades cannot be found and it cannot be analysed. "
                            "This happens when the eye tracker was not streaming during the run.")
    if not ref.meta_path.exists():
        raise FriendlyError("This run has no session.json, so the phone-to-computer clock offset is unknown and the flashes cannot be "
                            "placed against the saccades.")
    try:
        if not an.load_flashes(ref.trials_path):
            raise FriendlyError("This run has no saccade-phase flashes in trials.csv (it may have been stopped before the main trials).")
        with _LOCK:
            an.analyze(ref.key, data_dir=_data_dir())
    except FriendlyError:
        raise
    except ValueError as exc:
        if "clock_offset" in str(exc):
            raise FriendlyError("This run's session.json has no phone-to-computer clock offset (gaze_quality.clock_offset_ms), so the "
                                "flashes cannot be placed against the saccades. The offset is measured when the Neon starts streaming; "
                                "a run where that failed cannot be analysed from the dashboard.") from None
        raise _unexpected("analysis", exc) from None
    except Exception as exc:  # noqa: BLE001 - shown as a sentence, traceback to the console
        raise _unexpected("analysis", exc) from None


def _run_headline_summary(s: dict) -> dict:
    """suppression.json reshaped into the keys analyze_participant.headline()/meaning() read (one run, same wording)."""
    return {"runs_used": [{}], "n_flashes": {SACCADE: s.get("n_saccade", 0), FIXATION: s.get("n_fixation", 0)},
            "method": s.get("method"), "index": s.get("index"), "ci": s.get("ci"), "notes": s.get("notes") or []}


def run_page(timestamp: str, force: bool = False) -> dict:
    """Everything the results page shows for one run, analysing it first if forced or if no fresh analysis exists."""
    ref = run_ref(timestamp)
    if force or not run_is_fresh(ref):
        analyse_run(timestamp)
    suppression = session_files.read_json(run_file(ref, "suppression")) or {}
    summary = _run_headline_summary(suppression)
    meta = session_files.read_json(ref.meta_path) or {}
    participant = (meta.get("participant") or {}).get("id") or ref.participant
    return {
        "kind": "run",
        "key": ref.key,
        "title": f"Run {ref.label}",
        "participant": ref.participant,
        "participant_label": participant,
        "headline": ap.headline(summary),
        "meaning": ap.meaning(summary),
        "analysed_at": _stamp(_mtime(run_file(ref, "suppression"))),
        "quality": run_quality(meta, suppression),
        "runs": None,
        "downloads": [k for k in ("flashes", "suppression", "report", "plot") if run_file(ref, k).exists()],
        "report_text": _read_text(run_file(ref, "report")),
        "include_test": None,
    }


# -- one participant ---------------------------------------------------------------------------------


def participant_ids() -> list[str]:
    """Participant folder names that hold at least one run folder."""
    return sorted({r.participant for r in session_files.list_sessions(_data_dir())})


def check_participant(pid: str) -> str:
    if pid not in participant_ids():
        raise FriendlyError(f"There is no participant '{pid}' with runs in the data folder (the runs may have been deleted).", status=404)
    return pid


def participant_dir(pid: str) -> Path:
    return _data_dir() / pid / ap.ANALYSIS_DIR


def participant_file(pid: str, kind: str) -> Path:
    return participant_dir(pid) / PARTICIPANT_FILES[kind]


def _candidates(pid: str, include_test: bool) -> list:
    _, entries = ap.find_runs(pid, _data_dir())
    return [e for e in entries if include_test or e.mode != "test"]


def participant_is_fresh(pid: str, include_test: bool) -> bool:
    summary = session_files.read_json(participant_file(pid, "summary")) or {}
    if not summary or bool(summary.get("include_test")) != include_test:
        return False
    try:
        candidates = _candidates(pid, include_test)
    except ap.NoRunsError:
        return False
    analysed = {r.get("run") for r in (summary.get("runs_used") or []) + (summary.get("runs_excluded") or [])}
    if analysed != {e.name for e in candidates}:
        return False  # a run was added or removed since
    _, entries = ap.find_runs(pid, _data_dir())
    inputs = [p for e in entries for p in _run_inputs(e.ref)]
    return _is_fresh([participant_file(pid, k) for k in PARTICIPANT_FILES], inputs)


def overview() -> list[dict]:
    """One row per participant for the index page: run counts by mode and when the pooled analysis was last written."""
    rows = []
    for pid in participant_ids():
        try:
            _, entries = ap.find_runs(pid, _data_dir())
        except ap.NoRunsError:
            continue
        n_test = sum(e.mode == "test" for e in entries)
        summary = session_files.read_json(participant_file(pid, "summary")) or {}
        rows.append({"id": pid, "n_real": len(entries) - n_test, "n_test": n_test, "only_test": n_test == len(entries),
                     "analysed_at": _stamp(_mtime(participant_file(pid, "summary"))) if summary else None,
                     "analysed_with_test": bool(summary.get("include_test")) if summary else False})
    return rows


def analyse_participant(pid: str, include_test: bool) -> None:
    """Runs analyze_participant for this person (writes data/<pid>/analysis/). Raises FriendlyError."""
    check_participant(pid)
    try:
        _, entries = ap.find_runs(pid, _data_dir())
        if not include_test and all(e.mode == "test" for e in entries):
            raise FriendlyError(f"Participant '{pid}' has only test runs ({len(entries)}). Test runs are left out unless you tick "
                                "'Include test runs' and analyse again.", include_test_hint=True)
        with _LOCK:
            ap.analyze_participant(pid, _data_dir(), include_test=include_test, n_boot=N_BOOT)
    except FriendlyError:
        raise
    except ap.NoRunsError:
        raise FriendlyError(f"No runs were found for participant '{pid}'.", status=404) from None
    except Exception as exc:  # noqa: BLE001 - shown as a sentence, traceback to the console
        raise _unexpected("pooled analysis", exc) from None


def participant_page(pid: str, include_test: bool = False, force: bool = False) -> dict:
    check_participant(pid)
    if force or not participant_is_fresh(pid, include_test):
        analyse_participant(pid, include_test)
    summary = session_files.read_json(participant_file(pid, "summary")) or {}
    if not summary:
        raise FriendlyError("The pooled analysis wrote no summary.json; see the dashboard's console window.", status=500)
    runs = [{**r, "pooled": True} for r in summary.get("runs_used") or []] + [{**r, "pooled": False} for r in summary.get("runs_excluded") or []]
    runs.sort(key=lambda r: r.get("run") or "")
    for r in runs:
        r["index_text"] = ap._run_index_text(r)
        for k in ("n_saccade", "n_fixation", "n_excluded", "gaze_latency_ms_median", "longest_gap_ms", "reason"):
            r.setdefault(k, None)  # a run that could not be analysed has only run/date/mode/reason
    return {
        "kind": "participant",
        "key": pid,
        "title": f"Participant {pid}",
        "participant": pid,
        "participant_label": pid,
        "headline": ap.headline(summary),
        "meaning": ap.meaning(summary),
        "analysed_at": _stamp(_mtime(participant_file(pid, "summary"))),
        "quality": participant_quality(pid, summary),
        "runs": runs,
        "downloads": [k for k in ("flashes", "summary", "report", "plot") if participant_file(pid, k).exists()],
        "report_text": _read_text(participant_file(pid, "report")),
        "include_test": include_test,
    }
