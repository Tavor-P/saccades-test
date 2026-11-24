"""Post-hoc analysis of a saccade-phase session (`-m src.experiment.analyze_session [timestamp]`).

Finds the saccades in the raw gaze trace (`gaze_trace.csv` in the run folder; stamped with the phone's own capture
times, so the ~90 ms delivery delay and dropped samples don't matter), places every logged flash against the
nearest saccade onset, and reports detection accuracy as a function of flash time relative to onset - the
saccadic-suppression curve - plus a plot and a per-flash CSV. The live `flash_during_saccade` flag is not used.

Run it on the latest session with no arguments, or give a session timestamp (unix), a run folder name, `participant/folder`
or a path. Output goes to the run folder's `analysis/` (report.txt, plot.png, flashes.csv, suppression.json).
Units: gaze x is a 0-1 ratio of the scene (or screen) width per `x_px / NEON_SCENE_WIDTH` (traces recorded with the since-removed screen-mapped backends hold the screen ratio x NEON_SCENE_WIDTH in x_px, so old sessions analyse the same way); speeds are ratio/s.
The steps live in saccade_detection (trace, flashes, saccades), flash_gates + leg_gating (conditions), suppression_stats (counts,
index), analysis_report (text) and analysis_output (plot, per-flash CSV); thresholds in include/experiment/analysis_constants.py.
What they define and callers used from here is re-exported below, so `analyze_session.X` keeps working.
"""


import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.experiment import session_files
from src.experiment.session_files import DATA_DIR  # patched per module in tests, so always used by this bare name
from include.experiment.analysis_constants import (  # noqa: F401 (re-exported)
    BIN_EDGES_MS,
    CHANCE,
    FAR_MS,
    LAPSE,
    MIN_ACCURACY_TRIALS,
    MIN_AMPLITUDE,
    MIN_FIT_TRIALS,
    NEAR_ONSET_MS,
    OUT_SEARCH_S,
    RETURN_MIN_AMPLITUDE_REL,
    RETURN_SEARCH_AFTER_JUMP_S,
    SPEED_HIGH,
    SPEED_LOW,
)
from src.experiment.flash_gates import (  # noqa: F401 (re-exported)
    EXCLUDED,
    FIXATION,
    FIXATION_EXCLUSION_MS,
    RETURN_LATENCY_MAX_MS,
    SACCADE,
    classify_flashes,
    classify_leg_flash,
    summarize_gates,
)
from src.experiment.saccade_detection import Flash, Placed, Saccade, detect_saccades, load_flashes, load_trace, place_flashes  # noqa: F401
from src.experiment.suppression_stats import (  # noqa: F401 (re-exported)
    _EXTRAPOLATION,
    _SLOPES,
    _THRESHOLDS,
    _contrast_counts,
    _count,
    _fit_pair,
    _loglik_grid,
    _weibull,
    fisher_p,
    fit_pair_with_slope,
    summarize,
    suppression_index,
    wilson,
)
from src.experiment.leg_gating import (  # noqa: F401 (re-exported)
    LEGS,
    TrialSaccades,
    _robust_sd,
    classify_leg_flashes,
    find_trial_saccades,
    is_return_jump,
    leg_subset,
    place_leg_flashes,
    return_latency_stats,
    suppression_by_leg,
)
from src.experiment.analysis_report import (  # noqa: F401 (re-exported)
    _ms,
    _pct,
    format_gate_report,
    format_geometry,
    format_leg_suppression,
    format_report,
    format_return_latency,
    format_suppression,
    format_thresholds,
    thresholds_record,
)
from src.experiment.analysis_output import plot_report, write_flash_rows  # noqa: F401 (re-exported)


# -- driver --------------------------------------------------------------------------------------


def _analysable(ref: session_files.SessionRef) -> bool:
    return ref.gaze_trace_path.exists() and ref.trials_path.exists() and bool(load_flashes(ref.trials_path))


def latest_timestamp(data_dir: Path) -> str:
    """The newest session that has a gaze trace and at least one saccade-phase flash row."""
    ref = session_files.newest_session(Path(data_dir), _analysable)
    if ref is None:
        raise FileNotFoundError(f"no session with a gaze trace and saccade-phase flashes found in {data_dir}")
    return ref.key


@dataclass
class RunAnalysis:
    """One session's per-flash pipeline, before any file is written: saccades from the raw trace, every flash placed
    against its nearest onset and labelled by the pre-specified gates (each run is gated against its OWN trace and median
    saccade amplitude, so pooling several runs means concatenating these, never re-gating the combined data)."""

    ref: session_files.SessionRef
    meta: dict
    t: np.ndarray
    x: np.ndarray
    saccades: list[Saccade]
    placed: list[Placed]
    gates: list
    summary: dict  # summarize() plus "gates" and "suppression"
    suppression: dict
    geometry_line: str | None
    trials: list[TrialSaccades] | None = None  # return-jump runs: each flash's trial saccades (None for other runs)
    by_leg: dict | None = None  # return-jump runs: suppression_by_leg()
    return_latency: dict | None = None  # return-jump runs: return_latency_stats()

    @property
    def return_jump(self) -> bool:
        return self.trials is not None

    @property
    def stamp(self) -> str:
        return self.ref.label


def analyze_run(
    ref: session_files.SessionRef,
    clock_offset_ms: float | None = None,
    flash_delay_ms: float = 0.0,
    min_amplitude: float = MIN_AMPLITUDE,
    near_ms: float | None = None,
    far_ms: float | None = None,
) -> RunAnalysis:
    """The analysis numbers for one session (no files written). Raises ValueError when the meta file has no clock
    offset and none is given, OSError when a file is missing."""
    offset_history = None
    meta = session_files.read_json(ref.meta_path)  # None: missing or damaged
    if clock_offset_ms is None:
        if meta is None:
            raise ValueError(f"cannot read {ref.meta_path.name} of this session (missing or damaged), so there is no clock offset; pass --offset-ms")
        quality = meta.get("gaze_quality") or {}
        clock_offset_ms = quality.get("clock_offset_ms")
        offset_history = quality.get("clock_offset_history")
        if clock_offset_ms is None:
            status = quality.get("clock_offset_status")  # absent in older session.json files
            why = f" (clock_offset_status: {status})" if status else ""
            raise ValueError(f"this session has no clock_offset_ms in its meta file{why}; pass --offset-ms")
    t, x = load_trace(ref.gaze_trace_path, clock_offset_ms / 1000, offset_history)
    saccades = detect_saccades(t, x, min_amplitude=min_amplitude)
    flashes = load_flashes(ref.trials_path)
    trials = find_trial_saccades(flashes, saccades) if is_return_jump(flashes) else None
    if trials is None:
        placed = place_flashes(flashes, saccades, flash_delay_ms)
        gates = classify_flashes(placed, saccades, t, x, flash_delay_ms, [p.flash.go_cue for p in placed])
    else:
        placed = place_leg_flashes(flashes, trials, flash_delay_ms)
        gates = classify_leg_flashes(placed, trials, saccades, t, x, flash_delay_ms)
    summary = summarize(placed, saccades)
    summary["gates"] = summarize_gates(placed, gates)
    suppression = suppression_index(placed, near_ms, far_ms, gates=gates)
    summary["suppression"] = suppression
    meta = meta or {}
    try:
        geometry_line = format_geometry(meta.get("settings"))
    except (AttributeError, KeyError):
        geometry_line = None
    run = RunAnalysis(ref, meta, t, x, saccades, placed, gates, summary, suppression, geometry_line)
    if trials is not None:
        by_leg = suppression_by_leg(placed, gates, near_ms=near_ms, far_ms=far_ms)
        by_leg["pooled"] = suppression  # identical computation; keep the one object
        run.trials, run.by_leg = trials, by_leg
        run.return_latency = return_latency_stats(flashes, trials, meta.get("settings") if isinstance(meta.get("settings"), dict) else None)
        summary["by_leg"], summary["return_latency"] = by_leg, run.return_latency
    return run


def analyze(
    timestamp: str | None = None,
    data_dir: Path = DATA_DIR,
    clock_offset_ms: float | None = None,
    flash_delay_ms: float = 0.0,
    min_amplitude: float = MIN_AMPLITUDE,
    out_dir: Path | None = None,
    near_ms: float | None = None,
    far_ms: float | None = None,
) -> dict:
    """Runs the whole analysis for one session (`timestamp`: anything session_files.find_session accepts; default the
    newest). Writes the report text, plot, per-flash CSV and suppression json to `out_dir` (default: the run folder's
    analysis/ folder) and returns
    {"summary", "report", "report_path", "png", "csv", "suppression_json"}."""
    data_dir = Path(data_dir)
    ref = session_files.find_session(data_dir, timestamp or latest_timestamp(data_dir))
    if ref is None:
        raise FileNotFoundError(f"no session {timestamp!r} found in {data_dir}")
    run = analyze_run(ref, clock_offset_ms, flash_delay_ms, min_amplitude, near_ms, far_ms)
    stamp, summary, suppression, placed, gates = run.stamp, run.summary, run.suppression, run.placed, run.gates
    thresholds = thresholds_record(min_amplitude, flash_delay_ms, near_ms, far_ms)
    report = format_report(summary, label=f"(session {stamp})", geometry=run.geometry_line, thresholds=thresholds) + "\n" + format_gate_report(summary["gates"]) + "\n" + format_suppression(suppression)
    saved = suppression
    if run.return_jump:
        report = format_return_latency(run.return_latency) + "\n" + report + "\n" + format_leg_suppression(run.by_leg)
        saved = {**suppression, "by_leg": {leg: run.by_leg[leg] for leg in LEGS}, "return_latency": run.return_latency}

    out_dir = Path(out_dir) if out_dir is not None else ref.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = plot_report(placed, run.saccades, summary, out_dir / "plot.png", title=f"Session {stamp}")
    csv_path = write_flash_rows(out_dir / "flashes.csv", placed, gates)
    json_path = out_dir / "suppression.json"
    json_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    report_path = out_dir / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    return {"summary": summary, "report": report, "report_path": report_path, "png": png, "csv": csv_path, "suppression_json": json_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy vs flash time relative to saccade onset, from a session's raw gaze trace.")
    parser.add_argument("timestamp", nargs="?", help="session timestamp, run folder name, participant/folder or path; default: the newest session")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--offset-ms", type=float, default=None, help="phone-to-PC clock offset (default: from the meta file)")
    parser.add_argument("--flash-delay-ms", type=float, default=0.0, help="time from drawing the flash to light leaving the screen")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE, help="smallest saccade to count (fraction of the width)")
    parser.add_argument("--index-near-ms", type=float, default=None, help="suppression index: 'saccade' = within this many ms of an onset")
    parser.add_argument("--index-far-ms", type=float, default=None, help="suppression index: 'fixation' = at least this many ms from every onset")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.timestamp, args.data_dir, args.offset_ms, args.flash_delay_ms, args.min_amplitude, near_ms=args.index_near_ms, far_ms=args.index_far_ms)
    except (OSError, ValueError) as exc:  # e.g. a run without gaze_trace.csv, an unknown session: a message, not a traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    print(f"\nreport: {result['report_path']}\nplot: {result['png']}\nper-flash rows: {result['csv']}\nsuppression index: {result['suppression_json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
