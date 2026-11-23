"""Pooled analysis of ALL of one participant's runs (`-m src.experiment.analyze_participant <participant-id>`).

One run gives only ~30 saccade flashes, too few for a stable suppression index, so this pools every run of a person:
1. finds the run folders under data/<participant>/ (test runs only with --include-test; old loose `results_<ts>.csv` files
   have no participant and are ignored),
2. analyses each run exactly as `analyze_session` does (saccades from its own raw gaze trace, every flash gated by
   `flash_gates` against that run's own saccades - runs are never re-gated together),
3. pools the runs that used the same saccade contrasts and screen geometry (the largest such set; the others are listed
   with the reason) and fits the same suppression index as `analyze_session` (Weibull per condition, shared slope,
   bootstrap CI, same fallbacks and "cannot be computed" reasons).

Writes data/<participant>/analysis/: report.txt (plain language first), plot.png, flashes.csv (every pooled flash with its
run) and summary.json (the stable machine-readable input for comparing people; see SUMMARY_KEYS). Python callers (the
dashboard, a comparison engine) use `analyze_participant()`.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only renders PNG files
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import session_files
from src.experiment.analyze_session import (
    DATA_DIR,
    LEGS,
    MIN_AMPLITUDE,
    RunAnalysis,
    _contrast_counts,
    _weibull,
    analyze_run,
    fit_pair_with_slope,
    format_gate_report,
    format_leg_suppression,
    format_return_latency,
    format_suppression,
    load_flashes,
    return_latency_stats,
    summarize,
    suppression_by_leg,
    suppression_index,
    wilson,
    write_flash_rows,
)
from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE, summarize_gates

SCHEMA_VERSION = 2  # 2: run_kind, by_leg, return_latency (return-jump runs)
ANALYSIS_DIR = "analysis"  # data/<participant>/analysis/
CONDITIONS = (SACCADE, FIXATION, EXCLUDED)
# summary.json top-level keys. A comparison module reads these; add keys only with a SCHEMA_VERSION bump.
SUMMARY_KEYS = (
    "schema_version", "participant", "generated_at", "include_test",
    "index", "ci", "method", "label", "threshold_saccade", "threshold_fixation", "slope", "notes",
    "n_flashes", "correct", "by_contrast", "exclusion_reasons",
    "contrasts_pct", "geometry", "runs_used", "runs_excluded", "flashes_csv", "analysis",
    "run_kind", "by_leg", "return_latency",
)
# Run kinds never pool across each other: "single" (no return jump) and "return_jump/<trigger>" (the trigger changes the
# movement itself). The return flash spread SD and leg fraction only change flash scheduling, so they do not split runs.
SINGLE_RUN_KIND = "single"
_GEOMETRY_FIELDS = ("screen_height_cm", "viewing_distance_cm", "line_offset_deg", "line_sd_deg")


class NoRunsError(FileNotFoundError):
    """No run folder to analyse for this participant (the message says what was found instead)."""


@dataclass
class RunEntry:
    """One run folder of the participant, analysed or not."""

    ref: session_files.SessionRef
    mode: str  # "real" | "test" | "unknown"
    analysis: RunAnalysis | None = None
    contrasts_pct: tuple[float, ...] | None = None
    contrasts_from: str = ""  # "settings" (session.json) | "trials" (the contrasts actually shown; older runs)
    geometry: dict = field(default_factory=dict)
    excluded_reason: str | None = None
    run_kind: str = SINGLE_RUN_KIND  # SINGLE_RUN_KIND | "return_jump/<trigger>"

    @property
    def name(self) -> str:
        return self.ref.folder.name

    @property
    def date(self) -> str:
        try:
            return datetime.strptime("_".join(self.name.split("_")[:2]), "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return datetime.fromtimestamp(int(self.ref.key)).strftime("%Y-%m-%d %H:%M")

    @property
    def quality(self) -> dict:
        return (self.analysis.meta.get("gaze_quality") if self.analysis else None) or {}

    def counts(self) -> dict:
        gates = self.analysis.gates if self.analysis else []
        return {c: sum(g.condition == c for g in gates) for c in CONDITIONS}

    @property
    def compat_key(self) -> tuple:
        return (self.contrasts_pct, tuple(self.geometry.get(k) for k in _GEOMETRY_FIELDS), self.run_kind)


@dataclass
class ParticipantResult:
    participant: str  # the sanitized folder name
    summary: dict  # exactly what summary.json holds
    report: str
    runs_used: list[RunEntry]
    runs_excluded: list[RunEntry]
    placed: list  # pooled flashes (analyze_session.Placed), in run order
    gates: list  # flash_gates.GateResult per pooled flash
    run_of_flash: list[str]  # run folder name per pooled flash
    paths: dict = field(default_factory=dict)  # report / png / csv / summary (empty when write=False)


# -- discovery -----------------------------------------------------------------------------------


def _mode(ref: session_files.SessionRef, meta: dict) -> str:
    mode = (meta.get("run") or {}).get("mode")
    if mode in ("real", "test"):
        return mode
    match = session_files._FOLDER_NAME_RE.match(ref.folder.name)
    return "test" if match and match.group(1) else ("real" if match else "unknown")


def _read_meta(ref: session_files.SessionRef) -> dict:
    try:
        meta = json.loads(ref.meta_path.read_text())
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def find_runs(participant_id: str, data_dir: Path = DATA_DIR) -> tuple[str, list[RunEntry]]:
    """(sanitized id, every run folder of that participant, oldest first). Raises NoRunsError if there is none."""
    data_dir = Path(data_dir)
    pid = session_files.sanitize_participant_id(participant_id)
    folder = data_dir / pid
    refs = [r for r in session_files.list_sessions(data_dir) if r.folder is not None and r.participant == pid] if folder.is_dir() else []
    if not refs:
        others = sorted(p.name for p in data_dir.iterdir() if p.is_dir()) if data_dir.is_dir() else []
        where = f"no run folders in {folder}" if folder.is_dir() else f"no folder {folder}"
        hint = f"; participant folders found: {', '.join(others)}" if others else f"; {data_dir} has no participant folders"
        raise NoRunsError(f"No runs for participant '{participant_id}' ({where}{hint}).")
    refs.sort(key=lambda r: (int(r.key), r.label))
    return pid, [RunEntry(r, _mode(r, _read_meta(r))) for r in refs]


# -- per-run analysis and compatibility ------------------------------------------------------------


def _analyse_entry(entry: RunEntry, flash_delay_ms: float, min_amplitude: float) -> None:
    ref = entry.ref
    if not ref.gaze_trace_path.exists():
        entry.excluded_reason = "no gaze_trace.csv in the run folder"
        return
    try:
        if not load_flashes(ref.trials_path):
            entry.excluded_reason = "no saccade-phase flashes in trials.csv"
            return
        entry.analysis = analyze_run(ref, flash_delay_ms=flash_delay_ms, min_amplitude=min_amplitude)
    except Exception as exc:  # one damaged run must not stop the others
        entry.analysis = None
        entry.excluded_reason = f"could not be analysed ({type(exc).__name__}: {exc})"
        return
    settings = entry.analysis.meta.get("settings") or {}
    recorded = settings.get("saccade_contrasts_pct")
    if recorded:
        entry.contrasts_pct, entry.contrasts_from = tuple(sorted(round(float(c), 3) for c in recorded)), "settings"
    else:
        entry.contrasts_pct = tuple(sorted({round(100 * p.flash.contrast, 3) for p in entry.analysis.placed}))
        entry.contrasts_from = "trials"
    entry.geometry = {k: (None if settings.get(k) is None else round(float(settings[k]), 3)) for k in _GEOMETRY_FIELDS}
    entry.run_kind = run_kind(entry.analysis)


def run_kind(run: RunAnalysis) -> str:
    """SINGLE_RUN_KIND, or "return_jump/<trigger>" (trigger from session.json settings; 'unknown' if not recorded). The
    trials decide (a row logged a jump-back), not the setting alone, so a run is classed by what it actually did."""
    if not run.return_jump:
        return SINGLE_RUN_KIND
    return f"return_jump/{(run.meta.get('settings') or {}).get('return_jump_trigger') or 'unknown'}"


def _fmt_contrasts(levels) -> str:
    return "/".join(f"{c:g}" for c in levels) + " %" if levels else "none"


def _fmt_geometry(geometry: dict) -> str:
    if geometry.get("screen_height_cm") is None or geometry.get("viewing_distance_cm") is None:
        return "not recorded"
    text = f"{geometry['screen_height_cm']:g} cm high at {geometry['viewing_distance_cm']:g} cm"
    if geometry.get("line_offset_deg") is not None:
        text += f", line {geometry['line_offset_deg']:g} deg off, SD {geometry.get('line_sd_deg') or 0:g} deg"
    return text


def _choose_pool(analysed: list[RunEntry]) -> tuple:
    """The compatibility key of the largest set of runs: most runs, then most gated (saccade + fixation) flashes, then the newest run."""
    groups: dict[tuple, list[RunEntry]] = {}
    for e in analysed:
        groups.setdefault(e.compat_key, []).append(e)

    def rank(key):
        members = groups[key]
        flashes = sum(e.counts()[SACCADE] + e.counts()[FIXATION] for e in members)
        return (len(members), flashes, max(int(e.ref.key) for e in members))

    return max(groups, key=rank)


def _incompatibility(entry: RunEntry, pooled: RunEntry) -> str:
    parts = []
    if entry.contrasts_pct != pooled.contrasts_pct:
        parts.append(f"contrasts {_fmt_contrasts(entry.contrasts_pct)} differ from the pooled runs' {_fmt_contrasts(pooled.contrasts_pct)}")
    if entry.compat_key[1] != pooled.compat_key[1]:
        parts.append(f"screen geometry ({_fmt_geometry(entry.geometry)}) differs from the pooled runs' ({_fmt_geometry(pooled.geometry)})")
    if entry.run_kind != pooled.run_kind:
        parts.append(f"run kind {entry.run_kind} differs from the pooled runs' {pooled.run_kind} (return-jump and single-saccade runs, "
                     "or different jump triggers, are never pooled)")
    return "; ".join(parts)


# -- the analysis --------------------------------------------------------------------------------


def _pooled_slope(placed: list, gates: list) -> float | None:
    sac = [p for p, g in zip(placed, gates) if g.condition == SACCADE]
    fix = [p for p, g in zip(placed, gates) if g.condition == FIXATION]
    return fit_pair_with_slope(*_contrast_counts(sac), *_contrast_counts(fix))[3]


def _by_contrast(placed: list, gates: list) -> list[dict]:
    rows = []
    for c in sorted({p.flash.contrast for p in placed}):
        for cond in CONDITIONS:
            members = [p for p, g in zip(placed, gates) if g.condition == cond and p.flash.contrast == c]
            rows.append({"contrast_pct": round(100 * c, 4), "condition": cond, "n": len(members), "correct": sum(p.flash.correct for p in members)})
    return rows


def _leg_record(s: dict) -> dict:
    """The summary.json view of one suppression_index() result."""
    return {k: (list(s[k]) if k == "ci" and s[k] is not None else s[k]) for k in (
        "index", "ci", "method", "n_saccade", "n_fixation", "correct_saccade", "correct_fixation", "threshold_saccade", "threshold_fixation", "notes")}


def _run_record(e: RunEntry) -> dict:
    record = {"run": e.name, "run_key": e.ref.key, "date": e.date, "mode": e.mode}
    if e.analysis is not None:
        record["run_kind"] = e.run_kind
        n, s = e.counts(), e.analysis.suppression
        record.update(
            n_saccade=n[SACCADE], n_fixation=n[FIXATION], n_excluded=n[EXCLUDED],
            index=s["index"], ci=None if s["ci"] is None else list(s["ci"]), method=s["method"], notes=list(s["notes"]),
            contrasts_pct=list(e.contrasts_pct), contrasts_from=e.contrasts_from, geometry=e.geometry,
            gaze_latency_ms_median=e.quality.get("latency_ms_median"), longest_gap_ms=e.quality.get("longest_gap_ms"),
            gaps_over_threshold=e.quality.get("gaps_over_threshold"),
        )
    if e.excluded_reason:
        record["reason"] = e.excluded_reason
    return record


def analyze_participant(
    participant_id: str,
    data_dir: Path = DATA_DIR,
    include_test: bool = False,
    out_dir: Path | None = None,
    write: bool = True,
    flash_delay_ms: float = 0.0,
    min_amplitude: float = MIN_AMPLITUDE,
    n_boot: int = 1000,
) -> ParticipantResult:
    """Pools every compatible run of one participant into one suppression index (see the module docstring). Writes
    report.txt / plot.png / flashes.csv / summary.json to `out_dir` (default data/<participant>/analysis/) unless
    write=False. Raises NoRunsError when the participant has no run folder (or only test runs without include_test)."""
    data_dir = Path(data_dir)
    pid, entries = find_runs(participant_id, data_dir)
    tests = [e for e in entries if e.mode == "test"]
    candidates = entries if include_test else [e for e in entries if e.mode != "test"]
    if not candidates:
        raise NoRunsError(f"Participant '{pid}' has only test runs ({len(tests)}). To analyse them, run: "
                         f"launchers\\analyze_participant.bat {pid} --include-test  (or: .\\scripts\\py.ps1 -m src.experiment.analyze_participant {pid} --include-test)")
    for e in candidates:
        _analyse_entry(e, flash_delay_ms, min_amplitude)

    analysed = [e for e in candidates if e.analysis is not None]
    used: list[RunEntry] = []
    if analysed:
        key = _choose_pool(analysed)
        used = [e for e in analysed if e.compat_key == key]
        for e in analysed:
            if e.compat_key != key:
                e.excluded_reason = _incompatibility(e, used[0])
    excluded = [e for e in candidates if e not in used]

    placed, gates, run_of_flash = [], [], []
    for e in used:
        placed += e.analysis.placed
        gates += e.analysis.gates
        run_of_flash += [e.name] * len(e.analysis.placed)
    suppression = suppression_index(placed, gates=gates, n_boot=n_boot)
    gate_counts = summarize_gates(placed, gates)
    slope = _pooled_slope(placed, gates) if suppression["method"] == "psychometric" else None
    kind = used[0].run_kind if used else None
    by_leg = return_latency = None
    if kind is not None and kind != SINGLE_RUN_KIND:
        legs = suppression_by_leg(placed, gates, n_boot=n_boot)
        by_leg = {leg: _leg_record(legs[leg]) for leg in LEGS}
        by_leg["pooled"] = _leg_record(suppression)
        flashes = [p.flash for e in used for p in e.analysis.placed]
        trials = [tr for e in used for tr in e.analysis.trials]
        return_latency = return_latency_stats(flashes, trials, used[-1].analysis.meta.get("settings") or {})
        return_latency["return_flash_spread_sd_ms_by_run"] = {e.name: (e.analysis.meta.get("settings") or {}).get("return_flash_spread_sd_ms") for e in used}

    summary = {
        "schema_version": SCHEMA_VERSION,
        "participant": pid,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "include_test": include_test,
        "index": suppression["index"],
        "ci": None if suppression["ci"] is None else list(suppression["ci"]),
        "method": suppression["method"],
        "label": suppression["label"],
        "threshold_saccade": suppression["threshold_saccade"],
        "threshold_fixation": suppression["threshold_fixation"],
        "slope": slope,
        "notes": list(suppression["notes"]),
        "n_flashes": {**{c: gate_counts[c][1] for c in CONDITIONS}, "total": len(placed)},
        "correct": {c: gate_counts[c][0] for c in CONDITIONS},
        "by_contrast": _by_contrast(placed, gates),
        "exclusion_reasons": dict(gate_counts["reasons"]),
        "contrasts_pct": list(used[0].contrasts_pct) if used else None,
        "geometry": used[0].geometry if used else None,
        "runs_used": [_run_record(e) for e in used],
        "runs_excluded": [_run_record(e) for e in excluded],
        "flashes_csv": "flashes.csv",  # relative to summary.json
        "analysis": {
            "conditions_from": "pre-specified gates (src/experiment/flash_gates.py), each run gated against its own gaze trace",
            "flash_delay_ms": flash_delay_ms, "min_amplitude": min_amplitude, "n_boot": n_boot,
            "test_runs_skipped": 0 if include_test else len(tests),
            "legacy_loose_files": "ignored (old results_<ts>.csv sessions have no participant folder)",
            "pooling_rule": "same saccade contrasts, screen geometry and run kind (single vs return_jump/<trigger>)",
        },
        "run_kind": kind,
        "by_leg": by_leg,  # return-jump pools: {out, return, pooled}; each leg = its saccade flashes vs all fixation flashes
        "return_latency": return_latency,  # return-jump pools: return onset minus the jump (raw trace), all pooled trials
    }
    assert tuple(summary) == SUMMARY_KEYS
    result = ParticipantResult(pid, summary, "", used, excluded, placed, gates, run_of_flash)
    result.report = format_participant_report(result)
    if write:
        out = Path(out_dir) if out_dir is not None else data_dir / pid / ANALYSIS_DIR
        out.mkdir(parents=True, exist_ok=True)
        dates = {e.name: e.date for e in used}
        modes = {e.name: e.mode for e in used}
        result.paths = {
            "report": out / "report.txt",
            "png": plot_participant(result, out / "plot.png"),
            "csv": write_flash_rows(out / "flashes.csv", placed, gates, {
                "run": run_of_flash, "run_date": [dates[r] for r in run_of_flash], "mode": [modes[r] for r in run_of_flash]}),
            "summary": out / "summary.json",
        }
        result.paths["report"].write_text(result.report + "\n", encoding="utf-8")
        result.paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


# -- report --------------------------------------------------------------------------------------


def _fmt_ci(ci) -> str:
    if ci is None:
        return "95 % CI not available"
    lo, hi = ci
    return f"95 % CI {lo:.2f}-{hi:.2f}" if lo >= 0 else f"95 % CI {lo:.2f} to {hi:.2f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def headline(summary: dict) -> str:
    """The one-sentence result, in words."""
    n_runs = len(summary["runs_used"])
    n = summary["n_flashes"]
    source = f"from {_plural(n_runs, 'run')}, {n[SACCADE]} saccade and {n[FIXATION]} fixation flashes"
    if summary["method"] == "psychometric":
        return f"Suppression index {summary['index']:.2f} ({_fmt_ci(summary['ci'])}) {source}."
    if summary["method"] == "accuracy":
        reason = next((x.split(": ", 1)[1] for x in summary["notes"] if x.startswith("psychometric index not computed")), "")
        return (f"Accuracy-based suppression index {summary['index']:.2f} ({_fmt_ci(summary['ci'])}) {source}. This is a fallback "
                f"(accuracy difference, not a threshold ratio)" + (f" because of {reason}." if reason else "."))
    if not n_runs and summary.get("runs_excluded"):  # nothing pooled: say why the runs were left out, not "0 flashes"
        reasons = sorted({r.get("reason", "") for r in summary["runs_excluded"]} - {""})
        return "Suppression index cannot be computed yet: no run could be used" + (f" ({'; '.join(reasons)})." if reasons else ".")
    reason = summary["notes"][-1] if summary["notes"] else "no flashes"
    if reason.startswith("cannot compute:"):  # it only repeats the flash counts already in `source`
        return f"Suppression index cannot be computed yet {source}: both kinds of flashes are needed."
    return f"Suppression index cannot be computed yet {source}: {reason}."


def meaning(summary: dict) -> str | None:
    """How to read the index's scale, which differs between the threshold ratio and the accuracy fallback."""
    if summary["method"] == "psychometric":
        return "What it means: 0 = no suppression, 1 = the line is never seen during saccades, negative = seen better during saccades."
    if summary["method"] == "accuracy":
        return ("What it means: the difference in proportion correct (fixation minus saccade). 0 = no difference; about 0.5 = guessing "
                "during saccades while fixation is perfect (it cannot reach 1, because guessing still gives 50 % correct).")
    return None


def _run_index_text(record: dict) -> str:
    if record.get("index") is None:
        return "n/a"
    return f"{record['index']:+.2f}" + (" (accuracy)" if record["method"] == "accuracy" else "")


def _num(v, fmt="{:.0f}") -> str:
    return "?" if v is None else fmt.format(v)


def format_participant_report(result: ParticipantResult) -> str:
    s = result.summary
    lines = [f"Participant {s['participant']}: pooled saccadic-suppression analysis", "", headline(s)]
    if meaning(s):
        lines.append(meaning(s))
    skipped = s["analysis"]["test_runs_skipped"]
    if s["runs_excluded"]:
        lines += ["", f"NOT POOLED: {_plural(len(s['runs_excluded']), 'run')} left out:"]
        lines += [f"  - {r['run']} ({r['date']}, {r['mode']}): {r['reason']}" for r in s["runs_excluded"]]
    else:
        lines += ["", "All analysed runs were pooled."]
    if skipped:
        lines.append(f"{_plural(skipped, 'test run')} skipped (add --include-test to include them).")

    lines += ["", "Runs pooled:"]
    header = f"  {'run':<22} {'date':<16} {'mode':<5} {'saccade':>7} {'fixation':>8} {'excluded':>8}  {'run index':<16} {'gaze delay':>10} {'longest gap':>11}"
    lines.append(header)
    for r in s["runs_used"]:
        lines.append(f"  {r['run']:<22} {r['date']:<16} {r['mode']:<5} {r['n_saccade']:>7} {r['n_fixation']:>8} {r['n_excluded']:>8}  {_run_index_text(r):<16} "
                     f"{_num(r['gaze_latency_ms_median']) + ' ms':>10} {_num(r['longest_gap_ms']) + ' ms':>11}")
    if not s["runs_used"]:
        lines.append("  (none)")
    lines.append("  (run index = that run alone, usually too few flashes; n/a = cannot be computed from one run)")

    if s["runs_used"]:
        lines += ["", f"Settings shared by the pooled runs: contrasts {_fmt_contrasts(s['contrasts_pct'])}; screen {_fmt_geometry(s['geometry'])}."]
    lines += ["", "Details (all pooled runs):"]
    lines.append(format_gate_report({**{c: (s["correct"][c], s["n_flashes"][c]) for c in CONDITIONS}, "reasons": s["exclusion_reasons"]}))
    lines.append("  by contrast (correct/flashes):")
    for c in sorted({row["contrast_pct"] for row in s["by_contrast"]}):
        cells = {row["condition"]: row for row in s["by_contrast"] if row["contrast_pct"] == c}
        lines.append(f"    {c:5.1f}%: saccade {cells[SACCADE]['correct']}/{cells[SACCADE]['n']}, fixation {cells[FIXATION]['correct']}/{cells[FIXATION]['n']}, excluded {cells[EXCLUDED]['n']}")
    pooled = {"conditions_from": "gates", "near_ms": 0, "far_ms": 0, **{k: s[k] for k in ("index", "label", "threshold_saccade", "threshold_fixation", "notes")},
              "ci": s["ci"], "n_saccade": s["n_flashes"][SACCADE], "n_fixation": s["n_flashes"][FIXATION],
              "correct_saccade": s["correct"][SACCADE], "correct_fixation": s["correct"][FIXATION]}
    lines.append(format_suppression(pooled))
    if s["slope"] is not None:
        lines.append(f"    shared Weibull slope: {s['slope']:g}")
    if s["by_leg"] is not None:
        lines += ["", f"Return-jump runs ({s['run_kind']}):", format_return_latency(s["return_latency"]), format_leg_suppression(s["by_leg"])]
    lines += ["", "Files: report.txt (this), plot.png, flashes.csv (every pooled flash with its run), summary.json (machine-readable)."]
    return "\n".join(lines)


# -- plot ----------------------------------------------------------------------------------------


def plot_participant(result: ParticipantResult, out_path: Path) -> Path:
    s = result.summary
    fig, (ax_time, ax_psy) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")

    timed = summarize(result.placed, []) if result.placed else None
    bins = [b for b in (timed["bins"] if timed else []) if b["n"]]
    if bins:
        centres = [(b["lo_ms"] + b["hi_ms"]) / 2 for b in bins]
        acc = [b["correct"] / b["n"] for b in bins]
        ax_time.errorbar(centres, [100 * a for a in acc], yerr=[[100 * (a - b["ci"][0]) for a, b in zip(acc, bins)], [100 * (b["ci"][1] - a) for a, b in zip(acc, bins)]],
                         fmt="o-", color="#4C72B0", capsize=4)
        for x, b in zip(centres, bins):
            ax_time.annotate(f"n={b['n']}", (x, 3), ha="center", fontsize=8)
    ax_time.axvline(0, color="#999999", linewidth=1)
    ax_time.axhline(50, color="#999999", linestyle=":", linewidth=1)
    ax_time.set_ylim(0, 105)
    ax_time.set_xlim(-200, 200)
    ax_time.set_xlabel("flash time relative to saccade onset (ms)")
    ax_time.set_ylabel("correct (%), with 95% CI")
    ax_time.set_title("Accuracy vs flash time (all pooled flashes)")
    ax_time.grid(True, alpha=0.3)

    colours = {SACCADE: "#C44E52", FIXATION: "#4C72B0"}
    rows = s["by_contrast"]
    max_c = max([r["contrast_pct"] for r in rows], default=10.0)
    for cond, colour in colours.items():
        pts = [(r["contrast_pct"], r["correct"], r["n"]) for r in rows if r["condition"] == cond and r["n"]]
        if pts:
            cs = [c for c, _, _ in pts]
            acc = [k / n for _, k, n in pts]
            cis = [wilson(k, n) for _, k, n in pts]
            ax_psy.errorbar(cs, [100 * a for a in acc], yerr=[[100 * (a - lo) for a, (lo, _) in zip(acc, cis)], [100 * (hi - a) for a, (_, hi) in zip(acc, cis)]],
                            fmt="o", color=colour, capsize=4, label=f"{cond} (n={sum(n for _, _, n in pts)})")
        threshold = s[f"threshold_{cond}"]
        if s["method"] == "psychometric" and threshold is not None:
            grid = np.linspace(max_c / 200, max_c * 1.5, 200)
            ax_psy.plot(grid, 100 * _weibull(grid / 100, np.asarray(threshold), np.asarray(s["slope"])), color=colour, linewidth=1.5)
            ax_psy.axvline(100 * threshold, color=colour, linestyle="--", linewidth=1, alpha=0.6)
    ax_psy.axhline(50, color="#999999", linestyle=":", linewidth=1)
    ax_psy.set_ylim(0, 105)
    ax_psy.set_xlim(0, max_c * 1.5)
    ax_psy.set_xlabel("line contrast (%)")
    ax_psy.set_ylabel("correct (%), with 95% CI")
    ax_psy.set_title("By contrast and condition" + (" (curves: Weibull fit, dashed: thresholds)" if s["method"] == "psychometric" else ""))
    ax_psy.grid(True, alpha=0.3)
    if rows:
        ax_psy.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Participant {s['participant']}: {headline(s)}", fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -- CLI -----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One suppression index per participant, pooled over all of their compatible runs.")
    parser.add_argument("participant", help="participant ID (as typed in the startup dialog, or the folder name under data/)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--include-test", action="store_true", help="also pool test-mode runs (skipped by default)")
    parser.add_argument("--flash-delay-ms", type=float, default=0.0, help="time from drawing the flash to light leaving the screen")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE, help="smallest saccade to count (fraction of the width)")
    args = parser.parse_args(argv)
    try:
        result = analyze_participant(args.participant, args.data_dir, args.include_test, flash_delay_ms=args.flash_delay_ms, min_amplitude=args.min_amplitude)
    except NoRunsError as exc:
        print(f"Error: {exc}")
        return 1
    print(result.report)
    print(f"\nreport: {result.paths['report']}\nplot: {result.paths['png']}\nper-flash rows: {result.paths['csv']}\nsummary: {result.paths['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
