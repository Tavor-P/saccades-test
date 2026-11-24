"""The text of the post-hoc analysis report, written for the experimenter who runs the study (so terse, no explanations): a summary
first, then headed sections with aligned tables, then a "Technical details" appendix (one geometry line; the analysis thresholds that
differ from the defaults). The building blocks (gate counts, suppression index, return latency, per-leg indices, thresholds) are
shared with analyze_participant. Only ASCII (the report is also printed to Windows consoles). Presentation only: no number is computed here."""

from include.experiment import analysis_constants
from include.experiment.analysis_constants import FAR_MS, MIN_AMPLITUDE, MIN_FIT_TRIALS, NEAR_ONSET_MS
from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE
from src.experiment.leg_gating import LEGS


def _pct(k_n: tuple[int, int]) -> str:
    k, n = k_n
    return f"{k}/{n} = {100 * k / n:.0f}%" if n else "n/a"


def _ms(v, fmt="{:.0f} ms") -> str:
    return "n/a" if v is None else fmt.format(v)


def _p_text(p: float | None) -> str:
    if p is None:
        return "not available"
    return "p < 0.001" if round(p, 3) == 0 else f"p = {p:.3f}"


def _ci_text(ci) -> str:
    return "95% CI not available" if ci is None else f"95% CI {ci[0]:+.2f} to {ci[1]:+.2f}"


def heading(title: str) -> list[str]:
    return ["", title.upper(), "-" * len(title)]


def format_table(header: list[str], rows: list[list[str]], align: str, indent: int = 2) -> list[str]:
    """Fixed-width columns: `align` has one 'l' or 'r' per column. Trailing blanks are trimmed."""
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    out = []
    for row in [header, *rows]:
        cells = [str(c).ljust(w) if a == "l" else str(c).rjust(w) for c, w, a in zip(row, widths, align, strict=True)]
        out.append(" " * indent + "   ".join(cells).rstrip())
    return out


def _correct_cells(k_n: tuple[int, int]) -> tuple[str, str]:
    k, n = k_n
    return (f"{k} / {n}", f"{100 * k / n:.0f}%") if n else ("0 / 0", "n/a")


def index_meaning(method: str | None) -> str | None:
    """How to read the index's scale, which differs between the threshold ratio and the accuracy fallback (the dashboard shows it)."""
    if method == "psychometric":
        return "0 = no suppression, 1 = the line is never seen during saccades, negative = seen better during saccades."
    if method == "accuracy":
        return ("the difference in proportion correct (fixation minus saccade). 0 = no difference; about 0.5 = guessing "
                "during saccades while fixation is perfect (it cannot reach 1, because guessing still gives 50 % correct).")
    return None


# -- technical details: geometry and thresholds ----------------------------------------------------------------------------------


def format_geometry(settings: dict | None) -> str | None:
    """One geometry line from session.json `settings` (None for sessions that predate the geometry record)."""
    s = settings or {}
    if s.get("screen_height_deg") is None:
        return None
    note = " [placeholder, not measured]" if s.get("geometry_is_placeholder") else ""
    return (f"  Screen {s['screen_height_cm']:g} cm @ {s['viewing_distance_cm']:g} cm ({s['screen_height_deg']:.1f} deg), line {s['line_offset_deg']:g} deg "
            f"({s['line_offset_height_units']:.4f} screen heights), SD {s['line_sd_deg']:g} deg{note}")


def thresholds_record(min_amplitude: float = MIN_AMPLITUDE, flash_delay_ms: float = 0.0, near_ms: float | None = None, far_ms: float | None = None) -> dict:
    """Every analysis threshold that made a result (include/experiment/analysis_constants.py, lower-cased names), with the
    run's effective overrides: `min_amplitude` and `flash_delay_ms` (CLI / caller), `index_near_ms` / `index_far_ms` (the
    index's time windows, used only when no gates decide the conditions). Tuples become lists (JSON)."""
    out = {}
    for name in dir(analysis_constants):
        if name.isupper():
            value = getattr(analysis_constants, name)
            out[name.lower()] = list(value) if isinstance(value, tuple) else value
    out["min_amplitude"] = min_amplitude
    out["index_near_ms"] = NEAR_ONSET_MS if near_ms is None else near_ms
    out["index_far_ms"] = FAR_MS if far_ms is None else far_ms
    out["flash_delay_ms"] = flash_delay_ms
    return out


def _value_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value) + "]"
    return f"{value:g}" if isinstance(value, float) else str(value)


def format_thresholds(record: dict) -> str:
    """Only the thresholds that differ from include/experiment/analysis_constants.py, one per line by constant name
    (the full record is in suppression.json / summary.json `analysis_thresholds`)."""
    defaults = thresholds_record()
    changed = {k: v for k, v in record.items() if defaults.get(k) != v}
    if not changed:
        return "  Analysis thresholds: defaults (analysis_constants.py)"
    lines = ["  Analysis thresholds that differ from the defaults (analysis_constants.py):"]
    lines += [f"    {k} = {_value_text(v)}" + (f"  (default {_value_text(defaults[k])})" if k in defaults else "") for k, v in changed.items()]
    return "\n".join(lines)


# -- the shared blocks -----------------------------------------------------------------------------------------------------


def _index_text(s: dict) -> str:
    if s["index"] is None:
        return "cannot be computed"
    return f"{s['index']:+.2f}  ({_ci_text(s['ci'])})" + ("  [accuracy fallback]" if s["method"] == "accuracy" else "")


def index_details(s: dict) -> list[str]:
    """The index' threshold contrasts and notes (one line each)."""
    lines = []
    if s["index"] is not None and s["threshold_saccade"] is not None:
        lines.append(f"  Threshold contrast: saccade {100 * s['threshold_saccade']:.1f}%, fixation {100 * s['threshold_fixation']:.1f}%")
    lines += [f"  Note: {n}" for n in s["notes"]]
    return lines


def format_suppression(s: dict) -> str:
    """The suppression index with its 95% CI, threshold contrasts and notes."""
    return "\n".join([f"  Suppression index: {_index_text(s)}", *index_details(s)])


def format_gate_report(gates: dict) -> str:
    """The pre-specified gating (src/experiment/flash_gates.py): saccade vs fixation flashes and exclusions."""
    sac, fix = _correct_cells(tuple(gates[SACCADE])), _correct_cells(tuple(gates[FIXATION]))
    lines = heading("Main comparison (pre-specified gates)")
    lines += format_table(["Group", "Correct", "Accuracy"], [["Saccade", sac[0], sac[1]], ["Fixation", fix[0], fix[1]]], "lrr")
    lines.append(f"  Excluded: {gates[EXCLUDED][1]}" + "".join(f"\n    {n} x {reason}" for reason, n in gates["reasons"].items()))
    return "\n".join(lines)


def format_leg_suppression(by_leg: dict) -> str:
    """Suppression index for the out leg, the return leg (each: its saccade flashes vs ALL fixation flashes) and both pooled."""
    rows = []
    for leg in (*LEGS, "pooled"):
        s = by_leg[leg]
        rows.append([leg, f"{s['correct_saccade']} / {s['n_saccade']}", f"{s['correct_fixation']} / {s['n_fixation']}",
                     "cannot be computed" if s["index"] is None else f"{s['index']:+.2f}",
                     "" if s["ci"] is None else f"{s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}",
                     "" if s["index"] is None else s["method"]])
    return "\n".join([*heading("Suppression index by leg"), *format_table(["Leg", "Saccade", "Fixation", "Index", "95% CI", "Method"], rows, "lrrrrl")])


def format_return_latency(r: dict) -> str:
    """The return-latency block of a return-jump run (what RETURN_FLASH_SPREAD_SD_MS should be set from)."""
    lines = heading("Return latency")
    lines.append(f"  Trials jumped: {r['trials_jumped']} ({r['forced_jumps']} forced jumps left out, no return saccade found in {r['no_return_found']})")
    if r["n"]:
        lines.append(f"  n {r['n']}: median {_ms(r['median_ms'])}, SD {_ms(r['sd_ms'])} (robust SD {_ms(r['robust_sd_ms'])}), predictive {100 * r['share_predictive']:.0f}%")
    else:
        lines.append("  No return saccade found: no latency")
    if r["live_n"]:
        lines.append(f"  Live classifier (return_reaction_ms): n {r['live_n']}, median {_ms(r['live_median_ms'])}, SD {_ms(r['live_sd_ms'])}")
    if r.get("return_flash_spread_sd_ms") is not None:
        lines.append(f"  Return flash spread SD (RETURN_FLASH_SPREAD_SD_MS): {r['return_flash_spread_sd_ms']:g} ms; trigger {r.get('return_jump_trigger') or '?'}")
    return "\n".join(lines)


# -- the per-run report ----------------------------------------------------------------------------------------------------


def _summary(s: dict, test_run: bool) -> list[str]:
    """The top of the report: the main result, the usable-flash count and the short warnings."""
    gates = s["gates"]
    sac, fix = _correct_cells(tuple(gates[SACCADE])), _correct_cells(tuple(gates[FIXATION]))
    n_sac, n_fix, n_excl = gates[SACCADE][1], gates[FIXATION][1], gates[EXCLUDED][1]
    lines = heading("Summary")
    if test_run:
        lines.append("  TEST run, not a real recording")
    lines += [f"  Saccade:   {sac[0]} correct ({sac[1]})", f"  Fixation:  {fix[0]} correct ({fix[1]})", format_suppression(s["suppression"]),
              f"  Usable flashes: {n_sac + n_fix} of {s['flashes']} ({n_sac} saccade, {n_fix} fixation, {n_excl} excluded)"]
    if n_sac < MIN_FIT_TRIALS or n_fix < MIN_FIT_TRIALS:
        lines.append(f"  WARNING: only {n_sac} saccade and {n_fix} fixation flashes, too few for a firm conclusion (threshold fit needs {MIN_FIT_TRIALS}+ each)")
    return lines


def _data_quality(s: dict) -> list[str]:
    lines = heading("Data quality")
    found = f"{s['saccades']}" + (f" (median duration {s['median_saccade_ms']:.0f} ms, amplitude {s['median_amplitude']:.2f} of the width)" if s["median_saccade_ms"] else "")
    lines += [f"  Saccades found: {found}", f"  Flashes: {s['flashes']}; correct {_pct(s['overall'])}; no answer {s['misses']}"]
    return lines


def _time_windows(s: dict) -> list[str]:
    """Accuracy by the flash's time relative to the nearest saccade onset: windows (with Fisher p) and the fine bins."""
    lines = heading("Accuracy by flash time")
    rows = []
    for name, key in (("In a saccade", "inside"), ("Outside a saccade", "outside"), (f"+/-{NEAR_ONSET_MS:.0f} ms of onset", "near_onset"),
                      (f"{NEAR_ONSET_MS:.0f}-{FAR_MS:.0f} ms from onset", "beyond"), (f"Fixation control (>= {FAR_MS:.0f} ms)", "far")):
        c = _correct_cells(s[key])
        rows.append([name, c[0], c[1]])
    lines += format_table(["Group", "Correct", "Accuracy"], rows, "lrr")
    lines += [f"  In vs outside a saccade: Fisher {_p_text(s['p_inside_vs_outside'])}", f"  Near onset vs further away: Fisher {_p_text(s['p_near_vs_beyond'])}", ""]
    rows = []
    for b in s["bins"]:
        lo, hi = b["ci"]
        c = _correct_cells((b["correct"], b["n"])) if b["n"] else ("-", "-")
        rows.append([f"{b['lo_ms']:+d} to {b['hi_ms']:+d} ms", str(b["n"]), c[0], c[1], f"{100 * lo:.0f}-{100 * hi:.0f}%" if b["n"] else "-"])
    lines += format_table(["Onset-relative", "n", "Correct", "Accuracy", "95% CI"], rows, "lrrrr")
    return lines


def _by_contrast(s: dict) -> list[str]:
    lines = heading("Accuracy by contrast")
    rows = [[f"{100 * c:.1f}%", _pct(g["all"]), _pct(g["near_onset"]), _pct(g["beyond"])] for c, g in s["by_contrast"].items()]
    lines += format_table(["Contrast", "All", f"+/-{NEAR_ONSET_MS:.0f} ms", f"{NEAR_ONSET_MS:.0f}-{FAR_MS:.0f} ms"], rows, "rlll")
    return lines


def format_report(summary: dict, label: str = "", geometry: str | None = None, thresholds: dict | None = None, test_run: bool = False) -> str:
    """The whole per-run report from analyze_session's `summary` (summarize() plus "gates", "suppression" and, for return-jump
    runs, "by_leg" and "return_latency"). `geometry` is format_geometry()'s text, `thresholds` thresholds_record()."""
    title = f"Saccade-phase analysis {label}".strip()
    lines = [title, "=" * len(title), *_summary(summary, test_run), *_data_quality(summary), format_gate_report(summary["gates"]),
             *_time_windows(summary), *_by_contrast(summary)]
    if summary.get("by_leg"):
        lines.append(format_leg_suppression(summary["by_leg"]))
    if summary.get("return_latency"):
        lines.append(format_return_latency(summary["return_latency"]))
    lines += heading("Technical details")
    if geometry:
        lines.append(geometry)
    if thresholds is not None:
        lines.append(format_thresholds(thresholds))
    return "\n".join(lines)
