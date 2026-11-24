"""The text of the post-hoc analysis report: the per-run summary, the gate counts, the suppression index, the return-latency
and per-leg blocks (used by analyze_session and analyze_participant)."""

import textwrap

from include.experiment import analysis_constants
from include.experiment.analysis_constants import FAR_MS, FIXATION_EXCLUSION_MS, MIN_AMPLITUDE, NEAR_ONSET_MS, RETURN_LATENCY_MAX_MS
from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE
from src.experiment.leg_gating import LEGS


def _pct(k_n: tuple[int, int]) -> str:
    k, n = k_n
    return f"{k}/{n} = {100 * k / n:.0f}%" if n else "n/a"


def format_geometry(settings: dict | None) -> str | None:
    """One header line from session.json `settings` (None for sessions that predate the geometry record)."""
    s = settings or {}
    if s.get("screen_height_deg") is None:
        return None
    note = "  [PLACEHOLDER: not measured]" if s.get("geometry_is_placeholder") else ""
    return (
        f"  screen {s['screen_height_cm']:g} cm high at {s['viewing_distance_cm']:g} cm = {s['screen_height_deg']:.1f} deg; line offset "
        f"{s['line_offset_deg']:g} deg = {s['line_offset_height_units']:.4f} screen heights, SD {s['line_sd_deg']:g} deg{note}"
    )


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


def format_thresholds(record: dict) -> str:
    """The thresholds record as indented `name=value` lines for a report header."""
    text = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
    body = textwrap.wrap(text, width=116, initial_indent="    ", subsequent_indent="    ", break_on_hyphens=False)
    return "\n".join(["  analysis thresholds (include/experiment/analysis_constants.py; overrides applied):", *body])


def format_report(summary: dict, label: str = "", geometry: str | None = None, thresholds: dict | None = None) -> str:
    s = summary
    lines = [f"Saccade-phase analysis {label}".strip()]
    if geometry:
        lines.append(geometry)
    if thresholds is not None:
        lines.append(format_thresholds(thresholds))
    lines.append(f"  saccades found: {s['saccades']}" + (
        f" (median duration {s['median_saccade_ms']:.0f} ms, amplitude {s['median_amplitude']:.2f} of the width)" if s["median_saccade_ms"] else ""))
    lines.append(f"  flashes: {s['flashes']}; correct {_pct(s['overall'])}; no answer {s['misses']}")
    lines.append(f"  inside a saccade: {_pct(s['inside'])} | outside: {_pct(s['outside'])}" + (
        f" (Fisher p = {s['p_inside_vs_outside']:.3f})" if s["p_inside_vs_outside"] is not None else ""))
    lines.append(f"  within +-{NEAR_ONSET_MS:.0f} ms of onset: {_pct(s['near_onset'])} | {NEAR_ONSET_MS:.0f}-{FAR_MS:.0f} ms away: {_pct(s['beyond'])}" + (
        f" (Fisher p = {s['p_near_vs_beyond']:.3f})" if s["p_near_vs_beyond"] is not None else ""))
    lines.append(f"  fixation control (>= {FAR_MS:.0f} ms from any onset): {_pct(s['far'])}")
    lines.append("  accuracy by flash time relative to saccade onset:")
    for b in s["bins"]:
        lo, hi = b["ci"]
        lines.append(f"    {b['lo_ms']:+4d}..{b['hi_ms']:+4d} ms  n={b['n']:2d}  " + (f"{_pct((b['correct'], b['n']))}  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if b["n"] else "-"))
    lines.append("  by contrast:")
    for c, g in s["by_contrast"].items():
        lines.append(f"    {100 * c:5.1f}%: all {_pct(g['all'])} | near onset {_pct(g['near_onset'])} | beyond {_pct(g['beyond'])}")
    return "\n".join(lines)


def format_suppression(s: dict) -> str:
    how = ("conditions from the pre-specified gates, excluded flashes left out" if s.get("conditions_from") == "gates"
           else f"conditions from time windows: saccade = within +-{s['near_ms']:.0f} ms of onset, fixation = >= {s['far_ms']:.0f} ms away")
    lines = [f"  suppression index ({how}):",
             f"    trials: saccade {s['correct_saccade']}/{s['n_saccade']} correct, fixation {s['correct_fixation']}/{s['n_fixation']} correct"]
    if s["index"] is None:
        lines.append("    index: cannot be computed")
    else:
        ci = "95% CI not available" if s["ci"] is None else f"95% CI {s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}"
        lines.append(f"    index = {s['index']:+.2f} ({ci}); {s['label']}")
        if s["threshold_saccade"] is not None:
            lines.append(f"    thresholds: saccade {100 * s['threshold_saccade']:.1f}%, fixation {100 * s['threshold_fixation']:.1f}% contrast")
    lines += [f"    note: {n}" for n in s["notes"]]
    return "\n".join(lines)


def format_gate_report(gates: dict) -> str:
    """The pre-specified gating (src/experiment/flash_gates.py): saccade vs fixation flashes and exclusions."""
    lines = ["Gated analysis (pre-specified criteria, not tuned on the data):"]
    lines.append(f"  saccade flashes (eye moving fast, valid latency and amplitude): {_pct(tuple(gates[SACCADE]))}")
    lines.append(f"  fixation flashes (>= {FIXATION_EXCLUSION_MS:.0f} ms from any saccade, tracked throughout): {_pct(tuple(gates[FIXATION]))}")
    lines.append(f"  excluded: {gates[EXCLUDED][1]}" + "".join(f"\n    {n} x {reason}" for reason, n in gates["reasons"].items()))
    return "\n".join(lines)


def _ms(v, fmt="{:.0f} ms") -> str:
    return "n/a" if v is None else fmt.format(v)


def format_return_latency(r: dict) -> str:
    """The return-latency block of a return-jump run (what RETURN_FLASH_SPREAD_SD_MS should be set from)."""
    lines = ["Return latency (return saccade onset minus the jump-back frame, from the raw trace; < 0 = predictive):"]
    lines.append(f"  trials that jumped: {r['trials_jumped']} ({r['forced_jumps']} forced jumps left out, no return saccade found in {r['no_return_found']})")
    if r["n"]:
        lines.append(f"  n {r['n']}: median {_ms(r['median_ms'])}, SD {_ms(r['sd_ms'])} (robust SD {_ms(r['robust_sd_ms'])}), "
                     f"predictive {100 * r['share_predictive']:.0f}%")
    else:
        lines.append("  no return saccade found: no latency")
    if r["live_n"]:
        lines.append(f"  live classifier (return_reaction_ms, cross-check): n {r['live_n']}, median {_ms(r['live_median_ms'])}, SD {_ms(r['live_sd_ms'])}")
    if r.get("return_flash_spread_sd_ms") is not None:
        lines.append(f"  this run's return flash spread SD (RETURN_FLASH_SPREAD_SD_MS): {r['return_flash_spread_sd_ms']:g} ms; trigger: {r.get('return_jump_trigger') or '?'}")
    lines.append("  (the jump time is anchored on the scheduled flash delay, so it can be up to one frame late: latencies near 0 read slightly low)")
    return "\n".join(lines)


def format_leg_suppression(by_leg: dict) -> str:
    """Suppression index for the out leg, the return leg (each: its saccade flashes vs ALL fixation flashes) and both pooled."""
    lines = ["Suppression index by leg (each leg's saccade flashes against the shared fixation flashes):"]
    for leg in (*LEGS, "pooled"):
        s = by_leg[leg]
        ci = "" if s["ci"] is None else f", 95% CI {s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}"
        value = "cannot be computed" if s["index"] is None else f"{s['index']:+.2f} ({s['method']}{ci})"
        lines.append(f"  {leg:<7} saccade {s['correct_saccade']}/{s['n_saccade']}, fixation {s['correct_fixation']}/{s['n_fixation']}: {value}")
    lines.append(f"  (return-leg saccade gates: own return saccade, latency from the jump <= {RETURN_LATENCY_MAX_MS:.0f} ms, no lower bound)")
    return "\n".join(lines)
