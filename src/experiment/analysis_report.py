"""The text of the post-hoc analysis report, for the experimenter who reads one after every run (machines read the CSV / JSON files):
a three-line top block (QUALITY, RESULT, AMOUNT), then the same headed sections in the same order every run (a section with nothing
to show says so in one line), ending in "Technical details" (one geometry line; the thresholds that differ from the defaults).
The building blocks are shared with analyze_participant. Presentation only: no analysis number is computed here."""

from include.experiment import analysis_constants
from include.experiment.analysis_constants import FAR_MS, MIN_AMPLITUDE, MIN_FIT_TRIALS, NEAR_ONSET_MS
from src.experiment.flash_gates import EXCLUDED, FIXATION, NOT_MOVING_REASON, SACCADE
from src.experiment.leg_gating import LEGS

SPREAD_FLAG_SHARE = 0.25  # flag a return-flash spread SD that is off from the measured robust SD by more than this share


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

LABEL_WIDTH, STATUS_WIDTH = 9, 10  # the left columns of the three-line top block


def _top_line(label: str, status: str, text: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{status:<{STATUS_WIDTH}}{text}"


def format_suppression(s: dict) -> str:
    """The RESULT line: the index and its 95% CI (a suppression_index() result, or the pooled equivalent)."""
    if s["index"] is None:
        why = f" ({s['notes'][-1]})" if s["notes"] else ""
        return _top_line("RESULT", "n/a", f"suppression index cannot be computed{why}")
    fallback = " (accuracy fallback)" if s["method"] == "accuracy" else ""
    return _top_line("RESULT", f"{s['index']:+.2f}", f"suppression index, {_ci_text(s['ci'])}{fallback}")


def format_top_block(quality: tuple[str, str] | None, suppression: dict, n_saccade: int, n_fixation: int, total: int, tail: str = "") -> str:
    """The three lines read first, always in this order and format: QUALITY (OK / CHECK / BAD), RESULT (the index),
    AMOUNT (OK / TOO FEW). `quality` is run_quality.quality_line() (None: unknown)."""
    q_status, q_text = quality if quality is not None else ("CHECK", "quality unknown (no session data)")
    few = n_saccade < MIN_FIT_TRIALS or n_fixation < MIN_FIT_TRIALS
    amount = f"{n_saccade} saccade / {n_fixation} fixation flashes usable of {total} (need {MIN_FIT_TRIALS}+ each for a threshold fit)" + tail
    return "\n".join([_top_line("QUALITY", q_status, q_text), format_suppression(suppression), _top_line("AMOUNT", "TOO FEW" if few else "OK", amount)])


def format_index_fit(s: dict) -> list[str]:
    """INDEX FIT section: the threshold contrasts of the index and its notes (always present)."""
    fitted = s["index"] is not None and s["threshold_saccade"] is not None
    lines = heading("Index fit")
    lines.append(f"  Threshold contrast: saccade {100 * s['threshold_saccade']:.1f}%, fixation {100 * s['threshold_fixation']:.1f}%" if fitted
                 else "  Threshold contrast: no threshold fit")
    lines += [f"  Note: {n}" for n in s["notes"]] or ["  Notes: none"]
    return lines


def speed_exclusions(placed: list, gates: list) -> tuple[int, int, int]:
    """Flashes excluded as 'near a saccade but not moving fast yet': (before onset, after onset, no onset known)."""
    before = after = unknown = 0
    for p, g in zip(placed, gates, strict=True):
        if g.condition == EXCLUDED and g.reason == NOT_MOVING_REASON:
            if p.rel_ms is None:
                unknown += 1
            elif p.rel_ms < 0:
                before += 1
            else:
                after += 1
    return before, after, unknown


def format_excluded(gates: dict, speed: tuple[int, int, int] | None = None) -> list[str]:
    """One line with the excluded-flash breakdown, and one line on those excluded for 'not moving fast yet'."""
    n = gates[EXCLUDED][1]
    reasons = "; ".join(f"{k} x {reason}" for reason, k in gates["reasons"].items())
    lines = [f"  Excluded: {n}" + (f" ({reasons})" if reasons else "")]
    if speed is not None:
        before, after, unknown = speed
        lines.append(f"  Excluded as not moving fast yet: {before} before saccade onset, {after} after" + (f", {unknown} with no onset" if unknown else ""))
    return lines


def format_leg_suppression(by_leg: dict | None, leg_flashes: dict | None = None, what: str = "this run") -> str:
    """Suppression index for the out leg, the return leg (each: its saccade flashes vs ALL fixation flashes) and both pooled, with
    each leg's flash count. A run without return jumps (`by_leg` None) prints one line."""
    lines = heading("Suppression index by leg")
    if not by_leg:
        return "\n".join([*lines, f"  no return jump in {what}"])
    rows = []
    for leg in (*LEGS, "pooled"):
        s = by_leg[leg]
        flashes = None if leg_flashes is None else sum(leg_flashes.values()) if leg == "pooled" else leg_flashes.get(leg, 0)
        rows.append([leg, "?" if flashes is None else str(flashes), str(s["n_saccade"]), f"{s['correct_saccade']} / {s['n_saccade']}", f"{s['correct_fixation']} / {s['n_fixation']}",
                     "cannot be computed" if s["index"] is None else f"{s['index']:+.2f}",
                     "" if s["ci"] is None else f"{s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}",
                     "" if s["index"] is None else s["method"]])
    lines += format_table(["Leg", "Flashes", "In a saccade", "Saccade", "Fixation", "Index", "95% CI", "Method"], rows, "lrrrrrrl")
    return "\n".join(lines)


def format_return_latency(r: dict | None, what: str = "this run") -> str:
    """The return-latency block of a return-jump run (what RETURN_FLASH_SPREAD_SD_MS should be set from); one line without a jump."""
    lines = heading("Return latency")
    if not r:
        return "\n".join([*lines, f"  no return jump in {what}"])
    lines.append(f"  Trials jumped: {r['trials_jumped']} ({r['forced_jumps']} forced jumps left out, no return saccade found in {r['no_return_found']})")
    if r["n"]:
        lines.append(f"  n {r['n']}: median {_ms(r['median_ms'])}, SD {_ms(r['sd_ms'])} (robust SD {_ms(r['robust_sd_ms'])}), predictive {100 * r['share_predictive']:.0f}%")
    else:
        lines.append("  No return saccade found: no latency")
    if r["live_n"]:
        lines.append(f"  Live classifier (return_reaction_ms): n {r['live_n']}, median {_ms(r['live_median_ms'])}, SD {_ms(r['live_sd_ms'])}")
    spread = r.get("return_flash_spread_sd_ms")
    if spread is not None:
        lines.append(f"  Return flash spread SD (RETURN_FLASH_SPREAD_SD_MS): {spread:g} ms; trigger {r.get('return_jump_trigger') or '?'}")
        measured = r.get("robust_sd_ms")
        if r["n"] and measured is not None and spread > 0 and abs(measured - spread) / spread > SPREAD_FLAG_SHARE:
            lines.append(f"  CHECK: measured {measured:.0f} ms vs set {spread:g} ms: consider {'raising' if measured > spread else 'lowering'} RETURN_FLASH_SPREAD_SD_MS")
    return "\n".join(lines)


# -- the per-run report ----------------------------------------------------------------------------------------------------


def _data_quality(s: dict, speed: tuple[int, int, int] | None) -> list[str]:
    lines = heading("Data quality")
    found = f"{s['saccades']}" + (f" (median duration {s['median_saccade_ms']:.0f} ms, amplitude {s['median_amplitude']:.2f} of the width)" if s["median_saccade_ms"] else "")
    lines += [f"  Saccades found: {found}", f"  Flashes: {s['flashes']}; correct {_pct(s['overall'])}",
              f"  No answer: {s['misses']} (scored as not correct, like a wrong answer; not excluded)"]
    return lines + format_excluded(s["gates"], speed)


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


def format_technical(geometry: str | None, thresholds: dict | None) -> list[str]:
    """TECHNICAL DETAILS section: one geometry line, then the thresholds that differ from the defaults."""
    return [*heading("Technical details"), geometry or "  Screen geometry: not recorded",
            format_thresholds(thresholds) if thresholds is not None else "  Analysis thresholds: not recorded"]


def format_report(
    summary: dict,
    label: str = "",
    geometry: str | None = None,
    thresholds: dict | None = None,
    test_run: bool = False,
    quality: tuple[str, str] | None = None,
    leg_flashes: dict | None = None,
    speed: tuple[int, int, int] | None = None,
) -> str:
    """The whole per-run report from analyze_session's `summary` (summarize() plus "gates", "suppression" and, for return-jump
    runs, "by_leg" and "return_latency"): the three-line top block, then the same sections in the same order every run.
    `quality` is run_quality.quality_line(); `leg_flashes` {leg: flash count}; `speed` is speed_exclusions(); `geometry` is
    format_geometry()'s text, `thresholds` thresholds_record()."""
    title = f"Saccade-phase analysis {label}".strip()
    gates = summary["gates"]
    top = format_top_block(quality, summary["suppression"], gates[SACCADE][1], gates[FIXATION][1], summary["flashes"], " · TEST run" if test_run else "")
    lines = [title, "=" * len(title), "", top, *_data_quality(summary, speed), *format_index_fit(summary["suppression"]), *_time_windows(summary),
             *_by_contrast(summary), format_leg_suppression(summary.get("by_leg"), leg_flashes), format_return_latency(summary.get("return_latency")),
             *format_technical(geometry, thresholds)]
    return "\n".join(lines)
