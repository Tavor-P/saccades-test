"""The text of the post-hoc analysis report, written for a researcher who is not a programmer: a plain-language summary first, then
headed sections with aligned tables, then a "Technical details" appendix (screen geometry, every analysis threshold).
The building blocks (gate counts, suppression index, return latency, per-leg indices, thresholds) are shared with analyze_participant.
Only ASCII (the report is also printed to Windows consoles). Presentation only: no number is computed here."""

import textwrap

from include.experiment import analysis_constants
from include.experiment.analysis_constants import FAR_MS, FIXATION_EXCLUSION_MS, MIN_AMPLITUDE, MIN_FIT_TRIALS, NEAR_ONSET_MS, RETURN_LATENCY_MAX_MS
from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE
from src.experiment.leg_gating import LEGS

RULE_WIDTH = 78


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


def _wrap(text: str, indent: int = 2, first: str = "") -> list[str]:
    """`text` wrapped to the report's width, every line indented (the first one after `first`, e.g. 'Note: ')."""
    pad = " " * indent
    return textwrap.wrap(first + text, width=RULE_WIDTH + 30, initial_indent=pad, subsequent_indent=pad + " " * len(first), break_on_hyphens=False)


def _heading(title: str) -> list[str]:
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
    """How to read the index's scale, which differs between the threshold ratio and the accuracy fallback."""
    if method == "psychometric":
        return "0 = no suppression, 1 = the line is never seen during saccades, negative = seen better during saccades."
    if method == "accuracy":
        return ("the difference in proportion correct (fixation minus saccade). 0 = no difference; about 0.5 = guessing "
                "during saccades while fixation is perfect (it cannot reach 1, because guessing still gives 50 % correct).")
    return None


# -- technical details: geometry and thresholds ----------------------------------------------------------------------------------


def format_geometry(settings: dict | None) -> str | None:
    """The screen geometry as plain lines from session.json `settings` (None for sessions that predate the geometry record)."""
    s = settings or {}
    if s.get("screen_height_deg") is None:
        return None
    lines = [
        f"  Screen: {s['screen_height_cm']:g} cm high, viewed from {s['viewing_distance_cm']:g} cm = {s['screen_height_deg']:.1f} degrees of visual angle tall.",
        f"  Flash line: {s['line_offset_deg']:g} degrees above or below the target = {s['line_offset_height_units']:.4f} screen heights; "
        f"blur (Gaussian SD) {s['line_sd_deg']:g} degrees.",
    ]
    if s.get("geometry_is_placeholder"):
        lines += _wrap("the screen height and viewing distance were NOT measured; these are the program's default guesses, so the sizes in degrees "
                       "(and the line's real size on screen) are only approximate.", 2, "PLACEHOLDER: ")
    return "\n".join(lines)


# name -> (group, what it is, unit). Mirrors include/experiment/analysis_constants.py; a name missing here is still listed (by its raw name).
THRESHOLD_DOC: dict[str, tuple[str, str, str]] = {
    "speed_high": ("Saccade detection", "Eye speed that starts a saccade", "screen widths per second"),
    "speed_low": ("Saccade detection", "Eye speed below which a saccade starts/ends", "screen widths per second"),
    "min_amplitude": ("Saccade detection", "Smallest movement that counts as a saccade", "of the screen width"),
    "near_onset_ms": ("Time windows around saccade onset", "'Near onset' = flash within this of a saccade start", "ms"),
    "far_ms": ("Time windows around saccade onset", "Fixation control = flash at least this far from every onset", "ms"),
    "bin_edges_ms": ("Time windows around saccade onset", "Edges of the accuracy-by-flash-time bins", "ms from onset"),
    "index_near_ms": ("Time windows around saccade onset", "Index fallback: 'saccade' = within this of an onset", "ms"),
    "index_far_ms": ("Time windows around saccade onset", "Index fallback: 'fixation' = at least this far from every onset", "ms"),
    "flash_delay_ms": ("Time windows around saccade onset", "Delay between drawing the flash and light leaving the screen", "ms"),
    "fixation_exclusion_ms": ("Pre-specified flash gates", "A fixation flash must be this far from every saccade", "ms"),
    "tracking_gap_ms": ("Pre-specified flash gates", "Gaze gap around a flash counted as a blink / tracking loss", "ms"),
    "latency_range_ms": ("Pre-specified flash gates", "Valid saccade latency after the go cue (faster = anticipation, slower = lapse)", "ms"),
    "amplitude_range_rel": ("Pre-specified flash gates", "Valid saccade size relative to this run's median", "x median"),
    "min_flash_speed": ("Pre-specified flash gates", "Eye must be at least this fast when the flash was drawn", "screen widths per second"),
    "max_flash_speed": ("Pre-specified flash gates", "Faster than this is a tracking spike, not an eye", "screen widths per second"),
    "return_latency_max_ms": ("Pre-specified flash gates", "Return saccade must start within this after the jump (no lower bound)", "ms"),
    "out_search_s": ("Return-jump runs: finding each trial's saccades", "Out saccade: first onset within this of the go cue", "s"),
    "return_search_after_jump_s": ("Return-jump runs: finding each trial's saccades", "Return saccade starts at most this long after the jump", "s"),
    "return_min_amplitude_rel": ("Return-jump runs: finding each trial's saccades", "Return must move back at least this x the out amplitude", "x out amplitude"),
    "min_fit_trials": ("Suppression index", "Fewest flashes per condition for a threshold fit", "flashes"),
    "min_accuracy_trials": ("Suppression index", "Fewest flashes per condition for the accuracy fallback", "flashes"),
    "chance": ("Suppression index", "Guess rate of the two-choice task", "proportion"),
    "lapse": ("Suppression index", "Lapse rate assumed in the psychometric function", "proportion"),
    "threshold_extrapolation": ("Suppression index", "A fitted threshold this many times outside the tested contrasts is 'not identified'", "x"),
    "boot_min_identified_share": ("Suppression index", "Bootstrap CI needs at least this share of identified resamples...", "proportion"),
    "boot_min_identified": ("Suppression index", "...and at least this many of them", "resamples"),
    "accuracy_floor_margin": ("Suppression index", "Accuracy fallback refused when fixation accuracy < chance + this", "proportion"),
    "accuracy_ceiling": ("Suppression index", "Both conditions at or above this accuracy: index cannot show suppression", "proportion"),
}


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
        nums = [f"{v:g}" if isinstance(v, float) else str(v) for v in value]
        return f"{nums[0]} to {nums[1]}" if len(nums) == 2 else ", ".join(nums)
    return f"{value:g}" if isinstance(value, float) else str(value)


def format_thresholds(record: dict) -> str:
    """The thresholds record as grouped lines, one threshold per line: what it is, its value and unit, its constant's name."""
    by_group: dict[str, list[list[str]]] = {}
    order = list(THRESHOLD_DOC)
    for name, value in sorted(record.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)):
        group, text, unit = THRESHOLD_DOC.get(name, ("Other", name, ""))
        by_group.setdefault(group, []).append([text, f"{_value_text(value)} {'' if unit == 'proportion' else unit}".strip(), f"({name})"])
    lines = ["  Analysis thresholds (every rule that made these results; defined in include/experiment/analysis_constants.py,",
             "  with this run's overrides applied):"]
    for group in [g for g in dict.fromkeys(doc[0] for doc in THRESHOLD_DOC.values()) if g in by_group] + (["Other"] if "Other" in by_group else []):
        lines += ["", f"  {group}:"]
        lines += format_table(["", "", ""], by_group[group], "lll", indent=4)[1:]
    return "\n".join(lines)


# -- the shared blocks -----------------------------------------------------------------------------------------------------


def format_suppression(s: dict) -> str:
    """The suppression index of one condition split: counts, index with its 95% CI, thresholds, notes."""
    gates = s.get("conditions_from") == "gates"
    how = ("the pre-specified gates (excluded flashes are left out)" if gates
           else f"time windows (saccade = within +/-{s['near_ms']:.0f} ms of onset, fixation = at least {s['far_ms']:.0f} ms away)")
    lines = ["  Suppression index",
             f"    Flashes counted as 'saccade' and 'fixation' were chosen by: {how}.",
             f"    Saccade flashes:  {s['correct_saccade']} of {s['n_saccade']} correct",
             f"    Fixation flashes: {s['correct_fixation']} of {s['n_fixation']} correct"]
    if s["index"] is None:
        lines.append("    Index: cannot be computed")
    else:
        lines.append(f"    Index: {s['index']:+.2f}  ({_ci_text(s['ci'])})")
        lines += _wrap(s["label"], 4, "How it was computed: ")
        if s["threshold_saccade"] is not None:
            lines.append(f"    Contrast needed to see the line (threshold): during saccades {100 * s['threshold_saccade']:.1f}%, "
                         f"during fixation {100 * s['threshold_fixation']:.1f}%")
    for n in s["notes"]:
        lines += _wrap(n, 4, "Note: ")
    return "\n".join(lines)


def format_gate_report(gates: dict) -> str:
    """The pre-specified gating (src/experiment/flash_gates.py): saccade vs fixation flashes and exclusions."""
    sac, fix = _correct_cells(tuple(gates[SACCADE])), _correct_cells(tuple(gates[FIXATION]))
    lines = _heading("Main comparison: flashes during a saccade vs during fixation")
    lines += ["  Every flash was sorted by fixed rules, set before looking at any data (after Braun, Schutz & Gegenfurtner 2021),",
              "  and flashes that fit neither group were left out.", ""]
    lines += format_table(["Group", "Correct", "Accuracy", "Rule"], [
        ["Flashes during a saccade", sac[0], sac[1], "drawn while the eye moved fast, in a saccade of valid latency and size"],
        ["Flashes during fixation", fix[0], fix[1], f"at least {FIXATION_EXCLUSION_MS:.0f} ms from any saccade, tracking intact"],
    ], "lrrl")
    lines.append(f"  Excluded (fit neither group): {gates[EXCLUDED][1]}" + "".join(f"\n    {n} x {reason}" for reason, n in gates["reasons"].items()))
    return "\n".join(lines)


def format_leg_suppression(by_leg: dict) -> str:
    """Suppression index for the out leg, the return leg (each: its saccade flashes vs ALL fixation flashes) and both pooled."""
    lines = _heading("Suppression index by saccade leg")
    lines += ["  'out' = the saccade to the target, 'return' = the saccade back after the target jumped back, 'pooled' = both together",
              "  (the main result). Each leg's saccade flashes are compared with the same, shared fixation flashes.", ""]
    rows = []
    for leg in (*LEGS, "pooled"):
        s = by_leg[leg]
        rows.append([leg, f"{s['correct_saccade']} / {s['n_saccade']}", f"{s['correct_fixation']} / {s['n_fixation']}",
                     "cannot be computed" if s["index"] is None else f"{s['index']:+.2f}",
                     "" if s["ci"] is None else f"{s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}",
                     "" if s["index"] is None else s["method"]])
    lines += format_table(["Leg", "Saccade correct", "Fixation correct", "Index", "95% CI", "Method"], rows, "lrrrrl")
    lines += ["  Method: 'psychometric' = ratio of fitted contrast thresholds; 'accuracy' = difference in proportion correct (fallback when too few",
              "  flashes for a fit).",
              f"  Return-leg saccade rule: the eye's own return saccade, starting at most {RETURN_LATENCY_MAX_MS:.0f} ms after the jump (no lower bound)."]
    return "\n".join(lines)


def format_return_latency(r: dict) -> str:
    """The return-latency block of a return-jump run (what RETURN_FLASH_SPREAD_SD_MS should be set from)."""
    lines = _heading("Return latency")
    lines += ["  How long after the target jumped back the eye began its return saccade (from the raw gaze trace).",
              "  Negative = the eye left before the jump, i.e. it predicted it.", ""]
    lines.append(f"  Trials where the target jumped back: {r['trials_jumped']} ({r['forced_jumps']} forced jumps left out; "
                 f"no return saccade found in {r['no_return_found']})")
    if r["n"]:
        lines.append(f"  Return saccades measured: {r['n']}")
        lines.append(f"    Median latency:             {_ms(r['median_ms'])}")
        lines.append(f"    Spread (SD):                {_ms(r['sd_ms'])}  (robust SD {_ms(r['robust_sd_ms'])})")
        lines.append(f"    Share that were predictive:  {100 * r['share_predictive']:.0f}%")
    else:
        lines.append("  No return saccade found: no latency.")
    if r["live_n"]:
        lines.append(f"  Cross-check by the live classifier (return_reaction_ms): n {r['live_n']}, median {_ms(r['live_median_ms'])}, SD {_ms(r['live_sd_ms'])}")
    if r.get("return_flash_spread_sd_ms") is not None:
        lines.append(f"  This run's return-flash spread (RETURN_FLASH_SPREAD_SD_MS): {r['return_flash_spread_sd_ms']:g} ms; trigger: {r.get('return_jump_trigger') or '?'}")
    lines.append("  Note: the jump time is anchored on the scheduled flash delay, so it can be up to one frame late; latencies near 0 read slightly low.")
    return "\n".join(lines)


# -- the per-run report ----------------------------------------------------------------------------------------------------


def _summary(s: dict, test_run: bool) -> list[str]:
    """The plain-language top of the report: the main result, how many flashes were usable, and whether to trust it."""
    gates, sup = s["gates"], s["suppression"]
    sac, fix = _correct_cells(tuple(gates[SACCADE])), _correct_cells(tuple(gates[FIXATION]))
    n_sac, n_fix, n_excl = gates[SACCADE][1], gates[FIXATION][1], gates[EXCLUDED][1]
    lines = _heading("Summary")
    if test_run:
        lines += ["  This was a TEST run (a check of the setup, not a real recording): do not report these numbers as results.", ""]
    lines += [f"  Flashes during a saccade:  {sac[0].replace(' / ', ' of ')} correct ({sac[1]})",
              f"  Flashes during fixation:   {fix[0].replace(' / ', ' of ')} correct ({fix[1]})"]
    if sup["index"] is None:
        why = sup["notes"][0] if sup["notes"] else "too few flashes"
        lines += ["  Suppression index:         could not be computed", f"    ({why})"]
    else:
        lines.append(f"  Suppression index:         {sup['index']:+.2f}  ({_ci_text(sup['ci'])})" + ("  [accuracy-based fallback]" if sup["method"] == "accuracy" else ""))
        scale = index_meaning(sup["method"])
        if scale:
            lines += _wrap(scale, 4, "What it means: ")
        ci = sup["ci"]
        if ci is None:
            lines.append("    No confidence interval could be computed, so how certain the index is cannot be judged.")
        elif ci[0] > 0:
            lines.append("    The whole 95% CI is above 0: in this run the line was harder to see during saccades (suppression).")
        elif ci[1] < 0:
            lines.append("    The whole 95% CI is below 0: in this run the line was seen better during saccades than during fixation (unexpected).")
        else:
            lines.append("    The 95% CI includes 0: this run does not show clear suppression.")
    lines += ["", f"  Usable flashes: {n_sac + n_fix} of {s['flashes']} ({n_sac} during a saccade, {n_fix} during fixation, {n_excl} excluded)."]
    if n_sac < MIN_FIT_TRIALS or n_fix < MIN_FIT_TRIALS:
        lines += _wrap(f"only {n_sac} flashes during a saccade and {n_fix} during fixation. That is too few for a firm conclusion "
                       f"(a proper threshold fit needs at least {MIN_FIT_TRIALS} of each). Treat these numbers as a first look.", 2, "WARNING: ")
    else:
        lines.append("  One run is still a small sample: look at the confidence interval, not just the index.")
    lines.append("")
    lines += _wrap("a 95% CI (confidence interval) is the range of values the data are compatible with; a wide one means an uncertain result. "
                   "A p-value is the chance of a difference at least this large if saccades made no difference to seeing the line; below about "
                   "0.05 is usually called significant, and with few flashes a large p says very little.", 2, "Reading the statistics: ")
    return lines


def _data_quality(s: dict) -> list[str]:
    lines = _heading("Data quality")
    found = f"{s['saccades']}" + (f"  (median duration {s['median_saccade_ms']:.0f} ms, median size {s['median_amplitude']:.2f} of the screen width)" if s["median_saccade_ms"] else "")
    lines += [f"  Saccades found in the gaze trace:  {found}",
              f"  Flashes shown:                     {s['flashes']}",
              f"  Answered correctly:                {_pct(s['overall'])}",
              f"  No answer:                         {s['misses']}"]
    return lines


def _time_windows(s: dict) -> list[str]:
    """Accuracy by the flash's time relative to the nearest saccade onset: windows (with Fisher p) and the fine bins."""
    lines = _heading("Accuracy by flash time")
    lines += ["  Simpler sorts than the main comparison, using only the flash's time relative to the nearest saccade onset.", ""]
    rows = []
    for name, key in (("During a saccade (flash fell between a saccade's start and end)", "inside"), ("Outside a saccade", "outside"),
                      (f"Within +/-{NEAR_ONSET_MS:.0f} ms of a saccade onset", "near_onset"), (f"{NEAR_ONSET_MS:.0f}-{FAR_MS:.0f} ms from a saccade onset", "beyond"),
                      (f"Fixation control (at least {FAR_MS:.0f} ms from any onset)", "far")):
        c = _correct_cells(s[key])
        rows.append([name, c[0], c[1]])
    lines += format_table(["Group", "Correct", "Accuracy"], rows, "lrr")
    lines += ["",
              f"  Difference during vs outside a saccade:  Fisher exact test {_p_text(s['p_inside_vs_outside'])}",
              f"  Difference near onset vs further away:   Fisher exact test {_p_text(s['p_near_vs_beyond'])}",
              "", "  By flash time relative to saccade onset (negative = the flash came before the saccade began):", ""]
    rows = []
    for b in s["bins"]:
        lo, hi = b["ci"]
        c = _correct_cells((b["correct"], b["n"])) if b["n"] else ("-", "-")
        rows.append([f"{b['lo_ms']:+d} to {b['hi_ms']:+d} ms", str(b["n"]), c[0], c[1], f"{100 * lo:.0f}-{100 * hi:.0f}%" if b["n"] else "-"])
    lines += format_table(["Flash time", "Flashes", "Correct", "Accuracy", "95% CI"], rows, "lrrrr")
    return lines


def _by_contrast(s: dict) -> list[str]:
    lines = _heading("Accuracy by contrast")
    lines += [f"  'Near onset' = within +/-{NEAR_ONSET_MS:.0f} ms of a saccade onset; 'Away' = {NEAR_ONSET_MS:.0f}-{FAR_MS:.0f} ms from one.", ""]
    rows = [[f"{100 * c:.1f}%", _pct(g["all"]), _pct(g["near_onset"]), _pct(g["beyond"])] for c, g in s["by_contrast"].items()]
    lines += format_table(["Contrast", "All flashes", "Near onset", "Away"], rows, "rlll")
    return lines


def format_report(summary: dict, label: str = "", geometry: str | None = None, thresholds: dict | None = None, test_run: bool = False) -> str:
    """The whole per-run report from analyze_session's `summary` (summarize() plus "gates", "suppression" and, for return-jump
    runs, "by_leg" and "return_latency"). `geometry` is format_geometry()'s text, `thresholds` thresholds_record()."""
    title = f"Saccade-phase analysis {label}".strip()
    lines = [title, "=" * len(title), *_summary(summary, test_run), *_data_quality(summary), format_gate_report(summary["gates"]),
             "", format_suppression(summary["suppression"]), *_time_windows(summary), *_by_contrast(summary)]
    if summary.get("by_leg"):
        lines.append(format_leg_suppression(summary["by_leg"]))
    if summary.get("return_latency"):
        lines.append(format_return_latency(summary["return_latency"]))
    lines += _heading("Technical details")
    if geometry:
        lines += [geometry, ""]
    if thresholds is not None:
        lines.append(format_thresholds(thresholds))
    return "\n".join(lines)
