"""Data-quality rules for a run (gaze delay, tracking gaps, head movement, flashes per condition), in one place: the dashboard's traffic
lights (src/dashboard/results.py) and the QUALITY line at the top of report.txt both read them from here, so they never disagree.
Everything is read from session.json (`gaze_quality`, `head_motion`) and the flashes' `tracking_ok`; a missing or damaged field is
"unknown", never an error. Thresholds are the run summary's (run_summary.MAX_LATENCY_MS etc.) and the analysis' (analysis_constants)."""

from dataclasses import dataclass

from include.experiment.analysis_constants import MIN_ACCURACY_TRIALS, MIN_FIT_TRIALS
from include.eye_tracking.constants import NEON_GAP_THRESHOLD_MS
from src.experiment.flash_gates import TRACKING_GAP_MS
from src.experiment.run_summary import MAX_GAP_MS, MAX_LATENCY_MS, MAX_TRACKING_LOST_SHARE

GREEN, AMBER, RED, GREY = "green", "amber", "red", "grey"

LATENCY_RULE = (f"Green: the median gaze delay (phone to this computer) is at most {MAX_LATENCY_MS:.0f} ms. Amber: longer, or not recorded. "
                "Never red: the analysis times saccades with the phone's own capture times, so the delay only affects the live display "
                "(typical is about 15 ms in full runs).")
GAP_RULE = (f"Green: the longest pause in the gaze stream is at most {MAX_GAP_MS:.0f} ms. Amber: longer, or not recorded. "
            f"Never red: any flash with a pause over {TRACKING_GAP_MS:.0f} ms around it is already left out by the analysis.")
HEAD_RULE = ("Green: the head monitor ran and no trial was flagged. Amber: the monitor was unavailable, or some trials were flagged "
             "(those trials are already left out). Grey: not recorded (older runs, or a run made before the head monitor existed).")
FLASH_RULE = (f"Green: at least {MIN_FIT_TRIALS} saccade and {MIN_FIT_TRIALS} fixation flashes (enough for the threshold fit). "
              f"Amber: {MIN_ACCURACY_TRIALS}-{MIN_FIT_TRIALS - 1} in the smaller group (only the simpler accuracy index is possible). "
              f"Red: fewer than {MIN_ACCURACY_TRIALS} (no index at all). These are the limits the analysis itself uses.")


def latency_light(latency) -> str:
    return AMBER if latency is None or latency > MAX_LATENCY_MS else GREEN


def gap_light(gap) -> str:
    return AMBER if gap is None or gap > MAX_GAP_MS else GREEN


def ms_text(v) -> str:
    return "not recorded" if v is None else f"{v:.0f} ms"


def gap_text(gap, count) -> str:
    if gap is None:
        return "not recorded"
    return f"longest {gap:.0f} ms" + ("" if count is None else f", {count} pauses over {NEON_GAP_THRESHOLD_MS:.0f} ms")


def head_status(meta: dict) -> tuple[str, str]:
    """(light, text) of the head-movement monitor from session.json `head_motion`."""
    head = meta.get("head_motion")
    if not isinstance(head, dict):
        return GREY, "not recorded"
    status = str(head.get("status") or "unknown")
    flagged = head.get("flagged_trials")
    drift = head.get("max_drift_deg")
    extra = "" if drift is None else f", largest drift from the calibration position {drift:g} deg"
    if status != "ok":
        return AMBER, f"monitor {status}"
    if flagged:
        return AMBER, f"{flagged} trial(s) flagged for head movement and left out{extra}"
    return GREEN, f"ok, no trial flagged{extra}"


def flash_light(n_sac: int, n_fix: int) -> str:
    smaller = min(n_sac, n_fix)
    return GREEN if smaller >= MIN_FIT_TRIALS else (AMBER if smaller >= MIN_ACCURACY_TRIALS else RED)


# -- the one-line verdict of report.txt ------------------------------------------------------------------------------------------

OK, CHECK, BAD = "OK", "CHECK", "BAD"
_ORDER = {OK: 0, CHECK: 1, BAD: 2}


@dataclass(frozen=True)
class RunFacts:
    """One run's quality facts; None = unknown (field missing)."""

    latency_ms: float | None
    gap_ms: float | None
    clock: str | None  # session.json gaze_quality.clock_offset_status ("ok" or "failed (...)")
    head_light: str
    head_text: str
    tracking_lost: int | None  # flashes with tracking_ok False
    shown: int | None  # flashes shown (for the share)


def run_facts(meta: dict | None, tracking_lost: int | None, shown: int | None) -> RunFacts:
    meta = meta if isinstance(meta, dict) else {}
    quality = meta.get("gaze_quality")
    q: dict = quality if isinstance(quality, dict) else {}
    head_light, head_text = head_status(meta)
    status = q.get("clock_offset_status")
    return RunFacts(q.get("latency_ms_median"), q.get("longest_gap_ms"), None if status is None else str(status), head_light, head_text, tracking_lost, shown)


def _level(facts: list[RunFacts]) -> str:
    level = OK

    def raise_to(new: str) -> None:
        nonlocal level
        if _ORDER[new] > _ORDER[level]:
            level = new

    for f in facts:
        if f.latency_ms is None or f.latency_ms > MAX_LATENCY_MS:
            raise_to(CHECK)
        if f.gap_ms is not None and f.gap_ms > MAX_GAP_MS:
            raise_to(CHECK)
        if f.clock is None or not f.clock.startswith("ok"):
            raise_to(BAD if f.clock is not None and f.clock.startswith("failed") else CHECK)
        if f.head_light != GREEN:
            raise_to(CHECK)
        if f.tracking_lost is None:
            raise_to(CHECK)
        elif f.tracking_lost:
            raise_to(BAD if f.shown and f.tracking_lost / f.shown > MAX_TRACKING_LOST_SHARE else CHECK)
    return level


def quality_line(facts: list[RunFacts]) -> tuple[str, str]:
    """(status, text) for the QUALITY line of one run (one RunFacts) or of pooled runs (the worst status, ranges and totals)."""
    delays = [f.latency_ms for f in facts if f.latency_ms is not None]
    parts = []
    if len(delays) < len(facts) or not delays:
        parts.append("gaze delay unknown")
    elif len(facts) == 1 or min(delays) == max(delays):
        parts.append(f"gaze delay {delays[0]:.0f} ms")
    else:
        parts.append(f"gaze delay {min(delays):.0f}-{max(delays):.0f} ms")
    gaps = [f.gap_ms for f in facts if f.gap_ms is not None and f.gap_ms > MAX_GAP_MS]
    if gaps:
        parts.append(f"longest gaze gap {max(gaps):.0f} ms")
    clocks = [f.clock for f in facts]
    failed = sum(c is not None and c.startswith("failed") for c in clocks)
    unknown = sum(c is None for c in clocks)
    many = len(facts) > 1
    if failed:
        clock_text = f"failed in {failed} of {len(facts)} runs" if many else "failed"
    elif unknown:
        clock_text = f"unknown in {unknown} of {len(facts)} runs" if many else "unknown"
    else:
        clock_text = "ok"
    parts.append(f"clock offset {clock_text}")
    heads = [f for f in facts if f.head_light != GREEN]
    if not heads:
        parts.append("head still")
    elif all(f.head_light == GREY for f in heads):
        parts.append("head unknown")
    else:
        parts.append("head: " + "; ".join(f.head_text for f in heads if f.head_light != GREY))
    lost = [f.tracking_lost for f in facts]
    parts.append("tracking loss unknown" if any(x is None for x in lost) else f"{sum(x for x in lost if x is not None)} flashes lost to tracking")
    return _level(facts), " · ".join(parts)
