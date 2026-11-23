"""Post-hoc analysis of a saccade-phase session (`-m src.experiment.analyze_session [timestamp]`).

Finds the saccades in the raw gaze trace (`gaze_trace.csv` in the run folder, or the old `results_<ts>_gaze_trace.csv`; stamped with the phone's own capture
times, so the ~90 ms delivery delay and dropped samples don't matter), places every logged flash against the
nearest saccade onset, and reports detection accuracy as a function of flash time relative to onset - the
saccadic-suppression curve - plus a plot and a per-flash CSV. The live `flash_during_saccade` flag is not used.

Run it on the latest session with no arguments, or give a session timestamp (unix, either layout), a run folder name, `participant/folder`
or a path. Output goes to the run folder's `analysis/` (report.txt, plot.png, flashes.csv, suppression.json); old loose-file sessions get
`analysis_<ts>*` files next to them as before.
Units: gaze x is a 0-1 ratio of the scene (or screen) width per `x_px / NEON_SCENE_WIDTH` (traces recorded with the since-removed screen-mapped backends hold the screen ratio x NEON_SCENE_WIDTH in x_px, so old sessions analyse the same way); speeds are ratio/s.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only renders PNG files
import matplotlib.pyplot as plt
import numpy as np

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.experiment import session_files
from src.experiment.logger import read_trial_rows
from src.experiment.flash_gates import (
    EXCLUDED,
    FIXATION,
    FIXATION_EXCLUSION_MS,
    RETURN_LATENCY_MAX_MS,
    SACCADE,
    classify_flashes,
    classify_leg_flash,
    summarize_gates,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Saccade detection: a movement counts once its (3-sample smoothed) speed exceeds HIGH; its edges are where the
# speed falls back under LOW; only movements of at least MIN_AMPLITUDE (ratio of the width) are kept.
SPEED_HIGH = 2.0
SPEED_LOW = 0.7
MIN_AMPLITUDE = 0.12
NEAR_ONSET_MS = 50.0  # "near onset" = flash within this many ms of a saccade onset
FAR_MS = 300.0  # flashes at least this far from every onset are the fixation control
MIN_FIT_TRIALS = 12  # suppression_index: fewer flashes than this in a condition -> no threshold fit (the dashboard's quality box reads it)
MIN_ACCURACY_TRIALS = 5  # ... and fewer than this -> not even the accuracy fallback
BIN_EDGES_MS = (-200, -150, -100, -50, 0, 50, 100, 150, 200)


@dataclass(frozen=True)
class Saccade:
    onset: float  # PC unix seconds (phone capture time + clock offset)
    offset: float
    x0: float
    x1: float
    peak_speed: float

    @property
    def duration_ms(self) -> float:
        return 1000 * (self.offset - self.onset)

    @property
    def amplitude(self) -> float:
        return abs(self.x1 - self.x0)


@dataclass(frozen=True)
class Flash:
    trial: int
    time: float  # PC unix seconds the line was drawn (before any display-delay correction)
    contrast: float
    orientation: str
    outcome: str  # correct | incorrect | miss
    go_cue: float | None = None  # PC unix seconds the trial's target appeared (flash time minus its scheduled delay)
    leg: str | None = None  # return-jump runs: 'out' | 'return' (trials.csv flash_leg); None in older runs
    jump: float | None = None  # PC unix seconds the target jumped back (go cue + jump_back_ms_after_go_cue); None = no jump
    jump_forced: bool = False  # the jump came on the landing timeout (no out landing was seen live)
    return_reaction_ms: float | None = None  # the live classifier's signed return latency from the jump (cross-check only)

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"


@dataclass(frozen=True)
class Placed:
    """A flash placed against its nearest saccade."""

    flash: Flash
    rel_ms: float | None  # flash time minus the nearest onset (None when there is no saccade at all)
    inside: bool  # the flash fell between that saccade's onset and offset


# -- loading -------------------------------------------------------------------------------------


def load_trace(path: Path, clock_offset_s: float, offset_history: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(capture times in PC unix seconds, gaze x ratio), strictly increasing, without lost samples. With an `offset_history`
    of two or more {"phone_unix_s", "offset_ms"} entries (session.json gaze_quality.clock_offset_history) the offset is
    interpolated over the phone-clock time (the clocks drift); otherwise the single `clock_offset_s` applies throughout."""
    times, xs = [], []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("x_px") or not row.get("device_unix_s"):
                continue
            times.append(float(row["device_unix_s"]))
            xs.append(float(row["x_px"]) / NEON_SCENE_WIDTH)
    t, x = np.asarray(times), np.asarray(xs)
    if offset_history is not None and len(offset_history) >= 2:
        history = sorted(offset_history, key=lambda h: h["phone_unix_s"])
        t = t + np.interp(t, [h["phone_unix_s"] for h in history], [h["offset_ms"] / 1000 for h in history])  # held constant beyond the ends
    else:
        t = t + clock_offset_s
    keep = np.concatenate(([True], np.diff(t) > 0)) if len(t) else np.asarray([], dtype=bool)
    ok = keep & ~np.isnan(x)
    return t[ok], x[ok]


def load_flashes(path: Path) -> list[Flash]:
    """Every shown saccade-phase line with a logged wall-clock time (reads both trials.csv and the old results CSV)."""
    flashes = []
    for row in read_trial_rows(Path(path)):
        if row.get("phase") != "saccade" or row.get("grating_shown") != "True":
            continue
        if not row.get("grating_shown_at_unix_ms") or not row.get("contrast"):
            continue
        shown = float(row["grating_shown_at_unix_ms"]) / 1000
        delay = row.get("scheduled_delay_ms")
        outcome = row["outcome"]
        if outcome == "timeout":
            # The live trial ended on its no-saccade timeout although the line was shown (and maybe answered): score the answer.
            if row.get("responded") != "True":
                outcome = "miss"
            else:
                outcome = "correct" if row.get("response_orientation") == row.get("orientation") else "incorrect"
        go_cue = shown - float(delay) / 1000 if delay else None
        jump_ms = row.get("jump_back_ms_after_go_cue")
        reaction = row.get("return_reaction_ms")
        flashes.append(
            Flash(
                trial=int(row["trial_index"]),
                time=shown,
                contrast=float(row["contrast"]),
                orientation=row.get("orientation", ""),
                outcome=outcome,
                go_cue=go_cue,
                leg=row.get("flash_leg") or None,
                jump=go_cue + float(jump_ms) / 1000 if jump_ms and go_cue is not None else None,
                jump_forced=row.get("jump_back_forced") == "True",
                return_reaction_ms=float(reaction) if reaction else None,
            )
        )
    return flashes


# -- saccade detection and placement -------------------------------------------------------------


def detect_saccades(
    t: np.ndarray,
    x: np.ndarray,
    speed_high: float = SPEED_HIGH,
    speed_low: float = SPEED_LOW,
    min_amplitude: float = MIN_AMPLITUDE,
) -> list[Saccade]:
    if len(t) < 5:
        return []
    smooth = np.convolve(x, np.ones(3) / 3, mode="same")
    speed = np.abs(np.gradient(smooth, t))
    found, i = [], 0
    while i < len(speed):
        if speed[i] > speed_high:
            a = i
            while a > 0 and speed[a - 1] > speed_low:
                a -= 1
            b = i
            while b + 1 < len(speed) and speed[b + 1] > speed_low:
                b += 1
            if abs(x[b] - x[a]) >= min_amplitude:
                found.append(Saccade(float(t[a]), float(t[b]), float(x[a]), float(x[b]), float(speed[a : b + 1].max())))
            i = b + 1
        else:
            i += 1
    return found


def place_flashes(flashes: list[Flash], saccades: list[Saccade], flash_delay_ms: float = 0.0) -> list[Placed]:
    """Each flash against the saccade whose onset is nearest in time. `flash_delay_ms` shifts the flash later
    (the time from drawing it to light leaving the screen: up to one frame plus panel lag)."""
    if not saccades:
        return [Placed(f, None, False) for f in flashes]
    onsets = np.asarray([s.onset for s in saccades])
    placed = []
    for f in flashes:
        when = f.time + flash_delay_ms / 1000
        k = int(np.argmin(np.abs(onsets - when)))
        s = saccades[k]
        placed.append(Placed(f, 1000 * (when - s.onset), s.onset <= when <= s.offset))
    return placed


# -- return-jump runs: the two saccades of each trial ----------------------------------------------
# A trial whose row logs a jump-back has an out saccade (go cue -> target) and a return saccade (back to the start). Each
# flash is placed against and gated by the saccade of its own leg (flash_gates.classify_leg_flash). Runs with no jump
# logged in any row (older runs, or the jump setting off) never reach this code.

OUT_SEARCH_S = (-0.2, 1.5)  # the out saccade: the first one with onset in [go cue - 0.2 s, go cue + 1.5 s]
RETURN_SEARCH_AFTER_JUMP_S = 1.0  # the return saccade starts at most this long after the jump (and before the next go cue)
RETURN_MIN_AMPLITUDE_REL = 0.5  # ...and moves back toward the start by >= this x the out amplitude (an overshoot correction is not a return)
LEGS = ("out", "return")


@dataclass(frozen=True)
class TrialSaccades:
    """The saccades of one return-jump trial (None = not found)."""

    out: Saccade | None
    back: Saccade | None

    def of(self, leg: str | None) -> Saccade | None:
        return self.back if leg == "return" else self.out


def is_return_jump(flashes: list[Flash]) -> bool:
    """True when any row logged a jump-back: the run used the return-jump trials."""
    return any(f.jump is not None for f in flashes)


def find_trial_saccades(flashes: list[Flash], saccades: list[Saccade]) -> list[TrialSaccades]:
    """Per flash (one per trial): its trial's out and return saccade. The search for a trial ends at the next trial's go cue."""
    cues = sorted(f.go_cue for f in flashes if f.go_cue is not None)
    found = []
    for f in flashes:
        if f.go_cue is None:
            found.append(TrialSaccades(None, None))
            continue
        later = [c for c in cues if c > f.go_cue]
        end = later[0] if later else math.inf
        lo, hi = f.go_cue + OUT_SEARCH_S[0], min(f.go_cue + OUT_SEARCH_S[1], end)
        k = next((i for i, s in enumerate(saccades) if lo <= s.onset <= hi), None)
        if k is None:
            found.append(TrialSaccades(None, None))
            continue
        out = saccades[k]
        back = None
        if f.jump is not None:
            direction = math.copysign(1.0, out.x1 - out.x0)
            last = min(f.jump + RETURN_SEARCH_AFTER_JUMP_S, end)
            for s in saccades[k + 1 :]:
                if s.onset > last:
                    break
                if -direction * (s.x1 - s.x0) >= RETURN_MIN_AMPLITUDE_REL * out.amplitude:
                    back = s
                    break
        found.append(TrialSaccades(out, back))
    return found


def place_leg_flashes(flashes: list[Flash], trials: list[TrialSaccades], flash_delay_ms: float = 0.0) -> list[Placed]:
    """Each flash against its own leg's saccade (rel_ms None when that saccade was not found). A flash without a leg is an out flash."""
    placed = []
    for f, tr in zip(flashes, trials):
        s = tr.of(f.leg)
        when = f.time + flash_delay_ms / 1000
        placed.append(Placed(f, None, False) if s is None else Placed(f, 1000 * (when - s.onset), s.onset <= when <= s.offset))
    return placed


def classify_leg_flashes(placed: list[Placed], trials: list[TrialSaccades], saccades: list[Saccade], t: np.ndarray, x: np.ndarray, flash_delay_ms: float = 0.0) -> list:
    """flash_gates.classify_leg_flash for every flash; the amplitude gate uses each leg's own run median."""
    medians = {}
    for leg in LEGS:
        amps = [tr.of(leg).amplitude for tr in trials if tr.of(leg) is not None]
        medians[leg] = float(np.median(amps)) if amps else None
    out = []
    for p, tr in zip(placed, trials):
        leg = "return" if p.flash.leg == "return" else "out"
        anchor = p.flash.jump if leg == "return" else p.flash.go_cue
        out.append(classify_leg_flash(p.flash.time + flash_delay_ms / 1000, saccades, t, x, leg, tr.of(leg), anchor, medians[leg], p.flash.jump_forced))
    return out


def leg_subset(placed: list[Placed], gates: list, leg: str) -> tuple[list[Placed], list]:
    """This leg's saccade flashes plus EVERY fixation flash (the fixation control is shared by both legs: the eye was still,
    whichever leg the flash had been timed for)."""
    keep = [i for i, (p, g) in enumerate(zip(placed, gates)) if g.condition == FIXATION or (g.condition == SACCADE and (p.flash.leg or "out") == leg)]
    return [placed[i] for i in keep], [gates[i] for i in keep]


def suppression_by_leg(placed: list[Placed], gates: list, **kwargs) -> dict:
    """{'out', 'return', 'pooled'}: suppression_index per leg (that leg's saccade flashes vs all fixation flashes) and pooled."""
    result = {}
    for leg in LEGS:
        sub_placed, sub_gates = leg_subset(placed, gates, leg)
        result[leg] = suppression_index(sub_placed, gates=sub_gates, **kwargs)
    result["pooled"] = suppression_index(placed, gates=gates, **kwargs)
    return result


def _robust_sd(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def return_latency_stats(flashes: list[Flash], trials: list[TrialSaccades], settings: dict | None = None) -> dict:
    """Return-saccade latency from the jump-back frame, from the raw trace, over every trial that jumped (whichever leg its flash
    was on), forced jumps left out. Negative = the eye turned back before the jump was drawn (predictive). The logged live
    `return_reaction_ms` is summarised alongside as a cross-check. Caveat: the jump time is go cue + logged offset, and the go cue
    is the flash time minus its SCHEDULED delay, so the anchor can sit up to one frame late (latencies read up to ~17 ms short)."""
    jumped = [(f, tr) for f, tr in zip(flashes, trials) if f.jump is not None]
    usable = [(f, tr) for f, tr in jumped if not f.jump_forced]
    lat = np.asarray([1000 * (tr.back.onset - f.jump) for f, tr in usable if tr.back is not None])
    live = np.asarray([f.return_reaction_ms for f, _ in usable if f.return_reaction_ms is not None])
    s = settings or {}
    return {
        "trials_jumped": len(jumped),
        "forced_jumps": len(jumped) - len(usable),
        "no_return_found": sum(tr.back is None for _, tr in usable),
        "n": int(len(lat)),
        "median_ms": float(np.median(lat)) if len(lat) else None,
        "sd_ms": float(np.std(lat, ddof=1)) if len(lat) >= 2 else None,
        "robust_sd_ms": _robust_sd(lat) if len(lat) >= 2 else None,
        "share_predictive": float(np.mean(lat < 0)) if len(lat) else None,
        "live_n": int(len(live)),
        "live_median_ms": float(np.median(live)) if len(live) else None,
        "live_sd_ms": float(np.std(live, ddof=1)) if len(live) >= 2 else None,
        "return_flash_spread_sd_ms": s.get("return_flash_spread_sd_ms"),
        "return_jump_trigger": s.get("return_jump_trigger"),
    }


# -- statistics ----------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # rounding can leave the bounds a hair inside p (k == n gives 0.9999999999999999); keep p inside and the ends exact
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (0.0 if k == 0 else min(lo, p), 1.0 if k == n else max(hi, p))


def fisher_p(a_correct: int, a_total: int, b_correct: int, b_total: int) -> float | None:
    """Two-sided Fisher exact p for two accuracy groups (None if scipy is unavailable or a group is empty)."""
    if a_total == 0 or b_total == 0:
        return None
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return None
    return float(fisher_exact([[a_correct, a_total - a_correct], [b_correct, b_total - b_correct]])[1])


def _count(placed: list[Placed]) -> tuple[int, int]:
    return sum(p.flash.correct for p in placed), len(placed)


def summarize(placed: list[Placed], saccades: list[Saccade]) -> dict:
    """Counts and accuracy by condition; the numbers behind the printed report."""
    timed = [p for p in placed if p.rel_ms is not None]
    near = [p for p in timed if abs(p.rel_ms) < NEAR_ONSET_MS]
    beyond = [p for p in timed if NEAR_ONSET_MS <= abs(p.rel_ms) < FAR_MS]
    far = [p for p in timed if abs(p.rel_ms) >= FAR_MS]
    inside = [p for p in timed if p.inside]
    outside = [p for p in timed if not p.inside]
    bins = []
    for lo, hi in zip(BIN_EDGES_MS[:-1], BIN_EDGES_MS[1:]):
        members = [p for p in timed if lo <= p.rel_ms < hi]
        k, n = _count(members)
        bins.append({"lo_ms": lo, "hi_ms": hi, "n": n, "correct": k, "ci": wilson(k, n)})
    by_contrast = {}
    for c in sorted({p.flash.contrast for p in placed}):
        group = [p for p in placed if p.flash.contrast == c]
        by_contrast[c] = {"all": _count(group), "near_onset": _count([p for p in group if p in near]), "beyond": _count([p for p in group if p in beyond])}
    return {
        "flashes": len(placed),
        "saccades": len(saccades),
        "inside": _count(inside),
        "outside": _count(outside),
        "near_onset": _count(near),
        "beyond": _count(beyond),
        "far": _count(far),
        "overall": _count(placed),
        "misses": sum(p.flash.outcome == "miss" for p in placed),
        "bins": bins,
        "by_contrast": by_contrast,
        "p_near_vs_beyond": fisher_p(*_count(near), *_count(beyond)),
        "p_inside_vs_outside": fisher_p(*_count(inside), *_count(outside)),
        "median_saccade_ms": float(np.median([s.duration_ms for s in saccades])) if saccades else None,
        "median_amplitude": float(np.median([s.amplitude for s in saccades])) if saccades else None,
    }


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


def format_report(summary: dict, label: str = "", geometry: str | None = None) -> str:
    s = summary
    lines = [f"Saccade-phase analysis {label}".strip()]
    if geometry:
        lines.append(geometry)
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


# -- suppression index ---------------------------------------------------------------------------
# Braun, Schutz & Gegenfurtner (2021, J Vis 21(6):11): suppression = 1 - CS_saccade / CS_fixation, with contrast
# sensitivity CS = 1 / threshold. 0 = no suppression, 1 = flashes are never seen; negative = better during saccades.

CHANCE = 0.5  # 2AFC guess rate
LAPSE = 0.02  # small lapse rate in the psychometric function
_SLOPES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)  # Weibull slopes tried; one slope is shared by both conditions
_EXTRAPOLATION = 2.0  # a fitted threshold more than this factor outside the tested contrasts counts as not identified
_THRESHOLDS = np.logspace(-3, 1, 121)  # candidate thresholds as a contrast fraction (0.1 % .. 1000 %)


def _contrast_counts(placed: list[Placed]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(contrasts, trials, correct) per distinct contrast."""
    contrasts = sorted({p.flash.contrast for p in placed})
    n = np.asarray([sum(p.flash.contrast == c for p in placed) for c in contrasts], dtype=float)
    k = np.asarray([sum(p.flash.contrast == c and p.flash.correct for p in placed) for c in contrasts], dtype=float)
    return np.asarray(contrasts, dtype=float), n, k


def _weibull(contrast: np.ndarray, threshold: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """P(correct) = chance + (1 - chance - lapse) * (1 - exp(-(c / threshold) ** slope)); the threshold alpha
    is the contrast giving ~63 % of the way from chance to the ceiling (only the ratio of two alphas matters)."""
    return CHANCE + (1 - CHANCE - LAPSE) * (1 - np.exp(-((contrast / threshold) ** slope)))


def _loglik_grid(contrasts: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Log-likelihood over (slope, threshold) for one condition."""
    s = np.asarray(_SLOPES)[:, None, None]
    t = _THRESHOLDS[None, :, None]
    p = np.clip(_weibull(contrasts[None, None, :], t, s), 1e-9, 1 - 1e-9)
    return (k * np.log(p) + (n - k) * np.log(1 - p)).sum(axis=2)  # (slope, threshold)


def _fit_pair(contrasts_s, n_s, k_s, contrasts_f, n_f, k_f) -> tuple[float, float, bool]:
    """Maximum-likelihood thresholds (saccade, fixation) with a slope shared by the two conditions.
    The last value is True when either threshold sits on the edge of the search grid (not identified)."""
    return fit_pair_with_slope(contrasts_s, n_s, k_s, contrasts_f, n_f, k_f)[:3]


def fit_pair_with_slope(contrasts_s, n_s, k_s, contrasts_f, n_f, k_f) -> tuple[float, float, bool, float]:
    """`_fit_pair` plus the shared Weibull slope it chose (for drawing the fitted curves)."""
    ll_s, ll_f = _loglik_grid(contrasts_s, n_s, k_s), _loglik_grid(contrasts_f, n_f, k_f)
    best_s, best_f = ll_s.argmax(axis=1), ll_f.argmax(axis=1)
    rows = np.arange(len(_SLOPES))
    b = int((ll_s[rows, best_s] + ll_f[rows, best_f]).argmax())
    ts, tf = float(_THRESHOLDS[best_s[b]]), float(_THRESHOLDS[best_f[b]])
    edge = any(t in (_THRESHOLDS[0], _THRESHOLDS[-1]) for t in (ts, tf))
    # Not identified either when the threshold lies far outside the tested contrasts (flat likelihood: all near
    # chance or all near ceiling).
    for t, c in ((ts, contrasts_s), (tf, contrasts_f)):
        edge = edge or t > _EXTRAPOLATION * c.max() or t < c.min() / _EXTRAPOLATION
    return ts, tf, bool(edge), float(_SLOPES[b])


def suppression_index(
    placed: list[Placed],
    near_ms: float | None = None,
    far_ms: float | None = None,
    min_fit_trials: int = MIN_FIT_TRIALS,
    min_accuracy_trials: int = MIN_ACCURACY_TRIALS,
    n_boot: int = 1000,
    seed: int = 0,
    gates: list | None = None,
) -> dict:
    """Per-person saccadic suppression index from the placed flashes (see the section comment above).

    'saccade' = flashes with |time from the nearest onset| < near_ms; 'fixation' = flashes at least far_ms from
    every onset (defaults: NEAR_ONSET_MS / FAR_MS). Primary: a 2AFC Weibull fit per condition (shared slope) giving
    1 - threshold_fixation / threshold_saccade, with a bootstrap 95 % CI. When the fit is not possible or not
    identified (too few trials or contrasts, accuracy at ceiling or floor) the fallback is the clearly labelled
    accuracy index accuracy_fixation - accuracy_saccade with a Newcombe (Wilson-based) 95 % CI. If neither can be
    computed, "method" is None and "notes" says why. Never raises on sparse data."""
    near_ms = NEAR_ONSET_MS if near_ms is None else near_ms
    far_ms = FAR_MS if far_ms is None else far_ms
    if gates is not None:
        # Pre-specified gates (flash_gates.py) decide the conditions; excluded flashes are left out.
        sac = [p for p, g in zip(placed, gates) if g.condition == SACCADE]
        fix = [p for p, g in zip(placed, gates) if g.condition == FIXATION]
        source = "gates"
    else:
        timed = [p for p in placed if p.rel_ms is not None]
        sac = [p for p in timed if abs(p.rel_ms) < near_ms]
        fix = [p for p in timed if abs(p.rel_ms) >= far_ms]
        source = "windows"
    out = {
        "conditions_from": source, "near_ms": near_ms, "far_ms": far_ms, "n_saccade": len(sac), "n_fixation": len(fix),
        "correct_saccade": sum(p.flash.correct for p in sac), "correct_fixation": sum(p.flash.correct for p in fix),
        "method": None, "index": None, "ci": None, "label": "", "threshold_saccade": None, "threshold_fixation": None,
        "notes": [],
    }
    notes = out["notes"]
    if not sac or not fix:
        notes.append(f"cannot compute: {len(sac)} saccade flashes, {len(fix)} fixation flashes")
        return out

    # Primary: psychometric fit.
    cs, ns, ks = _contrast_counts(sac)
    cf, nf, kf = _contrast_counts(fix)
    fit_problem = None
    if len(sac) < min_fit_trials or len(fix) < min_fit_trials:
        fit_problem = f"fewer than {min_fit_trials} flashes per condition ({len(sac)} saccade, {len(fix)} fixation)"
    elif len(cs) < 2 or len(cf) < 2:
        fit_problem = "fewer than 2 distinct contrasts in a condition"
    else:
        ts, tf, edge = _fit_pair(cs, ns, ks, cf, nf, kf)
        if edge:
            fit_problem = "accuracy at ceiling or floor across the tested contrasts, so a threshold is not identified"
    if fit_problem is None:
        rng = np.random.default_rng(seed)
        boot = []
        edges = 0
        for _ in range(n_boot):
            bs = _fit_pair(cs, ns, rng.binomial(ns.astype(int), ks / ns), cf, nf, rng.binomial(nf.astype(int), kf / nf))
            edges += bs[2]
            if not bs[2]:
                boot.append(1 - bs[1] / bs[0])
        ci = None
        if len(boot) >= 0.75 * n_boot and len(boot) >= 200:
            ci = tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))
        else:
            notes.append(f"no CI: only {len(boot)} of {n_boot} bootstrap fits were identified (need >= 75% and >= 200)")
        out.update(method="psychometric", index=1 - tf / ts, ci=ci, threshold_saccade=ts, threshold_fixation=tf,
                   label="1 - CS_saccade/CS_fixation from per-condition Weibull thresholds (2AFC, guess 0.5, lapse 0.02, shared slope)")
        if edges:
            notes.append(f"{100 * edges / n_boot:.0f}% of bootstrap fits were unidentified and excluded from the CI")
        return out
    notes.append(f"psychometric index not computed: {fit_problem}")

    # Fallback: accuracy difference.
    if len(sac) < min_accuracy_trials or len(fix) < min_accuracy_trials:
        notes.append(f"accuracy index not computed: fewer than {min_accuracy_trials} flashes in a condition")
        return out
    k_sac, k_fix = out["correct_saccade"], out["correct_fixation"]
    p_sac, p_fix = k_sac / len(sac), k_fix / len(fix)
    if p_fix < CHANCE + 0.1:
        notes.append(f"accuracy index not computed: accuracy away from onset is only {100 * p_fix:.0f}% (floor), so the flashes are not visible even during fixation")
        return out
    (l_sac, u_sac), (l_fix, u_fix) = wilson(k_sac, len(sac)), wilson(k_fix, len(fix))
    d = p_fix - p_sac
    out.update(method="accuracy", index=d, label="ACCURACY index (not a threshold ratio): accuracy_fixation - accuracy_saccade",
               ci=(d - math.sqrt((p_fix - l_fix) ** 2 + (u_sac - p_sac) ** 2), d + math.sqrt((u_fix - p_fix) ** 2 + (p_sac - l_sac) ** 2)))
    if p_sac >= 0.95 and p_fix >= 0.95:
        notes.append("both conditions are at ceiling: this index cannot show suppression at these contrasts")
    return out


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


# -- plot ----------------------------------------------------------------------------------------


def plot_report(placed: list[Placed], saccades: list[Saccade], summary: dict, out_path: Path, title: str = "") -> Path:
    fig, (ax_curve, ax_scatter, ax_bars) = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    duration = summary["median_saccade_ms"] or 0.0

    for ax in (ax_curve, ax_scatter):
        ax.axvspan(0, duration, color="#C44E52", alpha=0.12, label=f"typical saccade (~{duration:.0f} ms)" if duration else None)
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_xlabel("flash time relative to saccade onset (ms)")
        ax.grid(True, alpha=0.3)

    centres = [(b["lo_ms"] + b["hi_ms"]) / 2 for b in summary["bins"] if b["n"]]
    acc = [b["correct"] / b["n"] for b in summary["bins"] if b["n"]]
    err = [[a - b["ci"][0] for a, b in zip(acc, [b for b in summary["bins"] if b["n"]])], [b["ci"][1] - a for a, b in zip(acc, [b for b in summary["bins"] if b["n"]])]]
    ax_curve.errorbar(centres, [100 * a for a in acc], yerr=[[100 * e for e in err[0]], [100 * e for e in err[1]]], fmt="o-", color="#4C72B0", capsize=4)
    for x, b in zip(centres, [b for b in summary["bins"] if b["n"]]):
        ax_curve.annotate(f"n={b['n']}", (x, 3), ha="center", fontsize=8)
    ax_curve.axhline(50, color="#999999", linestyle=":", linewidth=1)
    ax_curve.set_ylim(0, 105)
    ax_curve.set_ylabel("correct (%), with 95% CI")
    ax_curve.set_title("Accuracy vs flash time")
    if duration:
        ax_curve.legend(loc="lower center", bbox_to_anchor=(0.5, 0.07), fontsize=8)

    marker = {"correct": ("o", "#55A868"), "incorrect": ("x", "#C44E52"), "miss": ("s", "#999999")}
    for outcome, (m, colour) in marker.items():
        pts = [(p.rel_ms, 100 * p.flash.contrast) for p in placed if p.rel_ms is not None and p.flash.outcome == outcome and abs(p.rel_ms) <= 400]
        if pts:
            ax_scatter.scatter(*zip(*pts), marker=m, color=colour, alpha=0.8, label=outcome, s=34)
    ax_scatter.set_ylabel("contrast (%)")
    ax_scatter.set_xlim(-400, 400)
    ax_scatter.set_title("Every flash")
    ax_scatter.legend(fontsize=8, loc="best")

    groups = [("near onset", summary["near_onset"]), ("50-300 ms away", summary["beyond"]), ("inside saccade", summary["inside"]), ("outside saccade", summary["outside"])]
    labels = [f"{name}\n(n={n})" for name, (_, n) in groups]
    values = [100 * k / n if n else 0 for _, (k, n) in groups]
    cis = [wilson(k, n) for _, (k, n) in groups]
    ax_bars.bar(labels, values, color=["#C44E52", "#4C72B0", "#C44E52", "#4C72B0"], yerr=[[100 * (v / 100 - lo) for v, (lo, _) in zip(values, cis)], [100 * (hi - v / 100) for v, (_, hi) in zip(values, cis)]], capsize=4)
    ax_bars.set_ylim(0, 105)
    ax_bars.set_ylabel("correct (%)")
    p = summary["p_near_vs_beyond"]
    ax_bars.set_title("By condition" + (f" (near vs beyond: p={p:.3f})" if p is not None else ""))
    ax_bars.tick_params(axis="x", labelsize=8)
    ax_bars.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title or "Saccadic suppression analysis")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -- driver --------------------------------------------------------------------------------------


def _analysable(ref: session_files.SessionRef) -> bool:
    return ref.gaze_trace_path.exists() and ref.trials_path.exists() and bool(load_flashes(ref.trials_path))


def latest_timestamp(data_dir: Path) -> str:
    """The newest session (either layout) that has a gaze trace and at least one saccade-phase flash row."""
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
        return self.ref.key if self.ref.is_legacy else self.ref.label


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
    meta: dict = {}
    if clock_offset_ms is None:
        meta = json.loads(ref.meta_path.read_text())
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
    try:
        meta = json.loads(ref.meta_path.read_text())
        geometry_line = format_geometry(meta.get("settings"))
    except (OSError, ValueError, AttributeError, KeyError):
        geometry_line = None
    meta = meta if isinstance(meta, dict) else {}
    run = RunAnalysis(ref, meta, t, x, saccades, placed, gates, summary, suppression, geometry_line)
    if trials is not None:
        by_leg = suppression_by_leg(placed, gates, near_ms=near_ms, far_ms=far_ms)
        by_leg["pooled"] = suppression  # identical computation; keep the one object
        run.trials, run.by_leg = trials, by_leg
        run.return_latency = return_latency_stats(flashes, trials, meta.get("settings") if isinstance(meta.get("settings"), dict) else None)
        summary["by_leg"], summary["return_latency"] = by_leg, run.return_latency
    return run


def write_flash_rows(path: Path, placed: list[Placed], gates: list, extra: dict[str, list] | None = None) -> Path:
    """The per-flash CSV; `extra` columns (name -> one value per flash) go first (the participant analysis adds the run).
    flash_rel_onset_ms is relative to the flash's own leg's saccade in return-jump runs (the nearest onset otherwise);
    flash_leg is blank for runs without return jumps; saccade_latency_ms is from the go cue (out) or the jump (return)."""
    extra = extra or {}
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*extra, "trial", "contrast", "orientation", "outcome", "flash_rel_onset_ms", "inside_saccade", "condition", "gate_reason", "flash_leg", "saccade_latency_ms"])
        for i, (p, g) in enumerate(zip(placed, gates)):
            leg = (p.flash.leg or "out") if p.flash.jump is not None else ""
            writer.writerow([*(values[i] for values in extra.values()), p.flash.trial, p.flash.contrast, p.flash.orientation, p.flash.outcome, "" if p.rel_ms is None else round(p.rel_ms, 1), p.inside, g.condition, g.reason,
                             leg, "" if g.latency_ms is None else round(g.latency_ms, 1)])
    return Path(path)


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
    analysis/ folder; for an old loose-file session, data_dir with `analysis_<ts>*` names) and returns
    {"summary", "report", "report_path", "png", "csv", "suppression_json"}."""
    data_dir = Path(data_dir)
    ref = session_files.find_session(data_dir, timestamp or latest_timestamp(data_dir))
    if ref is None:
        raise FileNotFoundError(f"no session {timestamp!r} found in {data_dir}")
    run = analyze_run(ref, clock_offset_ms, flash_delay_ms, min_amplitude, near_ms, far_ms)
    stamp, summary, suppression, placed, gates = run.stamp, run.summary, run.suppression, run.placed, run.gates
    report = format_report(summary, label=f"(session {stamp})", geometry=run.geometry_line) + "\n" + format_gate_report(summary["gates"]) + "\n" + format_suppression(suppression)
    saved = suppression
    if run.return_jump:
        report = format_return_latency(run.return_latency) + "\n" + report + "\n" + format_leg_suppression(run.by_leg)
        saved = {**suppression, "by_leg": {leg: run.by_leg[leg] for leg in LEGS}, "return_latency": run.return_latency}

    out_dir = Path(out_dir) if out_dir is not None else ref.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = lambda plain, legacy: out_dir / (legacy.format(ts=ref.key) if ref.is_legacy else plain)  # noqa: E731
    png = plot_report(placed, run.saccades, summary, name("plot.png", "analysis_{ts}.png"), title=f"Session {stamp}")
    csv_path = write_flash_rows(name("flashes.csv", "analysis_{ts}_flashes.csv"), placed, gates)
    json_path = name("suppression.json", "analysis_{ts}_suppression.json")
    json_path.write_text(json.dumps(saved, indent=2))
    report_path = name("report.txt", "analysis_{ts}_report.txt")
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
