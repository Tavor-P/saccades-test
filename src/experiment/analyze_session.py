"""Post-hoc analysis of a saccade-phase session (`-m src.experiment.analyze_session [timestamp]`).

Finds the saccades in the raw gaze trace (`results_<ts>_gaze_trace.csv`, stamped with the phone's own capture
times, so the ~90 ms delivery delay and dropped samples don't matter), places every logged flash against the
nearest saccade onset, and reports detection accuracy as a function of flash time relative to onset - the
saccadic-suppression curve - plus a plot and a per-flash CSV. The live `flash_during_saccade` flag is not used.

Run it on the latest session with no arguments, or give a session timestamp (`results_<timestamp>.csv`) or a path.
Units: gaze x is a 0-1 ratio of the scene (or screen) width per `x_px / NEON_SCENE_WIDTH`; speeds are ratio/s.
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
from src.experiment.flash_gates import (
    EXCLUDED,
    FIXATION,
    FIXATION_EXCLUSION_MS,
    SACCADE,
    classify_flashes,
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
    time: float  # PC unix seconds the grating was drawn (before any display-delay correction)
    contrast: float
    orientation: str
    outcome: str  # correct | incorrect | miss
    go_cue: float | None = None  # PC unix seconds the trial's target appeared (flash time minus its scheduled delay)

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


def load_trace(path: Path, clock_offset_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(capture times in PC unix seconds, gaze x ratio), strictly increasing, without lost samples."""
    times, xs = [], []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("x_px") or not row.get("device_unix_s"):
                continue
            times.append(float(row["device_unix_s"]) + clock_offset_s)
            xs.append(float(row["x_px"]) / NEON_SCENE_WIDTH)
    t, x = np.asarray(times), np.asarray(xs)
    keep = np.concatenate(([True], np.diff(t) > 0)) if len(t) else np.asarray([], dtype=bool)
    ok = keep & ~np.isnan(x)
    return t[ok], x[ok]


def load_flashes(path: Path) -> list[Flash]:
    """Every shown saccade-phase grating with a logged wall-clock time."""
    flashes = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "saccade" or row.get("grating_shown") != "True":
                continue
            if not row.get("grating_shown_at_unix_ms") or not row.get("contrast"):
                continue
            shown = float(row["grating_shown_at_unix_ms"]) / 1000
            delay = row.get("scheduled_delay_ms")
            flashes.append(
                Flash(
                    trial=int(row["trial_index"]),
                    time=shown,
                    contrast=float(row["contrast"]),
                    orientation=row.get("orientation", ""),
                    outcome=row["outcome"],
                    go_cue=shown - float(delay) / 1000 if delay else None,
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


# -- statistics ----------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


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


def format_report(summary: dict, label: str = "") -> str:
    s = summary
    lines = [f"Saccade-phase analysis {label}".strip()]
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


def format_gate_report(gates: dict) -> str:
    """The pre-specified gating (src/experiment/flash_gates.py): saccade vs fixation flashes and exclusions."""
    lines = ["Gated analysis (pre-specified criteria, not tuned on the data):"]
    lines.append(f"  saccade flashes (eye moving fast, valid latency and amplitude): {_pct(tuple(gates[SACCADE]))}")
    lines.append(f"  fixation flashes (>= {FIXATION_EXCLUSION_MS:.0f} ms from any saccade, tracked throughout): {_pct(tuple(gates[FIXATION]))}")
    lines.append(f"  excluded: {gates[EXCLUDED][1]}" + "".join(f"\n    {n} x {reason}" for reason, n in gates["reasons"].items()))
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


def latest_timestamp(data_dir: Path) -> str:
    """The newest session that has a gaze trace and at least one saccade-phase flash row."""
    candidates = sorted(Path(data_dir).glob("results_*_gaze_trace.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for trace in candidates:
        stamp = trace.name[len("results_") : -len("_gaze_trace.csv")]
        csv_path = Path(data_dir) / f"results_{stamp}.csv"
        if csv_path.exists() and load_flashes(csv_path):
            return stamp
    raise FileNotFoundError(f"no session with a gaze trace and saccade-phase flashes found in {data_dir}")


def analyze(
    timestamp: str | None = None,
    data_dir: Path = DATA_DIR,
    clock_offset_ms: float | None = None,
    flash_delay_ms: float = 0.0,
    min_amplitude: float = MIN_AMPLITUDE,
    out_dir: Path | None = None,
) -> dict:
    """Runs the whole analysis for one session; writes `analysis_<ts>.png` and `analysis_<ts>_flashes.csv`
    to `out_dir` (default: data_dir) and returns {"summary", "report", "png", "csv"}."""
    data_dir = Path(data_dir)
    stamp = timestamp or latest_timestamp(data_dir)
    base = data_dir / f"results_{stamp}"
    if clock_offset_ms is None:
        meta = json.loads(Path(f"{base}_meta.json").read_text())
        clock_offset_ms = (meta.get("gaze_quality") or {}).get("clock_offset_ms")
        if clock_offset_ms is None:
            raise ValueError("this session has no clock_offset_ms in its meta file; pass --offset-ms")
    t, x = load_trace(Path(f"{base}_gaze_trace.csv"), clock_offset_ms / 1000)
    saccades = detect_saccades(t, x, min_amplitude=min_amplitude)
    flashes = load_flashes(Path(f"{base}.csv"))
    placed = place_flashes(flashes, saccades, flash_delay_ms)
    summary = summarize(placed, saccades)
    gates = classify_flashes(placed, saccades, t, x, flash_delay_ms, [p.flash.go_cue for p in placed])
    summary["gates"] = summarize_gates(placed, gates)
    report = format_report(summary, label=f"(session {stamp})") + "\n" + format_gate_report(summary["gates"])

    out_dir = Path(out_dir) if out_dir is not None else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = plot_report(placed, saccades, summary, out_dir / f"analysis_{stamp}.png", title=f"Session {stamp}")
    csv_path = out_dir / f"analysis_{stamp}_flashes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "contrast", "orientation", "outcome", "flash_rel_onset_ms", "inside_saccade", "condition", "gate_reason"])
        for p, g in zip(placed, gates):
            writer.writerow([p.flash.trial, p.flash.contrast, p.flash.orientation, p.flash.outcome, "" if p.rel_ms is None else round(p.rel_ms, 1), p.inside, g.condition, g.reason])
    return {"summary": summary, "report": report, "png": png, "csv": csv_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy vs flash time relative to saccade onset, from a session's raw gaze trace.")
    parser.add_argument("timestamp", nargs="?", help="session timestamp (results_<timestamp>.csv); default: the newest session")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--offset-ms", type=float, default=None, help="phone-to-PC clock offset (default: from the meta file)")
    parser.add_argument("--flash-delay-ms", type=float, default=0.0, help="time from drawing the flash to light leaving the screen")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE, help="smallest saccade to count (fraction of the width)")
    args = parser.parse_args(argv)
    result = analyze(args.timestamp, args.data_dir, args.offset_ms, args.flash_delay_ms, args.min_amplitude)
    print(result["report"])
    print(f"\nplot: {result['png']}\nper-flash rows: {result['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
