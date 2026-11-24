"""The post-hoc analysis' file outputs besides the report text: the plot (matplotlib, headless) and the per-flash CSV."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only renders PNG files
import matplotlib.pyplot as plt

from src.experiment.saccade_detection import Placed, Saccade
from src.experiment.suppression_stats import wilson

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


def write_flash_rows(path: Path, placed: list[Placed], gates: list, extra: dict[str, list] | None = None) -> Path:
    """The per-flash CSV; `extra` columns (name -> one value per flash) go first (the participant analysis adds the run).
    flash_rel_onset_ms is relative to the flash's own leg's saccade in return-jump runs (the nearest onset otherwise);
    flash_leg is blank for runs without return jumps; saccade_latency_ms is from the go cue (out) or the jump (return)."""
    extra = extra or {}
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*extra, "trial", "contrast", "orientation", "outcome", "flash_rel_onset_ms", "inside_saccade", "condition", "gate_reason", "flash_leg", "saccade_latency_ms"])
        for i, (p, g) in enumerate(zip(placed, gates)):
            leg = (p.flash.leg or "out") if p.flash.jump is not None else ""
            writer.writerow([*(values[i] for values in extra.values()), p.flash.trial, p.flash.contrast, p.flash.line_side, p.flash.outcome, "" if p.rel_ms is None else round(p.rel_ms, 1), p.inside, g.condition, g.reason,
                             leg, "" if g.latency_ms is None else round(g.latency_ms, 1)])
    return Path(path)
