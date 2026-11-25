"""The post-hoc analysis' file outputs besides the report text: the plot (matplotlib, headless) and the per-flash CSV.

The plot is the literature's standard figure (Diamond et al. 2000; Idrees et al. 2020; Braun et al. 2021): accuracy against flash time
from saccade ONSET (kernel curve, bootstrap band, fixation baseline, raw trials, flash-count histogram) beside accuracy against
contrast for the saccade and the fixation gate. The numbers behind it come from timecourse.py; this module only draws them."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only renders PNG files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from include.experiment.analysis_constants import FAR_MS, NEAR_ONSET_MS, TIMECOURSE_RANGE_MS
from src.experiment.saccade_detection import Placed, Saccade
from src.experiment.suppression_stats import _contrast_counts, _select_conditions, _weibull, fit_pair_with_slope, wilson
from src.experiment.timecourse import LOW, OK, timecourse_curve, timed_flashes

# -- plot ----------------------------------------------------------------------------------------


SACCADE_COLOUR, FIXATION_COLOUR, CURVE_COLOUR, CHANCE_COLOUR = "#C44E52", "#4C72B0", "#1B1B1B", "#555555"  # red = saccade, blue = fixation
HIST_BIN_MS = 25.0
RUN_NOTE = "Per-run plots are an indication only; the pooled per-person plot is the one to read."
POOLED_NOTE = "This pooled per-person plot is the one to read; single-run plots are an indication only."
Y_LIMITS = (-14.0, 114.0)  # room for the raw-trial rug above 100 % and below 0 %
RUG_BAND = 7.0  # height of the rug's jitter
BY_CONTRAST_RANGE_MS = (-300.0, 700.0)  # x range of timecourse_by_contrast.png (a plot setting, not an analysis threshold): wide enough for the flashes before onset and the +400/+500 ms controls, so accuracy outside the saccade is visible too


def _arr(values: list) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _draw_curve(ax, tc: dict, rel: np.ndarray, ok: np.ndarray, median_saccade_ms: float | None, range_ms: tuple[float, float] = TIMECOURSE_RANGE_MS) -> np.ndarray:
    """One time-course panel's contents (curve, band, fixation band, onset line, chance line, per-flash dots, axes); returns
    `in_range`, which flashes lie inside the x range `range_ms`. The caller adds title, x axis and any histogram."""
    lo_x, hi_x = range_ms
    grid, p, lo, hi = (_arr(tc[k]) for k in ("grid_ms", "p", "lo", "hi"))
    status = np.asarray(tc["status"])
    shown = np.isin(status, (LOW, OK)) & np.isfinite(p)
    solid = (status == OK) & np.isfinite(p)

    if median_saccade_ms:
        ax.axvspan(0, median_saccade_ms, color=SACCADE_COLOUR, alpha=0.10, zorder=0)
    ax.axvline(0, color="#777777", linewidth=1, zorder=1)
    ax.axhline(50, color=CHANCE_COLOUR, linestyle="--", linewidth=1, zorder=1)
    base = tc["baseline"]
    if base["n"]:
        b_lo, b_hi = (100 * v for v in base["ci"])
        ax.axhspan(b_lo, b_hi, color=FIXATION_COLOUR, alpha=0.18, zorder=1)
        ax.axhline(100 * base["p"], color=FIXATION_COLOUR, linewidth=1.2, zorder=2)

    band = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(grid, 100 * lo, 100 * hi, where=shown & band, color="#999999", alpha=0.18, linewidth=0, zorder=2)
    ax.fill_between(grid, 100 * lo, 100 * hi, where=solid & band, color=CURVE_COLOUR, alpha=0.14, linewidth=0, zorder=2)
    ax.plot(grid, np.where(shown, 100 * p, np.nan), color="#8A8A8A", linestyle="--", linewidth=1.6, zorder=3)  # "low" and "ok"; the solid line covers the "ok" part
    ax.plot(grid, np.where(solid, 100 * p, np.nan), color=CURVE_COLOUR, linewidth=2.2, zorder=4)

    in_range = (rel >= lo_x) & (rel <= hi_x)
    jitter = np.random.default_rng(0).random(len(rel)) * RUG_BAND
    y = np.where(ok > 0, 100 + 2 + jitter, 0 - 2 - jitter)
    ax.scatter(rel[in_range], y[in_range], s=7, color=CURVE_COLOUR, alpha=0.35, linewidths=0, zorder=3, clip_on=False)
    if not in_range.any():
        ax.text(0.5, 0.5, "no flash with a known saccade onset", transform=ax.transAxes, ha="center", va="center", color="#777777")

    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("correct (%)")
    ax.grid(True, alpha=0.3)
    return in_range


def _draw_time_course(ax, ax_hist, tc: dict, rel: np.ndarray, ok: np.ndarray, median_saccade_ms: float | None) -> None:
    in_range = _draw_curve(ax, tc, rel, ok, median_saccade_ms)
    lo_x, hi_x = TIMECOURSE_RANGE_MS
    ax.set_title("Accuracy vs flash time from saccade onset", fontsize=10)
    ax.tick_params(labelbottom=False)

    edges = np.arange(lo_x, hi_x + HIST_BIN_MS / 2, HIST_BIN_MS)
    counts, _ = np.histogram(rel[in_range], bins=edges)
    ax_hist.bar(edges[:-1], counts, width=HIST_BIN_MS * 0.9, align="edge", color="#9AA5B1")
    ax_hist.axvline(0, color="#777777", linewidth=1)
    ax_hist.set_xlim(lo_x, hi_x)
    ax_hist.set_ylabel("flashes", fontsize=8)
    ax_hist.set_xlabel("flash time relative to saccade onset (ms)")
    ax_hist.tick_params(labelsize=8)
    ax_hist.grid(True, alpha=0.3, axis="x")


def _draw_by_contrast(ax, sac: list[Placed], fix: list[Placed], suppression: dict | None) -> None:
    """Accuracy per contrast level for the two gates with Wilson CIs; the fitted Weibull curves when the index came from the fit."""
    contrasts = sorted({p.flash.contrast for p in sac + fix})
    top = 100 * max(contrasts, default=0.1)
    fit = suppression is not None and suppression.get("method") == "psychometric" and suppression.get("threshold_saccade") is not None
    slope = fit_pair_with_slope(*_contrast_counts(sac), *_contrast_counts(fix))[3] if fit and sac and fix else None
    for name, group, colour, dx, key in (("saccade gate", sac, SACCADE_COLOUR, -0.04, "threshold_saccade"), ("fixation gate", fix, FIXATION_COLOUR, 0.04, "threshold_fixation")):
        points = []
        for c in contrasts:
            members = [p for p in group if p.flash.contrast == c]
            if members:
                points.append((100 * c, sum(p.flash.correct for p in members), len(members)))
        if points:
            xs = np.asarray([x for x, _, _ in points]) * (1 + dx)
            acc = np.asarray([k / n for _, k, n in points])
            cis = [wilson(k, n) for _, k, n in points]
            ax.errorbar(xs, 100 * acc, yerr=[[100 * (a - lo) for a, (lo, _) in zip(acc, cis)], [100 * (hi - a) for a, (_, hi) in zip(acc, cis)]],
                        fmt="o", color=colour, capsize=4, zorder=3, label=f"{name} (n={len(group)})")
        threshold = (suppression or {}).get(key)
        if slope is not None and threshold is not None:
            grid = np.linspace(top / 200, top * 1.4, 200)
            ax.plot(grid, 100 * _weibull(grid / 100, np.asarray(threshold), np.asarray(slope)), color=colour, linewidth=1.5, alpha=0.8, zorder=2)
    ax.axhline(50, color=CHANCE_COLOUR, linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(0, top * 1.4)
    ax.set_xticks([100 * c for c in contrasts])
    ax.set_xticklabels([f"{100 * c:g}" for c in contrasts])
    ax.set_ylim(0, 105)
    ax.set_xlabel("line contrast (%)")
    ax.set_ylabel("correct (%), with 95% CI")
    ax.set_title("Accuracy vs contrast" + (" (curves: fitted Weibull)" if slope is not None else ""), fontsize=10)
    ax.grid(True, alpha=0.3)
    if contrasts:
        ax.legend(fontsize=8, loc="lower right")


def _legend_handles(tc: dict, median_saccade_ms: float | None, fixation_label: str | None = None) -> list:
    """The shared legend of both time-course figures; `fixation_label` None: the pooled-contrast baseline's numbers."""
    base = tc["baseline"]
    if fixation_label is None:
        fixation_label = f"fixation flashes: {100 * base['p']:.0f}% correct, n={base['n']} (95% CI band)" if base["n"] else "fixation flashes: none"
    handles = [
        Line2D([], [], color=CURVE_COLOUR, linewidth=2.2, label=f"accuracy (Gaussian kernel, SD {tc['sigma_ms']:g} ms; shaded: 95% bootstrap band)"),
        Line2D([], [], color="#8A8A8A", linestyle="--", linewidth=1.6, label=f"fewer than {tc['min_effective_n_solid']:g} effective flashes (hidden below {tc['min_effective_n_drawn']:g})"),
        Patch(color=FIXATION_COLOUR, alpha=0.4, label=fixation_label),
        Line2D([], [], color=CHANCE_COLOUR, linestyle="--", label="chance (50%)"),
    ]
    if median_saccade_ms:
        handles.append(Patch(color=SACCADE_COLOUR, alpha=0.18, label=f"typical saccade (~{median_saccade_ms:.0f} ms)"))
    return handles


def plot_timecourse_figure(
    placed: list[Placed],
    gates: list | None,
    timecourse: dict | None,
    suppression: dict | None,
    median_saccade_ms: float | None,
    out_path: Path,
    title: str = "",
    pooled: bool = False,
) -> Path:
    """The figure: accuracy vs time from onset (main panel, flash-count histogram beneath) and accuracy vs contrast (side panel).
    `timecourse` is timecourse.timecourse_curve() (computed here when None), `suppression` the index result of the same flashes."""
    tc = timecourse if timecourse is not None else timecourse_curve(placed, gates)
    sac, fix, _ = _select_conditions(placed, NEAR_ONSET_MS, FAR_MS, gates)
    rel, ok = timed_flashes(placed, gates)
    fig = plt.figure(figsize=(14, 6.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[4.2, 1])
    ax_main = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(grid[:, 1])
    _draw_time_course(ax_main, ax_hist, tc, rel, ok, median_saccade_ms)
    _draw_by_contrast(ax_side, sac, fix, suppression)

    handles = _legend_handles(tc, median_saccade_ms)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{title or 'Saccadic suppression analysis'}\n{POOLED_NOTE if pooled else RUN_NOTE}", fontsize=11)
    return _save(fig, out_path)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_contrast_panel(ax, contrast: float, placed: list[Placed], gates: list | None, median_saccade_ms: float | None) -> dict:
    """One contrast level's time course (that contrast's flashes only, same kernel / masking / fixation band as the main figure); returns its curve dict."""
    members = [i for i, p in enumerate(placed) if p.flash.contrast == contrast]
    sub = [placed[i] for i in members]
    sub_gates = None if gates is None else [gates[i] for i in members]
    tc = timecourse_curve(sub, sub_gates, range_ms=BY_CONTRAST_RANGE_MS)
    rel, ok = timed_flashes(sub, sub_gates)
    in_range = _draw_curve(ax, tc, rel, ok, median_saccade_ms, BY_CONTRAST_RANGE_MS)
    ax.set_title(f"Contrast {100 * contrast:g} %: {int(in_range.sum())} flashes shown, {tc['baseline']['n']} fixation flashes", fontsize=10, loc="left")
    if not (np.isin(np.asarray(tc["status"]), (LOW, OK)) & np.isfinite(_arr(tc["p"]))).any():
        ax.text(0.5, 0.62, "curve hidden: too few flashes at every time (dots = single flashes)", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#555555",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}, zorder=5)
    return tc


def plot_timecourse_by_contrast(
    placed: list[Placed],
    gates: list | None,
    median_saccade_ms: float | None,
    out_path: Path,
    title: str = "",
    pooled: bool = False,
) -> Path:
    """Accuracy vs flash time from saccade onset, one stacked panel per contrast level (shared x axis), each from that contrast's flashes only.
    Most of each curve is usually hidden by the effective-n rule (a run has only a few flashes near onset); the dots show every flash."""
    contrasts = sorted({p.flash.contrast for p in placed})
    fig, axes = plt.subplots(max(len(contrasts), 1), 1, sharex=True, figsize=(12, 3.0 * max(len(contrasts), 1) + 1.4), layout="constrained", squeeze=False)
    tc = None
    for ax, contrast in zip(axes[:, 0], contrasts):
        tc = _draw_contrast_panel(ax, contrast, placed, gates, median_saccade_ms)
    if tc is None:
        axes[0, 0].text(0.5, 0.5, "no flashes", transform=axes[0, 0].transAxes, ha="center", va="center", color="#777777")
        axes[0, 0].set_xlim(*BY_CONTRAST_RANGE_MS)
    axes[-1, 0].set_xlabel("flash time relative to saccade onset (ms)")
    if tc is not None:
        fig.legend(handles=_legend_handles(tc, median_saccade_ms, "fixation flashes of that contrast (95% CI band)"), loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{title or 'Saccadic suppression analysis'}: accuracy vs flash time, per contrast\n{POOLED_NOTE if pooled else RUN_NOTE}", fontsize=11)
    return _save(fig, out_path)


def plot_report(placed: list[Placed], saccades: list[Saccade], summary: dict, out_path: Path, title: str = "", gates: list | None = None) -> Path:
    """The per-run figure from analyze_session's `summary` (summarize() plus "timecourse" and "suppression" when present;
    `gates` as in suppression_index, None: the index's time windows decide the conditions)."""
    return plot_timecourse_figure(placed, gates, summary.get("timecourse"), summary.get("suppression"), summary.get("median_saccade_ms"), out_path, title)


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
