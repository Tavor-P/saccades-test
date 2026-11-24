"""gaze_timeline.png: horizontal gaze position over the whole run, stacked in rows of GAZE_TIMELINE_ROW_S seconds, with every
detected saccade shaded and every flash drawn as a vertical bar coloured by its contrast, so one can see when the eye moved
relative to each flash. It draws exactly what the analysis uses: the trace `analyze_run` loaded (PC timeline, after the clock
offset), the detected saccades and the placed flashes (+ the display delay, as the gates do). Units: gaze x is the 0-1 ratio of
the scene width. Nothing is computed here that the analysis would reuse. Thresholds: include/experiment/analysis_constants.py."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only renders PNG files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from include.experiment.analysis_constants import GAZE_TIMELINE_BIN_MS, GAZE_TIMELINE_MAX_ROWS, GAZE_TIMELINE_ROW_S
from src.experiment.flash_gates import EXCLUDED, TRACKING_GAP_REASON
from src.experiment.saccade_detection import Placed, Saccade
from src.timebase import ms_to_s

ROW_HEIGHT_IN, FIGURE_WIDTH_IN, HEADER_IN, FOOTER_IN, DPI = 1.0, 14.0, 1.2, 0.6, 100
LEAD_IN_S = 1.0  # the first row starts this long before the first flash (its saccade starts earlier than the flash)
TAIL_S = 1.5  # ...and the last row ends this long after the last flash
SACCADE_SHADE = "#B8B8B8"
CMAP = plt.get_cmap("YlGnBu")  # sequential: light = low contrast


def contrast_colours(contrasts: list[float]) -> dict[float, tuple]:
    """One colour per contrast level, light for the lowest, dark for the highest."""
    levels = sorted(set(contrasts))
    positions = [0.8] if len(levels) == 1 else list(np.linspace(0.3, 1.0, len(levels)))
    return {c: CMAP(p) for c, p in zip(levels, positions)}


def binned_trace(t: np.ndarray, x: np.ndarray, start: float, end: float, bin_ms: float = GAZE_TIMELINE_BIN_MS) -> tuple[np.ndarray, np.ndarray]:
    """(bin centre times, mean x) of the samples in [start, end) for bins of `bin_ms` anchored at `start`; empty bins are left out."""
    width = ms_to_s(bin_ms)
    keep = (t >= start) & (t < end)
    index = np.floor((t[keep] - start) / width).astype(int)
    if len(index) == 0:
        return np.empty(0), np.empty(0)
    counts = np.bincount(index)
    sums = np.bincount(index, weights=x[keep])
    filled = counts > 0
    return start + (np.nonzero(filled)[0] + 0.5) * width, sums[filled] / counts[filled]


def row_count(first_s: float, last_s: float, row_s: float = GAZE_TIMELINE_ROW_S) -> int:
    """Rows needed for a run whose first / last flash are `first_s` / `last_s` (any common origin), before the cap."""
    return max(1, math.ceil((last_s - first_s + LEAD_IN_S + TAIL_S) / row_s))


def plot_gaze_timeline(
    t: np.ndarray,
    x: np.ndarray,
    saccades: list[Saccade],
    placed: list[Placed],
    gates: list | None,
    out_path: Path,
    title: str = "",
    flash_delay_ms: float = 0.0,
) -> Path:
    """The stacked-rows figure (see the module docstring); returns `out_path`. A run with no flash gets one row of its first seconds."""
    delay_s = ms_to_s(flash_delay_ms)
    times = [p.flash.time + delay_s for p in placed]
    origin = min(times) if times else (float(t[0]) if len(t) else 0.0)  # seconds on the axis are measured from the first flash
    last = max(times) if times else origin + GAZE_TIMELINE_ROW_S
    needed = row_count(origin, last)
    n_rows = min(needed, GAZE_TIMELINE_MAX_ROWS)
    first_start = origin - LEAD_IN_S
    colours = contrast_colours([p.flash.contrast for p in placed])

    span_end = first_start + n_rows * GAZE_TIMELINE_ROW_S
    bt, bx = binned_trace(t, x, first_start, span_end)
    if len(bx) > 1:
        lo, hi = np.percentile(bx, [0.5, 99.5])
        pad = 0.08 * max(hi - lo, 0.1)
        y_lim = (float(lo - pad), float(hi + pad))
    else:
        y_lim = (0.0, 1.0)

    height = n_rows * ROW_HEIGHT_IN + HEADER_IN + FOOTER_IN
    fig, axes = plt.subplots(n_rows, 1, figsize=(FIGURE_WIDTH_IN, height), sharey=True, squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - HEADER_IN / height, bottom=FOOTER_IN / height, hspace=0.55)
    for r, ax in enumerate(axes):
        start = first_start + r * GAZE_TIMELINE_ROW_S
        end = start + GAZE_TIMELINE_ROW_S
        for s in saccades:
            if s.offset >= start and s.onset <= end:
                ax.axvspan(s.onset - origin, s.offset - origin, color=SACCADE_SHADE, alpha=0.6, linewidth=0, zorder=1)
        in_row = (bt >= start) & (bt < end)
        ax.plot(bt[in_row] - origin, bx[in_row], ".", color="#222222", markersize=2.2, zorder=3)
        for i, (p, when) in enumerate(zip(placed, times)):
            if not start <= when < end:
                continue
            gate = gates[i] if gates is not None else None
            excluded = gate is not None and gate.condition == EXCLUDED
            ax.vlines(when - origin, y_lim[0], y_lim[1], color=colours[p.flash.contrast], linewidth=2.2, linestyle=(0, (3, 2)) if excluded else "-",
                      alpha=0.55 if excluded else 0.95, zorder=2)
            if not p.flash.correct:
                answered = p.flash.outcome != "miss"
                ax.plot(when - origin, y_lim[1], "x" if answered else "o", color="#C44E52", markersize=6, markerfacecolor="none", zorder=4, clip_on=False)
            if gate is not None and gate.reason == TRACKING_GAP_REASON:
                ax.plot(when - origin, y_lim[0], "^", color="#555555", markersize=5, markerfacecolor="none", zorder=4, clip_on=False)
        ax.set_xlim(start - origin, end - origin)
        ax.set_ylim(*y_lim)
        ax.set_xticks(np.arange(math.ceil((start - origin) / 5) * 5, (end - origin) + 0.1, 5))
        ax.tick_params(labelsize=7, length=2)
        ax.grid(True, axis="x", alpha=0.25)
        ax.margins(x=0)
    axes[-1].set_xlabel("seconds from the first flash")
    fig.supylabel(f"gaze x (fraction of the scene width; each dot = mean of {GAZE_TIMELINE_BIN_MS:g} ms)", fontsize=9)

    handles: list = [Patch(color=colours[c], label=f"flash, contrast {100 * c:g}%") for c in sorted(colours)]
    handles += [
        Patch(color=SACCADE_SHADE, alpha=0.6, label="detected saccade"),
        Line2D([], [], color="#777777", linewidth=2.2, linestyle=(0, (3, 2)), label="dashed: flash excluded by the gates"),
        Line2D([], [], color="#C44E52", marker="x", linestyle="", markerfacecolor="none", label="wrong answer"),
        Line2D([], [], color="#C44E52", marker="o", linestyle="", markerfacecolor="none", label="no answer"),
        Line2D([], [], color="#555555", marker="^", linestyle="", markerfacecolor="none", label="tracking gap at the flash"),
    ]
    note = "" if needed <= n_rows else f"  (first {n_rows * GAZE_TIMELINE_ROW_S / 60:.0f} of {needed * GAZE_TIMELINE_ROW_S / 60:.0f} min shown)"
    fig.suptitle(f"{title or 'Gaze timeline'}: eye position and flashes{note}", fontsize=12, y=1 - 0.15 / height)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.5 / height), ncol=5, fontsize=8, frameon=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path
