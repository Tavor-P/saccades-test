from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: just rendering to a PNG file, no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np

from include.experiment.types import TrialResult
from src.experiment.zest import ZestStaircase, psychometric_function

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "accuracy_comparison.png"

_PHASE_STYLE = {
    "presaccade": {"label": "Presaccade (baseline)", "color": "#4C72B0"},
    "saccade": {"label": "Saccade", "color": "#C44E52"},
}

_Replay = tuple[ZestStaircase, list[TrialResult]]


def _replay_zest(results: list[TrialResult], phase: str) -> _Replay | None:
    """Replays this phase's grating-shown trials, in trial order, through a
    fresh ZEST staircase to recover the posterior it converged on - the same
    summary Diamond, Ross & Morrone (2000) report (a contrast threshold per
    condition), rather than a per-level accuracy curve, since contrast here is
    chosen adaptively rather than from a fixed set of levels."""
    trials = sorted(
        (r for r in results if r.phase == phase and r.grating_shown and r.contrast is not None),
        key=lambda r: r.index,
    )
    if not trials:
        return None
    zest = ZestStaircase()
    for trial in trials:
        zest.update(trial.contrast, detected=trial.outcome == "hit")
    return zest, trials


def _plot_threshold_bars(ax, replays: dict[str, _Replay]) -> None:
    labels, values, errors, colors = [], [], [], []
    for phase, (zest, _) in replays.items():
        style = _PHASE_STYLE[phase]
        threshold = zest.threshold_estimate
        lo, hi = zest.credible_interval(0.68)
        labels.append(style["label"].replace(" (", "\n("))
        values.append(threshold * 100)
        errors.append([(threshold - lo) * 100, (hi - threshold) * 100])
        colors.append(style["color"])

    bars = ax.bar(labels, values, yerr=list(zip(*errors)), capsize=6, color=colors)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    if "presaccade" in replays and "saccade" in replays:
        ratio = replays["saccade"][0].threshold_estimate / replays["presaccade"][0].threshold_estimate
        ax.set_title(f"Detection threshold\n({ratio:.1f}x elevation)")
    else:
        ax.set_title("Detection threshold")
    ax.set_ylabel("Contrast (%), with 68% credible interval")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_psychometric_curves(ax, replays: dict[str, _Replay]) -> None:
    contrasts = np.logspace(-3, 0, 200)  # 0.1% - 100%
    rng = np.random.default_rng(0)  # deterministic scatter jitter between graph renders

    for phase, (zest, trials) in replays.items():
        style = _PHASE_STYLE[phase]
        curve = [
            100 * psychometric_function(c, zest.threshold_estimate, zest.beta, zest.guess_rate, zest.lapse_rate)
            for c in contrasts
        ]
        ax.plot(contrasts * 100, curve, color=style["color"], label=style["label"])

        trial_contrasts = [t.contrast * 100 for t in trials]
        trial_hits = np.array([100 if t.outcome == "hit" else 0 for t in trials], dtype=float)
        jitter = rng.uniform(-3, 3, size=len(trials))
        ax.scatter(trial_contrasts, trial_hits + jitter, color=style["color"], alpha=0.4, s=18)

    ax.set_xscale("log")
    ax.set_xlabel("Contrast (%)")
    ax.set_ylabel("P(detect) (%)")
    ax.set_ylim(-8, 108)
    ax.set_title("Fitted psychometric function")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)


def build_comparison_graph(results: list[TrialResult], output_path: Path = OUTPUT_PATH) -> Path:
    """Two-panel figure: estimated contrast detection threshold (with a 68%
    credible interval) presaccade vs. saccade, and the fitted psychometric
    function each threshold came from, with actual trial responses scattered
    on top so the fit can be sanity-checked by eye."""
    replays = {}
    for phase in ("presaccade", "saccade"):
        replay = _replay_zest(results, phase)
        if replay is not None:
            replays[phase] = replay

    fig, (ax_bars, ax_curve) = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    _plot_threshold_bars(ax_bars, replays)
    if replays:
        _plot_psychometric_curves(ax_curve, replays)
    fig.suptitle("Contrast detection: presaccade vs. saccade")

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
