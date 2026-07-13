from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: just rendering to a PNG file, no GUI backend needed
import matplotlib.pyplot as plt

from include.experiment.types import TrialResult
from src.experiment.zest import ZestStaircase

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "accuracy_comparison.png"


def _threshold_estimate(results: list[TrialResult], phase: str) -> float | None:
    """Replays this phase's grating-shown trials, in trial order, through a
    fresh ZEST staircase to recover the threshold its posterior converged on -
    the same summary Diamond, Ross & Morrone (2000) report (a contrast
    threshold per condition), rather than a per-level accuracy curve, since
    contrast here is chosen adaptively rather than from a fixed set of levels."""
    trials = sorted(
        (r for r in results if r.phase == phase and r.grating_shown and r.contrast is not None),
        key=lambda r: r.index,
    )
    if not trials:
        return None
    zest = ZestStaircase()
    for trial in trials:
        zest.update(trial.contrast, detected=trial.outcome == "hit")
    return zest.threshold_estimate


def build_comparison_graph(results: list[TrialResult], output_path: Path = OUTPUT_PATH) -> Path:
    """Bar chart of estimated contrast detection threshold, presaccade vs.
    saccade, so the saccadic suppression effect reads directly as the height
    difference between the two bars (the paper reported roughly a 10-fold
    threshold elevation during saccades)."""
    presaccade_threshold = _threshold_estimate(results, "presaccade")
    saccade_threshold = _threshold_estimate(results, "saccade")

    labels = []
    thresholds = []
    if presaccade_threshold is not None:
        labels.append("Presaccade\n(baseline)")
        thresholds.append(presaccade_threshold * 100)
    if saccade_threshold is not None:
        labels.append("Saccade")
        thresholds.append(saccade_threshold * 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, thresholds, color=["#4C72B0", "#C44E52"][: len(labels)])
    for bar, value in zip(bars, thresholds):
        ax.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )

    if presaccade_threshold and saccade_threshold:
        ratio = saccade_threshold / presaccade_threshold
        ax.set_title(f"Contrast detection threshold: presaccade vs. saccade\n({ratio:.1f}x elevation)")
    else:
        ax.set_title("Contrast detection threshold: presaccade vs. saccade")

    ax.set_ylabel("Detection threshold (contrast, %)")
    ax.grid(True, alpha=0.3, axis="y")

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
