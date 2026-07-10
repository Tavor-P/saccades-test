from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: just rendering to a PNG file, no GUI backend needed
import matplotlib.pyplot as plt

from include.experiment.types import TrialResult

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "accuracy_comparison.png"


def _accuracy_by_contrast(results: list[TrialResult], phase: str) -> dict[float, float]:
    shown: dict[float, int] = {}
    hits: dict[float, int] = {}
    for result in results:
        if result.phase != phase or not result.square_shown or result.contrast is None:
            continue
        shown[result.contrast] = shown.get(result.contrast, 0) + 1
        if result.outcome == "hit":
            hits[result.contrast] = hits.get(result.contrast, 0) + 1
    return {level: hits.get(level, 0) / count for level, count in shown.items()}


def build_comparison_graph(results: list[TrialResult], output_path: Path = OUTPUT_PATH) -> Path:
    """Plots detection accuracy vs. contrast for the presaccade and saccade
    phases on one graph, so the saccadic suppression effect is directly
    visible as the gap between the two lines at matching contrast levels."""
    presaccade = _accuracy_by_contrast(results, "presaccade")
    saccade = _accuracy_by_contrast(results, "saccade")

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, data, marker in (("Presaccade (baseline)", presaccade, "o"), ("Saccade", saccade, "s")):
        if not data:
            continue
        levels = sorted(data)
        accuracies = [data[level] * 100 for level in levels]
        ax.plot(levels, accuracies, marker=marker, label=label)

    ax.set_xlabel("Contrast")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Detection accuracy by contrast: presaccade vs. saccade")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
