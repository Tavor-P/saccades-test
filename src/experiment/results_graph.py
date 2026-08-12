from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: just rendering to a PNG file, no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from include.experiment.types import Orientation, TrialResult
from src.experiment.scoring import is_valid_for_saccadic_analysis
from src.experiment.zest import ZestStaircase, psychometric_function

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "accuracy_comparison.png"

_PHASE_STYLE = {
    "presaccade": {"label": "Presaccade (baseline)", "color": "#4C72B0"},
    "saccade": {"label": "Saccade", "color": "#C44E52"},
}

_ORIENTATION_COLORS = {
    Orientation.VERTICAL: "#55A868",
    Orientation.HORIZONTAL: "#8172B2",
}

# False alarms (responded on a catch trial, where no grating was ever shown)
# have no real contrast to plot on the x-axis, so they're placed at a fixed
# spot near the low edge of the log-scaled axis - purely a "no stimulus was
# actually shown here" marker - in a color distinct from every phase's
# correct/incorrect scatter so they can't be mistaken for real detections.
FALSE_ALARM_DISPLAY_CONTRAST_PERCENT = 0.15
FALSE_ALARM_COLOR = "black"

_Replay = tuple[ZestStaircase, list[TrialResult]]


def _exclude_flashes_not_during_saccade(results: list[TrialResult]) -> list[TrialResult]:
    """Drops saccade-phase, flash-shown trials that fail
    is_valid_for_saccadic_analysis - the exact same validity gate
    ExperimentSession._finish_trial applies to the live ZEST update (see
    scoring.py), so the in-session estimate and this end-of-run graph can't
    silently diverge. Flash timing is open-loop now (scheduled off the
    participant's own reaction-time average, not triggered by detected onset
    - see ExperimentSession), so a flash landing outside the real saccade
    window is a structurally common outcome, not rare classifier lag - and
    unlike the old single-sample check, flash_during_saccade can now be None
    for a saccade-phase, grating-shown trial too (landing was never
    confirmed before the trial ended, so validity is genuinely
    undeterminable - see TrialResult.flash_during_saccade), which is just as
    unusable for analysis as an explicit False.

    flash_during_saccade is left alone (not excluded) for presaccade rows,
    catch trials, and rt_test rows - all of which are always None here since
    nothing ever flashes on them - so this never touches those, only
    saccade-phase rows where a grating was actually shown."""
    return [
        r
        for r in results
        if not (r.phase == "saccade" and r.grating_shown and not is_valid_for_saccadic_analysis(r.flash_during_saccade))
    ]


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
        zest.update(trial.contrast, detected=trial.outcome == "correct")
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


def _plot_psychometric_curves(ax, replays: dict[str, _Replay], results: list[TrialResult]) -> None:
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
        trial_correct = np.array([100 if t.outcome == "correct" else 0 for t in trials], dtype=float)
        jitter = rng.uniform(-3, 3, size=len(trials))
        ax.scatter(trial_contrasts, trial_correct + jitter, color=style["color"], alpha=0.4, s=18)

    catch_trials = [r for r in results if not r.grating_shown]
    false_alarms = [r for r in catch_trials if r.outcome == "false_alarm"]
    if false_alarms:
        y_jitter = rng.uniform(-3, 3, size=len(false_alarms))
        ax.scatter(
            [FALSE_ALARM_DISPLAY_CONTRAST_PERCENT] * len(false_alarms),
            100 + y_jitter,
            color=FALSE_ALARM_COLOR,
            marker="x",
            alpha=0.7,
            s=30,
            label=f"False alarm ({len(false_alarms)}/{len(catch_trials)})",
        )

    shown_trials = [r for r in results if r.grating_shown and r.outcome in ("correct", "incorrect", "miss")]
    if shown_trials:
        correct_count = sum(1 for r in shown_trials if r.outcome == "correct")
        ax.text(
            0.02,
            0.98,
            f"{correct_count}/{len(shown_trials)} correct overall",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Contrast (%)")
    ax.set_ylabel("P(correct) (%)")
    ax.set_ylim(-8, 108)
    ax.set_title("Fitted psychometric function")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _orientation_accuracy(trials: list[TrialResult]) -> tuple[int, int]:
    """(correct count, total count) among trials scored correct/incorrect -
    misses (no response at all) count against accuracy same as a wrong guess
    would, but aren't double-counted here since callers already filter to
    correct/incorrect/miss before calling this."""
    correct = sum(1 for r in trials if r.outcome == "correct")
    return correct, len(trials)


def _plot_orientation_v_curve(ax, results: list[TrialResult]) -> None:
    """Fitted detection-rate-vs-contrast curve, combined across both phases,
    for each grating orientation - vertical-striped gratings get smeared by
    the saccade's own (horizontal) motion in a way horizontal-striped ones
    don't, so this is worth checking directly rather than assuming
    orientation doesn't matter. Horizontal is mirrored onto the negative half
    of the x-axis and vertical onto the positive half, both anchored at
    (0, guess rate) - the two curves meet in the middle, forming a V/notch
    shape whose two arms' steepness and threshold position (readable off
    where each arm crosses the dotted 50% line) show the orientation gap at a
    glance instead of collapsing it into one number per orientation like a
    plain accuracy bar would."""
    by_orientation: dict[Orientation, list[TrialResult]] = {Orientation.HORIZONTAL: [], Orientation.VERTICAL: []}
    for r in results:
        if r.orientation in by_orientation and r.outcome in ("correct", "incorrect", "miss") and r.contrast is not None:
            by_orientation[r.orientation].append(r)

    fits = {}
    for orientation, trials in by_orientation.items():
        if not trials:
            continue
        zest = ZestStaircase()
        for trial in trials:
            zest.update(trial.contrast, detected=trial.outcome == "correct")
        fits[orientation] = zest

    if not fits:
        ax.set_visible(False)
        return

    max_contrast = max(t.contrast for trials in by_orientation.values() for t in trials) * 100
    xs = np.linspace(0, max_contrast * 1.05, 200)
    rng = np.random.default_rng(0)  # deterministic scatter jitter between graph renders

    for orientation, sign in ((Orientation.HORIZONTAL, -1), (Orientation.VERTICAL, 1)):
        if orientation not in fits:
            continue
        zest = fits[orientation]
        color = _ORIENTATION_COLORS[orientation]
        curve = [100 * psychometric_function(x / 100, zest.threshold_estimate, zest.beta, zest.guess_rate, zest.lapse_rate) for x in xs]
        ax.plot(
            sign * xs,
            curve,
            color=color,
            linewidth=2,
            label=f"{orientation.value.capitalize()} (threshold {zest.threshold_estimate:.1%})",
        )

        trials = by_orientation[orientation]
        contrasts = np.array([t.contrast * 100 for t in trials])
        correct = np.array([100 if t.outcome == "correct" else 0 for t in trials], dtype=float)
        jitter = rng.uniform(-3, 3, size=len(trials))
        ax.scatter(sign * contrasts, correct + jitter, color=color, alpha=0.35, s=22, edgecolors="none")

    guess_rate_percent = 100 * next(iter(fits.values())).guess_rate
    ax.axvline(0, color="#999999", linewidth=1, linestyle=":")
    ax.axhline(guess_rate_percent, color="#999999", linewidth=1, linestyle=":")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{abs(val):.0f}"))
    ax.set_xlabel("Contrast (%) - horizontal (left) vs. vertical (right)")
    ax.set_ylabel("Accuracy / detection rate (%)")
    ax.set_ylim(-8, 108)
    ax.set_title("Orientation discrimination vs. contrast")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_orientation_accuracy_by_phase(ax, results: list[TrialResult]) -> None:
    """Orientation-accuracy comparison broken out per phase - lets you check
    whether any vertical-vs-horizontal gap shows up in both the presaccade
    baseline and the saccade condition, or only appears once an actual
    saccade (and its motion smear) is involved."""
    phases = [p for p in ("presaccade", "saccade") if any(r.phase == p for r in results)]
    if not phases:
        ax.set_visible(False)
        return

    orientations = (Orientation.VERTICAL, Orientation.HORIZONTAL)
    width = 0.35
    x = np.arange(len(phases))

    for i, orientation in enumerate(orientations):
        values = []
        for phase in phases:
            trials = [
                r
                for r in results
                if r.phase == phase and r.orientation is orientation and r.outcome in ("correct", "incorrect", "miss")
            ]
            correct, total = _orientation_accuracy(trials)
            values.append(100 * correct / total if total else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=orientation.value.capitalize(), color=_ORIENTATION_COLORS[orientation])

    ax.set_xticks(x)
    ax.set_xticklabels([_PHASE_STYLE[p]["label"] for p in phases])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Correct (%)")
    ax.set_title("Accuracy by orientation × phase")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def build_comparison_graph(results: list[TrialResult], output_path: Path = OUTPUT_PATH) -> Path:
    """Four-panel figure: estimated contrast detection threshold (with a 68%
    credible interval) presaccade vs. saccade; the fitted psychometric
    function each threshold came from, with actual trial responses (and false
    alarms, marked in black) scattered on top so the fit can be sanity-checked
    by eye; a mirrored detection-rate-vs-contrast curve (horizontal vs.
    vertical orientation, combined across phases) meeting at chance level in
    the middle; and orientation accuracy broken out per phase."""
    results = _exclude_flashes_not_during_saccade(results)
    replays = {}
    for phase in ("presaccade", "saccade"):
        replay = _replay_zest(results, phase)
        if replay is not None:
            replays[phase] = replay

    fig, ((ax_bars, ax_curve), (ax_orientation, ax_orientation_phase)) = plt.subplots(
        2, 2, figsize=(12, 10), layout="constrained"
    )
    _plot_threshold_bars(ax_bars, replays)
    if replays:
        _plot_psychometric_curves(ax_curve, replays, results)
    _plot_orientation_v_curve(ax_orientation, results)
    _plot_orientation_accuracy_by_phase(ax_orientation_phase, results)
    fig.suptitle("Contrast detection: presaccade vs. saccade")

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
