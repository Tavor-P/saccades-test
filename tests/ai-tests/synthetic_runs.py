"""Anonymous synthetic run folders for the analysis tests (never under data/: always pass a pytest tmp_path).

`write_run()` lays out data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/ exactly as a real run does: trials.csv through the
real `TRIAL_COLUMNS` writers, a session.json with run / settings / gaze_quality, and a 200 Hz gaze_trace.csv whose saccade
onsets are known. Each trial: go cue, a 80 ms saccade 200 ms later (latency inside the gates' 100-500 ms), one flash either
mid-saccade (20-60 ms after onset: the eye is fast, so the gates call it `saccade`) or well away from any saccade
(`fixation`). Answers are drawn from a known 2AFC Weibull per condition, so the suppression index is known in advance.
"""

import csv
import json
from pathlib import Path

import numpy as np

from include.experiment.types import Side, TrialResult
from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.experiment import session_files
from src.experiment.analyze_session import _weibull
from src.experiment.logger import TRIAL_COLUMNS

RATE_HZ = 200
CLOCK_OFFSET_S = 0.25  # PC clock = phone clock + this
TRIAL_SPACING_S = 2.5
LATENCY_S = 0.2
SACCADE_S = 0.08
LO, HI = 0.40, 0.70
DEFAULT_GEOMETRY = {"screen_height_cm": 26.0, "viewing_distance_cm": 50.0}


def write_run(
    data_dir: Path,
    participant: str,
    start_unix: float,
    n_trials: int = 60,
    contrasts_pct: tuple[float, ...] = (4.0, 6.0, 8.0),
    saccade_fraction: float = 0.35,
    threshold_saccade: float = 0.08,
    threshold_fixation: float = 0.04,
    slope: float = 3.0,
    test_mode: bool = False,
    geometry: dict | None = None,
    seed: int = 0,
    record_contrasts: bool = True,
) -> Path:
    """One complete synthetic run folder; returns it."""
    rng = np.random.default_rng(seed)
    geometry = {**DEFAULT_GEOMETRY, **(geometry or {})}
    folder = session_files.create_session_folder(Path(data_dir), participant, start_unix, test_mode)
    t0 = float(start_unix) + 2.0
    cues = [t0 + TRIAL_SPACING_S * i for i in range(n_trials)]
    onsets = [g + LATENCY_S for g in cues]

    # Gaze trace (PC time), alternating between LO and HI with a smooth saccade at each onset.
    t = np.arange(float(start_unix), cues[-1] + 3.0, 1 / RATE_HZ)
    x = np.full_like(t, LO)
    level = LO
    for onset in onsets:
        target = HI if level == LO else LO
        u = np.clip((t - onset) / SACCADE_S, 0, 1)
        x = np.where(t >= onset, level + (target - level) * (3 * u**2 - 2 * u**3), x)
        level = target
    with (folder / session_files.GAZE_TRACE_FILE).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "device_unix_s", "x_px", "worn"])
        for ti, xi in zip(t, x):
            w.writerow([f"{ti - start_unix:.4f}", f"{ti - CLOCK_OFFSET_S:.4f}", f"{xi * NEON_SCENE_WIDTH:.3f}", 1])

    contrasts = [c / 100 for c in contrasts_pct]
    with (folder / session_files.TRIALS_FILE).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c.name for c in TRIAL_COLUMNS])
        w.writeheader()
        for i, (cue, onset) in enumerate(zip(cues, onsets)):
            contrast = contrasts[i % len(contrasts)]
            during = rng.random() < saccade_fraction
            flash = onset + rng.uniform(0.02, 0.06) if during else (cue + rng.uniform(0.0, 0.06) if rng.random() < 0.5 else onset + SACCADE_S + rng.uniform(0.3, 0.5))
            p = float(_weibull(np.asarray(contrast), np.asarray(threshold_saccade if during else threshold_fixation), np.asarray(slope)))
            correct = rng.random() < p
            side = Side.ABOVE if i % 2 == 0 else Side.BELOW
            answer = side if correct else (Side.BELOW if side == Side.ABOVE else Side.ABOVE)
            result = TrialResult(
                index=i, phase="saccade", source=None, target=None, saccade_duration_ms=SACCADE_S * 1000, grating_shown=True, contrast=contrast,
                orientation=side, responded=True, response_orientation=answer, response_time_ms=600.0, outcome="correct" if correct else "incorrect",
                reaction_latency_ms=LATENCY_S * 1000, flash_during_saccade=during, scheduled_delay_ms=round(1000 * (flash - cue), 1),
                grating_shown_at_unix_ms=round(1000 * flash, 1), tracking_ok=True,
            )
            w.writerow({c.name: c.write(result) for c in TRIAL_COLUMNS})

    settings = {
        **geometry,
        "screen_height_deg": 29.0,
        "line_offset_deg": 2.0,
        "line_sd_deg": 0.15,
        "geometry_is_placeholder": geometry == DEFAULT_GEOMETRY,
        "saccade_contrasts_pct": list(contrasts_pct) if record_contrasts else None,
    }
    meta = {
        "participant": {"id": participant},
        "run": {"mode": "test" if test_mode else "real", "unix_timestamp": int(start_unix), "folder": f"{folder.parent.name}/{folder.name}"},
        "settings": settings,
        "gaze_quality": {"clock_offset_ms": CLOCK_OFFSET_S * 1000, "sample_rate_hz": RATE_HZ, "latency_ms_median": 17.5, "longest_gap_ms": 12.0, "gaps_over_threshold": 0},
    }
    (folder / session_files.SESSION_FILE).write_text(json.dumps(meta, indent=2))
    return folder
