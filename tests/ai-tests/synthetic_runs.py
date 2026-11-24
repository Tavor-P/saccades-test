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


def _write_trace(folder: Path, t: np.ndarray, x: np.ndarray, start_unix: float) -> None:
    """gaze_trace.csv (t_s, device_unix_s, x_px, worn) for gaze x ratios `x` at PC times `t`: the bytes csv.writer would give, formatted in bulk."""
    lines = ["t_s,device_unix_s,x_px,worn"]
    lines += [f"{a:.4f},{b:.4f},{c:.3f},1" for a, b, c in zip((t - start_unix).tolist(), (t - CLOCK_OFFSET_S).tolist(), (x * NEON_SCENE_WIDTH).tolist())]
    (folder / session_files.GAZE_TRACE_FILE).write_bytes((chr(13) + chr(10)).join(lines + [""]).encode())


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
    return_jump: bool = False,
    return_trigger: str = "arrival",
    return_latency_ms: float = 150.0,
    return_latency_sd_ms: float = 0.0,
    return_fraction: float = 0.8,
    overshoot: bool = False,
    jump_delay_ms: float = 10.0,
) -> Path:
    """One complete synthetic run folder; returns it. `return_jump=True` writes return-jump trials instead (see
    `_return_jump_run`); the default path draws exactly the same random numbers as before, so seeded runs are unchanged."""
    if return_jump:
        return _return_jump_run(data_dir, participant, start_unix, n_trials, contrasts_pct, saccade_fraction, threshold_saccade, threshold_fixation,
                                slope, test_mode, geometry, seed, return_trigger, return_latency_ms, return_latency_sd_ms, return_fraction, overshoot,
                                jump_delay_ms)
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
    _write_trace(folder, t, x, start_unix)

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
                index=i, phase="saccade", source=None, target=None, saccade_duration_ms=SACCADE_S * 1000, flash_shown=True, contrast=contrast,
                line_side=side, responded=True, answer_side=answer, response_time_ms=600.0, outcome="correct" if correct else "incorrect",
                reaction_latency_ms=LATENCY_S * 1000, flash_during_saccade=during, scheduled_delay_ms=round(1000 * (flash - cue), 1),
                flash_unix_ms=round(1000 * flash, 1), tracking_ok=True,
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


def write_old_column_run(
    data_dir: Path,
    key: int | str,
    trials: Path | list[dict],
    fieldnames: list[str] | None = None,
    participant: str = "p1",
    meta: dict | None = None,
    test_mode: bool = False,
    gaze_trace_rows: list[list] | None = None,
) -> Path:
    """A run folder whose trials.csv still uses the grating-era column names (trial_index, grating_shown, orientation, ...)
    and whose session.json is `{"run": {"unix_timestamp": key}, **meta}`. `trials`: a CSV file to copy, or dict rows to write
    with `fieldnames`. `gaze_trace_rows` (t_s, device_unix_s, x_px, worn) writes gaze_trace.csv. Keeps session keys small and explicit."""
    folder = session_files.create_session_folder(Path(data_dir), participant, int(key), test_mode)
    target = folder / session_files.TRIALS_FILE
    if isinstance(trials, Path):
        target.write_bytes(trials.read_bytes())
    else:
        with target.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames or sorted({k for row in trials for k in row}))
            w.writeheader()
            w.writerows(trials)
    (folder / session_files.SESSION_FILE).write_text(json.dumps({"run": {"unix_timestamp": int(key)}, **(meta or {})}))
    if gaze_trace_rows is not None:
        with (folder / session_files.GAZE_TRACE_FILE).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "device_unix_s", "x_px", "worn"])
            w.writerows(gaze_trace_rows)
    return folder


# -- return-jump runs ----------------------------------------------------------------------------------------------------
# Every trial starts at LO: out saccade LO -> HI at go cue + LATENCY_S, the target jumps back `jump_delay_ms` after the
# out arrival, the return saccade HI -> LO starts `return_latency_ms` (+- SD; may be negative = predictive) after the jump.
# overshoot=True: the out saccade lands OVERSHOOT past HI and a small correction (larger than analyze_session.MIN_AMPLITUDE,
# smaller than its return threshold) brings it back to HI before the return. One flash per trial, on the return leg with
# probability `return_fraction`: mid-saccade (`saccade`) or well away from both saccades (`fixation`).
ARRIVAL_AFTER_ONSET_S = 0.06  # out_arrival: the first sample inside the target zone
OVERSHOOT = 0.16  # correction amplitude (> MIN_AMPLITUDE 0.12, < 0.5 x the 0.46 out amplitude)
CORRECTION_DELAY_S, CORRECTION_S = 0.05, 0.04


def _trace(t: np.ndarray, moves: list[tuple[float, float, float]]) -> np.ndarray:
    """Gaze x from sorted (onset, duration, target) smoothstep movements, starting at LO."""
    x = np.full_like(t, LO)
    level = LO
    for onset, dur, target in moves:
        u = np.clip((t - onset) / dur, 0, 1)
        x = np.where(t >= onset, level + (target - level) * (3 * u**2 - 2 * u**3), x)
        level = target
    return x


def _return_jump_run(data_dir, participant, start_unix, n_trials, contrasts_pct, saccade_fraction, threshold_saccade, threshold_fixation,
                     slope, test_mode, geometry, seed, trigger, return_latency_ms, return_latency_sd_ms, return_fraction, overshoot, jump_delay_ms) -> Path:
    rng = np.random.default_rng(seed)
    geometry = {**DEFAULT_GEOMETRY, **(geometry or {})}
    folder = session_files.create_session_folder(Path(data_dir), participant, start_unix, test_mode)
    t0 = float(start_unix) + 2.0
    trials = []
    moves = []
    for i in range(n_trials):
        cue = t0 + TRIAL_SPACING_S * i
        out_on = cue + LATENCY_S
        out_off = out_on + SACCADE_S
        jump = out_on + ARRIVAL_AFTER_ONSET_S + jump_delay_ms / 1000
        earliest = out_off + (CORRECTION_DELAY_S + CORRECTION_S + 0.03 if overshoot else 0.03)  # the eye must stop before it turns back
        back_on = max(jump + rng.normal(return_latency_ms, return_latency_sd_ms) / 1000, earliest)
        moves.append((out_on, SACCADE_S, HI + OVERSHOOT if overshoot else HI))
        if overshoot:
            moves.append((out_off + CORRECTION_DELAY_S, CORRECTION_S, HI))
        moves.append((back_on, SACCADE_S, LO))
        trials.append((cue, out_on, jump, back_on))

    t = np.arange(float(start_unix), t0 + TRIAL_SPACING_S * n_trials + 1.0, 1 / RATE_HZ)
    x = _trace(t, moves)
    _write_trace(folder, t, x, start_unix)

    contrasts = [c / 100 for c in contrasts_pct]
    with (folder / session_files.TRIALS_FILE).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c.name for c in TRIAL_COLUMNS])
        w.writeheader()
        for i, (cue, out_on, jump, back_on) in enumerate(trials):
            contrast = contrasts[i % len(contrasts)]
            leg = "return" if rng.random() < return_fraction else "out"
            during = rng.random() < saccade_fraction
            if leg == "out":
                flash = out_on + rng.uniform(0.02, 0.06) if during else cue + rng.uniform(0.0, 0.06)
            else:
                flash = back_on + rng.uniform(0.02, 0.06) if during else back_on + SACCADE_S + rng.uniform(0.3, 0.5)
            p = float(_weibull(np.asarray(contrast), np.asarray(threshold_saccade if during else threshold_fixation), np.asarray(slope)))
            correct = rng.random() < p
            side = Side.ABOVE if i % 2 == 0 else Side.BELOW
            answer = side if correct else (Side.BELOW if side == Side.ABOVE else Side.ABOVE)
            result = TrialResult(
                index=i, phase="saccade", source=None, target=None, saccade_duration_ms=SACCADE_S * 1000, flash_shown=True, contrast=contrast,
                line_side=side, responded=True, answer_side=answer, response_time_ms=600.0, outcome="correct" if correct else "incorrect",
                reaction_latency_ms=LATENCY_S * 1000, flash_during_saccade=during, scheduled_delay_ms=round(1000 * (flash - cue), 1),
                flash_unix_ms=round(1000 * flash, 1), tracking_ok=True,
                flash_leg=leg, jump_back_ms_after_go_cue=round(1000 * (jump - cue), 1), return_reaction_ms=round(1000 * (back_on - jump), 1),
                jump_back_forced=False, out_arrival_ms_after_go_cue=round(1000 * (LATENCY_S + ARRIVAL_AFTER_ONSET_S), 1),
                left_target_before_jump=back_on < jump,
            )
            w.writerow({c.name: c.write(result) for c in TRIAL_COLUMNS})

    settings = {
        **geometry,
        "screen_height_deg": 29.0,
        "line_offset_deg": 2.0,
        "line_sd_deg": 0.15,
        "geometry_is_placeholder": geometry == DEFAULT_GEOMETRY,
        "saccade_contrasts_pct": list(contrasts_pct),
        "return_jump": True,
        "return_jump_trigger": trigger,
        "return_flash_spread_sd_ms": 40.0,
        "return_leg_flash_fraction": return_fraction,
    }
    meta = {
        "participant": {"id": participant},
        "run": {"mode": "test" if test_mode else "real", "unix_timestamp": int(start_unix), "folder": f"{folder.parent.name}/{folder.name}"},
        "settings": settings,
        "gaze_quality": {"clock_offset_ms": CLOCK_OFFSET_S * 1000, "sample_rate_hz": RATE_HZ, "latency_ms_median": 17.5, "longest_gap_ms": 12.0, "gaps_over_threshold": 0},
    }
    (folder / session_files.SESSION_FILE).write_text(json.dumps(meta, indent=2))
    return folder
