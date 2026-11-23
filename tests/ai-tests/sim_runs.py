"""Simulated sessions written out as run folders, so analyze_session can be checked against the simulator's ground truth.

`simulate_run()` runs the real ExperimentSession against the simulator (src/experiment/simulation.py), records every gaze
sample the simulated tracker produced and writes trials.csv / gaze_trace.csv / session.json into a folder under `data_dir`
(always a pytest tmp folder or a temp dir - never data/). The trace is stamped with CAPTURE time (sample time minus the
profile's tracker latency), as NeonGazeSource does, on the simulated wall clock (`SimulatedClock.unix_ms`: 1.7e9 s + sim time),
with a clock offset of 0. Analysis results are compared per leg with `flash_truth` by `agreement()`.
"""

import csv
import json
import re
from pathlib import Path

from include.experiment.constants import RETURN_JUMP_TRIGGER
from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.experiment import session_files
from src.experiment import simulation as sim
from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE
from src.experiment.logger import TRIAL_COLUMNS

SIM_UNIX_BASE_S = 1_700_000_000.0  # SimulatedClock.unix_ms() / 1000 at sim time 0
_RETURN_ONSET = re.compile(r"\[true\] (learned )?return saccade onset")


def simulate_run(data_dir: Path, profile: sim.ParticipantProfile, seed: int = 0, n_trials: int = 150, participant: str = "SIM"):
    """(SimulationResult, SessionRef) for one simulated return-jump session written as a run folder."""
    samples: list[tuple[float, float | None]] = []
    original_push = sim.SimulatedGazeSource.push

    def recording_push(self, ratio, t):
        samples.append((t, ratio))
        original_push(self, ratio, t)

    sim.SimulatedGazeSource.push = recording_push
    try:
        result = sim.simulate_session(max_trials=n_trials, profile=profile, seed=seed, test_mode=False, return_jump=True)
    finally:
        sim.SimulatedGazeSource.push = original_push

    folder = session_files.create_session_folder(Path(data_dir), participant, SIM_UNIX_BASE_S, False)
    latency_s = profile.tracker_latency_ms / 1000
    with (folder / session_files.GAZE_TRACE_FILE).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "device_unix_s", "x_px", "worn"])
        for t, ratio in samples:
            x = "" if ratio is None else f"{ratio * NEON_SCENE_WIDTH:.3f}"
            w.writerow([f"{t:.4f}", f"{SIM_UNIX_BASE_S + t - latency_s:.5f}", x, 1])
    with (folder / session_files.TRIALS_FILE).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c.name for c in TRIAL_COLUMNS])
        w.writeheader()
        for row in result.results:
            w.writerow({c.name: c.write(row) for c in TRIAL_COLUMNS})
    meta = {
        "participant": {"id": participant},
        "run": {"mode": "real", "unix_timestamp": int(SIM_UNIX_BASE_S), "folder": f"{folder.parent.name}/{folder.name}"},
        "settings": {"return_jump": True, "return_jump_trigger": RETURN_JUMP_TRIGGER},
        "gaze_quality": {"clock_offset_ms": 0.0},
    }
    (folder / session_files.SESSION_FILE).write_text(json.dumps(meta, indent=2))
    return result, session_files.find_session(Path(data_dir), f"{folder.parent.name}/{folder.name}")


def agreement(result, run) -> dict:
    """Per leg: how the gates match the simulator's truth (flash inside a simulated saccade's flight).
    `saccade_true` = share of gate-saccade flashes that were truly in flight; `fixation_true` = share of gate-fixation flashes that
    were truly not; `missed` = truly-in-flight flashes the gates excluded; `as_fixation` = truly-in-flight flashes gated fixation."""
    out = {}
    for leg in ("out", "return"):
        rows = [(g.condition, result.flash_truth[("saccade", p.flash.trial)]) for p, g in zip(run.placed, run.gates)
                if (p.flash.leg or "out") == leg and ("saccade", p.flash.trial) in result.flash_truth]
        sac = [truth for cond, truth in rows if cond == SACCADE]
        fix = [truth for cond, truth in rows if cond == FIXATION]
        out[leg] = {
            "n": len(rows),
            "n_saccade": len(sac),
            "n_fixation": len(fix),
            "n_excluded": sum(cond == EXCLUDED for cond, _ in rows),
            "n_true": sum(truth for _, truth in rows),
            "saccade_true": sum(sac) / len(sac) if sac else None,
            "fixation_true": sum(not v for v in fix) / len(fix) if fix else None,
            "missed": sum(truth for cond, truth in rows if cond == EXCLUDED),
            "as_fixation": sum(fix),
        }
    return out


def true_return_onsets(result) -> list[float]:
    """Simulated (true) return-saccade onsets, PC unix s, from the simulator's '[true] ... return saccade onset' events."""
    return [SIM_UNIX_BASE_S + t for t, text in result.events if _RETURN_ONSET.search(text)]


def true_return_latencies(result, run) -> list[tuple[float, float]]:
    """(true latency ms, trace latency ms) from the jump, for every trial whose return saccade the analysis found and whose window
    holds exactly one true return onset."""
    onsets = true_return_onsets(result)
    cues = sorted(f.go_cue for f in (p.flash for p in run.placed) if f.go_cue is not None)
    pairs = []
    for p, tr in zip(run.placed, run.trials):
        f = p.flash
        if f.jump is None or f.jump_forced or tr.back is None:
            continue
        end = next((c for c in cues if c > f.go_cue), float("inf"))
        inside = [o for o in onsets if f.go_cue <= o < end]
        if len(inside) == 1:
            pairs.append((1000 * (inside[0] - f.jump), 1000 * (tr.back.onset - f.jump)))
    return pairs
