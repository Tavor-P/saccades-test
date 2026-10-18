"""Randomized bug-hunting campaign over the headless simulator.

    python tests/ai-tests/find_bugs.py                    # 40 random sessions
    python tests/ai-tests/find_bugs.py --sessions 300 --seed 7
    python tests/ai-tests/find_bugs.py --extreme          # only hostile participants/trackers

Each session draws a random ParticipantProfile (reaction time, noise, latency,
dropouts, response behaviour ...) and, optionally, random experiment-constant
overrides, runs the REAL ExperimentSession, then checks invariants.py and watches
for exceptions and hangs. Every finding prints a one-line command that reproduces
it, so a failure can be handed straight to `run_sim.py --timeline`.
Exit status is 1 if anything was found.
"""

import argparse
import dataclasses
import json
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # project root

from invariants import check_session  # noqa: E402
from sim_tools import override_constants  # noqa: E402
from src.experiment.simulation import ParticipantProfile, simulate_session  # noqa: E402

# (low, high) ranges for random profiles - wide on purpose, bugs hide at the edges.
PROFILE_RANGES = {
    "reaction_ms_mean": (120, 700),
    "reaction_ms_sd": (1, 150),
    "saccade_ms_min": (20, 60),
    "saccade_ms_max": (60, 140),
    "response_ms_mean": (150, 1100),  # beyond the response window = late responses
    "response_ms_sd": (5, 300),
    "threshold_contrast": (0.01, 0.4),
    "suppressed_threshold_contrast": (0.02, 0.6),
    "slope": (1.5, 14),
    "guess_response_probability": (0.0, 1.0),
    "gaze_noise_ratio": (0.0, 0.03),
    "tracker_latency_ms": (0, 150),
    "dropout_rate_per_s": (0.0, 4.0),
    "dropout_ms": (20, 400),
    "gaze_offset_ratio": (-0.05, 0.05),
}

# Hostile one-off situations that have to terminate cleanly.
EXTREME_PROFILES = {
    "never_responds": {"guess_response_probability": 0.0, "threshold_contrast": 1e6, "suppressed_threshold_contrast": 1e6},
    "always_sees": {"threshold_contrast": 1e-6, "suppressed_threshold_contrast": 1e-6},
    "responds_after_window": {"response_ms_mean": 1500, "response_ms_sd": 50},
    "very_slow_saccade_start": {"reaction_ms_mean": 2500, "reaction_ms_sd": 100},
    "tracker_dies_often": {"dropout_rate_per_s": 3.0, "dropout_ms": 250},
    "huge_latency": {"tracker_latency_ms": 400.0},
    "zero_noise": {"gaze_noise_ratio": 0.0},
    "very_noisy": {"gaze_noise_ratio": 0.05},
    "instant_saccades": {"saccade_ms_min": 8, "saccade_ms_max": 12},
}


def random_profile(rng: random.Random) -> dict:
    return {k: round(rng.uniform(lo, hi), 4) for k, (lo, hi) in PROFILE_RANGES.items() if rng.random() < 0.5}


def run_one(profile: dict, seed: int, trials: int, constants: dict | None = None) -> tuple[list[str], float]:
    """-> (findings, seconds). Findings include crashes and hangs as strings."""
    t0 = time.time()
    try:
        with override_constants(constants or {}):
            sim = simulate_session(max_trials=trials, profile=ParticipantProfile(**profile), seed=seed)
            return check_session(sim), time.time() - t0
    except Exception:  # noqa: BLE001 - the whole point is to catch anything
        return [f"EXCEPTION: {traceback.format_exc(limit=3).strip().splitlines()[-1]}"], time.time() - t0


def repro_command(profile: dict, seed: int, trials: int, constants: dict | None = None) -> str:
    parts = [f"python tests/ai-tests/run_sim.py --seed {seed} --trials {trials}"]
    parts += [f"--profile {k}={v}" for k, v in profile.items()]
    parts += [f"--set {k}={v}" for k, v in (constants or {}).items()]
    return " ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sessions", type=int, default=40)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--seed", type=int, default=0, help="campaign seed (sessions use seed, seed+1, ...)")
    p.add_argument("--extreme", action="store_true", help="only the hostile named profiles")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    rng = random.Random(a.seed)
    cases = [(name, prof) for name, prof in EXTREME_PROFILES.items()]
    if not a.extreme:
        cases += [(f"random_{i}", random_profile(rng)) for i in range(a.sessions)]

    findings = []
    for i, (name, profile) in enumerate(cases):
        seed = a.seed + i
        problems, secs = run_one(profile, seed, a.trials)
        status = "ok" if not problems else f"{len(problems)} PROBLEM(S)"
        if not a.json:
            print(f"[{i + 1}/{len(cases)}] {name:28s} {secs:5.1f}s  {status}", flush=True)
        for msg in problems:
            findings.append({"case": name, "seed": seed, "problem": msg, "repro": repro_command(profile, seed, a.trials)})

    if a.json:
        print(json.dumps(findings, indent=1))
    else:
        print(f"\n{len(cases)} sessions, {len(findings)} finding(s)")
        seen = set()
        for f in findings:
            key = (f["case"], f["problem"].split(":")[0])
            if key in seen:
                continue
            seen.add(key)
            print(f"\n- {f['case']}: {f['problem']}\n  repro: {f['repro']}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
