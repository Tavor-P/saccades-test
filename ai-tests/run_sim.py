"""One-command headless simulation runner. See README.md in this folder.

    python ai-tests/run_sim.py                                  # 30-trial session, default scenario
    python ai-tests/run_sim.py --timeline                       # one trial, event-by-event
    python ai-tests/run_sim.py --scenario dropouts --json
    python ai-tests/run_sim.py --set SACCADE_ONSET_STABILITY_MS=10
    python ai-tests/run_sim.py --sweep SACCADE_ONSET_STABILITY_MS=10,20,40 --seeds 5
    python ai-tests/run_sim.py --list
"""

import argparse
import json
import sys

import dataclasses

from sim_tools import SCENARIOS, coerce, parse_assignments, run, sweep  # also puts the project root on sys.path
from src.experiment.simulation import ParticipantProfile


def _print_table(rows: list[dict]) -> None:
    cols = list(rows[0])
    widths = [max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    p = argparse.ArgumentParser(description="Headless simulation of the saccade experiment (no window, no files).")
    p.add_argument("--scenario", default="default", help=f"one of {sorted(SCENARIOS)}")
    p.add_argument("--trials", type=int, default=30, help="saccade trials to run (session may end sooner)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", metavar="NAME=VALUE", help="override an experiment constant, e.g. SACCADE_ONSET_STABILITY_MS=10 (repeatable)")
    p.add_argument("--profile", action="append", metavar="FIELD=VALUE", help="override a participant field, e.g. reaction_ms_mean=300 (repeatable)")
    p.add_argument("--sweep", metavar="NAME=V1,V2,...", help="vary one constant over values; averaged over --seeds seeds")
    p.add_argument("--sweep-profile", metavar="FIELD=V1,V2,...", help="like --sweep, but for a participant field")
    p.add_argument("--seeds", type=int, default=5, help="seeds per value in a sweep")
    p.add_argument("--timeline", action="store_true", help="print the event timeline of a single trial (forces --trials 1)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--list", action="store_true", help="list scenarios, participant fields and what they mean")
    a = p.parse_args()

    if a.list:
        print("scenarios:", json.dumps(SCENARIOS, indent=1))
        print("participant fields (defaults):", json.dumps(dataclasses.asdict(ParticipantProfile()), indent=1))
        return

    constants, profile = parse_assignments(a.set), parse_assignments(a.profile)

    if a.sweep or a.sweep_profile:
        spec, target = (a.sweep, "constant") if a.sweep else (a.sweep_profile, "profile")
        name, _, values = spec.partition("=")
        parsed = [coerce(v) for v in values.split(",")]
        rows = sweep(name, parsed, a.scenario, a.trials, a.seeds, profile, constants, target)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            _print_table(rows)
        return

    trials = 1 if a.timeline else a.trials
    sim, summary = run(a.scenario, trials, a.seed, profile, constants)
    if a.timeline:
        from src.experiment.simulation import _print_trial_timeline

        _print_trial_timeline(sim)
        return
    if a.json:
        print(json.dumps(summary, indent=1))
    else:
        for k, v in summary.items():
            print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
