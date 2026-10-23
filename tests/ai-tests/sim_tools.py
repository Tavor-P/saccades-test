"""Helpers for running the headless experiment simulation (src/experiment/simulation.py)
the way an AI agent wants to: named scenarios, one-line constant overrides,
sweeps over a parameter, and machine-readable summaries. See README.md here.

Nothing in this module opens a window, touches a camera, or writes to data/.
"""

import contextlib
import dataclasses
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, so `src`/`include` import

from src.experiment.scoring import is_valid_for_saccadic_analysis  # noqa: E402
from src.experiment.simulation import ParticipantProfile, SimulationResult, simulate_session  # noqa: E402

# Named participant/tracker situations. Values are ParticipantProfile overrides.
SCENARIOS: dict[str, dict] = {
    "default": {},
    "fast_reactor": {"reaction_ms_mean": 170, "reaction_ms_sd": 20},
    "slow_reactor": {"reaction_ms_mean": 320, "reaction_ms_sd": 50},
    "variable_reactor": {"reaction_ms_mean": 230, "reaction_ms_sd": 80},
    "noisy_tracker": {"gaze_noise_ratio": 0.01},
    "high_latency": {"tracker_latency_ms": 100.0},
    "dropouts": {"dropout_rate_per_s": 1.0},
    "no_suppression": {"suppressed_threshold_contrast": 0.05},  # control: flash is equally visible in the saccade
}


def make_profile(scenario: str = "default", **overrides) -> ParticipantProfile:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    fields = {f.name for f in dataclasses.fields(ParticipantProfile)}
    merged = {**SCENARIOS[scenario], **overrides}
    unknown = set(merged) - fields
    if unknown:
        raise KeyError(f"unknown profile field(s) {sorted(unknown)}; valid: {sorted(fields)}")
    return ParticipantProfile(**merged)


def coerce(text: str):
    """'40' -> 40, '0.5' -> 0.5, 'true' -> True, otherwise the string."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_assignments(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        out[name.strip()] = coerce(value)
    return out


def parse_sweep(spec: str) -> tuple[str, list]:
    """'NAME=a,b,c' -> ('NAME', [a, b, c]) with each value coerced like --set values."""
    name, sep, values = spec.partition("=")
    if not sep or not name.strip():
        raise ValueError(f"expected NAME=V1,V2,..., got {spec!r}")
    return name.strip(), [coerce(v) for v in values.split(",")]


@contextlib.contextmanager
def override_constants(values: dict):
    """Temporarily sets module-level constants (e.g. SACCADE_ONSET_STABILITY_MS=10)
    in every already-imported include./src. module that defines them - needed
    because `from include... import NAME` copies the value into each importing
    module. Raises if a name exists nowhere (typo guard); restores on exit."""
    import src.experiment.session  # noqa: F401  (make sure the modules that bind constants are loaded)
    import src.experiment.trial_mechanics  # noqa: F401

    saved: list[tuple[object, str, object]] = []
    try:
        for name, value in values.items():
            hits = [
                m
                for modname, m in list(sys.modules.items())
                if m is not None and modname.startswith(("include.", "src.")) and hasattr(m, name)
            ]
            if not hits:
                raise KeyError(f"no constant named {name!r} in include/ or src/")
            for module in hits:
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, value)
        yield
    finally:
        for module, name, old in reversed(saved):
            setattr(module, name, old)


def _mean(xs, ndigits: int = 1):
    """Mean of the non-None values, rounded; None if there are none."""
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), ndigits) if xs else None


def summarize(sim: SimulationResult) -> dict:
    """One flat, JSON-friendly dict of what matters about a run."""
    rows = sim.saccade_results
    flashes = [r for r in rows if r.grating_shown]
    catches = [r for r in rows if not r.grating_shown]
    valid = [r for r in flashes if is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok)]
    valid_indices = {r.index for r in valid}
    truly_in = [r for r in flashes if sim.flash_truth.get((r.phase, r.index)) is True]
    truly_in_indices = {r.index for r in truly_in}
    return {
        "trials": len(rows),
        "session_completed": sim.session.render_state()["hud"]["phase"] == "COMPLETE",
        "simulated_seconds": round(sim.duration_s, 1),
        "outcomes": {o: sum(r.outcome == o for r in rows) for o in sorted({r.outcome for r in rows})},
        "flashes": len(flashes),
        "valid_flashes": len(valid),
        "valid_flash_rate": round(len(valid) / len(flashes), 3) if flashes else None,
        "flash_during_saccade": {
            "true": sum(r.flash_during_saccade is True for r in flashes),
            "false": sum(r.flash_during_saccade is False for r in flashes),
            "unknown": sum(r.flash_during_saccade is None for r in flashes),
        },
        "tracking_not_ok": sum(r.tracking_ok is False for r in flashes),
        "timeouts": sum(r.outcome == "timeout" for r in rows),
        "false_alarm_rate": round(sum(r.outcome == "false_alarm" for r in catches) / len(catches), 3) if catches else None,
        "mean_reaction_latency_ms": _mean([r.reaction_latency_ms for r in rows]),
        "mean_onset_detection_lag_ms": _mean([r.onset_detection_lag_ms for r in rows]),
        "zest_threshold_estimate": round(sim.session._zest.threshold_estimate, 4),
        # Ground truth (only the simulator knows): of the flashes the experiment
        # calls valid, how many REALLY hit the simulated saccade (precision), and
        # of the flashes that really hit it, how many the experiment kept (recall).
        "valid_precision": round(sum(r.index in truly_in_indices for r in valid) / len(valid), 3) if valid else None,
        "valid_recall": round(sum(r.index in valid_indices for r in truly_in) / len(truly_in), 3) if truly_in else None,
        "flashes_truly_in_saccade": len(truly_in),
    }


def simulate(scenario: str = "default", trials: int = 30, seed: int = 0, profile: dict | None = None, constants: dict | None = None) -> SimulationResult:
    """Run one simulated session (no summary) with an optional temporary constant override."""
    with override_constants(constants or {}):
        return simulate_session(max_trials=trials, profile=make_profile(scenario, **(profile or {})), seed=seed)


def run(scenario: str = "default", trials: int = 30, seed: int = 0, profile: dict | None = None, constants: dict | None = None):
    """Run one simulated session -> (SimulationResult, summary dict)."""
    sim = simulate(scenario, trials, seed, profile, constants)
    return sim, summarize(sim)


SWEEP_METRICS = (
    "valid_flash_rate", "valid_precision", "valid_recall", "mean_reaction_latency_ms",
    "mean_onset_detection_lag_ms", "false_alarm_rate", "zest_threshold_estimate", "tracking_not_ok", "timeouts",
)


def sweep(name: str, values: list, scenario: str = "default", trials: int = 30, seeds: int = 5, profile: dict | None = None, constants: dict | None = None, target: str = "constant") -> list[dict]:
    """Vary one constant (target='constant') or profile field (target='profile')
    over `values`, `seeds` seeds each; returns one averaged row per value."""
    rows = []
    for value in values:
        per_seed = []
        for seed in range(seeds):
            c, p = dict(constants or {}), dict(profile or {})
            (c if target == "constant" else p)[name] = value
            per_seed.append(run(scenario, trials, seed, p, c)[1])
        rows.append({name: value, "seeds": seeds, **{k: _mean([s[k] for s in per_seed], 3) for k in SWEEP_METRICS}})
    return rows
