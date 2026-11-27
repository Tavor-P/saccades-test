"""Return-jump runs: each trial's out and return saccade, every flash placed against and gated by its OWN leg's saccade,
the suppression index per leg, and the return-latency statistics (part of the post-hoc analysis; see analyze_session).
Thresholds: include/experiment/analysis_constants.py."""

import math
from dataclasses import dataclass

import numpy as np

from include.experiment.analysis_constants import MIN_FLASH_SPEED, OUT_SEARCH_S, RETURN_MIN_AMPLITUDE_REL, RETURN_SEARCH_AFTER_JUMP_S
from include.experiment.constants import MAD_TO_SD_FACTOR
from src.experiment.flash_gates import FIXATION, SACCADE, classify_leg_flash
from src.experiment.saccade_detection import Flash, Placed, Saccade
from src.experiment.suppression_stats import suppression_index
from src.timebase import ms_to_s, s_to_ms

# -- return-jump runs: the two saccades of each trial ----------------------------------------------
# A trial whose row logs a jump-back has an out saccade (go cue -> target) and a return saccade (back to the start). Each
# flash is placed against and gated by the saccade of its own leg (flash_gates.classify_leg_flash). Runs with no jump
# logged in any row (older runs, or the jump setting off) never reach this code.

LEGS = ("out", "return")


@dataclass(frozen=True)
class TrialSaccades:
    """The saccades of one return-jump trial (None = not found)."""

    out: Saccade | None
    back: Saccade | None

    def of(self, leg: str | None) -> Saccade | None:
        return self.back if leg == "return" else self.out


def is_return_jump(flashes: list[Flash]) -> bool:
    """True when any row logged a jump-back: the run used the return-jump trials."""
    return any(f.jump is not None for f in flashes)


def find_trial_saccades(flashes: list[Flash], saccades: list[Saccade]) -> list[TrialSaccades]:
    """Per flash (one per trial): its trial's out and return saccade. The search for a trial ends at the next trial's go cue."""
    cues = sorted(f.go_cue for f in flashes if f.go_cue is not None)
    found = []
    for f in flashes:
        if f.go_cue is None:
            found.append(TrialSaccades(None, None))
            continue
        later = [c for c in cues if c > f.go_cue]
        end = later[0] if later else math.inf
        lo, hi = f.go_cue + OUT_SEARCH_S[0], min(f.go_cue + OUT_SEARCH_S[1], end)
        k = next((i for i, s in enumerate(saccades) if lo <= s.onset <= hi), None)
        if k is None:
            found.append(TrialSaccades(None, None))
            continue
        out = saccades[k]
        back = None
        if f.jump is not None:
            direction = math.copysign(1.0, out.x1 - out.x0)
            last = min(f.jump + RETURN_SEARCH_AFTER_JUMP_S, end)
            for s in saccades[k + 1 :]:
                if s.onset > last:
                    break
                if -direction * (s.x1 - s.x0) >= RETURN_MIN_AMPLITUDE_REL * out.amplitude:
                    back = s
                    break
        found.append(TrialSaccades(out, back))
    return found


def place_leg_flashes(flashes: list[Flash], trials: list[TrialSaccades], flash_delay_ms: float = 0.0) -> list[Placed]:
    """Each flash against its own leg's saccade (rel_ms None when that saccade was not found). A flash without a leg is an out flash."""
    placed = []
    for f, tr in zip(flashes, trials):
        s = tr.of(f.leg)
        when = f.time + ms_to_s(flash_delay_ms)
        placed.append(Placed(f, None, False) if s is None else Placed(f, s_to_ms(when - s.onset), s.onset <= when <= s.offset))
    return placed


def classify_leg_flashes(  # noqa: PLR0913 - classify_leg_flash's inputs for every flash
    placed: list[Placed],
    trials: list[TrialSaccades],
    saccades: list[Saccade],
    t: np.ndarray,
    x: np.ndarray,
    flash_delay_ms: float = 0.0,
    min_speed: float = MIN_FLASH_SPEED,
) -> list:
    """flash_gates.classify_leg_flash for every flash; the amplitude gate uses each leg's own run median."""
    medians = {}
    for leg in LEGS:
        amps = [s.amplitude for tr in trials if (s := tr.of(leg)) is not None]
        medians[leg] = float(np.median(amps)) if amps else None
    out = []
    for p, tr in zip(placed, trials):
        leg = "return" if p.flash.leg == "return" else "out"
        anchor = p.flash.jump if leg == "return" else p.flash.go_cue
        out.append(classify_leg_flash(p.flash.time + ms_to_s(flash_delay_ms), saccades, t, x, leg, tr.of(leg), anchor, medians[leg], p.flash.jump_forced, min_speed))
    return out


def leg_subset(placed: list[Placed], gates: list, leg: str) -> tuple[list[Placed], list]:
    """This leg's saccade flashes plus EVERY fixation flash (the fixation control is shared by both legs: the eye was still,
    whichever leg the flash had been timed for)."""
    keep = [i for i, (p, g) in enumerate(zip(placed, gates)) if g.condition == FIXATION or (g.condition == SACCADE and (p.flash.leg or "out") == leg)]
    return [placed[i] for i in keep], [gates[i] for i in keep]


def suppression_by_leg(placed: list[Placed], gates: list, **kwargs) -> dict:
    """{'out', 'return', 'pooled'}: suppression_index per leg (that leg's saccade flashes vs all fixation flashes) and pooled."""
    result = {}
    for leg in LEGS:
        sub_placed, sub_gates = leg_subset(placed, gates, leg)
        result[leg] = suppression_index(sub_placed, gates=sub_gates, **kwargs)
    result["pooled"] = suppression_index(placed, gates=gates, **kwargs)
    return result


def _robust_sd(values: np.ndarray) -> float:
    return float(MAD_TO_SD_FACTOR * np.median(np.abs(values - np.median(values))))


def return_latency_stats(flashes: list[Flash], trials: list[TrialSaccades], settings: dict | None = None) -> dict:
    """Return-saccade latency from the jump-back frame, from the raw trace, over every trial that jumped (whichever leg its flash
    was on), forced jumps left out. Negative = the eye turned back before the jump was drawn (predictive). The logged live
    `return_reaction_ms` is summarised alongside as a cross-check. Caveat: the jump time is go cue + logged offset, and the go cue
    is the flash time minus its SCHEDULED delay, so the anchor can sit up to one frame late (latencies read up to ~17 ms short)."""
    jumped = [(f, tr, f.jump) for f, tr in zip(flashes, trials) if f.jump is not None]
    usable = [(f, tr, jump) for f, tr, jump in jumped if not f.jump_forced]
    latencies_ms = np.asarray([s_to_ms(tr.back.onset - jump) for f, tr, jump in usable if tr.back is not None])
    live = np.asarray([f.return_reaction_ms for f, _, _ in usable if f.return_reaction_ms is not None])
    s = settings or {}
    return {
        "trials_jumped": len(jumped),
        "forced_jumps": len(jumped) - len(usable),
        "no_return_found": sum(tr.back is None for _, tr, _ in usable),
        "n": int(len(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)) if len(latencies_ms) else None,
        "sd_ms": float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) >= 2 else None,
        "robust_sd_ms": _robust_sd(latencies_ms) if len(latencies_ms) >= 2 else None,
        "share_predictive": float(np.mean(latencies_ms < 0)) if len(latencies_ms) else None,
        "live_n": int(len(live)),
        "live_median_ms": float(np.median(live)) if len(live) else None,
        "live_sd_ms": float(np.std(live, ddof=1)) if len(live) >= 2 else None,
        "return_flash_spread_sd_ms": s.get("return_flash_spread_sd_ms"),
        "return_jump_trigger": s.get("return_jump_trigger"),
    }
