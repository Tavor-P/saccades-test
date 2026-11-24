"""Saccades from the raw gaze trace, and every logged flash placed against them (part of the post-hoc analysis; see
analyze_session). Units: gaze x is a 0-1 ratio of the scene (or screen) width per `x_px / NEON_SCENE_WIDTH` (traces recorded
with the since-removed screen-mapped backends hold the screen ratio x NEON_SCENE_WIDTH in x_px, so old sessions analyse the same
way); speeds are ratio/s. Thresholds: include/experiment/analysis_constants.py."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from include.experiment.analysis_constants import MIN_AMPLITUDE, SPEED_HIGH, SPEED_LOW
from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src import textio
from src.experiment.logger import read_trial_rows


@dataclass(frozen=True)
class Saccade:
    onset: float  # PC unix seconds (phone capture time + clock offset)
    offset: float
    x0: float
    x1: float
    peak_speed: float

    @property
    def duration_ms(self) -> float:
        return 1000 * (self.offset - self.onset)

    @property
    def amplitude(self) -> float:
        return abs(self.x1 - self.x0)


@dataclass(frozen=True)
class Flash:
    trial: int
    time: float  # PC unix seconds the line was drawn (before any display-delay correction)
    contrast: float
    line_side: str
    outcome: str  # correct | incorrect | miss
    go_cue: float | None = None  # PC unix seconds the trial's target appeared (flash time minus its scheduled delay)
    leg: str | None = None  # return-jump runs: 'out' | 'return' (trials.csv flash_leg); None in older runs
    jump: float | None = None  # PC unix seconds the target jumped back (go cue + jump_back_ms_after_go_cue); None = no jump
    jump_forced: bool = False  # the jump came on the landing timeout (no out landing was seen live)
    return_reaction_ms: float | None = None  # the live classifier's signed return latency from the jump (cross-check only)

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"


@dataclass(frozen=True)
class Placed:
    """A flash placed against its nearest saccade."""

    flash: Flash
    rel_ms: float | None  # flash time minus the nearest onset (None when there is no saccade at all)
    inside: bool  # the flash fell between that saccade's onset and offset


# -- loading -------------------------------------------------------------------------------------


def load_trace(path: Path, clock_offset_s: float, offset_history: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(capture times in PC unix seconds, gaze x ratio), strictly increasing, without lost samples. With an `offset_history`
    of two or more {"phone_unix_s", "offset_ms"} entries (session.json gaze_quality.clock_offset_history) the offset is
    interpolated over the phone-clock time (the clocks drift); otherwise the single `clock_offset_s` applies throughout."""
    # x_px / NEON_SCENE_WIDTH also reads traces recorded before 2026-10-25 with the removed screen-mapped backends (x_px = screen ratio x NEON_SCENE_WIDTH).
    times, xs = [], []
    for row in textio.read_csv(path):
        if not row.get("x_px") or not row.get("device_unix_s"):
            continue
        times.append(float(row["device_unix_s"]))
        xs.append(float(row["x_px"]) / NEON_SCENE_WIDTH)
    t, x = np.asarray(times), np.asarray(xs)
    if offset_history is not None and len(offset_history) >= 2:
        history = sorted(offset_history, key=lambda h: h["phone_unix_s"])
        t = t + np.interp(t, [h["phone_unix_s"] for h in history], [h["offset_ms"] / 1000 for h in history])  # held constant beyond the ends
    else:
        t = t + clock_offset_s
    keep = np.concatenate(([True], np.diff(t) > 0)) if len(t) else np.asarray([], dtype=bool)
    ok = keep & ~np.isnan(x)
    return t[ok], x[ok]


def load_flashes(path: Path) -> list[Flash]:
    """Every shown saccade-phase line with a logged wall-clock time (reads both trials.csv and the old results CSV)."""
    flashes = []
    for row in read_trial_rows(Path(path)):
        if row.get("phase") != "saccade" or row.get("grating_shown") != "True":
            continue
        if not row.get("grating_shown_at_unix_ms") or not row.get("contrast"):
            continue
        shown = float(row["grating_shown_at_unix_ms"]) / 1000
        delay = row.get("scheduled_delay_ms")
        outcome = row["outcome"]
        if outcome == "timeout":
            # The live trial ended on its no-saccade timeout although the line was shown (and maybe answered): score the answer.
            if row.get("responded") != "True":
                outcome = "miss"
            else:
                outcome = "correct" if row.get("response_orientation") == row.get("orientation") else "incorrect"
        go_cue = shown - float(delay) / 1000 if delay else None
        jump_ms = row.get("jump_back_ms_after_go_cue")
        reaction = row.get("return_reaction_ms")
        flashes.append(
            Flash(
                trial=int(row["trial_index"]),
                time=shown,
                contrast=float(row["contrast"]),
                line_side=row.get("orientation", ""),
                outcome=outcome,
                go_cue=go_cue,
                leg=row.get("flash_leg") or None,
                jump=go_cue + float(jump_ms) / 1000 if jump_ms and go_cue is not None else None,
                jump_forced=row.get("jump_back_forced") == "True",
                return_reaction_ms=float(reaction) if reaction else None,
            )
        )
    return flashes


# -- saccade detection and placement -------------------------------------------------------------


def detect_saccades(
    t: np.ndarray,
    x: np.ndarray,
    speed_high: float = SPEED_HIGH,
    speed_low: float = SPEED_LOW,
    min_amplitude: float = MIN_AMPLITUDE,
) -> list[Saccade]:
    if len(t) < 5:
        return []
    smooth = np.convolve(x, np.ones(3) / 3, mode="same")
    speed = np.abs(np.gradient(smooth, t))
    found, i = [], 0
    while i < len(speed):
        if speed[i] > speed_high:
            a = i
            while a > 0 and speed[a - 1] > speed_low:
                a -= 1
            b = i
            while b + 1 < len(speed) and speed[b + 1] > speed_low:
                b += 1
            if abs(x[b] - x[a]) >= min_amplitude:
                found.append(Saccade(float(t[a]), float(t[b]), float(x[a]), float(x[b]), float(speed[a : b + 1].max())))
            i = b + 1
        else:
            i += 1
    return found


def place_flashes(flashes: list[Flash], saccades: list[Saccade], flash_delay_ms: float = 0.0) -> list[Placed]:
    """Each flash against the saccade whose onset is nearest in time. `flash_delay_ms` shifts the flash later
    (the time from drawing it to light leaving the screen: up to one frame plus panel lag)."""
    if not saccades:
        return [Placed(f, None, False) for f in flashes]
    onsets = np.asarray([s.onset for s in saccades])
    placed = []
    for f in flashes:
        when = f.time + flash_delay_ms / 1000
        k = int(np.argmin(np.abs(onsets - when)))
        s = saccades[k]
        placed.append(Placed(f, 1000 * (when - s.onset), s.onset <= when <= s.offset))
    return placed
