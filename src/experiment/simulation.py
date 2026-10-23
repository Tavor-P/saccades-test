"""Headless simulation of the saccade experiment - no PsychoPy window, no camera,
no files, no real time. A simulated participant, who sees only what
`ExperimentSession.render_state()` would draw, drives the *real*
ExperimentSession through calibration, the reaction-time test and saccade
trials, on a simulated clock and a simulated gaze tracker.

    python -m src.experiment.simulation            # one trial, with an event timeline
    python -m src.experiment.simulation --trials 40 --seed 3

Useful for exercising session logic (onset detection, flash timing,
validity gates, ZEST) and for tuning constants like SACCADE_ONSET_STABILITY_MS
or TIMING_OFFSETS_MS against a participant with known behaviour. The
participant model is deliberately simple (see ParticipantProfile); it is a
sandbox for the experiment's plumbing, not a model of real saccadic
suppression.
"""

import argparse
import math
import random
import statistics
from collections import deque
from dataclasses import dataclass, field

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from include.experiment.types import Orientation, TrialResult
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import is_valid_for_saccadic_analysis
from src.experiment.session import ExperimentSession
from src.eye_tracking.gaze_mapping import GazeMapper

GAZE_HZ = 240  # simulated tracker rate
FRAME_HZ = 60  # simulated display/tick rate (ExperimentSession.tick() runs once per frame)
_STEPS_PER_FRAME = GAZE_HZ // FRAME_HZ
_MAX_SIM_SECONDS = 3600.0


@dataclass
class ParticipantProfile:
    """Behaviour of the simulated participant. Defaults roughly match a
    typical observer; every value is a plain knob."""

    reaction_ms_mean: float = 230.0  # go-cue to saccade start
    reaction_ms_sd: float = 35.0
    saccade_ms_min: float = 45.0
    saccade_ms_max: float = 60.0
    response_ms_mean: float = 450.0  # flash to key press
    response_ms_sd: float = 90.0
    # Psychometric function: P(see flash) is logistic in log10(contrast),
    # centred on the 50% threshold below. Flashes during the saccade use the
    # (higher) suppressed threshold - that is saccadic suppression.
    threshold_contrast: float = 0.05
    suppressed_threshold_contrast: float = 0.20
    slope: float = 6.0
    guess_response_probability: float = 0.5  # when the flash was not seen, chance of guessing instead of "miss"
    gaze_noise_ratio: float = 0.002  # sd of tracker noise on the raw gaze ratio (~0.2 deg)
    tracker_latency_ms: float = 16.0  # eye movement to sample availability (Neon: ~14-16 ms)
    dropout_rate_per_s: float = 0.0  # tracking losses (blinks etc.) per second
    dropout_ms: float = 80.0
    gaze_offset_ratio: float = 0.01  # constant offset of this participant's gaze readings


class MemoryLogger:
    """Stands in for ResultLogger: keeps rows in memory, touches no files."""

    def __init__(self) -> None:
        self.rows: list[TrialResult] = []

    def log(self, result: TrialResult) -> None:
        self.rows.append(result)


class SimulatedClock(PausableClock):
    """A clock advanced explicitly by the simulator - never reads real time."""

    def __init__(self) -> None:
        super().__init__()
        self._t = 0.0

    def advance(self, seconds: float) -> None:
        self._t += seconds

    def now(self) -> float:
        return self._t

    def from_perf_counter(self, timestamp: float) -> float:
        return timestamp  # simulated gaze samples are already stamped in this clock's time

    def unix_ms(self) -> float:
        return 1_700_000_000_000.0 + self._t * 1000  # simulated wall clock, never the real one


class SimulatedGazeSource(GazeSource):
    """Gaze source fed by the simulator; uses the real GazeMapper so
    calibration, zone classification and smoothing match the live Neon path."""

    def __init__(self) -> None:
        self._mapper = GazeMapper()
        self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=0.0)
        self._history: deque[GazeSample] = deque(maxlen=GAZE_HZ)  # ~1 s of samples, like a real source's buffer
        self.gaze_latency_ms: float | None = None  # what a real source (Neon) measures; set from the profile by the participant

    def push(self, ratio: float | None, t: float) -> None:
        if ratio is None:
            self._latest = GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=t)
        else:
            self._latest = self._mapper.map(ratio, t)
        self._history.append(self._latest)

    def samples_since(self, timestamp: float | None) -> list[GazeSample]:
        """Samples newer than `timestamp` (all buffered ones for None), oldest first."""
        return [s for s in self._history if timestamp is None or s.timestamp > timestamp]

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return True

    def latest_sample(self) -> GazeSample:
        return self._latest

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._mapper.calibrate(left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        return self._mapper.average_recent_ratio()

    def begin_calibration_sample(self) -> None:
        self._mapper.reset_ratio_history()


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class SimulatedParticipant:
    """Looks at the symbols the screen shows, saccades to a newly appearing
    target after a reaction time, and reports the flash with a psychometric
    probability that is lower during the saccade. Only reads render_state()."""

    def __init__(self, session: ExperimentSession, gaze: SimulatedGazeSource, profile: ParticipantProfile, rng: random.Random, events: list) -> None:
        self._session = session
        self._gaze = gaze
        self._p = profile
        gaze.gaze_latency_ms = profile.tracker_latency_ms
        self._rng = rng
        self._events = events
        self._phase: str | None = None
        self._eye_from = self._eye_to = 0.5  # screen x (0-1)
        self._move_start = 0.0
        self._move_dur = 0.05
        self._dropout_until = -1.0
        self._reset_phase_state()

    def _reset_phase_state(self) -> None:
        """State that belongs to one screen phase - cleared whenever a new phase starts."""
        self.saccade_onset: float | None = None
        self._pending_space_at: float | None = None  # re-armed for the phases where SPACE is used
        self._pending_response: tuple[float, Orientation] | None = None
        self._flash_seen = False
        # Ground truth for the flash currently in flight (None if none): True if it
        # really fell inside the participant's saccade. Collected by the simulator
        # and attached to the trial's logged row.
        self.pending_flash_truth: bool | None = None

    # -- eye ------------------------------------------------------------------

    def _eye_x(self, t: float) -> float:
        u = (t - self._move_start) / self._move_dur
        return self._eye_from + (self._eye_to - self._eye_from) * _smoothstep(u)

    def _fixate(self, x: float, t: float) -> None:
        self._eye_from = self._eye_to = x
        self._move_start = t

    def feed_gaze(self, t: float) -> None:
        """One tracker sample, taken from where the eye was tracker_latency ago."""
        if t < self._dropout_until:
            self._gaze.push(None, t)
            return
        if self._p.dropout_rate_per_s and self._rng.random() < self._p.dropout_rate_per_s / GAZE_HZ:
            self._dropout_until = t + self._p.dropout_ms / 1000
            self._gaze.push(None, t)
            return
        x = self._eye_x(t - self._p.tracker_latency_ms / 1000)
        ratio = 0.5 + (x - 0.5) * 0.5 + self._p.gaze_offset_ratio + self._rng.gauss(0, self._p.gaze_noise_ratio)
        self._gaze.push(ratio, t)

    # -- seeing the screen ----------------------------------------------------------

    @staticmethod
    def _visible_x(state: dict) -> float | None:
        for key in ("dot", "cross", "calibration_center"):
            if state[key]["visible"]:
                return state[key]["x"]
        return None

    def _log(self, t: float, text: str) -> None:
        self._events.append((t, text))

    def observe(self, state: dict, t: float) -> None:
        phase = state["hud"]["phase"]
        if phase != self._phase:
            self._on_phase_entry(phase, state, t)
            self._phase = phase

        if self._pending_space_at is not None and t >= self._pending_space_at:
            # A real participant presses SPACE again if nothing happened (calibration
            # ignores it until the gaze window has filled), so keep retrying until the
            # phase changes - which resets this in _on_phase_entry.
            self._pending_space_at = t + 0.7
            self._session.on_space()

        if phase == "TRIAL_ACTIVE" and state["grating"]["visible"] and not self._flash_seen:
            self._see_flash(state, t)

        if self._pending_response is not None and t >= self._pending_response[0]:
            _, orientation = self._pending_response
            self._pending_response = None
            self._log(t, f"key press: {orientation.value}")
            self._session.on_response_key(orientation)

    def _on_phase_entry(self, phase: str, state: dict, t: float) -> None:
        self._log(t, f"phase -> {phase}")
        self._reset_phase_state()
        x = self._visible_x(state)
        if phase in ("WAITING_TO_START", "CALIBRATE_LEFT", "CALIBRATE_CENTER", "CALIBRATE_RIGHT"):
            if x is not None:
                self._fixate(x, t)
            # Hold the fixation long enough for the calibration window to fill.
            self._pending_space_at = t + self._rng.uniform(0.5, 0.8)
        elif phase in ("RT_TEST_FOREPERIOD", "FOREPERIOD"):
            if x is not None:
                self._fixate(x, t)
        elif phase in ("RT_TEST_ACTIVE", "TRIAL_ACTIVE"):
            # The go-cue: the other symbol just appeared (the source fades out).
            if x is not None:
                reaction = max(0.12, self._rng.gauss(self._p.reaction_ms_mean, self._p.reaction_ms_sd)) / 1000
                dur = self._rng.uniform(self._p.saccade_ms_min, self._p.saccade_ms_max) / 1000
                self._eye_from = self._eye_x(t)
                self._eye_to = x
                self._move_start = t + reaction
                self._move_dur = dur
                self.saccade_onset = t + reaction
                self._log(t, "go-cue: target appeared")
                self._log(t + reaction, f"[true] saccade onset (reaction {reaction * 1000:.0f} ms, flight {dur * 1000:.0f} ms)")

    def _see_flash(self, state: dict, t: float) -> None:
        self._flash_seen = True
        contrast = state["grating"]["contrast"]
        orientation = state["grating"]["orientation"]
        onset = self.saccade_onset
        suppressed = onset is not None and onset <= t <= onset + self._move_dur
        self.pending_flash_truth = suppressed
        threshold = self._p.suppressed_threshold_contrast if suppressed else self._p.threshold_contrast
        p_seen = 1 / (1 + math.exp(-self._p.slope * (math.log10(max(contrast, 1e-6)) - math.log10(threshold))))
        seen = self._rng.random() < p_seen
        where = "during saccade (suppressed)" if suppressed else "outside saccade"
        self._log(t, f"flash shown: contrast {contrast:.3f}, {where}; P(seen)={p_seen:.2f} -> {'seen' if seen else 'not seen'}")
        if seen:
            answer = orientation
        elif self._rng.random() < self._p.guess_response_probability:
            answer = self._rng.choice([Orientation.VERTICAL, Orientation.HORIZONTAL])
        else:
            return  # no response -> "miss"
        delay = max(0.15, self._rng.gauss(self._p.response_ms_mean, self._p.response_ms_sd)) / 1000
        self._pending_response = (t + delay, answer)


@dataclass
class SimulationResult:
    session: ExperimentSession
    results: list[TrialResult]  # every logged row (rt_test + saccade)
    events: list[tuple[float, str]] = field(default_factory=list)  # (sim time s, description), chronological
    # (phase, trial index) -> whether that trial's flash REALLY fell inside the
    # simulated saccade (ground truth the real experiment can't know). Only for
    # logged flash trials.
    flash_truth: dict[tuple[str, int], bool] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def saccade_results(self) -> list[TrialResult]:
        return [r for r in self.results if r.phase == "saccade"]


def simulate_session(
    max_trials: int = 1,
    profile: ParticipantProfile | None = None,
    seed: int = 0,
    test_mode: bool = True,
    skip_practice_trials: bool = True,
) -> SimulationResult:
    """Runs the real ExperimentSession end-to-end against a simulated
    participant until `max_trials` saccade-phase trials have been logged (or
    the session completes on its own). Deterministic for a given seed. The
    global `random` state is saved and restored, since the session draws
    foreperiods and trial sequences from it."""
    profile = profile or ParticipantProfile()
    saved_state = random.getstate()
    random.seed(seed)
    try:
        clock = SimulatedClock()
        gaze = SimulatedGazeSource()
        logger = MemoryLogger()
        events: list[tuple[float, str]] = []
        session = ExperimentSession(
            gaze, logger=logger, clock=clock, test_mode=test_mode, skip_practice_trials=skip_practice_trials
        )
        participant = SimulatedParticipant(session, gaze, profile, random.Random(seed), events)

        step = 0
        last_zone = None
        logged = 0
        saccade_rows_logged = 0
        flash_truth: dict[tuple[str, int], bool] = {}
        while True:
            clock.advance(1 / GAZE_HZ)
            t = clock.now()
            if t > _MAX_SIM_SECONDS:
                raise RuntimeError("simulation did not finish within a simulated hour")
            participant.feed_gaze(t)
            step += 1
            if step % _STEPS_PER_FRAME:
                continue

            session.tick()
            state = session.render_state()
            zone = state["hud"]["gaze_zone"]
            if zone != last_zone:
                events.append((t, f"classifier zone -> {zone}"))
                last_zone = zone
            # Rows are consumed before the participant reacts to the new phase, so a
            # trial's ground truth attaches to its own row (practice trials log no
            # row, and their pending truth is cleared on the next phase entry).
            while logged < len(logger.rows):
                row = logger.rows[logged]
                if row.phase == "saccade":
                    saccade_rows_logged += 1
                    if row.grating_shown and participant.pending_flash_truth is not None:
                        flash_truth[(row.phase, row.index)] = participant.pending_flash_truth
                logged += 1
                events.append((t, f"trial logged: {row.phase} #{row.index} outcome={row.outcome}"))
            participant.observe(state, t)

            if state["hud"]["phase"] == "COMPLETE":
                break
            if saccade_rows_logged >= max_trials:
                break
        events.sort(key=lambda e: e[0])
        return SimulationResult(
            session=session, results=logger.rows, events=events, duration_s=clock.now(), flash_truth=flash_truth
        )
    finally:
        random.setstate(saved_state)


def simulate_trial(profile: ParticipantProfile | None = None, seed: int = 0) -> SimulationResult:
    """Calibration + reaction-time test, then exactly one saccade trial."""
    return simulate_session(max_trials=1, profile=profile, seed=seed)


def print_trial_timeline(sim: SimulationResult) -> None:
    result = sim.saccade_results[0]
    go = next(t for t, text in reversed(sim.events) if text == "go-cue: target appeared")
    print("Event timeline of the trial (ms relative to the go-cue):")
    for t, text in sim.events:
        if t >= go - 0.001:
            print(f"  {1000 * (t - go):8.1f}  {text}")
    print("\nLogged result:")
    for name in (
        "outcome", "grating_shown", "contrast", "orientation", "responded", "response_time_ms",
        "reaction_latency_ms", "onset_detection_lag_ms", "timing_offset_ms", "flash_during_saccade", "tracking_ok",
    ):
        print(f"  {name:24s} {getattr(result, name)}")
    print(f"\n(Simulated {sim.duration_s:.1f} s of session time, including calibration and the RT test.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sim = simulate_session(max_trials=args.trials, seed=args.seed)
    if args.trials == 1:
        print_trial_timeline(sim)
        return
    rows = sim.saccade_results
    print(f"{len(rows)} saccade trials, {sim.duration_s:.0f} s simulated")
    print("  outcome counts:", {o: sum(r.outcome == o for r in rows) for o in sorted({r.outcome for r in rows})})
    flashed = [r for r in rows if r.grating_shown]
    valid = [r for r in flashed if is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok)]
    print(f"  flashes: {len(flashed)}, flash during saccade & tracking ok: {len(valid)}")
    latencies = [r.reaction_latency_ms for r in rows if r.reaction_latency_ms]
    print(f"  mean reaction latency: {statistics.fmean(latencies) if latencies else 0:.0f} ms")


if __name__ == "__main__":
    main()
