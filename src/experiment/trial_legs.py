"""Gaze bookkeeping of one saccade trial, shared by ExperimentSession and TutorialSession: the per-tick sample feed and the legs
(out onset -> out landing/arrival -> jump back -> return onset -> return landing). PsychoPy-free; times are capture times on the
session clock. The sessions keep the policy: which leg flashes, when the flash fires, symbol visibility, the answer
that ends a trial and when it is logged."""

from include.experiment.constants import (
    GAZE_LANDING_STABILITY_MS,
    GAZE_LATENCY_MAX_MS,
    RETURN_JUMP_LANDING_TIMEOUT_MS,
    SACCADE_TIMEOUT_MS,
)
from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeSample, GazeZone
from src.experiment.pausable_clock import PausableClock
from src.experiment.trial_mechanics import OnsetDetector, sample_capture_time
from src.timebase import ms_to_s, s_to_ms


def clamped_gaze_latency_ms(gaze: GazeSource, fallback_ms: float | None = None) -> float:
    """Eye-to-sample latency: the source's own measurement if it has one, else `fallback_ms`, else 0; clamped to
    [0, GAZE_LATENCY_MAX_MS]."""
    value = getattr(gaze, "gaze_latency_ms", None)
    if value is None:
        value = fallback_ms
    if value is None or value != value:
        return 0.0
    return min(max(float(value), 0.0), GAZE_LATENCY_MAX_MS)


def timestamp_latency_s(gaze: GazeSource, latency_ms: float) -> float:
    """How much to subtract from sample timestamps: nothing if the source already stamps samples with their capture time
    (Neon with a clock offset, trace replay: `timestamps_are_capture_time`) - subtracting the latency again would put onset
    ~16 ms too early - else the measured latency."""
    if getattr(gaze, "timestamps_are_capture_time", False):
        return 0.0
    return ms_to_s(latency_ms)


class SampleFeed:
    """(sample, capture_time) for every gaze sample since the last call. `latency_s` is a snapshot taken at each attempt /
    trial start, so the per-sample timing path never queries the source and onset/landing share one offset. Set
    `last_sample_ts` to None (or the latest sample's timestamp at the go cue) when a new attempt starts."""

    def __init__(self, gaze: GazeSource, clock: PausableClock) -> None:
        self._gaze = gaze
        self._clock = clock
        self.latency_s = 0.0
        self.last_sample_ts: float | None = None

    def capture_time(self, sample, now: float) -> float:
        """Capture time of `sample` on the session clock, minus the snapshotted gaze latency (see sample_capture_time)."""
        return sample_capture_time(sample, now, self.latency_s, self._clock.from_perf_counter)

    def new_samples(self, latest) -> list[tuple[GazeSample, float]]:
        """Sources with a short history (`samples_since`, e.g. Neon) hand over all new samples; others contribute just the
        latest one, which is harmless to re-read."""
        now = self._clock.now()
        history = getattr(self._gaze, "samples_since", None)
        if history is not None and latest.timestamp is not None and latest.timestamp > 0:
            samples = history(self.last_sample_ts)
            if samples:
                self.last_sample_ts = samples[-1].timestamp
            elif not latest.face_found and latest.timestamp > (self.last_sample_ts or 0.0):
                # A stalled stream produces no new samples, only a synthetic "lost" latest_sample() - keep reporting it.
                samples = [latest]
        else:
            samples = [latest]
        return [(s, self.capture_time(s, now)) for s in samples]


class LandingTracker:
    """Landing on `zone`: gaze must stay there for GAZE_LANDING_STABILITY_MS of capture time. `since` = first sample of the
    current in-zone run (the landing time once `landed`). `stability_ms` replaces that default (ArrivalDwell)."""

    def __init__(self, zone: GazeZone, stability_ms: float | None = None) -> None:
        self.zone = zone
        self._stability_ms = GAZE_LANDING_STABILITY_MS if stability_ms is None else stability_ms
        self.since: float | None = None
        self.landed = False

    def feed(self, samples) -> None:
        for s, s_t in samples:
            if self.landed:
                break
            if s.face_found and s.zone is self.zone:
                if self.since is None:
                    self.since = s_t
                elif s_t - self.since >= ms_to_s(self._stability_ms):
                    self.landed = True
            else:
                self.since = None

    def mark(self, t: float) -> None:
        """Landed at `t` without the stability wait (return-jump mode: the out arrival is the out landing)."""
        self.since, self.landed = t, True


class ArrivalDwell:
    """Practice and reaction-time-test trials: the symbols stay until the eye has been on `zone` for `dwell_ms` without a
    break. `due`: None = keep waiting; False = the stay is complete; True = given up, RETURN_JUMP_LANDING_TIMEOUT_MS after
    `start` with the eye never on `zone` (one that arrived but did not stay gets `dwell_ms` on top)."""

    def __init__(self, zone: GazeZone, dwell_ms: float) -> None:
        self._dwell_ms = dwell_ms
        self._stay = LandingTracker(zone, dwell_ms)
        self._arrived = False

    def feed(self, samples) -> None:
        """Samples captured from the go cue on."""
        self._arrived = self._arrived or any(s.face_found and s.zone is self._stay.zone for s, _ in samples)
        self._stay.feed(samples)

    def due(self, now: float, start: float) -> bool | None:
        if self._stay.landed:
            return False
        limit_ms = RETURN_JUMP_LANDING_TIMEOUT_MS + (self._dwell_ms if self._arrived else 0)
        return True if now - start >= ms_to_s(limit_ms) else None


class TrialLegs:
    """The gaze side of one trial, from the go cue on.

    Single-leg trials (return jump off): out onset + out landing (stable stay on the target).
    Return-jump trials: the out ARRIVAL (first in-target sample) is the out landing; the target jumps back on `trigger`
    ("arrival", or "onset" = out onset confirmed) or at RETURN_JUMP_LANDING_TIMEOUT_MS (forced); the return leg is listened
    to from the arrival (or the jump, if that came first), so a predictive return can start before the jump frame
    (negative, signed return latency). With `dwell_ms` > 0 (practice) the jump waits for an ArrivalDwell, not `trigger`.

    Per tick the session calls, in this order: `measure_out_onset`, then `track_out_landing` (single leg) or
    `track_arrival` -> `jump_due` / `mark_jump` -> `track_return` (return jump)."""

    def __init__(
        self, source_zone: GazeZone, target_zone: GazeZone, go_cue_at: float, trigger: str = "arrival", dwell_ms: float = 0
    ) -> None:
        self.go_cue_at = go_cue_at
        self.dwell = ArrivalDwell(target_zone, dwell_ms) if dwell_ms > 0 else None
        self.trigger = trigger
        self._target_zone = target_zone
        self.onset = OnsetDetector(source_zone)
        self.out_landing = LandingTracker(target_zone)
        # Return-jump state (stays at these values with return jump off).
        self.out_arrived_at: float | None = None
        self.in_target_before_jump: bool | None = None  # zone of the last tracked sample captured before the jump frame
        self.jump_at: float | None = None  # session-clock time of the jump-back frame
        self.jump_forced: bool | None = None
        self.return_feed_from: float | None = None  # capture time from which samples go to the return-leg detectors
        self.return_onset: OnsetDetector | None = None
        self.return_landing = LandingTracker(source_zone)
        self.return_reaction_ms: float | None = None

    # -- out leg ---------------------------------------------------------------

    def measure_out_onset(self, samples) -> tuple[float, float] | None:
        """Out-leg onset detection (a measurement only). Returns (reaction latency ms, detection lag ms) on the call that
        confirms it, else None. Samples captured after a jump back belong to the return leg - unless the out saccade already
        arrived before the jump (the "arrival" trigger jumps on the first in-target sample, which can come before
        SACCADE_ONSET_STABILITY_MS has confirmed the out onset)."""
        for s, s_t in samples:
            if s_t < self.go_cue_at:
                continue  # captured before the go-cue (it just arrived after it): cannot be a reaction to it
            arrived_before_jump = self.out_arrived_at is not None and self.out_arrived_at <= (self.jump_at or 0.0)
            if self.jump_at is not None and s_t >= self.jump_at and not arrived_before_jump:
                break  # a late arrival after a FORCED jump must not let a > 1.5 s "reaction" into the RT average
            if not self.onset.confirmed and self.onset.update(s, s_t):
                since = self.onset.since
                assert since is not None  # update() returned True
                return s_to_ms(since - self.go_cue_at), s_to_ms(s_t - since)
        return None

    def track_out_landing(self, samples) -> None:
        """Single-leg trials: landing is flagged at once but the trial is not finalized here (a real reaction time can
        outlast how quickly landing is detected)."""
        self.out_landing.feed(samples)

    def out_timed_out(self, now: float) -> bool:
        """Single-leg trials: no landing within SACCADE_TIMEOUT_MS of the go cue."""
        return not self.out_landing.landed and now - self.go_cue_at >= ms_to_s(SACCADE_TIMEOUT_MS)

    # -- return jump -----------------------------------------------------------

    def track_arrival(self, samples) -> None:
        """The out arrival (= out landing in return-jump mode) and whether the eye was still on the target before the jump."""
        if self.dwell is not None:
            self.dwell.feed([(s, s_t) for s, s_t in samples if s_t >= self.go_cue_at])
        for s, s_t in samples:
            if s_t < self.go_cue_at or not s.face_found:
                continue
            if self.out_arrived_at is None and s.zone is self._target_zone:
                self.out_arrived_at = s_t
                self.out_landing.mark(s_t)
            if self.out_arrived_at is not None and (self.jump_at is None or s_t < self.jump_at):
                self.in_target_before_jump = s.zone is self._target_zone

    def jump_due(self, now: float, out_flash_done: bool) -> bool | None:
        """None = no jump yet; else jump now, with the value saying whether it is forced (no trigger by
        RETURN_JUMP_LANDING_TIMEOUT_MS; with a dwell, see ArrivalDwell.due). `out_flash_done`: an out-leg flash is drawn
        (and gone) before the target jumps."""
        if self.jump_at is not None or not out_flash_done:
            return None
        if self.dwell is not None:
            return self.dwell.due(now, self.go_cue_at)
        triggered = self.onset.confirmed if self.trigger == "onset" else self.out_arrived_at is not None
        if triggered:
            return False
        if now - self.go_cue_at >= ms_to_s(RETURN_JUMP_LANDING_TIMEOUT_MS):
            return True
        return None

    def mark_jump(self, now: float, forced: bool) -> None:
        self.jump_at = now
        self.jump_forced = forced

    def track_return(self, samples) -> float | None:
        """Feeds the return-leg onset detector and landing. Returns the signed return latency (ms) if this call measured it."""
        if self.return_feed_from is None:
            self.return_feed_from = self.out_arrived_at if self.out_arrived_at is not None else self.jump_at
        if self.return_feed_from is None:
            return None
        if self.return_onset is None:
            # wait_for_baseline: the eye may still be flying towards the target; approaching is not leaving.
            self.return_onset = OnsetDetector(self._target_zone, wait_for_baseline=True)
        feed = [(s, s_t) for s, s_t in samples if s_t >= self.return_feed_from]
        detector = self.return_onset
        for s, s_t in feed:
            if detector.confirmed:
                break
            detector.update(s, s_t)
        measured = None
        if detector.confirmed and self.jump_at is not None and self.return_reaction_ms is None:
            assert detector.since is not None  # confirmed
            self.return_reaction_ms = measured = s_to_ms(detector.since - self.jump_at)  # signed: < 0 = left before the jump
        self.return_landing.feed(feed)
        return measured

    def return_timed_out(self, now: float) -> bool:
        """No return landing within SACCADE_TIMEOUT_MS of the jump."""
        return self.jump_at is not None and not self.return_landing.landed and now - self.jump_at >= ms_to_s(SACCADE_TIMEOUT_MS)

    # -- end of the trial's movements ------------------------------------------

    def timed_out(self, now: float, return_jump: bool) -> bool:
        """The final landing (return-jump trials: back on the start symbol) never came in time."""
        return self.return_timed_out(now) if return_jump else self.out_timed_out(now)

    def movement_over(self, now: float, return_jump: bool) -> bool:
        """No further eye movement is awaited: the final landing (return-jump trials: after the jump) or its timeout."""
        landed = self.jump_at is not None and self.return_landing.landed if return_jump else self.out_landing.landed
        return landed or self.timed_out(now, return_jump)

    # -- derived values (trials.csv) ---------------------------------------------

    def onset_landing(self, flash_leg: str | None) -> tuple[float | None, float | None]:
        """(onset, first-in-zone landing) capture times of the saccade on the flash's leg."""
        if flash_leg == "return":
            return (self.return_onset.since if self.return_onset is not None else None), self.return_landing.since
        return self.onset.since, self.out_landing.since

    def leg_moved(self, flash_leg: str | None) -> bool:
        """Whether the eye moved on the flash's leg (onset confirmed or landed)."""
        if flash_leg == "return":
            return self.return_landing.landed or (self.return_onset is not None and self.return_onset.confirmed)
        return self.out_landing.landed or self.onset.confirmed

    def saccade_duration_ms(self) -> float | None:
        """Out landing minus out onset on the capture-time axis; None unless both were observed."""
        onset_at, landed_at = self.onset.since, self.out_landing.since
        if not (self.onset.confirmed and self.out_landing.landed and onset_at is not None and landed_at is not None):
            return None
        return s_to_ms(landed_at - onset_at) if landed_at >= onset_at else None

    def ms_after_go_cue(self, t: float | None) -> float | None:
        return s_to_ms(t - self.go_cue_at) if t is not None else None

    def left_target_before_jump(self) -> bool | None:
        """True = the eye had arrived and its last sample before the jump was outside the target; None without a jump."""
        if self.jump_at is None:
            return None
        return self.out_arrived_at is not None and self.in_target_before_jump is False
