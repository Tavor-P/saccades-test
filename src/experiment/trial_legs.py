"""Gaze bookkeeping of one saccade trial, shared by ExperimentSession and TutorialSession: the per-tick sample feed and the legs
(out onset -> out landing/arrival -> jump back -> return onset -> return landing). PsychoPy-free; times are capture times on the
session clock. The sessions keep the policy: which leg flashes, when the flash fires, symbol visibility, the response window
and when a trial is logged."""

from include.eye_tracking.interfaces import GazeSource
from include.experiment.constants import (
    GAZE_LANDING_STABILITY_MS,
    GAZE_LATENCY_MAX_MS,
    RETURN_JUMP_LANDING_TIMEOUT_MS,
    SACCADE_TIMEOUT_MS,
)
from include.eye_tracking.types import GazeZone
from src.experiment.pausable_clock import PausableClock
from src.experiment.trial_mechanics import OnsetDetector, sample_capture_time


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
    return latency_ms / 1000


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

    def new_samples(self, latest) -> list[tuple[object, float]]:
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
    current in-zone run (the landing time once `landed`)."""

    def __init__(self, zone: GazeZone) -> None:
        self.zone = zone
        self.since: float | None = None
        self.landed = False

    def feed(self, samples) -> None:
        for s, s_t in samples:
            if self.landed:
                break
            if s.face_found and s.zone is self.zone:
                if self.since is None:
                    self.since = s_t
                elif s_t - self.since >= GAZE_LANDING_STABILITY_MS / 1000:
                    self.landed = True
            else:
                self.since = None

    def mark(self, t: float) -> None:
        """Landed at `t` without the stability wait (return-jump mode: the out arrival is the out landing)."""
        self.since, self.landed = t, True


class TrialLegs:
    """The gaze side of one trial, from the go cue on.

    Single-leg trials (return jump off): out onset + out landing (stable stay on the target).
    Return-jump trials: the out ARRIVAL (first in-target sample) is the out landing; the target jumps back on `trigger`
    ("arrival", or "onset" = out onset confirmed) or at RETURN_JUMP_LANDING_TIMEOUT_MS (forced); the return leg is listened
    to from the arrival (or the jump, if that came first), so a predictive return can start before the jump frame
    (negative, signed return latency).

    Per tick the session calls, in this order: `measure_out_onset`, then `track_out_landing` (single leg) or
    `track_arrival` -> `jump_due` / `mark_jump` -> `track_return` (return jump)."""

    def __init__(self, source_zone: GazeZone, target_zone: GazeZone, go_cue_at: float, trigger: str = "arrival") -> None:
        self.go_cue_at = go_cue_at
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
                return (self.onset.since - self.go_cue_at) * 1000, (s_t - self.onset.since) * 1000
        return None

    def track_out_landing(self, samples) -> None:
        """Single-leg trials: landing is flagged at once but the trial is not finalized here (a real reaction time can
        outlast how quickly landing is detected)."""
        self.out_landing.feed(samples)

    def out_timed_out(self, now: float) -> bool:
        """Single-leg trials: no landing within SACCADE_TIMEOUT_MS of the go cue."""
        return not self.out_landing.landed and now - self.go_cue_at >= SACCADE_TIMEOUT_MS / 1000

    # -- return jump -----------------------------------------------------------

    def track_arrival(self, samples) -> None:
        """The out arrival (= out landing in return-jump mode) and whether the eye was still on the target before the jump."""
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
        RETURN_JUMP_LANDING_TIMEOUT_MS). `out_flash_done`: an out-leg flash is drawn (and gone) before the target jumps."""
        if self.jump_at is not None or not out_flash_done:
            return None
        triggered = self.onset.confirmed if self.trigger == "onset" else self.out_arrived_at is not None
        if triggered:
            return False
        if now - self.go_cue_at >= RETURN_JUMP_LANDING_TIMEOUT_MS / 1000:
            return True
        return None

    def mark_jump(self, now: float, forced: bool) -> None:
        self.jump_at = now
        self.jump_forced = forced

    def track_return(self, samples) -> tuple[float | None, bool]:
        """Feeds the return-leg onset detector and landing. Returns (signed return latency ms if measured by this call,
        whether the return landed in this call)."""
        if self.return_feed_from is None:
            self.return_feed_from = self.out_arrived_at if self.out_arrived_at is not None else self.jump_at
        if self.return_feed_from is None:
            return None, False
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
            self.return_reaction_ms = measured = (detector.since - self.jump_at) * 1000  # signed: < 0 = left before the jump
        was_landed = self.return_landing.landed
        self.return_landing.feed(feed)
        return measured, self.return_landing.landed and not was_landed

    def return_timed_out(self, now: float) -> bool:
        """No return landing within SACCADE_TIMEOUT_MS of the jump."""
        return self.jump_at is not None and not self.return_landing.landed and now - self.jump_at >= SACCADE_TIMEOUT_MS / 1000

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
        return (landed_at - onset_at) * 1000 if landed_at >= onset_at else None

    def ms_after_go_cue(self, t: float | None) -> float | None:
        return (t - self.go_cue_at) * 1000 if t is not None else None

    def left_target_before_jump(self) -> bool | None:
        """True = the eye had arrived and its last sample before the jump was outside the target; None without a jump."""
        if self.jump_at is None:
            return None
        return self.out_arrived_at is not None and self.in_target_before_jump is False
