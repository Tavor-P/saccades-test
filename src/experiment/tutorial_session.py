"""TutorialSession: the narrated first-timer on-ramp (calibration, how to answer, practice trials). Scoped-down mirror of ExperimentSession's trial logic."""

import random
from enum import Enum, auto

from include.eye_tracking.interfaces import GazeSource
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    CENTER_POSITION,
    CROSS_POSITION,
    DEFAULT_REACTION_TIME_MS,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GAZE_LANDING_STABILITY_MS,
    GAZE_LATENCY_MAX_MS,
    GRATING_DURATION_FRAMES,
    GRATING_POSITION,
    RESPONSE_WINDOW_MS,
    SACCADE_TIMEOUT_MS,
    TUTORIAL_PRACTICE_ATTEMPTS,
    TUTORIAL_PRACTICE_CONTRAST,
    TUTORIAL_QUIZ_CONTRAST,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Orientation, Target
from src.experiment.calibration import average_calibration_rounds
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import score_outcome
from src.experiment.space_input import CalibrationLatch, SpaceTally
from src.experiment.trial_mechanics import OnsetDetector, robust_average_ms
from src.experiment.trial_mechanics import random_orientation as _random_orientation
from src.experiment.trial_mechanics import zone_for as _zone_for


class Phase(Enum):
    WAITING_TO_START = auto()
    CALIBRATE_LEFT = auto()
    CALIBRATE_CENTER = auto()
    CALIBRATE_RIGHT = auto()
    QUIZ = auto()  # "how to answer": mixed gratings, streak-gated
    PRACTICE_AWAITING_SPACE = auto()  # each attempt waits for SPACE; the foreperiod (500-1000 ms) starts on the press
    PRACTICE_FOREPERIOD = auto()
    PRACTICE_ACTIVE = auto()
    COMPLETE = auto()


class TutorialSession:
    """A one-time, narrated on-ramp for first-time participants, run before
    both real phases (see run_experiment.py's main()). Three stages:

    1. Calibration - identical 3-round discard-first/average procedure as
       ExperimentSession's own self-calibration (see CALIBRATION_ROUNDS),
       so a participant who takes the tutorial gets the exact same quality
       calibration as one who doesn't. The result is exposed via
       `calibration_ratios` for the real saccade phase to reuse.
    2. QUIZ ("how to answer") - mixed-orientation, fixed-contrast gratings,
       streak-gated: TUTORIAL_QUIZ_STREAK_TARGET correct in a row, a miss
       resets the streak. The only stage gated by success.
    3. PRACTICE - TUTORIAL_PRACTICE_ATTEMPTS trials that look exactly like
       real saccade trials (SPACE -> dot -> cross -> clearly visible flash ->
       arrow key -> colour feedback), fixed count regardless of outcome. The
       flash fires at a scheduled delay (running robust average of this
       stage's own reaction times, DEFAULT_REACTION_TIME_MS until there is
       one). Each trial's saccade-onset latency is kept in `rt_samples_ms`
       and handed to ExperimentSession, which seeds its reaction-time average
       from them instead of running a separate RT test.

    Framework-agnostic like ExperimentSession/PresaccadeSession:
    `on_space()`/`on_response_key()`/`tick()` are called by the PsychoPy
    frame loop, `render_state()` returns a plain dict the runner applies to
    its stimuli. Never logs to the results CSV - none of this counts as
    data (matches the existing convention that practice trials aren't
    logged either).
    """

    def __init__(self, gaze_source: GazeSource, clock: PausableClock) -> None:
        self._gaze = gaze_source
        self._clock = clock
        self._phase = Phase.WAITING_TO_START

        # -- calibration --
        self._calibration_round = 1
        self._calibration_rounds_data: list[tuple[float, float, float]] = []
        self._pending_left_ratio: float | None = None
        self._calibration_latch = CalibrationLatch()  # a press made before the gaze window filled
        self.space_tally = SpaceTally(lambda: self._phase.name)
        self._pending_center_ratio: float | None = None
        self._calibration_ratios: tuple[float, float, float] | None = None

        self._dot_visible = False
        self._cross_visible = False
        self._foreperiod_start = 0.0
        self._foreperiod_duration = 0.0
        self._target_landed_since: float | None = None

        # -- quiz --
        self._quiz_streak = 0
        self._quiz_orientation: Orientation | None = None

        # -- feedback flash: color + a token that increments on every new
        # event, so the renderer can edge-detect "a new flash just happened"
        # the same way run_experiment.py already edge-detects FOREPERIOD
        # entry for the trial-start flash. --
        self._feedback_color: str | None = None
        self._feedback_token = 0

        # -- practice trials --
        self._attempt = 0
        self._rt_samples_ms: list[float] = []
        self._source = Target.DOT
        self._target = Target.CROSS
        self._orientation: Orientation | None = None
        self._onset_detector: OnsetDetector = OnsetDetector(_zone_for(self._source))
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation: Orientation | None = None
        self._landed = False
        self._trial_start = 0.0

    @property
    def calibration_ratios(self) -> tuple[float, float, float] | None:
        return self._calibration_ratios

    @property
    def rt_samples_ms(self) -> list[float]:
        """Saccade-onset latency (target appears -> eye leaves the source) of every practice trial that had one."""
        return list(self._rt_samples_ms)

    # -- key handling ------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            self._gaze.begin_calibration_sample()
            self._calibration_round = 1
            self._calibration_rounds_data = []
            self._phase = Phase.CALIBRATE_LEFT
        elif self._phase in (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT):
            self._press_calibration()
        elif self._phase is Phase.PRACTICE_AWAITING_SPACE:
            self._start_foreperiod()
        elif self._phase is not Phase.COMPLETE:
            # QUIZ/PRACTICE_ACTIVE respond to on_response_key, not space; PRACTICE_FOREPERIOD advances via tick().
            # COMPLETE: the runner decides when to move on.
            self.space_tally.record("not accepted in this phase")

    def _press_calibration(self) -> None:
        """One press is enough: if the gaze window is not full yet the press is latched and tick() completes it."""
        if self._calibration_latch.pending:
            self.space_tally.record("already waiting for gaze samples")
            return
        ratio = self._calibration_latch.press(self._gaze, self._clock.now())
        if ratio is not None:
            self._advance_calibration(ratio)

    def _advance_calibration(self, ratio: float) -> None:
        self._calibration_latch.clear()
        if self._phase is Phase.CALIBRATE_LEFT:
            self._pending_left_ratio = ratio
            self._gaze.begin_calibration_sample()
            self._phase = Phase.CALIBRATE_CENTER
        elif self._phase is Phase.CALIBRATE_CENTER:
            self._pending_center_ratio = ratio
            self._gaze.begin_calibration_sample()
            self._phase = Phase.CALIBRATE_RIGHT
        elif self._phase is Phase.CALIBRATE_RIGHT:
            if self._pending_left_ratio is None or self._pending_center_ratio is None:
                return
            self._calibration_rounds_data.append((self._pending_left_ratio, self._pending_center_ratio, ratio))
            if self._calibration_round < CALIBRATION_ROUNDS:
                self._calibration_round += 1
                self._gaze.begin_calibration_sample()
                self._phase = Phase.CALIBRATE_LEFT
            else:
                # Indexed from the end (not unpacked as exactly 3
                # elements) so this doesn't break if CALIBRATION_ROUNDS
                # is ever changed to something other than 3.
                round2, round3 = self._calibration_rounds_data[-2], self._calibration_rounds_data[-1]
                self._calibration_ratios = average_calibration_rounds(round2, round3)
                self._gaze.calibrate(*self._calibration_ratios)
                self._enter_quiz()

    def on_response_key(self, orientation: Orientation) -> None:
        if self._phase is Phase.QUIZ:
            if orientation == self._quiz_orientation:
                self._quiz_streak += 1
                if self._quiz_streak >= TUTORIAL_QUIZ_STREAK_TARGET:
                    self._set_feedback("green")
                    self._enter_practice()
                else:
                    self._quiz_orientation = _random_orientation()
            else:
                self._quiz_streak = 0
                self._set_feedback("red")
                self._quiz_orientation = _random_orientation()
        elif self._phase is Phase.PRACTICE_ACTIVE:
            if self._responded or not self._response_window_open:
                return
            now = self._clock.now()
            if now <= self._response_deadline:
                self._responded = True
                self._response_orientation = orientation

    def _set_feedback(self, color: str) -> None:
        self._feedback_color = color
        self._feedback_token += 1

    def _enter_quiz(self) -> None:
        self._quiz_streak = 0
        self._quiz_orientation = _random_orientation()
        self._phase = Phase.QUIZ

    # -- practice trials -----------------------------------------------------------

    def _enter_practice(self) -> None:
        self._attempt = 0
        self._rt_samples_ms = []
        self._source = Target.DOT
        self._dot_visible = True
        self._cross_visible = False
        self._begin_attempt()

    def _begin_attempt(self) -> None:
        self._phase = Phase.PRACTICE_AWAITING_SPACE
        self._target = Target.CROSS if self._source is Target.DOT else Target.DOT
        self._orientation = _random_orientation()
        self._onset_detector = OnsetDetector(_zone_for(self._source))
        self._target_landed_since = None
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation = None
        self._landed = False

    def _start_foreperiod(self) -> None:
        self._phase = Phase.PRACTICE_FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000

    def _tick_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration:
            self._dot_visible = self._target is Target.DOT
            self._cross_visible = self._target is Target.CROSS
            self._phase = Phase.PRACTICE_ACTIVE
            self._trial_start = self._clock.now()
            # Open-loop, like the real saccade phase: scheduled off the
            # reaction times measured so far (default until there are any).
            avg_ms = robust_average_ms(self._rt_samples_ms, default=DEFAULT_REACTION_TIME_MS)
            self._flash_scheduled_at = self._trial_start + avg_ms / 1000

    def _sample_time(self, sample, now: float) -> float:
        """When the eye was actually where `sample` says, on the session clock
        (same rule as ExperimentSession): the sample's capture timestamp, minus
        the tracker's latency when the source does not already stamp capture time."""
        latency_s = 0.0
        if not getattr(self._gaze, "timestamps_are_capture_time", False):
            latency_ms = getattr(self._gaze, "gaze_latency_ms", None)
            if latency_ms is not None and latency_ms == latency_ms:
                latency_s = min(max(float(latency_ms), 0.0), GAZE_LATENCY_MAX_MS) / 1000
        if sample.timestamp is None or sample.timestamp <= 0:
            return now - latency_s
        t = self._clock.from_perf_counter(sample.timestamp)
        if t > now or now - t > 1.0:
            t = now
        return t - latency_s

    def _tick_active(self) -> None:
        now = self._clock.now()
        sample = self._gaze.latest_sample()
        target_zone = _zone_for(self._target)
        sample_t = self._sample_time(sample, now)

        if not self._onset_detector.confirmed and sample_t >= self._trial_start:
            if self._onset_detector.update(sample, sample_t):
                self._rt_samples_ms.append((self._onset_detector.since - self._trial_start) * 1000)

        flash_fired_this_tick = False
        if not self._flash_decided and now >= self._flash_scheduled_at:
            self._flash_decided = True
            flash_fired_this_tick = True
            self._response_window_open = True
            self._response_deadline = now + RESPONSE_WINDOW_MS / 1000
            self._grating_visible = True
            self._grating_frames_remaining = GRATING_DURATION_FRAMES
            self._grating_actually_shown = True

        if self._grating_visible and not flash_fired_this_tick:
            self._grating_frames_remaining -= 1
            if self._grating_frames_remaining <= 0:
                self._grating_visible = False

        if self._response_window_open and now > self._response_deadline:
            self._response_window_open = False

        if not self._landed:
            if sample.face_found and sample.zone is target_zone:
                if self._target_landed_since is None:
                    self._target_landed_since = now
                elif now - self._target_landed_since >= GAZE_LANDING_STABILITY_MS / 1000:
                    self._landed = True
            else:
                self._target_landed_since = None

            if now - self._trial_start >= SACCADE_TIMEOUT_MS / 1000:
                self._finish_attempt(timed_out=True)
                return

        if self._landed and self._flash_decided and (self._responded or not self._response_window_open):
            self._finish_attempt(timed_out=False)

    def _finish_attempt(self, timed_out: bool) -> None:
        if timed_out and not self._onset_detector.confirmed:
            outcome = "timeout"
        else:
            outcome = score_outcome(self._grating_actually_shown, self._responded, self._response_orientation, self._orientation)
        self._set_feedback("green" if outcome == "correct" else "red")

        self._grating_visible = False
        self._source = self._target
        self._attempt += 1
        if self._attempt >= TUTORIAL_PRACTICE_ATTEMPTS:
            self._phase = Phase.COMPLETE
        else:
            self._begin_attempt()

    # -- tick ----------------------------------------------------------------

    def tick(self) -> None:
        ratio = self._calibration_latch.poll(self._gaze)
        if ratio is not None:
            self._advance_calibration(ratio)
        if self._phase is Phase.PRACTICE_FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.PRACTICE_ACTIVE:
            self._tick_active()

    # -- rendering -------------------------------------------------------------

    _QUIZ_TEXT = "Lines going up-and-down: press UP or DOWN. Side-to-side: press LEFT or RIGHT."
    _PRACTICE_INTRO = "Press SPACE, look at the dot, then at the cross. Then press an arrow key."

    def _instructions(self) -> str:
        if self._phase is Phase.WAITING_TO_START:
            return "Press SPACE to begin"
        if self._phase in (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT):
            # Only narrate round 1 of CALIBRATION_ROUNDS - by round 2 the
            # participant already knows the drill, so re-speaking it every
            # round just slows calibration down for no benefit.
            if self._calibration_round > 1:
                return ""
            return {
                Phase.CALIBRATE_LEFT: "Look at the dot, then press SPACE.",
                Phase.CALIBRATE_CENTER: "Look at the middle, then press SPACE.",
                Phase.CALIBRATE_RIGHT: "Look at the cross, then press SPACE.",
            }[self._phase]
        if self._phase is Phase.QUIZ:
            return self._QUIZ_TEXT
        if self._phase in (Phase.PRACTICE_AWAITING_SPACE, Phase.PRACTICE_FOREPERIOD, Phase.PRACTICE_ACTIVE):
            # Said once, before the first trial; silent afterwards (the narrator re-speaks on every text change).
            return self._PRACTICE_INTRO if self._attempt == 0 else ""
        return "Practice done. Press SPACE to continue."

    def _tutorial_text(self) -> tuple[str | None, str]:
        """(text, position) for the larger instructional text stim shown only
        while the participant is learning the answer keys; everywhere else the
        caption (see run_experiment.py's render_caption) carries the words."""
        if self._phase is Phase.QUIZ:
            return self._QUIZ_TEXT, "bottom"
        return None, "center"

    def _symbol_visibility(self) -> tuple[bool, bool, bool]:
        if self._phase is Phase.CALIBRATE_LEFT:
            return True, False, False
        if self._phase is Phase.CALIBRATE_CENTER:
            return False, False, True
        if self._phase is Phase.CALIBRATE_RIGHT:
            return False, True, False
        if self._phase in (Phase.PRACTICE_AWAITING_SPACE, Phase.PRACTICE_FOREPERIOD, Phase.PRACTICE_ACTIVE):
            return self._dot_visible, self._cross_visible, False
        return False, False, False

    def _grating_state(self) -> dict:
        if self._phase is Phase.QUIZ:
            return {
                "visible": True,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": TUTORIAL_QUIZ_CONTRAST,
                "orientation": self._quiz_orientation,
            }
        if self._phase is Phase.PRACTICE_ACTIVE:
            return {
                "visible": self._grating_visible,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": TUTORIAL_PRACTICE_CONTRAST if self._grating_visible else 0,
                "orientation": self._orientation,
            }
        return {"visible": False, "x": GRATING_POSITION[0], "y": GRATING_POSITION[1], "contrast": 0, "orientation": None}

    def _trial_label(self) -> str:
        if self._phase is Phase.QUIZ:
            return f"quiz streak {self._quiz_streak}/{TUTORIAL_QUIZ_STREAK_TARGET}"
        if self._phase in (Phase.PRACTICE_AWAITING_SPACE, Phase.PRACTICE_FOREPERIOD, Phase.PRACTICE_ACTIVE):
            return f"practice {self._attempt + 1}/{TUTORIAL_PRACTICE_ATTEMPTS}"
        return "-"

    def render_state(self) -> dict:
        sample = self._gaze.latest_sample()
        dot_visible, cross_visible, calibration_center_visible = self._symbol_visibility()
        text, position = self._tutorial_text()

        return {
            "instructions": self._instructions(),
            "tutorial_text": {"visible": text is not None, "text": text or "", "position": position},
            "feedback_flash": {"color": self._feedback_color, "token": self._feedback_token},
            "dot": {"visible": dot_visible, "x": DOT_POSITION[0], "y": DOT_POSITION[1]},
            "cross": {"visible": cross_visible, "x": CROSS_POSITION[0], "y": CROSS_POSITION[1]},
            "calibration_center": {
                "visible": calibration_center_visible,
                "x": CENTER_POSITION[0],
                "y": CENTER_POSITION[1],
            },
            "grating": self._grating_state(),
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": sample.zone.value,
                "face_found": sample.face_found,
                "source_available": self._gaze.is_available,
                "trial": self._trial_label(),
                "trial_index": 0,
                "calibration_wait_s": self._calibration_latch.waiting_s(self._clock.now()),
            },
        }
