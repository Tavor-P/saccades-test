"""TutorialSession: the narrated first-timer on-ramp (calibration, how to answer, practice trials). Scoped-down mirror of ExperimentSession's trial logic."""

import random
from enum import Enum, auto

from include.eye_tracking.interfaces import GazeSource
from include.experiment.constants import (
    CENTER_POSITION,
    CROSS_POSITION,
    DEFAULT_REACTION_TIME_MS,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GAZE_LANDING_STABILITY_MS,
    GAZE_LATENCY_MAX_MS,
    GRATING_DURATION_FRAMES,
    RESPONSE_WINDOW_MS,
    SACCADE_TIMEOUT_MS,
    TUTORIAL_PRACTICE_ATTEMPTS,
    TUTORIAL_PRACTICE_CONTRAST,
    TUTORIAL_QUIZ_CONTRAST,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Side, Target
from src.experiment.calibration import CalibrationRounds
from src.experiment.geometry import Geometry
from src.experiment.pausable_clock import PausableClock
from src.experiment.space_input import SpaceTally
from src.experiment.trial_mechanics import OnsetDetector, RollingReactionTime, line_position, sample_capture_time
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


_CALIBRATION_PHASES = (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT)


class TutorialSession:
    """A one-time, narrated on-ramp for first-time participants, run before both real phases. Stages:

    1. Calibration - the same 3-round discard-first/average procedure as ExperimentSession (CalibrationRounds); the result
       is exposed as `calibration_ratios` for the saccade phase to reuse.
    2. QUIZ ("how to answer") - a line above or below a dot, UP/DOWN to answer, TUTORIAL_QUIZ_STREAK_TARGET correct in a row
       (a miss resets the streak).
    3. PRACTICE - TUTORIAL_PRACTICE_ATTEMPTS trials that look like real saccade trials (SPACE -> dot -> cross -> visible flash
       -> key). The flash fires at the rolling robust average of this stage's own reaction times
       (DEFAULT_REACTION_TIME_MS until there are 3); each trial's onset latency is kept in `rt_samples_ms` to seed
       ExperimentSession's average instead of a separate RT test.

    Framework-agnostic like the other sessions; never logs to the results CSV (none of this counts as data).
    """

    def __init__(self, gaze_source: GazeSource, clock: PausableClock, geometry: Geometry | None = None) -> None:
        self._gaze = gaze_source
        self._geometry = geometry or Geometry.default()
        self._clock = clock
        self._phase = Phase.WAITING_TO_START

        # -- calibration --
        self._calibration = CalibrationRounds(gaze_source)
        self.space_tally = SpaceTally(lambda: self._phase.name)
        self._calibration_ratios: tuple[float, float, float] | None = None

        self._dot_visible = False
        self._cross_visible = False
        self._foreperiod_start = 0.0
        self._foreperiod_duration = 0.0
        self._target_landed_since: float | None = None

        # -- quiz --
        self._quiz_streak = 0
        self._quiz_orientation: Side | None = None

        # -- practice trials --
        self._attempt = 0
        self._rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS)
        self._source = Target.DOT
        self._target = Target.CROSS
        self._orientation: Side | None = None
        self._onset_detector: OnsetDetector = OnsetDetector(_zone_for(self._source))
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation: Side | None = None
        self._landed = False
        self._trial_start = 0.0

    @property
    def calibration_ratios(self) -> tuple[float, float, float] | None:
        return self._calibration_ratios

    @property
    def rt_samples_ms(self) -> list[float]:
        """Saccade-onset latency (target appears -> eye leaves the source) of every practice trial that had one."""
        return self._rt.samples

    # -- key handling ------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            self._calibration.start()
            self._phase = Phase.CALIBRATE_LEFT
        elif self._phase in _CALIBRATION_PHASES:
            self._on_calibration(self._calibration.press(self._clock.now(), self.space_tally))
        elif self._phase is Phase.PRACTICE_AWAITING_SPACE:
            self._start_foreperiod()
        elif self._phase is not Phase.COMPLETE:
            # QUIZ/PRACTICE_ACTIVE respond to on_response_key, not space; PRACTICE_FOREPERIOD advances via tick().
            # COMPLETE: the runner decides when to move on.
            self.space_tally.record("not accepted in this phase")

    def _on_calibration(self, final_ratios: tuple[float, float, float] | None) -> None:
        if final_ratios is not None:
            self._calibration_ratios = final_ratios
            self._enter_quiz()
        elif self._phase in _CALIBRATION_PHASES:
            self._phase = _CALIBRATION_PHASES[self._calibration.step]

    def on_response_key(self, orientation: Side) -> None:
        if self._phase is Phase.QUIZ:
            if orientation == self._quiz_orientation:
                self._quiz_streak += 1
                if self._quiz_streak >= TUTORIAL_QUIZ_STREAK_TARGET:
                    self._enter_practice()
                else:
                    self._quiz_orientation = _random_orientation()
            else:
                self._quiz_streak = 0
                self._quiz_orientation = _random_orientation()
        elif self._phase is Phase.PRACTICE_ACTIVE:
            if self._responded or not self._response_window_open:
                return
            now = self._clock.now()
            if now <= self._response_deadline:
                self._responded = True
                self._response_orientation = orientation

    def _enter_quiz(self) -> None:
        self._quiz_streak = 0
        self._quiz_orientation = _random_orientation()
        self._phase = Phase.QUIZ

    # -- practice trials -----------------------------------------------------------

    def _enter_practice(self) -> None:
        self._attempt = 0
        self._rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS)
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
            # Open-loop, like the real saccade phase: scheduled off the rolling average of the reaction
            # times measured so far (updated at every detected eye movement; the default until there are 3).
            self._flash_scheduled_at = self._trial_start + self._rt.average / 1000

    def _sample_time(self, sample, now: float) -> float:
        """Capture time of `sample` on the session clock (same rule as ExperimentSession): minus the tracker's latency unless
        the source already stamps capture time."""
        latency_s = 0.0
        if not getattr(self._gaze, "timestamps_are_capture_time", False):
            latency_ms = getattr(self._gaze, "gaze_latency_ms", None)
            if latency_ms is not None and latency_ms == latency_ms:
                latency_s = min(max(float(latency_ms), 0.0), GAZE_LATENCY_MAX_MS) / 1000
        return sample_capture_time(sample, now, latency_s, self._clock.from_perf_counter)

    def _tick_active(self) -> None:
        now = self._clock.now()
        sample = self._gaze.latest_sample()
        target_zone = _zone_for(self._target)
        sample_t = self._sample_time(sample, now)

        if not self._onset_detector.confirmed and sample_t >= self._trial_start:
            if self._onset_detector.update(sample, sample_t):
                self._rt.add((self._onset_detector.since - self._trial_start) * 1000)

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
                self._finish_attempt()
                return

        if self._landed and self._flash_decided and (self._responded or not self._response_window_open):
            self._finish_attempt()

    def _finish_attempt(self) -> None:
        self._grating_visible = False
        self._source = self._target
        self._attempt += 1
        if self._attempt >= TUTORIAL_PRACTICE_ATTEMPTS:
            self._phase = Phase.COMPLETE
        else:
            self._begin_attempt()

    # -- tick ----------------------------------------------------------------

    def tick(self) -> None:
        self._on_calibration(self._calibration.poll())
        if self._phase is Phase.PRACTICE_FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.PRACTICE_ACTIVE:
            self._tick_active()

    # -- rendering -------------------------------------------------------------

    _QUIZ_TEXT = "Press UP if the line is above the dot, DOWN if it is below."
    _PRACTICE_INTRO = "Press SPACE. Look at the dot, then the other dot. Answer UP or DOWN."

    def _instructions(self) -> str:
        if self._phase is Phase.WAITING_TO_START:
            return "Press SPACE to begin"
        if self._phase in (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT):
            # Only round 1 is narrated: later rounds would just re-speak the same drill.
            if self._calibration.round > 1:
                return ""
            return {
                Phase.CALIBRATE_LEFT: "Look at the dot, then press SPACE.",
                Phase.CALIBRATE_CENTER: "Look at the middle, then press SPACE.",
                Phase.CALIBRATE_RIGHT: "Look at the other dot, then press SPACE.",
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
        if self._phase is Phase.QUIZ:
            return True, False, False  # the line is shown just above/below the dot
        if self._phase in (Phase.PRACTICE_AWAITING_SPACE, Phase.PRACTICE_FOREPERIOD, Phase.PRACTICE_ACTIVE):
            return self._dot_visible, self._cross_visible, False
        return False, False, False

    def _grating_state(self) -> dict:
        """The flashed line (key kept as "grating"): in the quiz 2 deg above/below the dot, in a practice trial
        above/below the symbol the eye is moving to."""
        if self._phase is Phase.QUIZ:
            x, y = line_position(DOT_POSITION, self._quiz_orientation or Side.ABOVE, self._geometry)
            return {"visible": True, "x": x, "y": y, "contrast": TUTORIAL_QUIZ_CONTRAST, "orientation": self._quiz_orientation}
        target_position = DOT_POSITION if self._target is Target.DOT else CROSS_POSITION
        x, y = line_position(target_position, self._orientation or Side.ABOVE, self._geometry)
        if self._phase is Phase.PRACTICE_ACTIVE:
            return {
                "visible": self._grating_visible,
                "x": x,
                "y": y,
                "contrast": TUTORIAL_PRACTICE_CONTRAST if self._grating_visible else 0,
                "orientation": self._orientation,
            }
        return {"visible": False, "x": x, "y": y, "contrast": 0, "orientation": None}

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
                "calibration_wait_s": self._calibration.latch.waiting_s(self._clock.now()),
            },
        }
