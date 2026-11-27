"""TutorialSession: the narrated first-timer on-ramp (calibration, how to answer, practice trials). Practice trials use ExperimentSession's gaze rules (trial_legs.py);
only their flash scheduling and the absence of logging differ."""

import math
import random
from enum import Enum, auto

from include.experiment import participant_text as words
from include.experiment.constants import (
    CENTER_POSITION,
    CROSS_POSITION,
    DEFAULT_REACTION_TIME_MS,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    LINE_DURATION_FRAMES,
    RETURN_JUMP_TRIGGER,
    RETURN_LEG_FLASH_FRACTION,
    TUTORIAL_PRACTICE_ATTEMPTS,
    TUTORIAL_PRACTICE_CONTRAST,
    TUTORIAL_QUIZ_CONTRAST,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Side, Target
from include.eye_tracking.interfaces import GazeSource
from src.experiment.calibration import CalibrationRounds, calibration_text
from src.experiment.geometry import Geometry
from src.experiment.pausable_clock import PausableClock
from src.experiment.space_input import SpaceTally
from src.experiment.trial_legs import SampleFeed, TrialLegs, clamped_gaze_latency_ms, timestamp_latency_s
from src.experiment.trial_mechanics import RollingReactionTime, ShuffledBag, flash_leg_pool, line_position
from src.experiment.trial_mechanics import random_side as _random_side
from src.experiment.trial_mechanics import zone_for as _zone_for
from src.timebase import ms_to_s


class Phase(Enum):
    WAITING_TO_START = auto()
    INTRO = auto()  # a few narrated screens (words.TUTORIAL_INTRO), SPACE advances
    CALIBRATE_LEFT = auto()
    CALIBRATE_CENTER = auto()
    CALIBRATE_RIGHT = auto()
    QUIZ = auto()  # "how to answer": mixed line sides, streak-gated
    PRACTICE_INTRO = auto()  # "practice lines are clear, real ones fainter"; SPACE starts the first attempt
    PRACTICE_AWAITING_SPACE = auto()  # each attempt waits for SPACE; the foreperiod (500-1000 ms) starts on the press
    PRACTICE_FOREPERIOD = auto()
    PRACTICE_ACTIVE = auto()
    COMPLETE = auto()


_CALIBRATION_PHASES = (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT)


class TutorialSession:
    """A one-time, narrated on-ramp for first-time participants, run before the saccade phase. Stages:

    1. Calibration - the same 3-round discard-first/average procedure as ExperimentSession (CalibrationRounds); the result
       is exposed as `calibration_ratios` for the saccade phase to reuse.
    2. QUIZ ("how to answer") - a line above or below a dot, UP/DOWN to answer, TUTORIAL_QUIZ_STREAK_TARGET correct in a row
       (a miss resets the streak).
    3. PRACTICE - TUTORIAL_PRACTICE_ATTEMPTS trials that look like real saccade trials (SPACE -> dot -> cross -> visible flash
       -> key). The flash fires at the rolling robust average of this stage's own reaction times
       (DEFAULT_REACTION_TIME_MS until there are 3); each trial's onset latency is kept in `rt_samples_ms` to seed
       ExperimentSession's average instead of a separate RT test. With `return_jump` the practice trials also jump back
       (like the real ones): the flash goes on the out or the return leg, and the return latencies
       (`return_rt_samples_ms`) seed the saccade phase's return-latency average.

    A short narrated INTRO (what you will see, what to do) runs before the calibration. There is no feedback on answers.

    Framework-agnostic like the other sessions; never logs to the results CSV (none of this counts as data).
    """

    def __init__(
        self, gaze_source: GazeSource, clock: PausableClock, geometry: Geometry | None = None, return_jump: bool = False
    ) -> None:
        self._gaze = gaze_source
        self._return_jump = bool(return_jump)
        intro = words.TUTORIAL_INTRO
        self._intro_lines: tuple[str, ...] = (
            intro[:1] + (words.TUTORIAL_INTRO_RETURN_JUMP,) + intro[1:] if self._return_jump else intro
        )
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
        self._foreperiod_duration_s = 0.0

        # -- quiz --
        self._quiz_streak = 0
        self._quiz_side: Side | None = None
        self._intro_step = 0

        # -- practice trials --
        self._attempt = 0
        self._rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS)
        self._source = Target.DOT
        self._target = Target.CROSS
        self._line_side: Side | None = None
        # Gaze bookkeeping is shared with ExperimentSession (trial_legs): every sample since the last tick, capture-time
        # landing, the same leg rules. Created at the go cue of each attempt.
        self._feed = SampleFeed(gaze_source, clock)
        self._legs: TrialLegs | None = None
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._line_visible = False
        self._line_frames_remaining = 0
        self._line_actually_shown = False
        self._gaze_record_closed = False  # as in ExperimentSession: only the answer / SPACE is still awaited
        self._responded = False
        self._answer_side: Side | None = None
        self._space_ahead = False  # as in ExperimentSession: ends the running attempt, then starts the next one
        self._trial_start = 0.0
        # -- return jump (the same TrialLegs rules as ExperimentSession; read at construction like there) --
        self._return_jump_trigger = RETURN_JUMP_TRIGGER
        self._return_rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS, min_plausible_ms=None)  # signed, keeps early returns
        self._leg_bag: ShuffledBag[str] | None = (
            ShuffledBag(flash_leg_pool(RETURN_LEG_FLASH_FRACTION)) if self._return_jump else None
        )
        self._flash_leg = "out"

    @property
    def _active_legs(self) -> TrialLegs:
        """The running attempt's legs; only valid between the go cue and the attempt's end."""
        assert self._legs is not None, "no attempt running"
        return self._legs

    # Read-only views of the current attempt's legs (tests and the HUD read these).
    @property
    def _landed(self) -> bool:
        return self._legs is not None and self._legs.out_landing.landed

    @property
    def _jump_at(self) -> float | None:
        return self._legs.jump_at if self._legs is not None else None

    @property
    def _out_arrived(self) -> bool:
        return self._legs is not None and self._legs.out_arrived_at is not None

    @property
    def calibration_ratios(self) -> tuple[float, float, float] | None:
        return self._calibration_ratios

    @property
    def rt_samples_ms(self) -> list[float]:
        """Saccade-onset latency (target appears -> eye leaves the source) of every practice trial that had one."""
        return self._rt.samples

    @property
    def return_rt_samples_ms(self) -> list[float]:
        """Return-saccade latency (jump back -> eye leaves the target) of every practice trial that had one (not forced)."""
        return self._return_rt.samples

    # -- key handling ------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            self._intro_step = 0
            self._phase = Phase.INTRO
        elif self._phase is Phase.INTRO:
            self._intro_step += 1
            if self._intro_step >= len(self._intro_lines):
                self._calibration.start()
                self._phase = Phase.CALIBRATE_LEFT
        elif self._phase is Phase.PRACTICE_INTRO:
            self._begin_attempt()
        elif self._phase in _CALIBRATION_PHASES:
            self._on_calibration(self._calibration.press(self._clock.now(), self.space_tally))
        elif self._phase is Phase.PRACTICE_AWAITING_SPACE:
            self._start_foreperiod()
        elif self._phase is Phase.PRACTICE_ACTIVE and (self._responded or self._only_answer_pending()):
            self._space_ahead = True  # the real trials' rule (ExperimentSession.on_space)
        elif self._phase is not Phase.COMPLETE:
            # QUIZ responds to on_response_key, not space; a running attempt cannot be skipped; PRACTICE_FOREPERIOD
            # advances via tick(). COMPLETE: the runner decides when to move on.
            self.space_tally.record("during trial" if self._phase is Phase.PRACTICE_ACTIVE else "not accepted in this phase")

    def _on_calibration(self, final_ratios: tuple[float, float, float] | None) -> None:
        if final_ratios is not None:
            self._calibration_ratios = final_ratios
            self._enter_quiz()
        elif self._phase in _CALIBRATION_PHASES:
            self._phase = _CALIBRATION_PHASES[self._calibration.step]

    def on_response_key(self, answer: Side) -> None:
        if self._phase is Phase.QUIZ:
            correct = answer == self._quiz_side
            if correct:
                self._quiz_streak += 1
                if self._quiz_streak >= TUTORIAL_QUIZ_STREAK_TARGET:
                    self._enter_practice()
                else:
                    self._quiz_side = _random_side()
            else:
                self._quiz_streak = 0
                self._quiz_side = _random_side()
        elif self._phase is Phase.PRACTICE_ACTIVE and self._flash_decided and not self._responded:
            self._responded = True
            self._answer_side = answer

    def _enter_quiz(self) -> None:
        self._quiz_streak = 0
        self._quiz_side = _random_side()
        self._phase = Phase.QUIZ

    # -- practice trials -----------------------------------------------------------

    def _enter_practice(self) -> None:
        self._attempt = 0
        self._rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS)
        self._return_rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS, min_plausible_ms=None)
        self._source = Target.DOT
        self._dot_visible = True
        self._cross_visible = False
        self._phase = Phase.PRACTICE_INTRO

    def _begin_attempt(self) -> None:
        self._phase = Phase.PRACTICE_AWAITING_SPACE
        self._target = Target.CROSS if self._source is Target.DOT else Target.DOT
        self._line_side = _random_side()
        self._legs = None
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._line_visible = False
        self._line_frames_remaining = 0
        self._line_actually_shown = False
        self._gaze_record_closed = False
        self._responded = False
        self._answer_side = None
        self._flash_leg = self._leg_bag.draw() if self._leg_bag is not None else "out"

    def _start_foreperiod(self) -> None:
        self._phase = Phase.PRACTICE_FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration_s = ms_to_s(random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS))

    def _tick_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration_s:
            self._dot_visible = self._target is Target.DOT
            self._cross_visible = self._target is Target.CROSS
            self._phase = Phase.PRACTICE_ACTIVE
            self._trial_start = self._clock.now()
            # Same go-cue bookkeeping as ExperimentSession._reveal_target_and_start_trial: one latency snapshot per trial,
            # samples counted from the go cue.
            self._feed.latency_s = timestamp_latency_s(self._gaze, clamped_gaze_latency_ms(self._gaze))
            self._feed.last_sample_ts = self._gaze.latest_sample().timestamp
            self._legs = TrialLegs(
                _zone_for(self._source), _zone_for(self._target), self._trial_start, trigger=self._return_jump_trigger
            )
            # Open-loop, like the real saccade phase: scheduled off the rolling average of the reaction
            # times measured so far (updated at every detected eye movement; the default until there are 3).
            self._flash_scheduled_at = self._trial_start + ms_to_s(self._rt.average)
            if self._flash_leg == "return":
                self._flash_scheduled_at = math.inf  # scheduled at the jump back

    def _tick_active(self) -> None:
        now = self._clock.now()
        legs = self._active_legs
        new_samples = self._feed.new_samples(self._gaze.latest_sample())

        if not self._gaze_record_closed:
            measured = legs.measure_out_onset(new_samples)
            if measured is not None:
                self._rt.add(measured[0])

        flash_fired_this_tick = False
        if not self._flash_decided and now >= self._flash_scheduled_at:
            self._flash_decided = True
            flash_fired_this_tick = True
            self._line_visible = True
            self._line_frames_remaining = LINE_DURATION_FRAMES
            self._line_actually_shown = True

        if self._line_visible and not flash_fired_this_tick:
            self._line_frames_remaining -= 1
            if self._line_frames_remaining <= 0:
                self._line_visible = False

        if not self._gaze_record_closed:
            if self._return_jump:
                self._tick_return_jump(new_samples, now)
            else:
                legs.track_out_landing(new_samples)
        self._finish_if_over(now)

    def _finish_if_over(self, now: float) -> None:
        """Same rule as ExperimentSession._finish_if_over: no answer deadline."""
        legs = self._active_legs
        if legs.timed_out(now, self._return_jump) and not (legs.leg_moved(self._flash_leg) and self._line_actually_shown):
            self._finish_attempt()
            return
        self._gaze_record_closed = self._flash_decided and legs.movement_over(now, self._return_jump)
        if self._only_answer_pending() and (self._responded or self._space_ahead):
            self._finish_attempt()

    def _only_answer_pending(self) -> bool:
        return self._gaze_record_closed and not self._line_visible

    def _deliver_space_ahead(self) -> None:
        if not self._space_ahead or self._phase is Phase.PRACTICE_ACTIVE:
            return
        self._space_ahead = False
        if self._phase is Phase.PRACTICE_AWAITING_SPACE:
            self._start_foreperiod()
        elif self._responded:  # never skips the "practice done" screen; unanswered, the press ended the last attempt
            self.space_tally.record("early press, no trial left to start")

    def _tick_return_jump(self, new_samples, now: float) -> None:
        """Same steps as ExperimentSession._tick_return_jump (shared TrialLegs rules); only the logging differs."""
        legs = self._active_legs
        legs.track_arrival(new_samples)
        out_flash_done = self._flash_leg == "return" or (self._flash_decided and not self._line_visible)
        forced = legs.jump_due(now, out_flash_done)
        if forced is not None:
            self._jump_back(now, forced=forced)

        return_latency_ms = legs.track_return(new_samples)
        if return_latency_ms is not None and not legs.jump_forced:
            self._return_rt.add(return_latency_ms)

    def _jump_back(self, now: float, forced: bool) -> None:
        self._active_legs.mark_jump(now, forced)
        self._dot_visible = self._source is Target.DOT
        self._cross_visible = self._source is Target.CROSS
        if self._flash_leg == "return":
            average_ms = self._return_rt.average if self._return_rt.has_estimate else self._rt.average
            self._flash_scheduled_at = now + ms_to_s(max(0.0, average_ms))  # early returns: at the jump frame at the earliest

    def _finish_attempt(self) -> None:
        self._line_visible = False
        if not self._return_jump:  # with return jump the eye is back on the start symbol: every trial starts there
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
        self._deliver_space_ahead()

    # -- rendering -------------------------------------------------------------

    def _instructions(self) -> str:
        if self._phase is Phase.WAITING_TO_START:
            return words.TUTORIAL_WELCOME
        if self._phase is Phase.INTRO:
            return self._intro_lines[self._intro_step]
        if self._phase in (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT):
            return calibration_text(self._calibration.step, self._calibration.round)
        if self._phase is Phase.QUIZ:
            return words.QUIZ
        if self._phase is Phase.PRACTICE_INTRO:
            return words.PRACTICE_INTRO
        if self._phase in (Phase.PRACTICE_AWAITING_SPACE, Phase.PRACTICE_FOREPERIOD, Phase.PRACTICE_ACTIVE):
            # Said once, before the first trial; silent afterwards (the narrator re-speaks on every text change).
            if self._attempt > 0:
                return ""
            return words.PRACTICE_START_RETURN_JUMP if self._return_jump else words.RT_TEST_INTRO
        return words.PRACTICE_DONE

    def _tutorial_text(self) -> tuple[str | None, str]:
        """(text, position) for the larger instructional text stim shown only
        while the participant is learning the answer keys; everywhere else the
        caption (see run_experiment.py's render_caption) carries the words."""
        if self._phase is Phase.QUIZ:
            return words.QUIZ, "bottom"
        if self._phase is Phase.WAITING_TO_START:
            return words.TUTORIAL_WELCOME, "center"
        if self._phase is Phase.INTRO:
            return self._intro_lines[self._intro_step], "center"
        if self._phase is Phase.PRACTICE_INTRO:
            return words.PRACTICE_INTRO, "center"
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
        if self._phase in (Phase.PRACTICE_INTRO, Phase.PRACTICE_AWAITING_SPACE, Phase.PRACTICE_FOREPERIOD, Phase.PRACTICE_ACTIVE):
            return self._dot_visible, self._cross_visible, False
        return False, False, False

    def _line_state(self) -> dict:
        """The flashed line: in the quiz 2 deg above/below the dot, in a practice trial
        above/below the symbol the eye is moving to."""
        if self._phase is Phase.QUIZ:
            x, y = line_position(DOT_POSITION, self._quiz_side or Side.ABOVE, self._geometry)
            return {"visible": True, "x": x, "y": y, "contrast": TUTORIAL_QUIZ_CONTRAST, "side": self._quiz_side}
        aim = self._source if self._flash_leg == "return" else self._target  # the target of the flash's leg
        target_position = DOT_POSITION if aim is Target.DOT else CROSS_POSITION
        x, y = line_position(target_position, self._line_side or Side.ABOVE, self._geometry)
        if self._phase is Phase.PRACTICE_ACTIVE:
            return {
                "visible": self._line_visible,
                "x": x,
                "y": y,
                "contrast": TUTORIAL_PRACTICE_CONTRAST if self._line_visible else 0,
                "side": self._line_side,
            }
        return {"visible": False, "x": x, "y": y, "contrast": 0, "side": None}

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
            "line": self._line_state(),
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
