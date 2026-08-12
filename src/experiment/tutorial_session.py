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
    GRATING_DURATION_FRAMES,
    GRATING_POSITION,
    RESPONSE_WINDOW_MS,
    SACCADE_TIMEOUT_MS,
    TUTORIAL_DEMO_CONTRAST,
    TUTORIAL_DRESS_REHEARSAL_ATTEMPTS,
    TUTORIAL_GAZE_PRACTICE_ATTEMPTS,
    TUTORIAL_QUIZ_CONTRAST,
    TUTORIAL_QUIZ_STREAK_TARGET,
    TUTORIAL_RT_TEST_ATTEMPTS,
)
from include.experiment.types import Orientation, Target
from src.experiment.calibration import average_calibration_rounds
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import score_outcome
from src.experiment.trial_mechanics import OnsetDetector, average_ms
from src.experiment.trial_mechanics import random_orientation as _random_orientation
from src.experiment.trial_mechanics import zone_for as _zone_for
from src.experiment.zest import ZestStaircase


class Phase(Enum):
    WAITING_TO_START = auto()
    CALIBRATE_LEFT = auto()
    CALIBRATE_CENTER = auto()
    CALIBRATE_RIGHT = auto()
    DEMO_VERTICAL = auto()
    DEMO_HORIZONTAL = auto()
    QUIZ = auto()
    GAZE_PRACTICE_FOREPERIOD = auto()
    GAZE_PRACTICE_ACTIVE = auto()
    RT_TEST_FOREPERIOD = auto()
    RT_TEST_ACTIVE = auto()
    DRESS_FOREPERIOD = auto()
    DRESS_ACTIVE = auto()
    COMPLETE = auto()


class TutorialSession:
    """A one-time, narrated on-ramp for first-time participants, run before
    both real phases (see run_experiment.py's main()). Stages, in order:

    1. Calibration - identical 3-round discard-first/average procedure as
       ExperimentSession's own self-calibration (see CALIBRATION_ROUNDS),
       so a participant who takes the tutorial gets the exact same quality
       calibration as one who doesn't. The result is exposed via
       `calibration_ratios` for the real saccade phase to reuse, skipping
       its own calibration entirely.
    2. DEMO_VERTICAL / DEMO_HORIZONTAL - one static, high-contrast grating
       of each orientation, teaching the up/down vs left/right response
       mapping. Any response key advances - these aren't scored.
    3. QUIZ - mixed-orientation, fixed-contrast, streak-gated: 5 correct in
       a row to pass, a miss resets the streak. The only stage in this
       tutorial gated by success rather than a fixed attempt count.
    4. GAZE_PRACTICE - alternating dot/cross fixation, no grating, fixed at
       TUTORIAL_GAZE_PRACTICE_ATTEMPTS attempts regardless of outcome
       (early tracking is noisy - a streak gate here could trap a
       participant with imperfect-but-workable calibration).
    5. RT_TEST - a scoped-down version of ExperimentSession's own
       reaction-time test (see its module for the full design rationale):
       TUTORIAL_RT_TEST_ATTEMPTS beep+target-appear -> saccade-onset
       attempts, no grating, averaged into an avg_reaction_time_ms used only
       by this tutorial's own dress rehearsal below - never handed off to
       the real saccade phase, which always measures its own.
    6. DRESS_REHEARSAL - the real trial mechanics (dot/cross alternation +
       grating + 2AFC scoring) against a throwaway ZEST staircase, fixed at
       TUTORIAL_DRESS_REHEARSAL_ATTEMPTS attempts; a miss resets the
       throwaway staircase (contrast back to the top of its range) so the
       lesson "you need to actually be careful" lands, without an
       open-ended retry loop. Like the real saccade phase, the grating fires
       at a scheduled delay (avg_reaction_time_ms from this stage's own
       RT_TEST, no timing offset - a demo doesn't need TIMING_OFFSETS_MS
       coverage) rather than being triggered by detected saccade onset.
       Replaces the saccade phase's own built-in practice trials (see
       ExperimentSession's skip_practice_trials).

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
        self._pending_center_ratio: float | None = None
        self._calibration_ratios: tuple[float, float, float] | None = None

        # -- shared fixation-symbol state (gaze practice + dress rehearsal) --
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

        # -- gaze practice --
        self._gaze_practice_attempt = 0
        self._gaze_practice_source = Target.DOT
        self._gaze_practice_target = Target.CROSS
        self._practice_active_start: float | None = None

        # -- reaction-time test (see ExperimentSession for the full design) --
        self._rt_test_attempt = 0
        self._rt_samples_ms: list[float] = []
        self._rt_test_onset_detector: OnsetDetector = OnsetDetector(_zone_for(Target.DOT))
        self._rt_test_start_time = 0.0
        self._avg_reaction_time_ms: float | None = None

        # -- dress rehearsal --
        self._dress_attempt = 0
        self._dress_zest = ZestStaircase()
        self._dress_source = Target.DOT
        self._dress_target = Target.CROSS
        self._dress_orientation: Orientation | None = None
        self._dress_contrast: float | None = None
        self._onset_detector: OnsetDetector = OnsetDetector(_zone_for(self._dress_source))
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._grating_shown_at: float | None = None
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation: Orientation | None = None
        self._landed = False
        self._dress_trial_start = 0.0

    @property
    def calibration_ratios(self) -> tuple[float, float, float] | None:
        return self._calibration_ratios

    # -- key handling ------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            self._gaze.begin_calibration_sample()
            self._calibration_round = 1
            self._calibration_rounds_data = []
            self._phase = Phase.CALIBRATE_LEFT
        elif self._phase is Phase.CALIBRATE_LEFT:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None:
                self._pending_left_ratio = ratio
                self._gaze.begin_calibration_sample()
                self._phase = Phase.CALIBRATE_CENTER
        elif self._phase is Phase.CALIBRATE_CENTER:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None:
                self._pending_center_ratio = ratio
                self._gaze.begin_calibration_sample()
                self._phase = Phase.CALIBRATE_RIGHT
        elif self._phase is Phase.CALIBRATE_RIGHT:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None and self._pending_left_ratio is not None and self._pending_center_ratio is not None:
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
                    self._phase = Phase.DEMO_VERTICAL
        # DEMO_*/QUIZ/DRESS_ACTIVE respond to on_response_key, not space.
        # GAZE_PRACTICE_*/DRESS_FOREPERIOD advance automatically via tick().
        # COMPLETE: space does nothing here - the runner decides when to move on.

    def on_response_key(self, orientation: Orientation) -> None:
        if self._phase is Phase.DEMO_VERTICAL:
            self._phase = Phase.DEMO_HORIZONTAL
        elif self._phase is Phase.DEMO_HORIZONTAL:
            self._enter_quiz()
        elif self._phase is Phase.QUIZ:
            if orientation == self._quiz_orientation:
                self._quiz_streak += 1
                if self._quiz_streak >= TUTORIAL_QUIZ_STREAK_TARGET:
                    self._set_feedback("green")
                    self._enter_gaze_practice()
                else:
                    self._quiz_orientation = _random_orientation()
            else:
                self._quiz_streak = 0
                self._set_feedback("red")
                self._quiz_orientation = _random_orientation()
        elif self._phase is Phase.DRESS_ACTIVE:
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

    # -- gaze practice -------------------------------------------------------

    def _enter_gaze_practice(self) -> None:
        self._gaze_practice_attempt = 0
        self._gaze_practice_source = Target.DOT
        self._dot_visible = True
        self._cross_visible = False
        self._begin_gaze_practice_foreperiod()

    def _begin_gaze_practice_foreperiod(self) -> None:
        self._phase = Phase.GAZE_PRACTICE_FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000
        self._target_landed_since = None

    def _tick_gaze_practice_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration:
            target = Target.CROSS if self._gaze_practice_source is Target.DOT else Target.DOT
            self._gaze_practice_target = target
            if target is Target.DOT:
                self._dot_visible = True
                self._cross_visible = False
            else:
                self._dot_visible = False
                self._cross_visible = True
            self._target_landed_since = None
            self._practice_active_start = self._clock.now()
            self._phase = Phase.GAZE_PRACTICE_ACTIVE

    def _tick_gaze_practice_active(self) -> None:
        now = self._clock.now()
        sample = self._gaze.latest_sample()
        target_zone = _zone_for(self._gaze_practice_target)

        landed = False
        if sample.face_found and sample.zone is target_zone:
            if self._target_landed_since is None:
                self._target_landed_since = now
            elif now - self._target_landed_since >= GAZE_LANDING_STABILITY_MS / 1000:
                landed = True
        else:
            self._target_landed_since = None

        timed_out = now - self._practice_active_start >= SACCADE_TIMEOUT_MS / 1000
        if landed or timed_out:
            self._gaze_practice_source = self._gaze_practice_target
            self._gaze_practice_attempt += 1
            if self._gaze_practice_attempt >= TUTORIAL_GAZE_PRACTICE_ATTEMPTS:
                self._enter_rt_test()
            else:
                self._begin_gaze_practice_foreperiod()

    # -- reaction-time test ---------------------------------------------------

    def _enter_rt_test(self) -> None:
        self._rt_test_attempt = 0
        self._rt_samples_ms = []
        self._begin_rt_test_foreperiod()

    def _begin_rt_test_foreperiod(self) -> None:
        self._phase = Phase.RT_TEST_FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000
        self._rt_test_onset_detector = OnsetDetector(_zone_for(Target.DOT))
        self._dot_visible = True
        self._cross_visible = False

    def _tick_rt_test_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration:
            self._dot_visible = False
            self._cross_visible = True
            self._phase = Phase.RT_TEST_ACTIVE
            self._rt_test_start_time = self._clock.now()

    def _tick_rt_test_active(self) -> None:
        now = self._clock.now()
        sample = self._gaze.latest_sample()

        if not self._rt_test_onset_detector.confirmed and self._rt_test_onset_detector.update(sample, now):
            self._finish_rt_test_attempt((self._rt_test_onset_detector.since - self._rt_test_start_time) * 1000)
            return

        if now - self._rt_test_start_time >= SACCADE_TIMEOUT_MS / 1000:
            self._finish_rt_test_attempt(None)

    def _finish_rt_test_attempt(self, reaction_time_ms: float | None) -> None:
        if reaction_time_ms is not None:
            self._rt_samples_ms.append(reaction_time_ms)
        self._rt_test_attempt += 1
        if self._rt_test_attempt >= TUTORIAL_RT_TEST_ATTEMPTS:
            self._avg_reaction_time_ms = average_ms(self._rt_samples_ms, default=DEFAULT_REACTION_TIME_MS)
            self._enter_dress_rehearsal()
        else:
            self._begin_rt_test_foreperiod()

    # -- dress rehearsal -------------------------------------------------------

    def _enter_dress_rehearsal(self) -> None:
        self._dress_attempt = 0
        self._dress_zest = ZestStaircase()
        self._dress_source = Target.DOT
        self._dot_visible = True
        self._cross_visible = False
        self._begin_dress_foreperiod()

    def _begin_dress_foreperiod(self) -> None:
        self._phase = Phase.DRESS_FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000
        self._dress_target = Target.CROSS if self._dress_source is Target.DOT else Target.DOT
        self._dress_orientation = _random_orientation()
        self._onset_detector = OnsetDetector(_zone_for(self._dress_source))
        self._target_landed_since = None
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._grating_shown_at = None
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._dress_contrast = None
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation = None
        self._landed = False

    def _tick_dress_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration:
            if self._dress_target is Target.DOT:
                self._dot_visible = True
                self._cross_visible = False
            else:
                self._dot_visible = False
                self._cross_visible = True
            self._phase = Phase.DRESS_ACTIVE
            self._dress_trial_start = self._clock.now()
            # Open-loop, like the real saccade phase: scheduled off this
            # tutorial's own RT_TEST average, no timing offset - a demo
            # doesn't need TIMING_OFFSETS_MS coverage.
            self._flash_scheduled_at = self._dress_trial_start + self._avg_reaction_time_ms / 1000

    def _tick_dress_active(self) -> None:
        now = self._clock.now()
        sample = self._gaze.latest_sample()
        target_zone = _zone_for(self._dress_target)

        if not self._onset_detector.confirmed:
            self._onset_detector.update(sample, now)

        flash_fired_this_tick = False
        if not self._flash_decided and now >= self._flash_scheduled_at:
            self._flash_decided = True
            flash_fired_this_tick = True
            self._response_window_open = True
            self._response_deadline = now + RESPONSE_WINDOW_MS / 1000
            self._grating_shown_at = now
            self._grating_visible = True
            self._grating_frames_remaining = GRATING_DURATION_FRAMES
            self._grating_actually_shown = True
            self._dress_contrast = self._dress_zest.next_contrast()

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

            if now - self._dress_trial_start >= SACCADE_TIMEOUT_MS / 1000:
                self._finish_dress_attempt(timed_out=True)
                return

        if self._landed and self._flash_decided and (self._responded or not self._response_window_open):
            self._finish_dress_attempt(timed_out=False)

    def _finish_dress_attempt(self, timed_out: bool) -> None:
        if timed_out and not self._onset_detector.confirmed:
            outcome = "timeout"
        else:
            outcome = score_outcome(
                self._grating_actually_shown, self._responded, self._response_orientation, self._dress_orientation
            )

        correct = outcome == "correct"
        if self._grating_actually_shown:
            self._dress_zest.update(self._dress_contrast, detected=correct)
        if not correct:
            self._set_feedback("red")
            self._dress_zest = ZestStaircase()  # start over from the top of the range

        self._grating_visible = False
        self._dress_source = self._dress_target
        self._dress_attempt += 1
        if self._dress_attempt >= TUTORIAL_DRESS_REHEARSAL_ATTEMPTS:
            self._phase = Phase.COMPLETE
        else:
            self._begin_dress_foreperiod()

    # -- tick ----------------------------------------------------------------

    def tick(self) -> None:
        if self._phase is Phase.GAZE_PRACTICE_FOREPERIOD:
            self._tick_gaze_practice_foreperiod()
        elif self._phase is Phase.GAZE_PRACTICE_ACTIVE:
            self._tick_gaze_practice_active()
        elif self._phase is Phase.RT_TEST_FOREPERIOD:
            self._tick_rt_test_foreperiod()
        elif self._phase is Phase.RT_TEST_ACTIVE:
            self._tick_rt_test_active()
        elif self._phase is Phase.DRESS_FOREPERIOD:
            self._tick_dress_foreperiod()
        elif self._phase is Phase.DRESS_ACTIVE:
            self._tick_dress_active()

    # -- rendering -------------------------------------------------------------

    def _instructions(self) -> str:
        if self._phase is Phase.WAITING_TO_START:
            return "Press SPACE to begin the tutorial"
        if self._phase in (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT):
            # Only narrate round 1 of CALIBRATION_ROUNDS - by round 2 the
            # participant already knows the drill, so re-speaking it every
            # round just slows calibration down for no benefit.
            if self._calibration_round > 1:
                return ""
            return {
                Phase.CALIBRATE_LEFT: "Look at the dot, then press SPACE",
                Phase.CALIBRATE_CENTER: "Now look at the center, then press SPACE",
                Phase.CALIBRATE_RIGHT: "Now look at the other circle, then press SPACE",
            }[self._phase]
        if self._phase is Phase.DEMO_VERTICAL:
            return "This is a vertical grating. Press UP or DOWN."
        if self._phase is Phase.DEMO_HORIZONTAL:
            return "This is a horizontal grating. Press LEFT or RIGHT."
        if self._phase is Phase.QUIZ:
            return "UP/DOWN if the grating is vertical, LEFT/RIGHT if horizontal. Get 5 in a row to continue."
        if self._phase in (Phase.GAZE_PRACTICE_FOREPERIOD, Phase.GAZE_PRACTICE_ACTIVE):
            return "Follow the dots on the screen"
        if self._phase in (Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE):
            return "Measuring your reaction time — look at the dot, then look at the cross as soon as it appears"
        if self._phase in (Phase.DRESS_FOREPERIOD, Phase.DRESS_ACTIVE):
            return "Now just like the real thing - follow the dots and report the grating"
        return "Tutorial complete! Press SPACE to begin the real session"

    def _tutorial_text(self) -> tuple[str | None, str]:
        """(text, position) for the tutorial's larger, centered/bottom
        instructional text stim - only shown for stages that aren't
        gaze-contingent, so the gaze-tracking-sensitive stages aren't
        drawing attention away from the target they're teaching. This is
        separate from (and in addition to) the smaller generic caption
        every phase now gets (see run_experiment.py's render_caption)."""
        if self._phase is Phase.WAITING_TO_START:
            return "Look at the dots, then press SPACE to begin calibration", "center"
        if self._phase is Phase.DEMO_VERTICAL:
            return "This is a vertical grating - press UP or DOWN", "bottom"
        if self._phase is Phase.DEMO_HORIZONTAL:
            return "This is a horizontal grating - press LEFT or RIGHT", "bottom"
        return None, "center"

    def _symbol_visibility(self) -> tuple[bool, bool, bool]:
        if self._phase is Phase.CALIBRATE_LEFT:
            return True, False, False
        if self._phase is Phase.CALIBRATE_CENTER:
            return False, False, True
        if self._phase is Phase.CALIBRATE_RIGHT:
            return False, True, False
        if self._phase in (
            Phase.GAZE_PRACTICE_FOREPERIOD,
            Phase.GAZE_PRACTICE_ACTIVE,
            Phase.RT_TEST_FOREPERIOD,
            Phase.RT_TEST_ACTIVE,
            Phase.DRESS_FOREPERIOD,
            Phase.DRESS_ACTIVE,
        ):
            return self._dot_visible, self._cross_visible, False
        return False, False, False

    def _grating_state(self) -> dict:
        if self._phase is Phase.DEMO_VERTICAL:
            return {
                "visible": True,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": TUTORIAL_DEMO_CONTRAST,
                "orientation": Orientation.VERTICAL,
            }
        if self._phase is Phase.DEMO_HORIZONTAL:
            return {
                "visible": True,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": TUTORIAL_DEMO_CONTRAST,
                "orientation": Orientation.HORIZONTAL,
            }
        if self._phase is Phase.QUIZ:
            return {
                "visible": True,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": TUTORIAL_QUIZ_CONTRAST,
                "orientation": self._quiz_orientation,
            }
        if self._phase is Phase.DRESS_ACTIVE:
            return {
                "visible": self._grating_visible,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": self._dress_contrast if self._grating_visible else 0,
                "orientation": self._dress_orientation,
            }
        return {"visible": False, "x": GRATING_POSITION[0], "y": GRATING_POSITION[1], "contrast": 0, "orientation": None}

    def _trial_label(self) -> str:
        if self._phase is Phase.QUIZ:
            return f"quiz streak {self._quiz_streak}/{TUTORIAL_QUIZ_STREAK_TARGET}"
        if self._phase in (Phase.GAZE_PRACTICE_FOREPERIOD, Phase.GAZE_PRACTICE_ACTIVE):
            return f"gaze practice {self._gaze_practice_attempt + 1}/{TUTORIAL_GAZE_PRACTICE_ATTEMPTS}"
        if self._phase in (Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE):
            return f"RT test {self._rt_test_attempt + 1}/{TUTORIAL_RT_TEST_ATTEMPTS}"
        if self._phase in (Phase.DRESS_FOREPERIOD, Phase.DRESS_ACTIVE):
            return f"dress rehearsal {self._dress_attempt + 1}/{TUTORIAL_DRESS_REHEARSAL_ATTEMPTS}"
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
            },
        }
