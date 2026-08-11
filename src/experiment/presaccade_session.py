import math
import random
from enum import Enum, auto

from include.experiment.constants import (
    CENTER_POSITION,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GRATING_DURATION_FRAMES,
    GRATING_POSITION,
    NUM_PRACTICE_TRIALS_REAL,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_TRIALS_PER_PHASE_REAL,
    NUM_TRIALS_PER_PHASE_TEST,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
)
from include.experiment.types import Orientation, TrialResult
from src.experiment.logger import ResultLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import score_outcome
from src.experiment.trial_factory import build_presaccade_sequence
from src.experiment.zest import ZestStaircase


class Phase(Enum):
    WAITING_TO_START = auto()
    FOREPERIOD = auto()  # fixating center, waiting out the foreperiod before the flash-or-not moment
    FLASH_WINDOW = auto()  # flash (or catch) has fired; response window open
    COMPLETE = auto()


class PresaccadeSession:
    """Baseline (non-saccade) contrast detection: fixate the center dot the
    whole time; after a foreperiod, a grating flashes (or not, on catch
    trials); score a yes/no button response. No eye tracking involved at all -
    this measures detection accuracy without any saccade in the mix, for
    comparison against the saccade-phase results. Contrast is picked trial by
    trial by its own ZEST staircase (see src.experiment.zest), run
    independently from the saccade phase's staircase so each condition
    converges on its own threshold. A few practice trials (fixed, easily-visible
    contrast) run first and are excluded from the staircase and logged results.

    Framework-agnostic like ExperimentSession: `on_space()`/`tick()` are called
    synchronously by the PsychoPy frame loop, `render_state()` returns a plain
    dict the runner applies to its stimuli.
    """

    def __init__(
        self,
        logger: ResultLogger,
        clock: PausableClock,
        contrast_floor: float | None = None,
        test_mode: bool = False,
    ) -> None:
        self._logger = logger
        self._clock = clock
        num_trials = NUM_TRIALS_PER_PHASE_TEST if test_mode else NUM_TRIALS_PER_PHASE_REAL
        num_practice = NUM_PRACTICE_TRIALS_TEST if test_mode else NUM_PRACTICE_TRIALS_REAL
        self._trials = build_presaccade_sequence(num_trials=num_trials, num_practice=num_practice)
        self._num_practice = sum(1 for t in self._trials if t.practice)
        self._trial_index = 0
        self._phase = Phase.WAITING_TO_START
        self._results: list[TrialResult] = []
        zest_kwargs = {"log_contrast_min": math.log10(contrast_floor)} if contrast_floor is not None else {}
        self._zest = ZestStaircase(**zest_kwargs)
        self._clear_trial_state()

    def _clear_trial_state(self) -> None:
        self._foreperiod_start = 0.0
        self._foreperiod_duration = 0.0
        self._flash_triggered_at = 0.0
        self._grating_shown_at: float | None = None
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._trial_contrast: float | None = None
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation: Orientation | None = None
        self._response_time_ms: float | None = None

    @property
    def results(self) -> list[TrialResult]:
        return self._results

    def _current_trial(self):
        return self._trials[self._trial_index]

    # -- key handling ----------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            self._begin_foreperiod()
        # FOREPERIOD: nothing to respond to yet. FLASH_WINDOW: responses come
        # from arrow keys (see on_response_key), not SPACE. COMPLETE: the
        # runner is responsible for advancing past it.

    def on_response_key(self, orientation: Orientation) -> None:
        """Called when an arrow key is pressed during the flash window:
        up/down report `Orientation.VERTICAL`, left/right report
        `Orientation.HORIZONTAL` - see run_experiment.py's key dispatch."""
        if self._phase is not Phase.FLASH_WINDOW:
            return
        if self._responded or not self._response_window_open:
            return
        now = self._clock.now()
        if now <= self._response_deadline:
            self._responded = True
            self._response_orientation = orientation
            self._response_time_ms = (now - self._grating_shown_at) * 1000 if self._grating_shown_at else None

    # -- trial state machine ----------------------------------------------------

    def _begin_foreperiod(self) -> None:
        self._clear_trial_state()
        self._phase = Phase.FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000

    def tick(self) -> None:
        if self._phase is Phase.FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.FLASH_WINDOW:
            self._tick_flash_window()

    def _tick_foreperiod(self) -> None:
        now = self._clock.now()
        if now - self._foreperiod_start >= self._foreperiod_duration:
            trial = self._current_trial()
            self._flash_triggered_at = now
            self._response_window_open = True
            self._response_deadline = now + RESPONSE_WINDOW_MS / 1000
            if trial.grating_shown:
                self._trial_contrast = PRACTICE_CONTRAST if trial.practice else self._zest.next_contrast()
                self._grating_shown_at = now
                self._grating_visible = True
                self._grating_frames_remaining = GRATING_DURATION_FRAMES
            self._phase = Phase.FLASH_WINDOW

    def _tick_flash_window(self) -> None:
        now = self._clock.now()

        if self._grating_visible:
            self._grating_frames_remaining -= 1
            if self._grating_frames_remaining <= 0:
                self._grating_visible = False

        if self._response_window_open and now > self._response_deadline:
            self._response_window_open = False

        if not self._response_window_open:
            self._finish_trial()

    def _finish_trial(self) -> None:
        trial = self._current_trial()
        outcome = score_outcome(trial.grating_shown, self._responded, self._response_orientation, trial.orientation)

        if not trial.practice:
            # "miss" (no response within the window) counts as a non-detection
            # here too, not just "correct"/"incorrect" - see session.py's
            # identical change for why, and results_graph.py's replay, which
            # already treated misses this way when rebuilding the staircase
            # from the logged CSV.
            if trial.grating_shown and outcome in ("correct", "incorrect", "miss"):
                self._zest.update(self._trial_contrast, detected=outcome == "correct")

            result = TrialResult(
                index=trial.index,
                phase="presaccade",
                source=None,
                target=None,
                saccade_duration_ms=None,
                grating_shown=trial.grating_shown,
                contrast=self._trial_contrast,
                orientation=trial.orientation if trial.grating_shown else None,
                responded=self._responded,
                response_orientation=self._response_orientation,
                response_time_ms=self._response_time_ms,
                outcome=outcome,
            )
            self._logger.log(result)
            self._results.append(result)

        self._trial_index += 1
        if self._trial_index >= len(self._trials):
            self._phase = Phase.COMPLETE
        else:
            self._begin_foreperiod()

    # -- rendering ----------------------------------------------------------

    def _instructions(self) -> str:
        if self._phase is Phase.COMPLETE:
            return "Baseline done — starting the saccade test next"
        if self._phase is Phase.WAITING_TO_START:
            return "Fixate the center of the screen. Press SPACE to begin"
        prefix = "Practice (doesn't count) — " if self._current_trial().practice else ""
        return f"{prefix}UP/DOWN if the grating is vertical, LEFT/RIGHT if horizontal — not sure? Guess"

    def _trial_label(self) -> str:
        real_total = len(self._trials) - self._num_practice
        if self._phase is Phase.COMPLETE:
            return f"{real_total}/{real_total}"
        trial = self._current_trial()
        if trial.practice:
            return f"practice {self._trial_index + 1}/{self._num_practice}"
        real_index = self._trial_index - self._num_practice
        return f"{real_index + 1}/{real_total}"

    def render_state(self) -> dict:
        # No fixation dot: it sits at the same position as the flash (both
        # center), so a bright opaque dot there would either mask the flash or
        # itself be a huge, confounding contrast step - neither is a clean
        # contrast-detection trial. The participant just holds gaze on center
        # without a marker.
        trial = self._current_trial() if self._phase is not Phase.COMPLETE and self._trials else None
        return {
            "instructions": self._instructions(),
            "dot": {"visible": False, "x": CENTER_POSITION[0], "y": CENTER_POSITION[1]},
            "cross": {"visible": False, "x": CENTER_POSITION[0], "y": CENTER_POSITION[1]},
            "grating": {
                "visible": self._grating_visible,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": self._trial_contrast if (self._grating_visible and self._trials) else 0,
                "orientation": trial.orientation if trial is not None else None,
            },
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": "n/a",
                "face_found": True,
                "source_available": True,
                "trial": self._trial_label(),
            },
        }
