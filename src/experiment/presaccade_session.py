import random
from enum import Enum, auto

from include.experiment.constants import (
    CENTER_POSITION,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    RESPONSE_WINDOW_MS,
    SQUARE_DURATION_FRAMES,
    SQUARE_POSITION,
    TRIALS_PER_PHASE,
)
from include.experiment.types import TrialResult
from src.experiment.logger import ResultLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.trial_factory import build_presaccade_sequence


class Phase(Enum):
    WAITING_TO_START = auto()
    FOREPERIOD = auto()  # fixating center, waiting out the foreperiod before the flash-or-not moment
    FLASH_WINDOW = auto()  # flash (or catch) has fired; response window open
    COMPLETE = auto()


class PresaccadeSession:
    """Baseline (non-saccade) contrast detection: fixate the center dot the
    whole time; after a foreperiod, a square flashes (or not, on catch
    trials); score a yes/no button response. No eye tracking involved at all -
    this measures detection accuracy without any saccade in the mix, for
    comparison against the saccade-phase results at the same contrast levels.

    Framework-agnostic like ExperimentSession: `on_space()`/`tick()` are called
    synchronously by the PsychoPy frame loop, `render_state()` returns a plain
    dict the runner applies to its stimuli.
    """

    def __init__(self, logger: ResultLogger, clock: PausableClock) -> None:
        self._logger = logger
        self._clock = clock
        self._trials = build_presaccade_sequence()
        self._trial_index = 0
        self._phase = Phase.WAITING_TO_START
        self._results: list[TrialResult] = []
        self._clear_trial_state()

    def _clear_trial_state(self) -> None:
        self._foreperiod_start = 0.0
        self._foreperiod_duration = 0.0
        self._flash_triggered_at = 0.0
        self._square_shown_at: float | None = None
        self._square_visible = False
        self._square_frames_remaining = 0
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
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
        elif self._phase is Phase.FLASH_WINDOW:
            self._on_response()
        # FOREPERIOD and COMPLETE: space does nothing (nothing to respond to yet
        # / the runner is responsible for advancing past COMPLETE)

    def _on_response(self) -> None:
        if self._responded or not self._response_window_open:
            return
        now = self._clock.now()
        if now <= self._response_deadline:
            self._responded = True
            self._response_time_ms = (now - self._square_shown_at) * 1000 if self._square_shown_at else None

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
            if trial.square_shown:
                self._square_shown_at = now
                self._square_visible = True
                self._square_frames_remaining = SQUARE_DURATION_FRAMES
            self._phase = Phase.FLASH_WINDOW

    def _tick_flash_window(self) -> None:
        now = self._clock.now()

        if self._square_visible:
            self._square_frames_remaining -= 1
            if self._square_frames_remaining <= 0:
                self._square_visible = False

        if self._response_window_open and now > self._response_deadline:
            self._response_window_open = False

        if not self._response_window_open:
            self._finish_trial()

    def _finish_trial(self) -> None:
        trial = self._current_trial()

        if trial.square_shown and self._responded:
            outcome = "hit"
        elif trial.square_shown and not self._responded:
            outcome = "miss"
        elif not trial.square_shown and self._responded:
            outcome = "false_alarm"
        else:
            outcome = "correct_rejection"

        result = TrialResult(
            index=trial.index,
            phase="presaccade",
            source=None,
            target=None,
            saccade_duration_ms=None,
            square_shown=trial.square_shown,
            contrast=trial.contrast,
            responded=self._responded,
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
        return {
            Phase.WAITING_TO_START: "Fixate the center of the screen. Press SPACE to begin",
            Phase.FOREPERIOD: "Press SPACE if you see the square",
            Phase.FLASH_WINDOW: "Press SPACE if you see the square",
            Phase.COMPLETE: "Baseline done — starting the saccade test next",
        }[self._phase]

    def render_state(self) -> dict:
        # No fixation dot: it sits at the same position as the flash (both
        # center), so a bright opaque dot there would either mask the flash or
        # itself be a huge, confounding contrast step - neither is a clean
        # contrast-detection trial. The participant just holds gaze on center
        # without a marker.
        return {
            "instructions": self._instructions(),
            "dot": {"visible": False, "x": CENTER_POSITION[0], "y": CENTER_POSITION[1]},
            "cross": {"visible": False, "x": CENTER_POSITION[0], "y": CENTER_POSITION[1]},
            "square": {
                "visible": self._square_visible,
                "x": SQUARE_POSITION[0],
                "y": SQUARE_POSITION[1],
                "contrast": self._current_trial().contrast if (self._square_visible and self._trials) else 0,
            },
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": "n/a",
                "face_found": True,
                "source_available": True,
                "trial": f"{self._trial_index + 1}/{len(self._trials)}"
                if self._phase != Phase.COMPLETE
                else f"{len(self._trials)}/{len(self._trials)}",
            },
        }
