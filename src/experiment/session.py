import random
import time
from enum import Enum, auto

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CROSS_POSITION,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GAZE_LANDING_STABILITY_MS,
    NUM_TRIALS,
    RESPONSE_WINDOW_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
    SQUARE_DURATION_FRAMES,
    SQUARE_POSITION,
)
from include.experiment.types import Target, TrialResult
from src.experiment.logger import ResultLogger
from src.experiment.trial_factory import build_trial_sequence


class Phase(Enum):
    WAITING_TO_START = auto()
    CALIBRATE_LEFT = auto()
    CALIBRATE_RIGHT = auto()
    FOREPERIOD = auto()  # only the current fixation symbol shown; next target not revealed yet
    TRIAL_ACTIVE = auto()
    COMPLETE = auto()


def _zone_for(target: Target) -> GazeZone:
    return GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT


class ExperimentSession:
    """Gaze-contingent saccade experiment: calibration, an alternating dot/cross
    trial loop, perisaccadic square flashes, response scoring, and CSV logging.

    Trial structure follows Diamond, Ross & Morrone (2000, J Neurosci
    20:3449-3455): a jittered foreperiod during which only the current
    fixation symbol is shown, then the other symbol's appearance is the go-cue
    for the saccade; a brief flash may occur around saccade onset (contrast
    varied, ~50% zero-contrast catch trials) and is scored yes/no against a
    button response, with false-alarm rate tracked as a data-quality check
    exactly as in that paper.

    Framework-agnostic: `on_space()` and `tick()` are called synchronously by
    the PsychoPy frame loop, and `render_state()` returns a plain dict the
    runner applies to its stimuli. It owns no camera/tracker state itself
    (that lives in the injected GazeSource) so ticking never blocks on camera
    IO or MediaPipe inference.
    """

    def __init__(self, gaze_source: GazeSource) -> None:
        self._gaze = gaze_source
        self._logger = ResultLogger()
        self._reset()

    def _reset(self) -> None:
        self._phase = Phase.WAITING_TO_START
        self._trials = build_trial_sequence(NUM_TRIALS)
        self._trial_index = 0
        self._dot_visible = False
        self._cross_visible = False
        self._pending_left_ratio: float | None = None
        self._results: list[TrialResult] = []
        self._clear_trial_state()

    def _clear_trial_state(self) -> None:
        self._trial_start_time = 0.0
        self._gaze_left_source = False
        self._away_from_source_since: float | None = None
        self._target_landed_since: float | None = None
        self._square_shown_at: float | None = None
        self._square_visible = False
        self._square_frames_remaining = 0
        self._square_actually_shown = False
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_time_ms: float | None = None
        self._foreperiod_start = 0.0
        self._foreperiod_duration = 0.0

    def close(self) -> None:
        self._logger.close()

    # -- key handling ----------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            self._phase = Phase.CALIBRATE_LEFT
        elif self._phase is Phase.CALIBRATE_LEFT:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None:
                self._pending_left_ratio = ratio
                self._phase = Phase.CALIBRATE_RIGHT
        elif self._phase is Phase.CALIBRATE_RIGHT:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None and self._pending_left_ratio is not None:
                self._gaze.calibrate(self._pending_left_ratio, ratio)
                self._dot_visible = True  # trial 0's source
                self._cross_visible = False
                self._begin_foreperiod()
        elif self._phase is Phase.TRIAL_ACTIVE:
            self._on_response()
        elif self._phase is Phase.COMPLETE:
            self._logger.close()
            self._logger = ResultLogger()
            self._reset()

    def _on_response(self) -> None:
        if self._responded or not self._response_window_open:
            return
        now = time.monotonic()
        if now <= self._response_deadline:
            self._responded = True
            self._response_time_ms = (now - self._square_shown_at) * 1000 if self._square_shown_at else None

    # -- trial state machine ----------------------------------------------------

    def _begin_foreperiod(self) -> None:
        self._clear_trial_state()
        self._phase = Phase.FOREPERIOD
        self._foreperiod_start = time.monotonic()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000

    def _reveal_target_and_start_trial(self) -> None:
        trial = self._current_trial()
        if trial.target is Target.DOT:
            self._dot_visible = True
        else:
            self._cross_visible = True
        self._phase = Phase.TRIAL_ACTIVE
        self._trial_start_time = time.monotonic()

    def _current_trial(self):
        return self._trials[self._trial_index]

    def tick(self) -> None:
        if self._phase is Phase.FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.TRIAL_ACTIVE:
            self._tick_trial_active()

    def _tick_foreperiod(self) -> None:
        if time.monotonic() - self._foreperiod_start >= self._foreperiod_duration:
            self._reveal_target_and_start_trial()

    def _tick_trial_active(self) -> None:
        trial = self._current_trial()
        sample = self._gaze.latest_sample()
        now = time.monotonic()
        source_zone = _zone_for(trial.source)
        target_zone = _zone_for(trial.target)

        onset_this_tick = False
        if not self._gaze_left_source:
            if sample.face_found and sample.zone not in (source_zone, GazeZone.UNKNOWN):
                if self._away_from_source_since is None:
                    self._away_from_source_since = now
                elif now - self._away_from_source_since >= SACCADE_ONSET_STABILITY_MS / 1000:
                    self._gaze_left_source = True
                    self._response_window_open = True
                    self._response_deadline = now + RESPONSE_WINDOW_MS / 1000
                    onset_this_tick = True
                    if trial.square_shown:
                        self._square_shown_at = now
                        self._square_visible = True
                        self._square_frames_remaining = SQUARE_DURATION_FRAMES
                        self._square_actually_shown = True
            else:
                self._away_from_source_since = None

        # Frame-counted (not time-based) so the flash lasts exactly N drawn
        # frames regardless of the display's refresh rate. Skipped on the tick
        # that just turned the square on, so it isn't docked a frame before
        # it's even been drawn once.
        if self._square_visible and not onset_this_tick:
            self._square_frames_remaining -= 1
            if self._square_frames_remaining <= 0:
                self._square_visible = False

        if self._response_window_open and now > self._response_deadline:
            self._response_window_open = False

        if sample.face_found and sample.zone is target_zone:
            if self._target_landed_since is None:
                self._target_landed_since = now
            elif now - self._target_landed_since >= GAZE_LANDING_STABILITY_MS / 1000:
                self._finish_trial(landed=True)
                return
        else:
            self._target_landed_since = None

        if now - self._trial_start_time >= SACCADE_TIMEOUT_MS / 1000:
            self._finish_trial(landed=False)

    def _finish_trial(self, landed: bool) -> None:
        trial = self._current_trial()
        now = time.monotonic()

        saccade_duration_ms = (now - self._trial_start_time) * 1000 if self._gaze_left_source else None
        if not landed and not self._gaze_left_source:
            outcome = "timeout"
        elif self._square_actually_shown and self._responded:
            outcome = "hit"
        elif self._square_actually_shown and not self._responded:
            outcome = "miss"
        elif not self._square_actually_shown and self._responded:
            outcome = "false_alarm"
        else:
            outcome = "correct_rejection"

        result = TrialResult(
            index=trial.index,
            source=trial.source,
            target=trial.target,
            saccade_duration_ms=saccade_duration_ms,
            square_shown=self._square_actually_shown,
            contrast=trial.contrast,
            responded=self._responded,
            response_time_ms=self._response_time_ms,
            outcome=outcome,
        )
        self._logger.log(result)
        self._results.append(result)

        if trial.source is Target.DOT:
            self._dot_visible = False
        else:
            self._cross_visible = False
        self._square_visible = False

        self._trial_index += 1
        if self._trial_index >= len(self._trials):
            self._phase = Phase.COMPLETE
        else:
            self._begin_foreperiod()

    # -- rendering ----------------------------------------------------------

    def _summary(self) -> str:
        # Hit rate = (times SPACE was pressed while the square was actually shown)
        # divided by (times the square was actually shown).
        hits = sum(1 for r in self._results if r.outcome == "hit")
        misses = sum(1 for r in self._results if r.outcome == "miss")
        false_alarms = sum(1 for r in self._results if r.outcome == "false_alarm")
        rejections = sum(1 for r in self._results if r.outcome == "correct_rejection")

        square_shown_count = hits + misses
        hit_rate = hits / square_shown_count if square_shown_count else 0.0
        fa_rate = false_alarms / (false_alarms + rejections) if (false_alarms + rejections) else 0.0
        return f"Hit rate: {hits}/{square_shown_count} ({hit_rate:.0%}) | False alarms: {fa_rate:.0%}"

    def _instructions(self) -> str:
        if self._phase is Phase.COMPLETE:
            return f"Session complete — thank you! {self._summary()} — Press SPACE to run again"
        return {
            Phase.WAITING_TO_START: "Press SPACE to begin calibration",
            Phase.CALIBRATE_LEFT: "Look at the dot, then press SPACE",
            Phase.CALIBRATE_RIGHT: "Now look at the other circle, then press SPACE",
            Phase.FOREPERIOD: "Hold your gaze — get ready",
            Phase.TRIAL_ACTIVE: "Press SPACE if you see the square",
        }[self._phase]

    def _symbol_visibility(self) -> tuple[bool, bool]:
        if self._phase is Phase.CALIBRATE_LEFT:
            return True, False
        if self._phase is Phase.CALIBRATE_RIGHT:
            return False, True
        if self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE):
            return self._dot_visible, self._cross_visible
        return False, False

    def render_state(self) -> dict:
        sample = self._gaze.latest_sample()
        dot_visible, cross_visible = self._symbol_visibility()
        trial = self._current_trial() if self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE) else None

        return {
            "instructions": self._instructions(),
            "dot": {"visible": dot_visible, "x": DOT_POSITION[0], "y": DOT_POSITION[1]},
            "cross": {"visible": cross_visible, "x": CROSS_POSITION[0], "y": CROSS_POSITION[1]},
            "square": {
                "visible": self._square_visible,
                "x": SQUARE_POSITION[0],
                "y": SQUARE_POSITION[1],
                "contrast": trial.contrast if (trial is not None and self._square_visible) else 0,
            },
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": sample.zone.value,
                "face_found": sample.face_found,
                "source_available": self._gaze.is_available,
                "trial": f"{self._trial_index + 1}/{len(self._trials)}" if trial is not None else "-",
            },
        }
