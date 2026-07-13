import random
from enum import Enum, auto

from include.eye_tracking.interfaces import GazeSource
from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CROSS_POSITION,
    DOT_POSITION,
    FALSE_ALARM_RATE_THRESHOLD,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GAZE_LANDING_STABILITY_MS,
    GRATING_DURATION_FRAMES,
    GRATING_POSITION,
    MIN_CATCH_TRIALS_FOR_RELIABILITY,
    RESPONSE_WINDOW_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
)
from include.experiment.types import Target, TrialResult
from src.experiment.logger import ResultLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.trial_factory import build_saccade_sequence
from src.experiment.zest import ZestStaircase


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
    trial loop, perisaccadic grating flashes, response scoring, and CSV logging.

    Trial structure follows Diamond, Ross & Morrone (2000, J Neurosci
    20:3449-3455): a jittered foreperiod during which only the current
    fixation symbol is shown, then the other symbol's appearance is the go-cue
    for the saccade; a brief flash may occur around saccade onset and is
    scored yes/no against a button response, with false-alarm rate tracked as
    a data-quality check exactly as in that paper. Contrast is picked trial by
    trial by a ZEST staircase (see src.experiment.zest), the same adaptive
    procedure the paper used, run independently from the presaccade phase's
    staircase so each condition converges on its own threshold.

    Framework-agnostic: `on_space()` and `tick()` are called synchronously by
    the PsychoPy frame loop, and `render_state()` returns a plain dict the
    runner applies to its stimuli. It owns no camera/tracker state itself
    (that lives in the injected GazeSource) so ticking never blocks on camera
    IO or MediaPipe inference. Doesn't own the logger either - it's shared
    with the presaccade phase so both land in one CSV file.
    """

    def __init__(self, gaze_source: GazeSource, logger: ResultLogger, clock: PausableClock) -> None:
        self._gaze = gaze_source
        self._logger = logger
        self._clock = clock
        self._phase = Phase.WAITING_TO_START
        self._trials = build_saccade_sequence()
        self._trial_index = 0
        self._dot_visible = False
        self._cross_visible = False
        self._pending_left_ratio: float | None = None
        self._results: list[TrialResult] = []
        self._zest = ZestStaircase()
        self._clear_trial_state()

    def _clear_trial_state(self) -> None:
        self._trial_start_time = 0.0
        self._gaze_left_source = False
        self._away_from_source_since: float | None = None
        self._target_landed_since: float | None = None
        self._grating_shown_at: float | None = None
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._trial_contrast: float | None = None  # contrast actually used for this trial's flash, if any
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_time_ms: float | None = None
        self._landed = False
        self._foreperiod_start = 0.0
        self._foreperiod_duration = 0.0

    @property
    def results(self) -> list[TrialResult]:
        return self._results

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
        # COMPLETE: space does nothing here - the runner decides when to move on

    def _on_response(self) -> None:
        if self._responded or not self._response_window_open:
            return
        now = self._clock.now()
        if now <= self._response_deadline:
            self._responded = True
            self._response_time_ms = (now - self._grating_shown_at) * 1000 if self._grating_shown_at else None

    # -- trial state machine ----------------------------------------------------

    def _begin_foreperiod(self) -> None:
        self._clear_trial_state()
        self._phase = Phase.FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000

    def _reveal_target_and_start_trial(self) -> None:
        trial = self._current_trial()
        if trial.target is Target.DOT:
            self._dot_visible = True
        else:
            self._cross_visible = True
        self._phase = Phase.TRIAL_ACTIVE
        self._trial_start_time = self._clock.now()

    def _current_trial(self):
        return self._trials[self._trial_index]

    def tick(self) -> None:
        if self._phase is Phase.FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.TRIAL_ACTIVE:
            self._tick_trial_active()

    def _tick_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration:
            self._reveal_target_and_start_trial()

    def _tick_trial_active(self) -> None:
        trial = self._current_trial()
        sample = self._gaze.latest_sample()
        now = self._clock.now()
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
                    if trial.grating_shown:
                        self._grating_shown_at = now
                        self._grating_visible = True
                        self._grating_frames_remaining = GRATING_DURATION_FRAMES
                        self._grating_actually_shown = True
                        self._trial_contrast = self._zest.next_contrast()
            else:
                self._away_from_source_since = None

        # Frame-counted (not time-based) so the flash lasts exactly N drawn
        # frames regardless of the display's refresh rate. Skipped on the tick
        # that just turned the grating on, so it isn't docked a frame before
        # it's even been drawn once.
        if self._grating_visible and not onset_this_tick:
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
                    # Land immediately (visual feedback: hide the source symbol),
                    # but don't finalize/log the trial yet - a real reaction time
                    # can easily outlast how quickly landing gets detected, so
                    # ending the trial here would cut off a still-valid response.
                    self._landed = True
                    self._hide_source_symbol()
            else:
                self._target_landed_since = None

            if now - self._trial_start_time >= SACCADE_TIMEOUT_MS / 1000:
                self._finish_trial(landed=False)
                return

        if self._landed and (self._responded or not self._response_window_open):
            self._finish_trial(landed=True)

    def _hide_source_symbol(self) -> None:
        trial = self._current_trial()
        if trial.source is Target.DOT:
            self._dot_visible = False
        else:
            self._cross_visible = False

    def _finish_trial(self, landed: bool) -> None:
        trial = self._current_trial()
        now = self._clock.now()

        saccade_duration_ms = (now - self._trial_start_time) * 1000 if self._gaze_left_source else None
        if not landed and not self._gaze_left_source:
            outcome = "timeout"
        elif self._grating_actually_shown and self._responded:
            outcome = "hit"
        elif self._grating_actually_shown and not self._responded:
            outcome = "miss"
        elif not self._grating_actually_shown and self._responded:
            outcome = "false_alarm"
        else:
            outcome = "correct_rejection"

        if self._grating_actually_shown and outcome in ("hit", "miss"):
            self._zest.update(self._trial_contrast, detected=outcome == "hit")

        result = TrialResult(
            index=trial.index,
            phase="saccade",
            source=trial.source,
            target=trial.target,
            saccade_duration_ms=saccade_duration_ms,
            grating_shown=self._grating_actually_shown,
            contrast=self._trial_contrast,
            responded=self._responded,
            response_time_ms=self._response_time_ms,
            outcome=outcome,
        )
        self._logger.log(result)
        self._results.append(result)

        self._hide_source_symbol()  # no-op if landing already hid it; needed for the timeout path
        self._grating_visible = False

        self._trial_index += 1
        if self._trial_index >= len(self._trials):
            self._phase = Phase.COMPLETE
        else:
            self._begin_foreperiod()

    # -- rendering ----------------------------------------------------------

    def _summary(self) -> str:
        # Hit rate = (times SPACE was pressed while the grating was actually shown)
        # divided by (times the grating was actually shown).
        hits = sum(1 for r in self._results if r.outcome == "hit")
        misses = sum(1 for r in self._results if r.outcome == "miss")
        false_alarms = sum(1 for r in self._results if r.outcome == "false_alarm")
        rejections = sum(1 for r in self._results if r.outcome == "correct_rejection")

        grating_shown_count = hits + misses
        hit_rate = hits / grating_shown_count if grating_shown_count else 0.0
        catch_trial_count = false_alarms + rejections
        fa_rate = false_alarms / catch_trial_count if catch_trial_count else 0.0

        # Diamond, Ross & Morrone (2000) report false-alarm rates <1/200 as their
        # bar for a reliably attentive observer (not guessing/spamming responses).
        # That assumes hundreds of catch trials though - with only a handful,
        # even a genuinely careless observer could show 0 false alarms by luck,
        # so don't claim "reliable" until there's enough data to back it up.
        if catch_trial_count < MIN_CATCH_TRIALS_FOR_RELIABILITY:
            reliability = "not enough data"
        elif fa_rate < FALSE_ALARM_RATE_THRESHOLD:
            reliability = "reliable"
        else:
            reliability = "unreliable"
        return (
            f"Hit rate: {hits}/{grating_shown_count} ({hit_rate:.0%}) | "
            f"False alarms: {false_alarms}/{catch_trial_count} ({fa_rate:.1%}, {reliability}) | "
            f"Threshold estimate: {self._zest.threshold_estimate:.1%} contrast"
        )

    def _instructions(self) -> str:
        if self._phase is Phase.COMPLETE:
            return f"Saccade test complete! {self._summary()} — Press SPACE to see your results"
        # FOREPERIOD and TRIAL_ACTIVE deliberately share one constant message -
        # switching text every trial would flicker in peripheral vision, which
        # is exactly the kind of onset we've been trying to avoid elsewhere.
        if self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE):
            return "Press SPACE if you see the grating"
        return {
            Phase.WAITING_TO_START: "Press SPACE to begin calibration",
            Phase.CALIBRATE_LEFT: "Look at the dot, then press SPACE",
            Phase.CALIBRATE_RIGHT: "Now look at the other circle, then press SPACE",
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
            "grating": {
                "visible": self._grating_visible,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": self._trial_contrast if self._grating_visible else 0,
            },
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": sample.zone.value,
                "face_found": sample.face_found,
                "source_available": self._gaze.is_available,
                "trial": f"{self._trial_index + 1}/{len(self._trials)}" if trial is not None else "-",
            },
        }
