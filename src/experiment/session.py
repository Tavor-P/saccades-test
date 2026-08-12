import math
import random
import time
from collections import deque
from enum import Enum, auto

from include.eye_tracking.interfaces import GazeSource
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    CENTER_POSITION,
    CROSS_POSITION,
    DEFAULT_REACTION_TIME_MS,
    DOT_POSITION,
    FALSE_ALARM_RATE_THRESHOLD,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GAZE_LANDING_STABILITY_MS,
    GRATING_DURATION_FRAMES,
    GRATING_POSITION,
    MAX_SACCADE_TRIALS_REAL,
    MAX_SACCADE_TRIALS_TEST,
    MIN_CATCH_TRIALS_FOR_RELIABILITY,
    NUM_PRACTICE_TRIALS_REAL,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_RT_TEST_TRIALS_REAL,
    NUM_RT_TEST_TRIALS_TEST,
    PRACTICE_CONTRAST,
    RECALIBRATION_ROUNDS,
    RESPONSE_WINDOW_MS,
    RT_AVERAGE_RECOMPUTE_EVERY,
    RT_AVERAGE_ROLLING_WINDOW,
    SACCADE_TIMEOUT_MS,
    TIMING_OFFSETS_MS,
    ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH,
    ZEST_MIN_VALID_TRIALS_REAL,
    ZEST_MIN_VALID_TRIALS_TEST,
)
from include.experiment.types import Orientation, Target, TrialResult, TrialSpec
from src.experiment.calibration import average_calibration_rounds
from src.experiment.logger import ResultLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import is_valid_for_saccadic_analysis, score_outcome
from src.experiment.trial_factory import build_saccade_sequence, generate_next_saccade_trial
from src.experiment.trial_mechanics import OnsetDetector, ShuffledBag, average_ms, zone_for
from src.experiment.zest import ZestStaircase


class Phase(Enum):
    WAITING_TO_START = auto()
    CALIBRATE_LEFT = auto()
    CALIBRATE_CENTER = auto()
    CALIBRATE_RIGHT = auto()
    RT_TEST_FOREPERIOD = auto()  # upfront (and post-recalibration) reaction-time measurement
    RT_TEST_ACTIVE = auto()
    FOREPERIOD = auto()  # only the current fixation symbol shown; next target not revealed yet
    TRIAL_ACTIVE = auto()
    COMPLETE = auto()


class ExperimentSession:
    """Gaze-contingent saccade experiment: calibration, an alternating dot/cross
    trial loop, perisaccadic grating flashes, response scoring, and CSV logging.

    Trial structure follows Diamond, Ross & Morrone (2000, J Neurosci
    20:3449-3455): a jittered foreperiod during which only the current
    fixation symbol is shown, then the other symbol's appearance is the go-cue
    for the saccade; a brief flash may occur around saccade onset, oriented
    either vertically or horizontally at random, and is scored as a
    2-alternative forced-choice orientation discrimination (up/down arrow =
    vertical, left/right arrow = horizontal) rather than plain yes/no
    detection, with false-alarm rate tracked as a data-quality check exactly
    as in that paper. Contrast is picked trial by trial by a ZEST staircase
    (see src.experiment.zest), the same adaptive procedure the paper used, run
    independently from the presaccade phase's staircase so each condition
    converges on its own threshold.

    Uses a "step" gap/overlap paradigm: the old fixation symbol starts
    fading out the instant the new one appears (both crossfade over
    FADE_DURATION_S in run_experiment.py), rather than staying fully lit
    until gaze lands on the target - two identical, fully-opaque symbols on
    screen at once made it genuinely ambiguous which one was the new go-cue.

    Framework-agnostic: `on_space()` and `tick()` are called synchronously by
    the PsychoPy frame loop, and `render_state()` returns a plain dict the
    runner applies to its stimuli. It owns no camera/tracker state itself
    (that lives in the injected GazeSource) so ticking never blocks on camera
    IO or MediaPipe inference. Doesn't own the logger either - it's shared
    with the presaccade phase so both land in one CSV file.
    """

    def __init__(
        self,
        gaze_source: GazeSource,
        logger: ResultLogger,
        clock: PausableClock,
        contrast_floor: float | None = None,
        show_gaze_indicator: bool = False,
        test_mode: bool = False,
        precomputed_calibration_ratios: tuple[float, float, float] | None = None,
        skip_practice_trials: bool = False,
    ) -> None:
        self._gaze = gaze_source
        self._logger = logger
        self._clock = clock
        self._show_gaze_indicator = show_gaze_indicator
        self._phase = Phase.WAITING_TO_START

        # -- practice (fixed list) then dynamically-generated real trials --
        num_practice = 0 if skip_practice_trials else (NUM_PRACTICE_TRIALS_TEST if test_mode else NUM_PRACTICE_TRIALS_REAL)
        self._practice_trials = build_saccade_sequence(num_practice=num_practice)
        self._num_practice = len(self._practice_trials)
        self._practice_index = 0
        self._trial_source: Target = self._practice_trials[-1].target if self._practice_trials else Target.DOT
        self._main_trial_counter = 0
        self._timing_offset_bag: ShuffledBag[int] = ShuffledBag(TIMING_OFFSETS_MS)
        self._current_trial_spec: TrialSpec | None = None

        # -- reaction-time test + rolling average (see constants.py) --
        self._rt_test_target_attempts = NUM_RT_TEST_TRIALS_TEST if test_mode else NUM_RT_TEST_TRIALS_REAL
        self._rt_test_attempt = 0
        self._rt_samples_ms: list[float] = []
        self._rt_test_onset_detector: OnsetDetector = OnsetDetector(zone_for(Target.DOT))
        self._rt_test_start_time = 0.0
        self._avg_reaction_time_ms: float | None = None
        self._recent_rt_ms: deque[float] = deque(maxlen=RT_AVERAGE_ROLLING_WINDOW)

        # -- stopping criterion for the real block (see constants.py) --
        self._min_valid_trials = ZEST_MIN_VALID_TRIALS_TEST if test_mode else ZEST_MIN_VALID_TRIALS_REAL
        self._max_saccade_trials = MAX_SACCADE_TRIALS_TEST if test_mode else MAX_SACCADE_TRIALS_REAL

        self._dot_visible = False
        self._cross_visible = False
        self._pending_left_ratio: float | None = None
        self._pending_center_ratio: float | None = None
        self._calibration_round = 1
        self._calibration_round_target = CALIBRATION_ROUNDS
        self._calibration_rounds_data: list[tuple[float, float, float]] = []
        self._results: list[TrialResult] = []
        zest_kwargs = {"log_contrast_min": math.log10(contrast_floor)} if contrast_floor is not None else {}
        self._zest = ZestStaircase(**zest_kwargs)
        self._clear_trial_state()

        # A tutorial run before this session already did its own 3-round
        # calibration - reuse that result instead of calibrating again. Its
        # own scoped-down reaction-time test doesn't carry over though - this
        # session always measures its own.
        self._precomputed_calibration = precomputed_calibration_ratios is not None
        self._calibration_ratios: tuple[float, float, float] | None = None
        if precomputed_calibration_ratios is not None:
            self._gaze.calibrate(*precomputed_calibration_ratios)
            self._calibration_ratios = precomputed_calibration_ratios

    @property
    def calibration_ratios(self) -> tuple[float, float, float] | None:
        """(left_ratio, center_ratio, right_ratio) from the 3-point webcam
        calibration, once it's completed - None before that. Session metadata
        reads this."""
        return self._calibration_ratios

    def _clear_trial_state(self) -> None:
        self._trial_start_time = 0.0
        # Constructed properly (with the right zone to avoid) in
        # _reveal_target_and_start_trial, once the trial's source is known -
        # this placeholder is never read before then.
        self._onset_detector: OnsetDetector | None = None
        self._target_landed_since: float | None = None
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._grating_shown_at: float | None = None
        self._grating_shown_at_unix_ms: float | None = None
        self._grating_visible = False
        self._grating_frames_remaining = 0
        self._grating_actually_shown = False
        self._flash_during_saccade: bool | None = None
        self._trial_contrast: float | None = None  # contrast actually used for this trial's flash, if any
        self._reaction_latency_ms: float | None = None
        self._onset_detection_lag_ms: float | None = None
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._response_orientation: Orientation | None = None
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
            if self._precomputed_calibration:
                # A tutorial already calibrated - skip straight to the
                # reaction-time test.
                self._begin_rt_test_foreperiod()
            else:
                self._gaze.begin_calibration_sample()  # fresh window before the dot appears
                self._phase = Phase.CALIBRATE_LEFT
        elif self._phase is Phase.CALIBRATE_LEFT:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None:
                self._pending_left_ratio = ratio
                self._gaze.begin_calibration_sample()  # fresh window before the center target appears
                self._phase = Phase.CALIBRATE_CENTER
        elif self._phase is Phase.CALIBRATE_CENTER:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None:
                self._pending_center_ratio = ratio
                self._gaze.begin_calibration_sample()  # fresh window before the cross appears
                self._phase = Phase.CALIBRATE_RIGHT
        elif self._phase is Phase.CALIBRATE_RIGHT:
            ratio = self._gaze.average_recent_ratio()
            if ratio is not None and self._pending_left_ratio is not None and self._pending_center_ratio is not None:
                self._calibration_rounds_data.append((self._pending_left_ratio, self._pending_center_ratio, ratio))
                if self._calibration_round < self._calibration_round_target:
                    # First attempts tend to be the least reliable - run
                    # another round rather than trusting this one alone.
                    self._calibration_round += 1
                    self._gaze.begin_calibration_sample()  # fresh window before the next round's dot
                    self._phase = Phase.CALIBRATE_LEFT
                else:
                    # Round 1 is discarded when there are >=3 rounds (the
                    # normal case); a 2-round recalibration (see
                    # resume_from_pause) has no round to discard, so this
                    # naturally averages both instead - indexed from the
                    # end rather than unpacked as exactly 3 elements, so it
                    # doesn't break either way.
                    round2, round3 = self._calibration_rounds_data[-2], self._calibration_rounds_data[-1]
                    final_ratios = average_calibration_rounds(round2, round3)
                    self._gaze.calibrate(*final_ratios)
                    self._calibration_ratios = final_ratios
                    self._begin_rt_test_foreperiod()
        # RT_TEST_*/TRIAL_ACTIVE: responses come from gaze/arrow keys, not
        # SPACE. COMPLETE: space does nothing here - the runner decides when
        # to move on.

    def on_response_key(self, orientation: Orientation) -> None:
        """Called when an arrow key is pressed during a trial: up/down report
        `Orientation.VERTICAL`, left/right report `Orientation.HORIZONTAL` -
        see run_experiment.py's key dispatch."""
        if self._phase is not Phase.TRIAL_ACTIVE:
            return
        if self._responded or not self._response_window_open:
            return
        now = self._clock.now()
        if now <= self._response_deadline:
            self._responded = True
            self._response_orientation = orientation
            self._response_time_ms = (now - self._grating_shown_at) * 1000 if self._grating_shown_at else None

    # -- pause / recalibration ----------------------------------------------

    def resume_from_pause(self, recalibrate: bool) -> None:
        """Called by the runner when a paused participant clicks a pause-menu
        button. Any trial (RT-test or main/practice) in flight at the moment
        of pausing is discarded entirely - no CSV row, no ZEST feed, no
        RT-average feed - because gaze during a pause is consciously aimed at
        a UI button, not a spontaneous task response, so any onset/landing
        timestamps straddling the pause would be contaminated regardless of
        PausableClock's own timer protection. `recalibrate` triggers a fast
        2-round eye-position recalibration followed by a fresh reaction-time
        test (see RECALIBRATION_ROUNDS); either way, resuming during
        calibration itself, WAITING_TO_START, or COMPLETE is a no-op beyond
        the runner's own clock pause/resume - there's nothing in-flight to
        discard or restart there."""
        discardable = (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE, Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE)
        if self._phase not in discardable:
            return
        was_rt_test = self._phase in (Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE)
        if recalibrate:
            self._begin_recalibration()
        elif was_rt_test:
            self._begin_rt_test_foreperiod()  # retry the same (uncounted) attempt slot
        else:
            self._begin_next_trial_or_complete()  # fresh practice/main trial

    def _begin_recalibration(self) -> None:
        self._calibration_round = 1
        self._calibration_round_target = RECALIBRATION_ROUNDS
        self._calibration_rounds_data = []
        self._pending_left_ratio = None
        self._pending_center_ratio = None
        # A fresh RT-test batch, same as the initial one - the whole point of
        # recalibrating is that the current avg_reaction_time_ms is presumed
        # stale, so it gets fully overwritten (not blended) once this
        # completes. RT_AVERAGE_ROLLING_WINDOW/_completed_main_trial_count
        # are untouched - only the average value itself is replaced.
        self._rt_test_attempt = 0
        self._rt_samples_ms = []
        self._gaze.begin_calibration_sample()
        self._phase = Phase.CALIBRATE_LEFT

    # -- reaction-time test ---------------------------------------------------

    def _begin_rt_test_foreperiod(self) -> None:
        self._clear_trial_state()
        self._phase = Phase.RT_TEST_FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000
        self._rt_test_onset_detector = OnsetDetector(zone_for(Target.DOT))
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
        result = TrialResult(
            index=self._rt_test_attempt,
            phase="rt_test",
            source=Target.DOT,
            target=Target.CROSS,
            saccade_duration_ms=None,
            grating_shown=False,
            contrast=None,
            orientation=None,
            responded=False,
            response_orientation=None,
            response_time_ms=None,
            outcome="timeout" if reaction_time_ms is None else "detected",
            reaction_latency_ms=reaction_time_ms,
        )
        self._logger.log(result)
        self._results.append(result)
        if reaction_time_ms is not None:
            self._rt_samples_ms.append(reaction_time_ms)

        self._rt_test_attempt += 1
        if self._rt_test_attempt >= self._rt_test_target_attempts:
            self._avg_reaction_time_ms = average_ms(self._rt_samples_ms, default=DEFAULT_REACTION_TIME_MS)
            self._begin_next_trial_or_complete()
        else:
            self._begin_rt_test_foreperiod()

    # -- trial state machine ----------------------------------------------------

    def _begin_next_trial_or_complete(self) -> None:
        """Selects and starts the next practice trial (fixed list), the next
        dynamically-generated real trial, or ends the session - called both
        to advance normally (from _finish_trial) and to resume after a pause
        or recalibration, where it naturally continues from wherever the
        practice/main cursors already were rather than restarting."""
        if self._practice_index < len(self._practice_trials):
            self._current_trial_spec = self._practice_trials[self._practice_index]
            self._practice_index += 1
            self._begin_foreperiod()
            return
        if self._should_stop_main_block():
            self._phase = Phase.COMPLETE
            return
        trial, self._trial_source = generate_next_saccade_trial(
            self._main_trial_counter, self._trial_source, self._timing_offset_bag
        )
        self._main_trial_counter += 1
        self._current_trial_spec = trial
        self._begin_foreperiod()

    @property
    def _completed_main_trial_count(self) -> int:
        """Derived from self._results rather than hand-maintained - every
        non-practice main-block trial that finishes gets exactly one
        "saccade"-phase row appended there (see _finish_trial), so counting
        those directly can't drift out of sync with what's actually been
        logged."""
        return sum(1 for r in self._results if r.phase == "saccade")

    @property
    def _valid_trial_count(self) -> int:
        return sum(1 for r in self._results if r.phase == "saccade" and is_valid_for_saccadic_analysis(r.flash_during_saccade))

    def _should_stop_main_block(self) -> bool:
        """An efficient, well-estimated threshold rather than a fixed trial
        count: stop once ZEST's 68% credible interval is narrow enough (in
        log-contrast space - credible_interval returns linear bounds) AND
        enough valid trials have actually shaped it, or unconditionally once
        the safety cap is hit (open-loop misses are structural now, not rare
        lag noise, so this needs real headroom to guarantee termination)."""
        if self._completed_main_trial_count >= self._max_saccade_trials:
            return True
        if self._valid_trial_count < self._min_valid_trials:
            return False
        lo, hi = self._zest.credible_interval(0.68)
        log_width = math.log10(hi) - math.log10(lo)
        return log_width <= ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH

    def _begin_foreperiod(self) -> None:
        self._clear_trial_state()
        self._phase = Phase.FOREPERIOD
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration = random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS) / 1000
        # Explicit rather than relying on whatever _dot_visible/_cross_visible
        # was left showing by the previous phase - that's usually already
        # correct (a finishing trial's target becomes the next trial's
        # source, via _hide_source_symbol/_reveal_target_and_start_trial),
        # but isn't when the previous phase was RT-test or recalibration,
        # neither of which tracks a TrialSpec at all.
        trial = self._current_trial_spec
        self._dot_visible = trial.source is Target.DOT
        self._cross_visible = trial.source is Target.CROSS

    def _reveal_target_and_start_trial(self) -> None:
        trial = self._current_trial_spec
        if trial.target is Target.DOT:
            self._dot_visible = True
        else:
            self._cross_visible = True
        # Hide the source the instant the target appears (not at landing) so
        # the two crossfade - the renderer already fades opacity toward
        # whatever `visible` says over FADE_DURATION_S, so this alone turns
        # "new symbol pops in next to an already-full-brightness old one"
        # into "old fades out as new fades in", with no ambiguity about which
        # one is the one to look at.
        self._hide_source_symbol()
        self._phase = Phase.TRIAL_ACTIVE
        self._trial_start_time = self._clock.now()
        self._onset_detector = OnsetDetector(zone_for(trial.source))
        # Open-loop: the flash (or catch) fires at a fixed delay from this
        # instant - avg_reaction_time_ms + this trial's offset - rather than
        # whenever the real-time gaze classifier detects onset (see
        # _tick_trial_active). avg_reaction_time_ms is always set by this
        # point - the RT-test phase always runs before any FOREPERIOD.
        # Not `trial.timing_offset_ms or 0` - 0 is itself a valid member of
        # TIMING_OFFSETS_MS, so that pattern would only coincidentally give
        # the right answer today (both branches happen to be 0) and silently
        # break if the fallback value or the offsets ever changed.
        offset_ms = trial.timing_offset_ms if trial.timing_offset_ms is not None else 0
        self._flash_scheduled_at = self._trial_start_time + (self._avg_reaction_time_ms + offset_ms) / 1000

    def tick(self) -> None:
        if self._phase is Phase.RT_TEST_FOREPERIOD:
            self._tick_rt_test_foreperiod()
        elif self._phase is Phase.RT_TEST_ACTIVE:
            self._tick_rt_test_active()
        elif self._phase is Phase.FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.TRIAL_ACTIVE:
            self._tick_trial_active()

    def _tick_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration:
            self._reveal_target_and_start_trial()

    def _tick_trial_active(self) -> None:
        trial = self._current_trial_spec
        sample = self._gaze.latest_sample()
        now = self._clock.now()
        target_zone = zone_for(trial.target)

        # Onset detection still runs every tick, purely as a measurement now
        # (feeds reaction_latency_ms - and via that, the rolling RT average -
        # plus one bound of the flash_during_saccade window check in
        # _finish_trial) - it no longer triggers the flash. See the
        # scheduled-time block below for that.
        if not self._onset_detector.confirmed and self._onset_detector.update(sample, now):
            self._reaction_latency_ms = (self._onset_detector.since - self._trial_start_time) * 1000
            self._onset_detection_lag_ms = (now - self._onset_detector.since) * 1000

        # Open-loop: fires once, at its scheduled absolute time, independent
        # of the onset detection above. Catch trials still "decide" here (no
        # grating) purely so the response window has a well-defined open
        # point, matching shown trials.
        flash_fired_this_tick = False
        if not self._flash_decided and now >= self._flash_scheduled_at:
            self._flash_decided = True
            flash_fired_this_tick = True
            self._response_window_open = True
            self._response_deadline = now + RESPONSE_WINDOW_MS / 1000
            if trial.grating_shown:
                self._grating_shown_at = now
                self._grating_shown_at_unix_ms = time.time() * 1000
                self._grating_visible = True
                self._grating_frames_remaining = GRATING_DURATION_FRAMES
                self._grating_actually_shown = True
                self._trial_contrast = PRACTICE_CONTRAST if trial.practice else self._zest.next_contrast()

        # Frame-counted (not time-based) so the flash lasts exactly N drawn
        # frames regardless of the display's refresh rate. Skipped on the tick
        # that just turned the grating on, so it isn't docked a frame before
        # it's even been drawn once.
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
                    # Land immediately, but don't finalize/log the trial yet -
                    # a real reaction time can easily outlast how quickly
                    # landing gets detected, so ending the trial here would
                    # cut off a still-valid response. (The source symbol is
                    # already gone by now - it started fading out back when
                    # the target first appeared - so there's nothing to hide.)
                    self._landed = True
            else:
                self._target_landed_since = None

            if now - self._trial_start_time >= SACCADE_TIMEOUT_MS / 1000:
                self._finish_trial(landed=False)
                return

        # Also requires the scheduled flash to have already fired - a fast
        # lander (common at a +40ms offset against a stale average) can't
        # finalize before their scheduled flash even happens.
        if self._landed and self._flash_decided and (self._responded or not self._response_window_open):
            self._finish_trial(landed=True)

    def _hide_source_symbol(self) -> None:
        trial = self._current_trial_spec
        if trial.source is Target.DOT:
            self._dot_visible = False
        else:
            self._cross_visible = False

    def _compute_flash_during_saccade(self) -> bool | None:
        """Whether this trial's scheduled flash time actually fell within
        [onset, landing] - the real-time-detected saccade window.
        _onset_detector.since/_target_landed_since are both already low-lag
        "first observed" timestamps (not the later stability-confirmed
        ones), so no continuous gaze-trace logging is needed to compute
        this, just a comparison against the already-known schedule. Only
        meaningful for trials that actually flashed. False if onset was
        never detected (no saccade evidence at flash time at all);
        None if landing was never confirmed before the trial ended -
        genuinely undeterminable, not merely invalid."""
        if not self._grating_actually_shown:
            return None
        if self._onset_detector.since is None:
            return False
        if self._target_landed_since is None:
            return None
        return self._onset_detector.since <= self._flash_scheduled_at <= self._target_landed_since

    def _update_rt_tracking(self) -> None:
        if self._reaction_latency_ms is not None:
            self._recent_rt_ms.append(self._reaction_latency_ms)
        if self._completed_main_trial_count % RT_AVERAGE_RECOMPUTE_EVERY == 0:
            # default=current average: a window that's entirely timeouts
            # shouldn't zero it out, just leave it unchanged.
            self._avg_reaction_time_ms = average_ms(self._recent_rt_ms, default=self._avg_reaction_time_ms)

    def _finish_trial(self, landed: bool) -> None:
        trial = self._current_trial_spec
        now = self._clock.now()

        saccade_duration_ms = (now - self._trial_start_time) * 1000 if self._onset_detector.confirmed else None
        if not landed and not self._onset_detector.confirmed:
            outcome = "timeout"
        else:
            outcome = score_outcome(
                self._grating_actually_shown, self._responded, self._response_orientation, trial.orientation
            )
        self._flash_during_saccade = self._compute_flash_during_saccade()

        if not trial.practice:
            # "miss" (no response within the window) counts as a non-detection
            # here too, not just "correct"/"incorrect" - a participant who
            # never saw the grating well enough to answer in time is exactly
            # the signal ZEST needs to stop drifting the contrast down further.
            # Matches what results_graph.py's replay already does from the
            # logged CSV, so the live in-session estimate and the end-of-run
            # graph agree. Also requires flash_during_saccade is True now -
            # open-loop scheduling makes a missed saccade window structurally
            # common (not rare lag noise like the old gaze-contingent
            # trigger), so feeding those trials in would corrupt the
            # threshold estimate.
            if (
                self._grating_actually_shown
                and outcome in ("correct", "incorrect", "miss")
                and is_valid_for_saccadic_analysis(self._flash_during_saccade)
            ):
                self._zest.update(self._trial_contrast, detected=outcome == "correct")

            result = TrialResult(
                index=trial.index,
                phase="saccade",
                source=trial.source,
                target=trial.target,
                saccade_duration_ms=saccade_duration_ms,
                grating_shown=self._grating_actually_shown,
                contrast=self._trial_contrast,
                orientation=trial.orientation if self._grating_actually_shown else None,
                responded=self._responded,
                response_orientation=self._response_orientation,
                response_time_ms=self._response_time_ms,
                outcome=outcome,
                reaction_latency_ms=self._reaction_latency_ms,
                onset_detection_lag_ms=self._onset_detection_lag_ms,
                flash_during_saccade=self._flash_during_saccade,
                timing_offset_ms=trial.timing_offset_ms,
                grating_shown_at_unix_ms=self._grating_shown_at_unix_ms,
            )
            self._logger.log(result)
            self._results.append(result)
            self._update_rt_tracking()

        self._grating_visible = False
        self._begin_next_trial_or_complete()

    # -- rendering ----------------------------------------------------------

    def _summary(self) -> str:
        # Accuracy = (times the arrow-key response correctly reported the
        # grating's orientation) divided by (times the grating was actually
        # shown) - "incorrect" (wrong orientation guessed) and "miss" (no
        # response at all) both count against it.
        correct = sum(1 for r in self._results if r.outcome == "correct")
        incorrect = sum(1 for r in self._results if r.outcome == "incorrect")
        misses = sum(1 for r in self._results if r.outcome == "miss")
        false_alarms = sum(1 for r in self._results if r.outcome == "false_alarm")
        rejections = sum(1 for r in self._results if r.outcome == "correct_rejection")

        grating_shown_count = correct + incorrect + misses
        accuracy = correct / grating_shown_count if grating_shown_count else 0.0
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
            f"Accuracy: {correct}/{grating_shown_count} ({accuracy:.0%}) | "
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
            prefix = "Practice (doesn't count) — " if self._current_trial_spec.practice else ""
            return f"{prefix}UP/DOWN if the grating is vertical, LEFT/RIGHT if horizontal — not sure? Guess"
        if self._phase in (Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE):
            return "Measuring your reaction time — look at the dot, then look at the cross as soon as it appears"
        calibration_targets = {
            Phase.CALIBRATE_LEFT: "Look at the dot, then press SPACE",
            Phase.CALIBRATE_CENTER: "Now look at the center, then press SPACE",
            Phase.CALIBRATE_RIGHT: "Now look at the other circle, then press SPACE",
        }
        if self._phase in calibration_targets:
            # Only narrate round 1 of CALIBRATION_ROUNDS - by round 2 the
            # participant already knows the drill, so re-speaking it every
            # round just slows calibration down for no benefit.
            return calibration_targets[self._phase] if self._calibration_round == 1 else ""
        return {
            Phase.WAITING_TO_START: "Press SPACE to begin calibration",
        }[self._phase]

    def _trial_label(self) -> str:
        if self._phase in (Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE):
            return f"RT test {self._rt_test_attempt + 1}/{self._rt_test_target_attempts}"
        trial = self._current_trial_spec
        if trial.practice:
            return f"practice {self._practice_index}/{self._num_practice}"
        # Open-ended now that contrast stays ZEST-adaptive rather than a
        # fixed trial count (see _should_stop_main_block) - shows progress
        # toward the stopping criterion instead of "N/total".
        return f"trial {self._main_trial_counter} (valid: {self._valid_trial_count})"

    def _symbol_visibility(self) -> tuple[bool, bool, bool]:
        """(dot, cross, calibration_center) - a dedicated stimulus for the
        center calibration target, rather than repositioning "dot" onto it:
        reusing "dot" made it teleport instantly from the dot position to
        center while staying fully opaque throughout, with no fade cue that a
        new step had even started (easy to mistake for the step being
        skipped) - the same ambiguity this app already fixed for the
        dot/cross source/target crossfade."""
        if self._phase is Phase.CALIBRATE_LEFT:
            return True, False, False
        if self._phase is Phase.CALIBRATE_CENTER:
            return False, False, True
        if self._phase is Phase.CALIBRATE_RIGHT:
            return False, True, False
        if self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE, Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE):
            return self._dot_visible, self._cross_visible, False
        return False, False, False

    def _gaze_indicator_state(self, sample) -> dict:
        """Live gaze cursor, opt-in via show_gaze_indicator: continuously
        tracks the participant's actual estimated gaze position rather than
        snapping to one of three fixed zones, using
        GazeSample.smoothed_position (see include/eye_tracking/types.py) -
        median-filtered + EMA-smoothed, since MediaPipe's raw per-frame
        position is noisy enough frame-to-frame to look jittery even when
        tracking is working correctly. Deliberately NOT what onset/landing
        detection uses for scoring - that stays on the debounced `zone`
        classification (see _tick_trial_active) and the raw, unsmoothed
        `position`, so this indicator's smoothing can never affect trial
        timing."""
        show = (
            self._show_gaze_indicator
            and self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE)
            and sample.face_found
            and sample.smoothed_position is not None
            and self._calibration_ratios is not None
        )
        if not show:
            return {"visible": False, "x": 0.0, "y": 0.0}

        # DOT_POSITION/CROSS_POSITION share the same y (this experiment only
        # tracks horizontal gaze) - interpolate x between them by the
        # smoothed position (0=dot/left target, 1=cross/right target).
        x = DOT_POSITION[0] + (CROSS_POSITION[0] - DOT_POSITION[0]) * sample.smoothed_position
        return {"visible": True, "x": x, "y": DOT_POSITION[1]}

    def render_state(self) -> dict:
        sample = self._gaze.latest_sample()
        dot_visible, cross_visible, calibration_center_visible = self._symbol_visibility()
        trial = self._current_trial_spec if self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE) else None

        return {
            "instructions": self._instructions(),
            "dot": {"visible": dot_visible, "x": DOT_POSITION[0], "y": DOT_POSITION[1]},
            "cross": {"visible": cross_visible, "x": CROSS_POSITION[0], "y": CROSS_POSITION[1]},
            "calibration_center": {
                "visible": calibration_center_visible,
                "x": CENTER_POSITION[0],
                "y": CENTER_POSITION[1],
            },
            "grating": {
                "visible": self._grating_visible,
                "x": GRATING_POSITION[0],
                "y": GRATING_POSITION[1],
                "contrast": self._trial_contrast if self._grating_visible else 0,
                "orientation": trial.orientation if trial is not None else None,
            },
            "gaze_indicator": self._gaze_indicator_state(sample),
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": sample.zone.value,
                "face_found": sample.face_found,
                "source_available": self._gaze.is_available,
                "trial": self._trial_label() if self._has_trial_label() else "-",
                "trial_index": self._main_trial_counter,
            },
        }

    def _has_trial_label(self) -> bool:
        return self._phase in (
            Phase.FOREPERIOD,
            Phase.TRIAL_ACTIVE,
            Phase.RT_TEST_FOREPERIOD,
            Phase.RT_TEST_ACTIVE,
        )
