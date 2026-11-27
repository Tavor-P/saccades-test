"""ExperimentSession: the saccade phase as a PsychoPy-free state machine (calibration -> reaction-time test -> practice -> real block, generated trial by trial; stops at `SACCADE_FIXED_TRIALS_*`).
Drive it with on_space/on_response_key/tick/render_state. Flash timing is open-loop (scheduled off the measured reaction time); see CLAUDE.md. Headless
runs of this class: src/experiment/simulation.py."""

import math
import random
from enum import Enum, auto
from typing import Protocol

from include.experiment import participant_text as words
from include.experiment.constants import (
    CENTER_POSITION,
    CONTROL_OFFSETS_MS,
    CROSS_POSITION,
    DEFAULT_REACTION_TIME_MS,
    DOT_POSITION,
    FLASH_SPREAD_SD_MS,
    FLASH_WINDOW_MAX_MS,
    FLASH_WINDOW_MIN_MS,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    LINE_DURATION_FRAMES,
    NUM_PRACTICE_TRIALS_REAL,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_RT_TEST_TRIALS_REAL,
    NUM_RT_TEST_TRIALS_TEST,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
    RETURN_FLASH_SPREAD_SD_MS,
    RETURN_JUMP_TRIGGER,
    RETURN_LEG_FLASH_FRACTION,
    SACCADE_FIXED_CONTRASTS,
    SACCADE_FIXED_TRIALS_REAL,
    SACCADE_FIXED_TRIALS_TEST,
    SACCADE_TIMEOUT_MS,
    SHARP_GO_CUE,
    TIMING_JITTER_MS,
    TIMING_OFFSETS_MS,
    TRACKING_LOSS_WINDOW_MS,
)
from include.experiment.types import Side, Target, TrialResult, TrialSpec
from include.eye_tracking.constants import HEAD_DRIFT_WARN_DEG, HEAD_MAX_SPEED_DPS, HEAD_WARN_MIN_INTERVAL_S, HEAD_WARN_TEXT
from include.eye_tracking.interfaces import GazeSource
from src.experiment.calibration import CalibrationRounds, calibration_text
from src.experiment.geometry import Geometry
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import is_valid_for_saccadic_analysis, score_outcome
from src.experiment.session_config import SessionConfig, TutorialHandoff
from src.experiment.space_input import SpaceTally
from src.experiment.trial_factory import build_saccade_sequence, generate_next_saccade_trial
from src.experiment.trial_legs import SampleFeed, TrialLegs, clamped_gaze_latency_ms, timestamp_latency_s
from src.experiment.trial_mechanics import (
    OnsetDetector,
    RollingReactionTime,
    ShuffledBag,
    flash_leg_pool,
    line_position,
    spread_flash_delay_ms,
    zone_for,
)
from src.experiment.trial_record import saccade_trial_result
from src.timebase import ms_to_s, s_to_ms


class Phase(Enum):
    WAITING_TO_START = auto()
    CALIBRATE_LEFT = auto()
    CALIBRATE_CENTER = auto()
    CALIBRATE_RIGHT = auto()
    RT_TEST_AWAITING_SPACE = auto()  # upfront reaction-time measurement: waits for SPACE before each attempt
    RT_TEST_FOREPERIOD = auto()
    RT_TEST_ACTIVE = auto()
    AWAITING_SPACE = auto()  # fixation symbol shown; the participant presses SPACE when ready for the next trial
    FOREPERIOD = auto()  # 500-1000 ms after SPACE: only the current fixation symbol shown; target not revealed yet
    TRIAL_ACTIVE = auto()
    COMPLETE = auto()


_CALIBRATION_PHASES = (Phase.CALIBRATE_LEFT, Phase.CALIBRATE_CENTER, Phase.CALIBRATE_RIGHT)
_FOREPERIOD_PHASES = (Phase.RT_TEST_FOREPERIOD, Phase.FOREPERIOD)
_RT_TEST_PHASES = (Phase.RT_TEST_AWAITING_SPACE, Phase.RT_TEST_FOREPERIOD, Phase.RT_TEST_ACTIVE)
_MAIN_TRIAL_PHASES = (Phase.AWAITING_SPACE, Phase.FOREPERIOD, Phase.TRIAL_ACTIVE)
_TRIAL_PHASES = _RT_TEST_PHASES + _MAIN_TRIAL_PHASES


_ANSWER_TEXT = words.ANSWER


class TrialLogger(Protocol):
    """What the session needs from a logger: ResultLogger, or the simulator's in-memory stand-in."""

    def log(self, result: TrialResult) -> None: ...


class ExperimentSession:
    """Gaze-contingent saccade experiment: calibration, an alternating dot/cross trial loop, perisaccadic line flashes,
    response scoring and logging.

    Trial structure follows Diamond, Ross & Morrone (2000): a jittered foreperiod showing only the fixation symbol, then the
    other symbol's appearance is the go-cue. A brief flash (here a thin line just above or below the saccade target, after
    Braun, Schutz & Gegenfurtner 2021) is scored as a 2AFC above/below discrimination (UP/DOWN). Contrast comes from fixed
    levels (SACCADE_FIXED_CONTRASTS, method of constant stimuli).
    With SHARP_GO_CUE (default) the dot/cross swap is instant (render_state()["snap_symbols"]) so the go-cue has a definite
    onset; otherwise the old symbol crossfades out as the new one appears.

    Framework-agnostic: the runner calls `on_space()`/`on_response_key()`/`tick()` and applies `render_state()` to its
    stimuli. Gaze state lives in the injected GazeSource and the logger records the run.
    """

    def __init__(  # noqa: PLR0915 - live-session state machine, unvalidated on glasses; not refactored
        self,
        gaze_source: GazeSource,
        logger: TrialLogger,
        clock: PausableClock,
        config: SessionConfig | None = None,
        handoff: TutorialHandoff | None = None,
    ) -> None:
        config = config or SessionConfig()
        handoff = handoff or TutorialHandoff()
        test_mode = config.test_mode
        return_jump = config.return_jump
        flash_spread_sd_ms = config.flash_spread_sd_ms
        fixed_contrasts = config.fixed_contrasts
        precomputed_calibration_ratios = handoff.calibration_ratios
        precomputed_rt_samples_ms = handoff.rt_samples_ms
        self._gaze = gaze_source
        # Return-jump trials (startup dialog "Return jump"): once the out landing is confirmed the target jumps back to the
        # start symbol and the participant looks back; each trial's one flash goes on the out or the return leg (shuffled bag,
        # RETURN_LEG_FLASH_FRACTION on the return leg). Off = the old alternating dot<->cross trials, unchanged.
        self._return_jump = bool(return_jump)
        # Read once per session (a run never changes them; checkers read these copies after the simulator restores constants).
        self._return_jump_trigger = RETURN_JUMP_TRIGGER
        self._return_flash_sd_ms = float(RETURN_FLASH_SPREAD_SD_MS)
        # Bell-curve width for the flash times (startup dialog, Advanced); None = the FLASH_SPREAD_SD_MS default.
        self._flash_spread_sd_ms = FLASH_SPREAD_SD_MS if flash_spread_sd_ms is None else float(flash_spread_sd_ms)
        self._geometry = config.geometry or Geometry.default()  # sizes the line's offset (screen height / viewing distance)
        self._gaze_latency_fallback_ms = config.gaze_latency_ms
        # New samples with capture times; its latency is snapshotted at each RT attempt / trial start.
        self._feed = SampleFeed(gaze_source, clock)
        self._logger = logger
        self._clock = clock
        self._show_gaze_indicator = config.show_gaze_indicator
        self._phase = Phase.WAITING_TO_START
        self.head_flagged_trials = 0  # trials excluded for a head-speed spike near the flash (session.json head_motion)
        self.head_warnings = 0
        self._head_last_warning_at: float | None = None

        # -- practice (fixed list) then dynamically-generated real trials --
        # Reaction times already measured (the tutorial's practice trials) replace both the RT test and this
        # session's own practice trials; [] means "measured, but nothing usable" (-> DEFAULT_REACTION_TIME_MS).
        self._rt_seeded = precomputed_rt_samples_ms is not None
        num_practice = 0 if (config.skip_practice_trials or self._rt_seeded) else (NUM_PRACTICE_TRIALS_TEST if test_mode else NUM_PRACTICE_TRIALS_REAL)
        self._practice_trials = build_saccade_sequence(num_practice=num_practice, alternate=not self._return_jump)
        self._num_practice = len(self._practice_trials)
        self._practice_index = 0
        # With return jump every trial starts at the dot (the target always jumps back to it).
        self._trial_source: Target = (
            self._practice_trials[-1].target if self._practice_trials and not self._return_jump else Target.DOT
        )
        self._main_trial_counter = 0
        # Fixed probe contrasts (method of constant stimuli) - see SACCADE_FIXED_CONTRASTS. Practice trials use PRACTICE_CONTRAST.
        # `fixed_contrasts` (from the startup dialog) overrides the SACCADE_FIXED_CONTRASTS default.
        contrasts = SACCADE_FIXED_CONTRASTS if fixed_contrasts is None else tuple(fixed_contrasts)
        if not contrasts:
            raise ValueError("at least one saccade contrast is required")
        self._fixed_contrast_bag: ShuffledBag[float] = ShuffledBag(contrasts)
        self._fixed_trials = SACCADE_FIXED_TRIALS_TEST if test_mode else SACCADE_FIXED_TRIALS_REAL
        self._timing_offset_bag: ShuffledBag[int] = ShuffledBag(TIMING_OFFSETS_MS + CONTROL_OFFSETS_MS)
        self._current_trial_spec: TrialSpec | None = None
        # Which leg each trial's flash goes on: practice and main trials have separate bags so the main block's split is exact
        # per pass. Only built (and drawn from) with return jump on, so the off case consumes no extra random numbers.
        self._flash_leg_bag: ShuffledBag[str] | None = None
        self._practice_flash_leg_bag: ShuffledBag[str] | None = None
        if self._return_jump:
            self._flash_leg_bag = ShuffledBag(flash_leg_pool(RETURN_LEG_FLASH_FRACTION))
            self._practice_flash_leg_bag = ShuffledBag(flash_leg_pool(RETURN_LEG_FLASH_FRACTION))
        self.forced_jump_backs = 0  # main trials whose jump back came at RETURN_JUMP_LANDING_TIMEOUT_MS (session.json summary)

        # -- reaction-time test + rolling average (see constants.py) --
        self._rt_test_target_attempts = NUM_RT_TEST_TRIALS_TEST if test_mode else NUM_RT_TEST_TRIALS_REAL
        self._rt_test_attempt = 0
        self._rt_test_onset_detector: OnsetDetector = OnsetDetector(zone_for(Target.DOT))
        self._rt_test_start_time = 0.0
        self._avg_reaction_time_ms: float | None = None
        # One rolling window for the RT test, practice and main trials (and the tutorial's practice samples): every
        # detected eye movement is added and _avg_reaction_time_ms follows at once (see _record_reaction_time).
        self._rt = RollingReactionTime(DEFAULT_REACTION_TIME_MS, seed=precomputed_rt_samples_ms or ())
        if precomputed_rt_samples_ms is not None:
            self._avg_reaction_time_ms = self._rt.average
        # Return-saccade latencies (jump-back frame -> return onset, SIGNED) have their own rolling average: return saccades
        # to a just-left, well-known location are expected to be faster, often anticipatory. It keeps every latency,
        # negative and near-zero ones included (min_plausible_ms=None): an early return is the behaviour being tracked, not
        # a bad measurement. Until it has RT_MIN_SAMPLES_FOR_AVERAGE samples, return-leg flashes are centred on the
        # out-leg average instead (see _return_average_ms).
        self._return_rt = RollingReactionTime(
            DEFAULT_REACTION_TIME_MS, seed=handoff.return_rt_samples_ms or (), min_plausible_ms=None
        )

        self._dot_visible = False
        self._cross_visible = False
        self._calibration = CalibrationRounds(gaze_source)
        self._space_ahead = False  # SPACE pressed after answering, before the next trial is ready
        self.space_tally = SpaceTally(lambda: self._phase.name)
        self._results: list[TrialResult] = []
        self._clear_trial_state()

        # A tutorial run before this session already calibrated: reuse that result.
        self._precomputed_calibration = precomputed_calibration_ratios is not None
        self._calibration_ratios: tuple[float, float, float] | None = None
        if precomputed_calibration_ratios is not None:
            self._gaze.calibrate(*precomputed_calibration_ratios)
            self._calibration_ratios = precomputed_calibration_ratios

    @property
    def calibration_ratios(self) -> tuple[float, float, float] | None:
        """(left, center, right) ratios once calibration has completed (or was seeded); None before."""
        return self._calibration_ratios

    @property
    def gaze_latency_ms(self) -> float:
        """Eye-to-sample latency: the source's own measurement if it has one, else the constructor's value, else 0; clamped to
        [0, GAZE_LATENCY_MAX_MS]."""
        return clamped_gaze_latency_ms(self._gaze, self._gaze_latency_fallback_ms)

    def _clear_trial_state(self) -> None:
        self._trial_start_time = 0.0
        self._feed.last_sample_ts = None
        self._trial_avg_rt_ms: float | None = None
        self._trial_scheduled_delay_ms: float | None = None
        self._trial_gaze_latency_ms: float | None = None
        # The trial's gaze legs (onset/landing/jump back/return): built in _reveal_target_and_start_trial, once the trial's
        # source is known - never read before then.
        self._legs: TrialLegs | None = None
        self._flash_scheduled_at = 0.0
        self._flash_decided = False
        self._flash_shown_at: float | None = None
        self._flash_unix_ms: float | None = None
        self._line_visible = False
        self._line_frames_remaining = 0
        self._line_actually_shown = False
        self._flash_during_saccade: bool | None = None
        self._tracking_lost_near_flash = False
        self._trial_contrast: float | None = None  # contrast actually used for this trial's flash, if any
        self._reaction_latency_ms: float | None = None
        self._onset_detection_lag_ms: float | None = None
        self._response_window_open = False
        self._response_deadline = 0.0
        self._responded = False
        self._answer_side: Side | None = None
        self._response_time_ms: float | None = None
        self._foreperiod_start = 0.0
        self._foreperiod_duration_s = 0.0
        self._flash_leg: str | None = None  # 'out' | 'return': which leg this trial's flash goes on ('out' with return jump off)

    @property
    def _active_legs(self) -> TrialLegs:
        """The running trial's legs; only valid between the go cue and the trial's end."""
        assert self._legs is not None, "no trial running"
        return self._legs

    @property
    def _average_ms(self) -> float:
        """The out-leg average reaction time; set (calibration / RT test / seed) before any main trial starts."""
        assert self._avg_reaction_time_ms is not None, "no reaction-time average yet"
        return self._avg_reaction_time_ms

    @property
    def _active_trial(self) -> TrialSpec:
        """The running trial's spec; only valid once a trial has been started."""
        assert self._current_trial_spec is not None, "no trial started"
        return self._current_trial_spec

    # Read-only views of the trial's legs (tests and the rendering code read these names).
    @property
    def _onset_detector(self) -> OnsetDetector | None:
        return self._legs.onset if self._legs is not None else None

    @property
    def _landed(self) -> bool:
        return self._legs is not None and self._legs.out_landing.landed

    @property
    def _target_landed_since(self) -> float | None:
        return self._legs.out_landing.since if self._legs is not None else None

    @property
    def _jump_at(self) -> float | None:
        return self._legs.jump_at if self._legs is not None else None

    @property
    def _jump_forced(self) -> bool | None:
        return self._legs.jump_forced if self._legs is not None else None

    @property
    def _return_onset_detector(self) -> OnsetDetector | None:
        return self._legs.return_onset if self._legs is not None else None

    @property
    def _return_landed(self) -> bool:
        return self._legs is not None and self._legs.return_landing.landed

    @property
    def results(self) -> list[TrialResult]:
        return self._results

    # -- key handling ----------------------------------------------------------

    def on_space(self) -> None:
        if self._phase is Phase.WAITING_TO_START:
            if self._precomputed_calibration:
                # A tutorial already calibrated - skip straight to the
                # reaction-time test.
                self._begin_after_calibration()
            else:
                self._calibration.start()
                self._phase = Phase.CALIBRATE_LEFT
        elif self._phase in _CALIBRATION_PHASES:
            self._on_calibration(self._calibration.press(self._clock.now(), self.space_tally))
        elif self._phase is Phase.RT_TEST_AWAITING_SPACE:
            self._start_rt_test_foreperiod()
        elif self._phase is Phase.AWAITING_SPACE:
            self._start_foreperiod()
        elif self._phase is Phase.TRIAL_ACTIVE and self._responded:
            # Answered already: this press starts the next trial the moment it is ready (see _deliver_space_ahead).
            self._space_ahead = True
        elif self._phase is not Phase.COMPLETE:
            # Foreperiod: must not restart it or start the next trial. Active phases: responses come from
            # gaze/arrow keys. COMPLETE: the runner decides when to move on.
            self.space_tally.record("during foreperiod" if self._phase in _FOREPERIOD_PHASES else "during trial")

    def _on_calibration(self, final_ratios: tuple[float, float, float] | None) -> None:
        if final_ratios is not None:
            self._calibration_ratios = final_ratios
            self._begin_after_calibration()
        elif self._phase in _CALIBRATION_PHASES:
            self._phase = _CALIBRATION_PHASES[self._calibration.step]

    def _deliver_space_ahead(self) -> None:
        if not self._space_ahead or self._phase is Phase.TRIAL_ACTIVE:
            return
        self._space_ahead = False
        if self._phase in (Phase.AWAITING_SPACE, Phase.RT_TEST_AWAITING_SPACE):
            self.on_space()
        else:
            self.space_tally.record("early press, no trial left to start")

    def _line_state(self, trial: TrialSpec | None) -> dict:
        side = trial.line_side if trial is not None else None
        # The saccade target of the flash's leg: the start symbol for a return-leg flash.
        aim = None if trial is None else (trial.source if self._flash_leg == "return" else trial.target)
        target_position = DOT_POSITION if aim is None or aim is Target.DOT else CROSS_POSITION
        x, y = line_position(target_position, side or Side.ABOVE, self._geometry)
        return {
            "visible": self._line_visible,
            "x": x,
            "y": y,
            "contrast": self._trial_contrast if self._line_visible else 0,
            "side": side,
        }

    def on_response_key(self, answer: Side) -> None:
        """Called when an arrow key is pressed during a trial: UP reports
        `Side.ABOVE`, DOWN reports `Side.BELOW` - see run_experiment.py's key dispatch."""
        if self._phase is not Phase.TRIAL_ACTIVE:
            return
        if self._responded or not self._response_window_open:
            return
        now = self._clock.now()
        if now <= self._response_deadline:
            self._responded = True
            self._answer_side = answer
            self._response_time_ms = s_to_ms(now - self._flash_shown_at) if self._flash_shown_at else None

    def _begin_after_calibration(self) -> None:
        """Measures the participant's reaction time first, unless the tutorial already did (seeded
        samples), in which case it goes straight to the trials."""
        if self._rt_seeded:
            self._begin_next_trial_or_complete()
        else:
            self._begin_rt_test_foreperiod()

    # -- reaction-time test ---------------------------------------------------

    def _begin_rt_test_foreperiod(self) -> None:
        """Shows the dot and waits for SPACE; the foreperiod itself starts on the press."""
        self._clear_trial_state()
        self._phase = Phase.RT_TEST_AWAITING_SPACE
        self._rt_test_onset_detector = OnsetDetector(zone_for(Target.DOT))
        self._dot_visible = True
        self._cross_visible = False

    def _start_foreperiod_timer(self, phase: Phase) -> None:
        self._phase = phase
        self._foreperiod_start = self._clock.now()
        self._foreperiod_duration_s = ms_to_s(random.uniform(FOREPERIOD_MIN_MS, FOREPERIOD_MAX_MS))

    def _start_rt_test_foreperiod(self) -> None:
        self._start_foreperiod_timer(Phase.RT_TEST_FOREPERIOD)
        self._feed.latency_s = timestamp_latency_s(self._gaze, self.gaze_latency_ms)

    def _tick_rt_test_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration_s:
            self._dot_visible = False
            self._cross_visible = True
            self._phase = Phase.RT_TEST_ACTIVE
            self._rt_test_start_time = self._clock.now()
            self._feed.last_sample_ts = getattr(self._gaze.latest_sample(), "timestamp", None)

    def _record_reaction_time(self, reaction_ms: float) -> None:
        """A saccade onset was just confirmed (RT test, practice or main trial, valid or not): add its latency
        to the rolling window and refresh the average at once. The running trial's flash is already scheduled
        and stays put; the next trial's schedule uses the new average."""
        self._avg_reaction_time_ms = self._rt.add(reaction_ms)

    def _tick_rt_test_active(self) -> None:
        now = self._clock.now()
        for sample, sample_t in self._feed.new_samples(self._gaze.latest_sample()):
            if sample_t < self._rt_test_start_time:
                continue  # captured before the go-cue (it just arrived after it): cannot be a reaction to it
            if self._rt_test_onset_detector.update(sample, sample_t):
                since = self._rt_test_onset_detector.since
                assert since is not None  # update() returned True
                self._finish_rt_test_attempt(s_to_ms(since - self._rt_test_start_time))
                return

        if now - self._rt_test_start_time >= ms_to_s(SACCADE_TIMEOUT_MS):
            self._finish_rt_test_attempt(None)

    def _finish_rt_test_attempt(self, reaction_time_ms: float | None) -> None:
        result = TrialResult(
            index=self._rt_test_attempt,
            phase="rt_test",
            source=Target.DOT,
            target=Target.CROSS,
            saccade_duration_ms=None,
            flash_shown=False,
            contrast=None,
            line_side=None,
            responded=False,
            answer_side=None,
            response_time_ms=None,
            outcome="timeout" if reaction_time_ms is None else "detected",
            reaction_latency_ms=reaction_time_ms,
        )
        self._logger.log(result)
        self._results.append(result)
        if reaction_time_ms is not None:
            self._record_reaction_time(reaction_time_ms)

        self._rt_test_attempt += 1
        if self._rt_test_attempt >= self._rt_test_target_attempts:
            # The window already holds these attempts (the cue is the same sharp onset as in the trials), so the
            # first trials blend with them instead of jumping to a single trial's latency.
            self._avg_reaction_time_ms = self._rt.average  # DEFAULT_REACTION_TIME_MS if too few were usable
            self._begin_next_trial_or_complete()
        else:
            self._begin_rt_test_foreperiod()

    # -- trial state machine ----------------------------------------------------

    def _begin_next_trial_or_complete(self) -> None:
        """Starts the next practice trial (fixed list), the next generated real trial, or ends the session."""
        if self._practice_index < len(self._practice_trials):
            self._current_trial_spec = self._practice_trials[self._practice_index]
            self._practice_index += 1
            self._begin_foreperiod()
            return
        if self._should_stop_main_block():
            self._phase = Phase.COMPLETE
            self._log_phase_summary()
            return
        trial, self._trial_source = generate_next_saccade_trial(
            self._main_trial_counter, self._trial_source, self._timing_offset_bag, alternate=not self._return_jump
        )
        self._main_trial_counter += 1
        self._current_trial_spec = trial
        self._begin_foreperiod()

    @property
    def _completed_main_trial_count(self) -> int:
        """Counted from self._results, so it cannot drift from what was logged."""
        return sum(1 for r in self._results if r.phase == "saccade")

    @property
    def _valid_trial_count(self) -> int:
        return sum(1 for r in self._results if r.phase == "saccade" and is_valid_for_saccadic_analysis(r.flash_during_saccade, r.tracking_ok))

    def _log_phase_summary(self) -> None:
        """Final per-phase numbers for _meta.json (loggers without it, like the
        simulator's in-memory one, are skipped)."""
        log = getattr(self._logger, "set_phase_summary", None)
        if log is None:
            return
        log(
            "saccade",
            {
                "valid_trials": self._valid_trial_count,
                "completed_trials": self._completed_main_trial_count,
                "final_avg_reaction_time_ms": self._avg_reaction_time_ms,
                "rt_samples_total": self._rt.total_added,
                "rt_samples_in_window": len(self._rt.samples),
                "rt_samples_dropped_in_window": self._rt.n_dropped,
                "gaze_latency_ms_used": self.gaze_latency_ms,
                **self._return_jump_summary(),
            },
        )

    def _return_jump_summary(self) -> dict:
        if not self._return_jump:
            return {}
        return {
            "final_avg_return_reaction_time_ms": self._return_rt.average if self._return_rt.has_estimate else None,
            "return_rt_samples_total": self._return_rt.total_added,
            "return_rt_samples_in_window": len(self._return_rt.samples),
            "return_rt_samples_dropped_in_window": self._return_rt.n_dropped,
            "forced_jump_backs": self.forced_jump_backs,
        }

    @property
    def return_rt_samples_ms(self) -> list[float]:
        """Return-saccade latencies currently in the return rolling window (oldest first)."""
        return self._return_rt.samples

    def _should_stop_main_block(self) -> bool:
        """Constant stimuli: stop after SACCADE_FIXED_TRIALS_* main trials."""
        return self._completed_main_trial_count >= self._fixed_trials

    def _begin_foreperiod(self) -> None:
        """Shows the current fixation symbol and waits for the participant's SPACE; the
        500-1000 ms foreperiod before the swap starts on the press (_start_foreperiod)."""
        self._clear_trial_state()
        self._phase = Phase.AWAITING_SPACE
        # Explicit: the previous phase may have been the RT test, which tracks no TrialSpec.
        trial = self._active_trial
        self._dot_visible = trial.source is Target.DOT
        self._cross_visible = trial.source is Target.CROSS

    def _start_foreperiod(self) -> None:
        self._start_foreperiod_timer(Phase.FOREPERIOD)

    def _reveal_target_and_start_trial(self) -> None:
        trial = self._active_trial
        if trial.target is Target.DOT:
            self._dot_visible = True
        else:
            self._cross_visible = True
        # The source hides the instant the target appears (not at landing), so the two crossfade (or snap, see SHARP_GO_CUE).
        self._hide_source_symbol()
        self._phase = Phase.TRIAL_ACTIVE
        self._trial_start_time = self._clock.now()
        self._feed.last_sample_ts = self._gaze.latest_sample().timestamp
        self._legs = TrialLegs(
            zone_for(trial.source), zone_for(trial.target), self._trial_start_time, trigger=self._return_jump_trigger
        )
        # Open-loop: the flash fires at a fixed delay from this instant, not when the gaze classifier detects onset. The
        # average is always set here: the RT test (or seeded samples) comes first. A return-leg flash is scheduled the same
        # way but from the jump-back frame, with the return average (_jump_back).
        self._trial_gaze_latency_ms = self.gaze_latency_ms
        self._feed.latency_s = timestamp_latency_s(self._gaze, self._trial_gaze_latency_ms)
        if self._return_jump:
            bag = self._practice_flash_leg_bag if trial.practice else self._flash_leg_bag
            assert bag is not None  # built whenever the return jump is on
            self._flash_leg = bag.draw()
        else:
            self._flash_leg = "out"
        if self._flash_leg == "return":
            self._flash_scheduled_at = math.inf  # scheduled at the jump back
            return
        self._trial_avg_rt_ms = self._avg_reaction_time_ms
        self._trial_scheduled_delay_ms = self._flash_delay_ms(trial, self._average_ms)
        self._flash_scheduled_at = self._trial_start_time + ms_to_s(self._trial_scheduled_delay_ms)

    def _flash_delay_ms(self, trial: TrialSpec, average_ms: float, sd_ms: float | None = None) -> float:
        """Delay (ms) from the leg's start (go cue or jump back) to the flash. A TIMING_OFFSETS_MS label picks a slice of a
        bell curve centred on `average_ms` with SD `sd_ms` (default: the session's out-leg spread; see the constant); a
        control offset (or a practice trial's None = 0) is average + offset + jitter."""
        if trial.timing_offset_ms is not None and trial.timing_offset_ms in TIMING_OFFSETS_MS:
            return spread_flash_delay_ms(
                trial.timing_offset_ms, TIMING_OFFSETS_MS, average_ms,
                self._flash_spread_sd_ms if sd_ms is None else sd_ms, FLASH_WINDOW_MIN_MS, FLASH_WINDOW_MAX_MS,
            )
        offset_ms = trial.timing_offset_ms if trial.timing_offset_ms is not None else 0  # 0 is itself a valid offset
        jitter_ms = random.uniform(-TIMING_JITTER_MS, TIMING_JITTER_MS) if TIMING_JITTER_MS else 0.0
        return average_ms + offset_ms + jitter_ms

    def _return_average_ms(self) -> float:
        """Centre for a return-leg flash: the return-saccade average once it rests on enough samples, else the out-leg one."""
        return self._return_rt.average if self._return_rt.has_estimate else self._average_ms

    def tick(self) -> None:
        self._on_calibration(self._calibration.poll())
        if self._phase is Phase.RT_TEST_FOREPERIOD:
            self._tick_rt_test_foreperiod()
        elif self._phase is Phase.RT_TEST_ACTIVE:
            self._tick_rt_test_active()
        elif self._phase is Phase.FOREPERIOD:
            self._tick_foreperiod()
        elif self._phase is Phase.TRIAL_ACTIVE:
            self._tick_trial_active()
        self._deliver_space_ahead()

    def _tick_foreperiod(self) -> None:
        if self._clock.now() - self._foreperiod_start >= self._foreperiod_duration_s:
            self._reveal_target_and_start_trial()

    def _tick_trial_active(self) -> None:
        trial = self._active_trial
        sample = self._gaze.latest_sample()
        now = self._clock.now()
        new_samples = self._feed.new_samples(sample)

        self._measure_reaction(new_samples)
        flash_fired_this_tick = self._fire_flash_if_due(trial, now)

        # Frame-counted (not time-based) so the flash lasts exactly N drawn frames whatever the refresh rate; skipped on the
        # tick that just turned it on, so it is not docked a frame before it has been drawn once.
        if self._line_visible and not flash_fired_this_tick:
            self._line_frames_remaining -= 1
            if self._line_frames_remaining <= 0:
                self._line_visible = False

        if self._response_window_open and now > self._response_deadline:
            self._response_window_open = False

        if any(
            not s.face_found and abs(s_t - self._flash_scheduled_at) <= ms_to_s(TRACKING_LOSS_WINDOW_MS)
            for s, s_t in new_samples
        ):
            self._tracking_lost_near_flash = True

        if self._return_jump:
            self._tick_return_jump(trial, new_samples, now)
            return

        # Landing is flagged at once but the trial is not finalized: a real reaction time can outlast how quickly landing is
        # detected, so ending the trial here would cut off a still-valid response.
        self._active_legs.track_out_landing(new_samples)

        if self._active_legs.out_timed_out(now):
            self._finish_trial()
            return

        # Also requires the scheduled flash to have fired: a fast lander (common at a +40 ms offset against a stale
        # average) cannot finalize before their flash happens.
        if self._landed and self._flash_decided and (self._responded or not self._response_window_open):
            self._finish_trial()

    def _tick_return_jump(self, trial: TrialSpec, new_samples, now: float) -> None:
        """The rest of a return-jump trial tick: out arrival / onset -> jump back -> return onset/landing -> finish."""
        legs = self._active_legs
        legs.track_arrival(new_samples)
        # An out-leg flash is drawn (and gone) before the target jumps, so the jump never overlaps it.
        out_flash_done = self._flash_leg == "return" or (self._flash_decided and not self._line_visible)
        forced = legs.jump_due(now, out_flash_done)
        if forced is not None:
            self._jump_back(trial, now, forced=forced)

        # The return leg is listened to from the out arrival (or the jump, if that came first): a predictive return
        # saccade may start BEFORE the jump frame (negative return latency), e.g. while a late out-leg flash holds the jump.
        return_latency_ms, just_landed = legs.track_return(new_samples)
        if return_latency_ms is not None and not legs.jump_forced:
            # After a forced jump the eye may not be on the target: no clean return latency.
            self._return_rt.add(return_latency_ms)
        if legs.jump_at is None:
            return
        if just_landed and self._flash_decided and not self._responded:
            # The answer is asked once the eye is back: the full window counts from the return landing.
            self._response_window_open = True
            self._response_deadline = max(self._response_deadline, now + ms_to_s(RESPONSE_WINDOW_MS))

        if legs.return_timed_out(now):
            self._finish_trial()
            return
        if legs.return_landing.landed and self._flash_decided and (self._responded or not self._response_window_open):
            self._finish_trial()

    def _jump_back(self, trial: TrialSpec, now: float, forced: bool) -> None:
        """The target jumps back: the start symbol reappears (the new saccade target), the one just reached disappears.
        A return-leg flash is scheduled from this frame."""
        self._active_legs.mark_jump(now, forced)
        if forced and not trial.practice:
            self.forced_jump_backs += 1
        self._dot_visible = trial.source is Target.DOT
        self._cross_visible = trial.source is Target.CROSS
        if self._flash_leg == "return":
            average_ms = self._return_average_ms()
            if average_ms <= 0 and trial.timing_offset_ms in TIMING_OFFSETS_MS:
                delay_ms = 0.0  # anticipatory returns: the jump frame is the earliest a return flash can be
            else:
                delay_ms = max(0.0, self._flash_delay_ms(trial, average_ms, self._return_flash_sd_ms))
            self._trial_avg_rt_ms = average_ms
            # Still flash minus GO CUE (analyze_session derives the go cue from it); the jump is in jump_back_ms_after_go_cue.
            self._trial_scheduled_delay_ms = s_to_ms(now - self._trial_start_time) + delay_ms
            self._flash_scheduled_at = now + ms_to_s(delay_ms)

    def _head_tracker(self):
        """The source's head-movement monitor (Neon IMU), or None when the source has none / it switched itself off."""
        return getattr(self._gaze, "head_motion", None)

    def _head_speed_spike_near_flash(self) -> bool:
        """True if head angular speed exceeded HEAD_MAX_SPEED_DPS within TRACKING_LOSS_WINDOW_MS of the flash: the scene-camera gaze is then
        shifted by head, not eye, movement, so the trial counts as tracking-not-ok (excluded by the same gates as a tracking loss)."""
        tracker = self._head_tracker()
        if tracker is None or not self._line_actually_shown:
            return False
        window = ms_to_s(TRACKING_LOSS_WINDOW_MS)
        return tracker.speed_spike_between(self._flash_scheduled_at - window, self._flash_scheduled_at + window, HEAD_MAX_SPEED_DPS)

    def pop_head_warning(self) -> str | None:
        """The "keep your head still" text, once, when the head has rotated more than HEAD_DRIFT_WARN_DEG from its calibration orientation;
        only between trials (waiting for SPACE) and at most once per HEAD_WARN_MIN_INTERVAL_S. The runner speaks it (speech only, no
        caption: real trials hide the caption) and prints it. None otherwise."""
        tracker = self._head_tracker()
        if tracker is None or self._phase is not Phase.AWAITING_SPACE:
            return None
        drift = tracker.drift_deg
        now = self._clock.now()
        if drift is None or drift <= HEAD_DRIFT_WARN_DEG:
            return None
        if self._head_last_warning_at is not None and now - self._head_last_warning_at < HEAD_WARN_MIN_INTERVAL_S:
            return None
        self._head_last_warning_at = now
        self.head_warnings += 1
        return HEAD_WARN_TEXT

    def _measure_reaction(self, new_samples) -> None:
        """Onset detection runs every tick purely as a measurement (reaction_latency_ms, the rolling RT average and one
        bound of the flash_during_saccade window); it no longer triggers the flash. Out leg only (TrialLegs.measure_out_onset)."""
        measured = self._active_legs.measure_out_onset(new_samples)
        if measured is not None:
            self._reaction_latency_ms, lag_ms = measured
            self._record_reaction_time(self._reaction_latency_ms)
            self._onset_detection_lag_ms = lag_ms

    def _fire_flash_if_due(self, trial: TrialSpec, now: float) -> bool:
        """Open-loop: the flash fires once, at its scheduled absolute time, independent of onset detection."""
        if self._flash_decided or now < self._flash_scheduled_at:
            return False
        self._flash_decided = True
        self._response_window_open = True
        self._response_deadline = now + ms_to_s(RESPONSE_WINDOW_MS)
        self._flash_shown_at = now
        self._flash_unix_ms = self._clock.unix_ms()
        self._line_visible = True
        self._line_frames_remaining = LINE_DURATION_FRAMES
        self._line_actually_shown = True
        if trial.practice:
            self._trial_contrast = PRACTICE_CONTRAST
        else:
            self._trial_contrast = self._fixed_contrast_bag.draw()
        return True

    def _hide_source_symbol(self) -> None:
        trial = self._active_trial
        if trial.source is Target.DOT:
            self._dot_visible = False
        else:
            self._cross_visible = False

    def _flash_leg_onset_landing(self) -> tuple[float | None, float | None]:
        """(onset, first-in-zone landing) capture times of the saccade on the flash's leg."""
        return self._active_legs.onset_landing(self._flash_leg)

    def _compute_flash_during_saccade(self) -> bool | None:
        """Whether the scheduled flash time fell within [onset, landing] (the "first observed" capture times) of the flash's
        leg. Only meaningful for trials that flashed. False if onset was never detected; None if landing was never confirmed
        (undeterminable)."""
        if not self._line_actually_shown:
            return None
        onset, landed_since = self._flash_leg_onset_landing()
        if onset is None:
            return False
        if landed_since is None:
            return None
        return onset <= self._flash_scheduled_at <= landed_since

    def _finish_trial(self) -> None:
        trial = self._active_trial
        if not self._active_legs.leg_moved(self._flash_leg) or not self._line_actually_shown:
            outcome = "timeout"  # no saccade (on the flash's leg), or the trial ended before its flash fired
        else:
            outcome = score_outcome(self._responded, self._answer_side, trial.line_side)
        self._flash_during_saccade = self._compute_flash_during_saccade()
        head_moved = self._head_speed_spike_near_flash()
        if head_moved and not trial.practice:
            self.head_flagged_trials += 1
        tracking_ok = (not (self._tracking_lost_near_flash or head_moved)) if self._line_actually_shown else None

        if not trial.practice:
            result = saccade_trial_result(
                trial,
                self._active_legs,
                outcome=outcome,
                flash_shown=self._line_actually_shown,
                flash_leg=self._flash_leg,
                flash_scheduled_at=self._flash_scheduled_at,
                flash_unix_ms=self._flash_unix_ms,
                contrast=self._trial_contrast,
                flash_during_saccade=self._flash_during_saccade,
                tracking_ok=tracking_ok,
                responded=self._responded,
                answer_side=self._answer_side,
                response_time_ms=self._response_time_ms,
                reaction_latency_ms=self._reaction_latency_ms,
                onset_detection_lag_ms=self._onset_detection_lag_ms,
                avg_rt_used_ms=self._trial_avg_rt_ms,
                scheduled_delay_ms=self._trial_scheduled_delay_ms,
                gaze_latency_ms=self._trial_gaze_latency_ms,
            )
            self._logger.log(result)
            self._results.append(result)

        self._line_visible = False
        self._begin_next_trial_or_complete()

    # -- rendering ----------------------------------------------------------

    def _summary(self) -> str:
        # Accuracy = (times the arrow-key response correctly reported the
        # line's side) divided by (times the line was actually
        # shown) - "incorrect" (wrong side guessed) and "miss" (no
        # response at all) both count against it.
        correct = sum(1 for r in self._results if r.outcome == "correct")
        incorrect = sum(1 for r in self._results if r.outcome == "incorrect")
        misses = sum(1 for r in self._results if r.outcome == "miss")

        flash_shown_count = correct + incorrect + misses
        accuracy = correct / flash_shown_count if flash_shown_count else 0.0
        return f"Accuracy: {correct}/{flash_shown_count} ({accuracy:.0%})"

    def _instructions(self) -> str:
        if self._phase is Phase.COMPLETE:
            return f"{words.FINISHED} {self._summary()} — Press SPACE to see your results"
        # The answer text is shared by AWAITING_SPACE (after the first trial), FOREPERIOD and TRIAL_ACTIVE
        # so it is spoken once and never flickers or repeats from trial to trial.
        if self._phase is Phase.AWAITING_SPACE:
            if self._is_first_trial():
                return words.FIRST_TRIAL_RETURN_JUMP if self._return_jump else words.FIRST_TRIAL
            return _ANSWER_TEXT
        if self._phase in (Phase.FOREPERIOD, Phase.TRIAL_ACTIVE):
            return _ANSWER_TEXT
        if self._phase in _RT_TEST_PHASES:
            # Said once, before the first attempt; silent afterwards (the narrator re-speaks on every text change).
            return words.RT_TEST_INTRO if self._rt_test_attempt == 0 else ""
        if self._phase in _CALIBRATION_PHASES:
            return calibration_text(self._calibration.step, self._calibration.round)
        return {
            Phase.WAITING_TO_START: "Press SPACE to begin" if self._precomputed_calibration else "Press SPACE to begin calibration",
        }[self._phase]

    def _trial_label(self) -> str:
        if self._phase in _RT_TEST_PHASES:
            return f"RT test {self._rt_test_attempt + 1}/{self._rt_test_target_attempts}"
        trial = self._active_trial
        if trial.practice:
            return f"practice {self._practice_index}/{self._num_practice}"
        return f"trial {self._main_trial_counter} (valid: {self._valid_trial_count})"

    def _symbol_visibility(self) -> tuple[bool, bool, bool]:
        """(dot, cross, calibration_center). The center calibration target has its own stimulus: reusing "dot" made it teleport
        with no fade cue that a new step had started."""
        if self._phase is Phase.CALIBRATE_LEFT:
            return True, False, False
        if self._phase is Phase.CALIBRATE_CENTER:
            return False, False, True
        if self._phase is Phase.CALIBRATE_RIGHT:
            return False, True, False
        if self._phase in _TRIAL_PHASES:
            return self._dot_visible, self._cross_visible, False
        return False, False, False

    def _gaze_indicator_state(self, sample) -> dict:
        """Opt-in live gaze cursor, interpolated from GazeSample.smoothed_position. Display only: onset/landing detection
        uses the debounced zone and the raw position, never this smoothing."""
        show = (
            self._show_gaze_indicator
            and self._phase in _MAIN_TRIAL_PHASES
            and sample.face_found
            and sample.smoothed_position is not None
            and self._calibration_ratios is not None
        )
        if not show:
            return {"visible": False, "x": 0.0, "y": 0.0}

        # Dot and cross share y (only horizontal gaze is tracked): interpolate x by the smoothed position (0=dot, 1=cross).
        x = DOT_POSITION[0] + (CROSS_POSITION[0] - DOT_POSITION[0]) * sample.smoothed_position
        return {"visible": True, "x": x, "y": DOT_POSITION[1]}

    def render_state(self) -> dict:
        sample = self._gaze.latest_sample()
        dot_visible, cross_visible, calibration_center_visible = self._symbol_visibility()
        trial = self._current_trial_spec if self._phase in _MAIN_TRIAL_PHASES else None

        return {
            "instructions": self._instructions(),
            "hide_caption": self._hide_caption(),
            "dot": {"visible": dot_visible, "x": DOT_POSITION[0], "y": DOT_POSITION[1]},
            "cross": {"visible": cross_visible, "x": CROSS_POSITION[0], "y": CROSS_POSITION[1]},
            "calibration_center": {
                "visible": calibration_center_visible,
                "x": CENTER_POSITION[0],
                "y": CENTER_POSITION[1],
            },
            # The flashed line: centred LINE_OFFSET_DEG (constants.py) above/below the
            # saccade TARGET (the symbol the eye is moving to), at the trial's contrast.
            "line": self._line_state(trial),
            "gaze_indicator": self._gaze_indicator_state(sample),
            # Renderer hint: show/hide dot and cross instantly (no crossfade) so
            # the go-cue has a definite onset - see SHARP_GO_CUE. Calibration
            # steps keep the fade (it signals "new step").
            "snap_symbols": SHARP_GO_CUE and self._phase in _TRIAL_PHASES,
            "hud": {
                "phase": self._phase.name,
                "gaze_zone": sample.zone.value,
                "face_found": sample.face_found,
                "source_available": self._gaze.is_available,
                "trial": self._trial_label() if self._has_trial_label() else "-",
                "trial_index": self._main_trial_counter,
                "calibration_wait_s": self._calibration.latch.waiting_s(self._clock.now()),
                **self._head_hud(),
            },
        }

    def _head_hud(self) -> dict:
        tracker = self._head_tracker()
        return {} if tracker is None else {"head_drift_deg": tracker.drift_deg}

    def _is_first_trial(self) -> bool:
        return self._practice_index + self._main_trial_counter <= 1

    def _hide_caption(self) -> bool:
        """True during the real (non-practice) main-block trials: the runner then neither draws nor speaks the bottom caption.
        Not on the very first screen of the block (after the tutorial there are no practice trials), which must say to press SPACE."""
        if self._phase is Phase.AWAITING_SPACE and self._is_first_trial():
            return False
        return self._phase in _MAIN_TRIAL_PHASES and not self._active_trial.practice

    def _has_trial_label(self) -> bool:
        return self._phase in _TRIAL_PHASES
