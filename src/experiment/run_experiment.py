"""Entry point (`-m src.experiment.run_experiment`): startup dialog, PsychoPy window/stimuli, and the per-phase frame loops that call each session's
tick() + render_state() and draw the result. All experiment logic lives in the session classes, not here - this is the only module that needs PsychoPy."""

import sys
from functools import partial

import cv2
import pyglet
from PIL import Image as PILImage
from psychopy import core, event, gui, sound, visual

from include.experiment.constants import (
    BACKGROUND_LUMINANCE,
    CIRCLE_RADIUS_RATIO,
    FEEDBACK_FLASH_DURATION_FRAMES,
    FEEDBACK_FLASH_GREEN,
    FEEDBACK_FLASH_RED,
    GAZE_INDICATOR_OPACITY,
    GAZE_INDICATOR_RADIUS_RATIO,
    ABOVE_RESPONSE_KEYS,
    BELOW_RESPONSE_KEYS,
    LINE_HEIGHT_HEIGHT_UNITS,
    LINE_WIDTH_SCREENS,
    WARNING_TONE_DURATION_S,
    WARNING_TONE_HZ,
)
from include.experiment.types import Side
from include.eye_tracking.constants import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH
from src.eye_tracking.gaze_factory import (
    BACKEND_NEON,
    BACKENDS,
    NEON_BACKENDS,
    build_gaze_source,
    list_cameras,
)
from src.eye_tracking.video_recorder import VideoRecorder
from src import timebase
from src.experiment.glasses_video import END_FLASH_COUNT, SYNC_FLASH_COUNT, sync_flash_frames
from src.experiment.logger import ResultLogger
from src.experiment.narrator import Narrator
from src.experiment.pausable_clock import PausableClock
from src.experiment.presaccade_session import PresaccadeSession
from src.experiment.results_graph import build_comparison_graph
from src.experiment.run_summary import finish_run_summary
from src.experiment.session import ExperimentSession
from src.experiment.session_schedule import PHASE_BASELINE, PHASE_SACCADE, phase_order
from src.experiment.space_input import NarrationKeyGate, deliver_space, leaves_complete_screen
from src.experiment.settings import (
    DEFAULT_SACCADE_CONTRAST_MAX_PERCENT,
    DEFAULT_SACCADE_CONTRAST_MIN_PERCENT,
    load_contrast_floor_percent,
    resolve_saccade_contrasts,
)
from src.experiment.tutorial_session import TutorialSession

FADE_DURATION_S = 0.4  # dot/cross opacity fade; softens onset so it doesn't trigger a reflexive saccade

# Full-screen pure-green rectangle: the glasses-video sync flash (run_sync_flashes) - never shown during trials.
START_FLASH_COLOR = [-1, 1, -1]  # pure green
SYNC_FLASH_ANNOUNCEMENT = "A green flash will show."  # the only spoken line of the sync step (the flashes themselves are silent)
SYNC_LEAD_IN_FRAMES = 30  # dark frames after the announcement / before the first flash, so the phone is certainly recording

# Debug camera preview (top-left corner) showing the raw feed the tracker is
# seeing, for calibration/testing - not shown to real participants mid-run.
CAMERA_PREVIEW_HEIGHT = 0.22  # fraction of window height
CAMERA_PREVIEW_MARGIN = 0.02
CAMERA_PREVIEW_FRAME_ASPECT = FRAME_WIDTH / FRAME_HEIGHT


def camera_preview_enabled(trial_index: int) -> bool:
    """Single toggle point for the debug camera preview (on for trial 0 only)."""
    return trial_index == 0


# Answer keys: UP = the line was above the target, DOWN = below (LEFT/RIGHT are not used).
_RESPONSE_KEY_SIDE = {key: Side.ABOVE for key in ABOVE_RESPONSE_KEYS} | {key: Side.BELOW for key in BELOW_RESPONSE_KEYS}
RESPONSE_KEYS = list(_RESPONSE_KEY_SIDE)

# Phase names (ExperimentSession.Phase.name / TutorialSession.Phase.name)
# whose entry is a target/cross appearing and should play the go-cue beep -
# see phase_just_became_active/build_go_cue_tone.
_GO_CUE_PHASES = frozenset({"TRIAL_ACTIVE", "RT_TEST_ACTIVE"})
_TUTORIAL_GO_CUE_PHASES = frozenset({"PRACTICE_ACTIVE"})


def dispatch_response_keys(session, keys: list[str]) -> None:
    for key in keys:
        side = _RESPONSE_KEY_SIDE.get(key)
        if side is not None:
            session.on_response_key(side)


def narrate_if_changed(narrator: Narrator, text: str, last_spoken: str | None) -> str | None:
    """Speaks `text` only if it differs from what was last spoken and returns the new last-spoken value. Instructions stay
    constant across many frames, so speaking unconditionally would queue the same phrase over and over. Blank text is never
    spoken and does not count as spoken (the caption keeps showing the previous, already-read text)."""
    if not text:
        return last_spoken
    if text != last_spoken:
        narrator.speak(text)
    return text


def update_caption(
    narrator: Narrator, caption_stim, state: dict, last_spoken: str | None, last_caption: str
) -> tuple[str | None, str]:
    """Per-frame caption: shown and spoken in lockstep. While state["hide_caption"] is set (the real measured trials) it is
    hidden and silent, and both memories reset so the next caption (e.g. the end-of-phase message) is shown and spoken even
    if it repeats an earlier string. Returns (last_spoken, last_caption)."""
    if state.get("hide_caption"):
        caption_stim.text = ""
        caption_stim.opacity = 0.0
        return None, ""
    last_spoken = narrate_if_changed(narrator, state["instructions"], last_spoken)
    last_caption = render_caption(caption_stim, state["instructions"], last_caption)
    return last_spoken, last_caption


def render_caption(caption_stim, text: str, last_caption: str) -> str:
    """Shows the last non-blank instructions text and returns it. state["instructions"] is "" to stop the narrator
    re-announcing something unchanged, not to say "nothing to show", so a blank keeps the previous caption on screen."""
    caption = text if text else last_caption
    caption_stim.text = caption
    caption_stim.opacity = 1.0 if caption else 0.0
    return caption


def build_go_cue_tone() -> sound.Sound:
    """The go-cue beep, played the instant the target/cross appears: it anchors the reaction-time measurement (see WARNING_TONE_HZ)."""
    return sound.Sound(value=WARNING_TONE_HZ, secs=WARNING_TONE_DURATION_S)


def phase_just_became_active(phase: str, last_phase: str | None, active_phases: frozenset[str]) -> bool:
    """True on the frame `phase` enters `active_phases` (a target/cross appearing): fires the go-cue beep."""
    return phase in active_phases and last_phase not in active_phases


def luminance_to_color(luminance: float) -> list[float]:
    """Convert a 0 (black) - 1 (white) luminance into PsychoPy's -1..1 color space."""
    value = luminance * 2 - 1
    return [value, value, value]


def ratio_to_pos(x_ratio: float, y_ratio: float, aspect: float) -> tuple[float, float]:
    """Convert an (xRatio, yRatio) 0-1 fraction-of-screen position into PsychoPy
    'height' units, where +y is up and 1.0 horizontal unit spans `aspect` (the
    window's width/height ratio) so circles drawn with a plain radius stay round."""
    x = (x_ratio - 0.5) * aspect
    y = (0.5 - y_ratio) * 1.0
    return x, y


def _second_monitor_screen_index() -> int:
    """PsychoPy's Window(screen=N) is 0-indexed into pyglet's screen list -
    1 selects the second monitor if one is connected, falling back to the
    primary (0) if there's only one."""
    screens = pyglet.canvas.get_display().get_screens()
    return 1 if len(screens) > 1 else 0


def build_window(fullscreen: bool = True) -> visual.Window:
    kwargs = {
        "color": luminance_to_color(BACKGROUND_LUMINANCE),
        "units": "height",
        "allowGUI": False,
        "screen": _second_monitor_screen_index(),
    }
    if fullscreen:
        kwargs["fullscr"] = True
    else:
        kwargs["fullscr"] = False
        kwargs["size"] = (800, 600)
    return visual.Window(**kwargs)


def build_stimuli(win: visual.Window, show_hud: bool = False) -> dict:
    """`show_hud`: draw the top-left debug text (Test mode only; apply_render_state honours it)."""
    aspect = win.size[0] / win.size[1]
    return {
        "aspect": aspect,
        "show_hud": show_hud,
        "start_flash": visual.Rect(
            win, width=aspect * 2 + 1, height=2, fillColor=START_FLASH_COLOR, lineColor=None, opacity=0
        ),
        # Tutorial-only correctness feedback: a full-screen rect whose colour is set per frame.
        "feedback_flash": visual.Rect(win, width=aspect * 2 + 1, height=2, lineColor=None, opacity=0),
        # Tutorial-only larger instruction text; `pos` is set per frame from render_state()'s "center"/"bottom".
        "tutorial_text": visual.TextStim(win, text="", pos=(0, -0.4), color="white", height=0.04, opacity=0),
        # The narrated text (state["instructions"]), shown in every phase. Bottom-anchored so the longest (multi-line)
        # results summary grows upward instead of off-screen.
        "caption": visual.TextStim(
            win, text="", pos=(0, -0.46), color="white", height=0.028, wrapWidth=1.6, opacity=0, anchorVert="bottom"
        ),
        "dot": visual.Circle(win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0),
        "cross": visual.Circle(win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0),
        "calibration_center": visual.Circle(
            win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0
        ),
        "gaze_indicator": visual.Circle(
            win, radius=GAZE_INDICATOR_RADIUS_RATIO, fillColor="white", lineColor=None, opacity=0
        ),
        # Braun, Schutz & Gegenfurtner (2021)'s probe: a thin horizontal line with a vertical Gaussian profile, a texture-less
        # GratingStim under the 'gauss' mask sized (very wide, 6 SDs tall). `contrast` scales the white field around the
        # mid-grey background, so the peak luminance step is contrast/2 (a light line).
        "grating": visual.GratingStim(
            win,
            tex=None,
            mask="gauss",
            size=(LINE_WIDTH_SCREENS * aspect, LINE_HEIGHT_HEIGHT_UNITS),
            color="white",
            contrast=0,
            opacity=0,
        ),
        "hud": visual.TextStim(
            win,
            text="",
            pos=(-aspect / 2 + 0.02, 0.47),
            color="white",
            height=0.02,
            alignText="left",
            anchorHoriz="left",
        ),
    }


def _move_toward(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def apply_render_state(state: dict, stimuli: dict, fade_opacities: dict, dt: float) -> None:
    aspect = stimuli["aspect"]
    fade_step = dt / FADE_DURATION_S

    # calibration_center is only in the saccade phase's render_state(); default it to hidden.
    for name in ("dot", "cross", "calibration_center"):
        symbol_state = state.get(name, {"visible": False, "x": 0.0, "y": 0.0})
        target_opacity = 1.0 if symbol_state["visible"] else 0.0
        if state.get("snap_symbols"):
            fade_opacities[name] = target_opacity  # sharp go-cue: instant on/off (see SHARP_GO_CUE)
        else:
            fade_opacities[name] = _move_toward(fade_opacities[name], target_opacity, fade_step)
        stim = stimuli[name]
        stim.opacity = fade_opacities[name]
        stim.pos = ratio_to_pos(symbol_state["x"], symbol_state["y"], aspect)

    # Saccade phase only. Snaps (no fade): it is a live cursor, smoothing would make it lag.
    gaze_indicator = stimuli["gaze_indicator"]
    indicator_state = state.get("gaze_indicator")
    if indicator_state and indicator_state["visible"]:
        gaze_indicator.opacity = GAZE_INDICATOR_OPACITY
        gaze_indicator.pos = ratio_to_pos(indicator_state["x"], indicator_state["y"], aspect)
    else:
        gaze_indicator.opacity = 0.0

    grating = stimuli["grating"]
    grating.opacity = 1.0 if state["grating"]["visible"] else 0.0
    if state["grating"]["visible"]:
        # Full screen width, so only the vertical position matters (x is just the target the line belongs to).
        grating.pos = (0.0, ratio_to_pos(state["grating"]["x"], state["grating"]["y"], aspect)[1])
        # The window background is mid-grey, so the line's contrast is its peak contrast around it.
        grating.contrast = state["grating"]["contrast"]

    # Debug text: Test mode only (stimuli["show_hud"], set in build_stimuli); never drawn in a real run.
    if stimuli.get("show_hud"):
        hud = state["hud"]
        stimuli["hud"].text = (
            f"phase: {hud['phase']} | trial: {hud['trial']} | gaze: {hud['gaze_zone']} | "
            f"face: {'yes' if hud['face_found'] else 'no'} | source: {'ok' if hud['source_available'] else 'unavailable'}"
            f"{_calibration_wait_text(hud.get('calibration_wait_s'))}"
        )
        stimuli["hud"].opacity = 1.0
    else:
        stimuli["hud"].opacity = 0.0

    # Tutorial only (absent from the other sessions' render_state()).
    tutorial_text = state.get("tutorial_text")
    tutorial_text_stim = stimuli["tutorial_text"]
    if tutorial_text and tutorial_text["visible"]:
        tutorial_text_stim.text = tutorial_text["text"]
        tutorial_text_stim.pos = (0, 0) if tutorial_text["position"] == "center" else (0, -0.4)
        tutorial_text_stim.opacity = 1.0
    else:
        tutorial_text_stim.opacity = 0.0


def _calibration_wait_text(waited_s: float | None) -> str:
    """HUD suffix while a calibration SPACE press is latched, waiting for the gaze window to fill."""
    if waited_s is None:
        return ""
    return f" | SPACE pressed, waiting for gaze samples {waited_s:.1f}s" + (" - TRACKING LOST?" if waited_s > 3.0 else "")


def drain_stale_space(session) -> None:
    """Space pressed before the phase's first screen existed (during start-up) answers nothing on screen:
    discard it, but count it so it is visible in the console and the meta file."""
    session.space_tally.record("pressed before the phase started", len(event.getKeys(keyList=["space"])), phase="(start-up)")


def report_space_presses(session, label: str, logger: ResultLogger | None = None) -> None:
    print(session.space_tally.summary(label))
    if logger is not None and session.space_tally.total:
        logger.set_ignored_space_presses(label, session.space_tally.as_dict())


def measure_refresh_hz(win: visual.Window) -> float | None:
    """The display's real refresh rate (None if PsychoPy cannot measure it): flash lengths are counted in frames from this."""
    try:
        return float(win.getActualFrameRate())
    except Exception:  # noqa: BLE001
        return None


def run_sync_flashes(win: visual.Window, stimuli: dict, logger: ResultLogger, narrator: Narrator, count: int, label: str, announce: bool = True) -> None:
    """A short step of its own, outside any trial: `count` full-screen green flashes (150 ms on, 300 ms dark, counted in
    frames) that the glasses' scene camera sees, so the phone's video can be lined up with the PC clock afterwards (see
    glasses_video.py). The PC wall-clock time of each flash's first frame is logged right after the flip that shows it.
    Silent except for one optional announcement line."""
    flash = stimuli["start_flash"]
    on_frames, gap_frames = sync_flash_frames(logger.display_refresh_hz)

    def blank_frames(n: int) -> None:
        flash.opacity = 0.0
        for _ in range(n):
            win.flip()

    if announce:
        narrator.speak(SYNC_FLASH_ANNOUNCEMENT)
        waited = core.Clock()
        while narrator.is_speaking and waited.getTime() < 5.0:
            win.flip()
    blank_frames(SYNC_LEAD_IN_FRAMES)
    for i in range(count):
        flash.opacity = 1.0
        for frame in range(on_frames):
            flash.draw()
            win.flip()
            if frame == 0:
                logger.add_sync_flash(f"{label} {i + 1}", timebase.unix_ms())
        blank_frames(gap_frames)
    flash.opacity = 0.0


class FramePresenter:
    """The per-frame drawing state every phase loop shares: caption/narration memory, symbol fades and the frame clock."""

    def __init__(self, stimuli: dict, narrator: Narrator) -> None:
        self.stimuli = stimuli
        self.narrator = narrator
        self.fade_opacities = {"dot": 0.0, "cross": 0.0, "calibration_center": 0.0}
        self.frame_clock = core.Clock()
        self.last_spoken: str | None = None
        self.last_caption = ""

    def update(self, state: dict) -> None:
        """Caption + speech, then every stimulus from `state` (call draw_all and flip afterwards)."""
        self.last_spoken, self.last_caption = update_caption(
            self.narrator, self.stimuli["caption"], state, self.last_spoken, self.last_caption
        )
        dt = self.frame_clock.getTime()
        self.frame_clock.reset()
        apply_render_state(state, self.stimuli, self.fade_opacities, dt)
        self.stimuli["start_flash"].opacity = 0.0  # only run_sync_flashes shows it, never a phase's own frames


def deliver_keys(session, keys: list[str]) -> bool:
    """Forward SPACE and the answer keys to a baseline/saccade session; True if a SPACE press was delivered."""
    space_delivered = deliver_space(session, keys.count("space"))  # no-op if already COMPLETE
    dispatch_response_keys(session, keys)
    return space_delivered


def draw_all(stimuli: dict) -> None:
    # start_flash/feedback_flash first so they sit behind the fixation
    # stimuli, which stay visible/fixatable on top of them during a flash.
    for name in (
        "start_flash",
        "feedback_flash",
        "dot",
        "cross",
        "calibration_center",
        "gaze_indicator",
        "grating",
        "hud",
        "tutorial_text",
        "caption",
    ):
        stimuli[name].draw()


def run_presaccade_phase(
    win: visual.Window,
    stimuli: dict,
    logger: ResultLogger,
    narrator: Narrator,
    contrast_floor: float | None = None,
    test_mode: bool = False,
    gaze_backend: str = BACKEND_NEON,
    camera_index: int = CAMERA_INDEX,
    calibration_ratios: tuple[float, float, float] | None = None,
) -> PresaccadeSession | None:
    """Phase 1: fixate center, detect flashes. No eye tracking, unless `calibration_ratios` is known (tutorial ran, or the saccade phase went
    first): then a gaze source is built, calibrated and used only to hold a due flash until gaze is on center (fixation gating - see
    PresaccadeSession); if it can't start, the baseline runs ungated. Returns the completed
    session, or None if the participant quit early."""
    gaze = _start_gating_gaze(win, gaze_backend, camera_index, calibration_ratios)
    try:
        return _run_presaccade_loop(win, stimuli, logger, narrator, contrast_floor, test_mode, gaze)
    finally:
        if gaze is not None:
            gaze.stop()


def _start_gating_gaze(win, gaze_backend: str, camera_index: int, calibration_ratios):
    """The calibrated gaze source for baseline fixation gating, or None (ungated baseline) when there is no calibration to classify zones with
    or the source won't start - gating is a refinement, so a failure here must not cost the participant their session."""
    if calibration_ratios is None:
        return None
    gaze = None
    try:
        gaze = build_gaze_source(gaze_backend, camera_index)
        gaze.start()
        gaze.calibrate(*calibration_ratios)
        return gaze
    except Exception as exc:  # noqa: BLE001 - any hardware/SDK failure falls back to the ungated baseline
        print(f"WARNING: baseline fixation gating unavailable ({exc!r}); running the baseline ungated", file=sys.stderr)
        if gaze is not None:
            try:
                gaze.stop()
            except Exception:  # noqa: BLE001
                pass
        return None


def _run_presaccade_loop(win, stimuli, logger, narrator, contrast_floor, test_mode, gaze) -> PresaccadeSession | None:
    session = PresaccadeSession(
        logger=logger, clock=PausableClock(), contrast_floor=contrast_floor, test_mode=test_mode, gaze=gaze
    )
    presenter = FramePresenter(stimuli, narrator)
    last_phase: str | None = None
    try:
        drain_stale_space(session)
        while True:
            keys = event.getKeys(keyList=["space", "escape", *RESPONSE_KEYS])
            if "escape" in keys:
                return None

            space_delivered = deliver_keys(session, keys)
            session.tick()
            state = session.render_state()

            if leaves_complete_screen(last_phase, space_delivered):
                return session  # a space press on the results screen moves on
            last_phase = state["hud"]["phase"]

            presenter.update(state)
            draw_all(stimuli)
            win.flip()
    finally:
        report_space_presses(session, "baseline", logger)


def _start_glasses_recording(gaze, logger: ResultLogger) -> bool:
    """The phone records the glasses' video itself; the green sync flashes (before any trial) let it be lined up with the PC clock.
    True if a recording started."""
    if not hasattr(gaze, "start_recording"):
        logger.update_glasses_video(recording="not available (webcam backend)")
        return False
    status = gaze.start_recording()
    logger.update_glasses_video(**status)
    return status.get("recording") == "started"


def _stop_glasses_recording(win, stimuli: dict, logger: ResultLogger, narrator: Narrator, gaze) -> None:
    try:  # one more flash at the very end (also after Escape) pins down clock drift; then stop and save the phone recording
        run_sync_flashes(win, stimuli, logger, narrator, END_FLASH_COUNT, "end", announce=False)
        gaze.send_event("saccade_phase_end")
    except Exception as exc:  # noqa: BLE001 - the end flash is a refinement; never let it block saving the recording
        print(f"WARNING: end sync flash failed ({exc!r})", file=sys.stderr)
    logger.update_glasses_video(**gaze.stop_recording())


def run_saccade_phase(
    win: visual.Window,
    stimuli: dict,
    logger: ResultLogger,
    narrator: Narrator,
    camera_index: int = CAMERA_INDEX,
    gaze_backend: str = BACKEND_NEON,
    contrast_floor: float | None = None,
    show_gaze_indicator: bool = False,
    test_mode: bool = False,
    precomputed_calibration_ratios: tuple[float, float, float] | None = None,
    skip_practice_trials: bool = False,
    precomputed_rt_samples_ms: list[float] | None = None,
    record_video: bool = False,
    saccade_contrasts: tuple[float, ...] | None = None,
) -> ExperimentSession | None:
    """Phase 2: the gaze-contingent saccade test. `precomputed_calibration_ratios`/
    `skip_practice_trials`/`precomputed_rt_samples_ms` come from a completed
    TutorialSession (see main()) - when set, this phase reuses that calibration
    and its practice reaction times instead of calibrating, running its own RT
    test and practising again. `record_video` writes logger.video_path
    (+ a companion per-frame-timestamp CSV, see VideoRecorder) alongside the
    results CSV - only this phase, since it's the only one with a running
    camera and real (non-practice, non-tutorial) data to replay against.
    Every trial waits for the participant's SPACE (the built-in break). Returns the completed session, or None if the participant quit early."""
    # Raw gaze is recorded alongside the results CSV (Neon only) so the session can be replayed offline.
    gaze = build_gaze_source(gaze_backend, camera_index, trace_path=logger.gaze_trace_path)
    gaze.start()
    recording_started = False
    video_recorder = None
    try:  # a failure before the main loop must still stop the phone recording (gaze.stop() does) and the video recorder
        recording_started = _start_glasses_recording(gaze, logger)
        # Neon has no latest_frame() (gaze only, no Mono8 camera feed), so video + preview are webcam-only.
        has_frames = hasattr(gaze, "latest_frame")
        video_recorder = (
            VideoRecorder(logger.video_path, logger.video_timestamps_path, gaze) if record_video and has_frames else None
        )
        if video_recorder is not None:
            video_recorder.start()
        clock = PausableClock()
        session = ExperimentSession(
            gaze,
            logger=logger,
            clock=clock,
            contrast_floor=contrast_floor,
            show_gaze_indicator=show_gaze_indicator,
            test_mode=test_mode,
            precomputed_calibration_ratios=precomputed_calibration_ratios,
            skip_practice_trials=skip_practice_trials,
            precomputed_rt_samples_ms=precomputed_rt_samples_ms,
            fixed_contrasts=saccade_contrasts,
        )
        go_cue_tone = build_go_cue_tone()

        aspect = stimuli["aspect"]
        preview_width = CAMERA_PREVIEW_HEIGHT * CAMERA_PREVIEW_FRAME_ASPECT
        preview_pos = (
            -aspect / 2 + preview_width / 2 + CAMERA_PREVIEW_MARGIN,
            0.5 - CAMERA_PREVIEW_HEIGHT / 2 - CAMERA_PREVIEW_MARGIN,
        )
        camera_preview = visual.ImageStim(
            win, size=(preview_width, CAMERA_PREVIEW_HEIGHT), pos=preview_pos, units="height"
        )

        presenter = FramePresenter(stimuli, narrator)
        last_phase = None
    except BaseException:
        if video_recorder is not None:
            video_recorder.stop()
        gaze.stop()
        raise

    try:
        if recording_started:  # its own short step before the first screen/trial; SPACE pressed meanwhile is discarded (and counted) below
            run_sync_flashes(win, stimuli, logger, narrator, SYNC_FLASH_COUNT, "start")
            gaze.send_event("pc_sync_flashes_done")
        drain_stale_space(session)
        while True:
            keys = event.getKeys(keyList=["space", "escape", *RESPONSE_KEYS])
            if "escape" in keys:
                return None

            space_delivered = deliver_keys(session, keys)
            session.tick()
            state = session.render_state()
            phase = state["hud"]["phase"]

            if leaves_complete_screen(last_phase, space_delivered):
                return session  # a space press on the results screen moves on

            if phase_just_became_active(phase, last_phase, _GO_CUE_PHASES):
                go_cue_tone.play()
            last_phase = phase

            presenter.update(state)
            draw_all(stimuli)
            if has_frames and camera_preview_enabled(state["hud"]["trial_index"]):
                frame = gaze.latest_frame()  # raw Mono8 (grayscale)
                if frame is not None:
                    camera_preview.image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
                camera_preview.draw()
            win.flip()
    finally:
        report_space_presses(session, "saccade", logger)
        if hasattr(gaze, "quality_summary"):
            logger.set_gaze_quality(gaze.quality_summary())
        if video_recorder is not None:
            video_recorder.stop()
        if recording_started:
            _stop_glasses_recording(win, stimuli, logger, narrator, gaze)
        gaze.stop()


def run_tutorial_phase(
    win: visual.Window,
    stimuli: dict,
    narrator: Narrator,
    camera_index: int = CAMERA_INDEX,
    gaze_backend: str = BACKEND_NEON,
) -> TutorialSession | None:
    """A one-time, narrated on-ramp for first-time participants, run before
    both real phases (see main()). Unlike the other two phases, no key or
    tick is forwarded to the session while narrator.is_speaking is true -
    the tutorial holds on the current screen until narration finishes,
    rather than letting a fast participant outrun instructions they haven't
    heard yet. Returns the completed session (its calibration_ratios feeds
    the real saccade phase, skipping its own calibration), or None if the
    participant quit early."""
    gaze = build_gaze_source(gaze_backend, camera_index)
    gaze.start()
    clock = PausableClock()
    session = TutorialSession(gaze, clock)
    go_cue_tone = build_go_cue_tone()

    presenter = FramePresenter(stimuli, narrator)
    last_phase = None
    last_feedback_token = 0
    feedback_frames_remaining = 0

    try:
        drain_stale_space(session)
        key_gate = NarrationKeyGate()
        while True:
            keys = event.getKeys(keyList=["space", "escape", *RESPONSE_KEYS])
            if "escape" in keys:
                return None

            speaking = narrator.is_speaking
            # A SPACE pressed during narration is held and delivered when speech ends, never discarded.
            space_delivered, response_keys, duplicates = key_gate.route(keys, speaking)
            session.space_tally.record("duplicate press (same frame or while narrating)", duplicates)
            if not speaking:
                if space_delivered:
                    session.on_space()
                # At most one response key per frame - unlike the real
                # trial phases (where a repeat is harmless: a response,
                # once recorded, is ignored for the rest of that trial),
                # the tutorial's quiz stage advances immediately on
                # any correct key with no such guard, so an OS key-repeat
                # event landing in the same frame as the original press
                # could otherwise fire two stage transitions - or silently
                # cost a quiz streak - off a single keypress.
                for key in response_keys:
                    side = _RESPONSE_KEY_SIDE.get(key)
                    if side is not None:
                        session.on_response_key(side)
                        break
                session.tick()

            state = session.render_state()
            phase = state["hud"]["phase"]

            if leaves_complete_screen(last_phase, space_delivered):
                return session

            if phase_just_became_active(phase, last_phase, _TUTORIAL_GO_CUE_PHASES):
                go_cue_tone.play()
            last_phase = phase

            presenter.update(state)

            feedback_token = state["feedback_flash"]["token"]
            if feedback_token != last_feedback_token:
                last_feedback_token = feedback_token
                color = state["feedback_flash"]["color"]
                stimuli["feedback_flash"].fillColor = FEEDBACK_FLASH_GREEN if color == "green" else FEEDBACK_FLASH_RED
                feedback_frames_remaining = FEEDBACK_FLASH_DURATION_FRAMES

            stimuli["feedback_flash"].opacity = 1.0 if feedback_frames_remaining > 0 else 0.0
            feedback_frames_remaining = max(0, feedback_frames_remaining - 1)

            draw_all(stimuli)
            win.flip()
    finally:
        gaze.stop()


def _camera_labels(camera_indices: list[int]) -> list[str]:
    return [f"Camera {i}" + (" (default)" if i == CAMERA_INDEX else "") for i in camera_indices]


def _resolve_camera_choice(camera_indices: list[int], camera_labels: list[str], selected_label: str) -> tuple[int, str]:
    position = camera_labels.index(selected_label)
    return camera_indices[position], camera_labels[position]


def _resolve_show_gaze_indicator(selected_label: str) -> bool:
    return selected_label == "Yes"


MODE_TEST = "Test"
MODE_REAL = "Real"


def _resolve_test_mode(mode_label: str) -> bool:
    """The startup dialog's Mode dropdown decides this explicitly: Test is a
    quick smoke run (fewer trials, saccade phase first); Real is a data-collection session."""
    return mode_label == MODE_TEST


def _session_info_error(mode_label: str, raw_participant_id: str) -> str | None:
    """A real run must be attributable to a participant, so it needs a
    non-empty ID; a test run may leave it blank (saved as "anonymous")."""
    if mode_label == MODE_REAL and not raw_participant_id.strip():
        return "Real mode needs a Participant ID"
    return None


def _selected_first(options: list[str], selected: str | None) -> list[str]:
    """A DlgFromDict choice list with `selected` moved to the front (its
    default), used to restore the picks when the dialog is re-shown."""
    if selected is None or selected not in options:
        return list(options)
    return [selected] + [o for o in options if o != selected]


def _resolve_run_tutorial(selected_label: str) -> bool:
    return selected_label == "Yes"


def _resolve_record_video(selected_label: str) -> bool:
    return selected_label == "Yes"


def prompt_session_info() -> tuple[str, str, int, str, float, bool, bool, bool, bool, tuple[float, ...]] | None:
    """The startup dialog: Mode (Test/Real), Participant ID, gaze source, webcam picker, the saccade phase's contrast range
    (min/max, see SACCADE_FIXED_CONTRASTS; the baseline staircase floor comes from src/experiment/settings.py), and the
    Show gaze indicator / Run tutorial / Record video toggles. A real run with a blank Participant ID re-shows the dialog
    with the problem in its title. Returns None if the dialog was cancelled."""
    try:
        camera_indices = list_cameras() or [CAMERA_INDEX]
    except ImportError:  # PySpin not installed (Neon-only machine)
        camera_indices = [CAMERA_INDEX]
    camera_labels = _camera_labels(camera_indices)
    default_contrast_floor_percent = load_contrast_floor_percent()

    previous: dict = {}
    error: str | None = None
    while True:
        info = {
            "Mode": _selected_first([MODE_REAL, MODE_TEST], previous.get("Mode")),
            "Participant ID": previous.get("Participant ID", ""),
            "Gaze source": _selected_first(BACKENDS, previous.get("Gaze source")),
            "Camera (webcam source only)": _selected_first(camera_labels, previous.get("Camera (webcam source only)")),
            # The saccade phase flashes fixed contrasts evenly spaced between these two (SACCADE_FIXED_CONTRASTS).
            "Saccade contrast min (%)": previous.get("Saccade contrast min (%)", f"{DEFAULT_SACCADE_CONTRAST_MIN_PERCENT:g}"),
            "Saccade contrast max (%)": previous.get("Saccade contrast max (%)", f"{DEFAULT_SACCADE_CONTRAST_MAX_PERCENT:g}"),
            "Show gaze indicator": _selected_first(["No", "Yes"], previous.get("Show gaze indicator")),
            "Run tutorial": _selected_first(["No", "Yes"], previous.get("Run tutorial")),
            "Record video": _selected_first(["Yes", "No"], previous.get("Record video")),
        }
        dlg = gui.DlgFromDict(info, title="Saccade experiment" + (f" - {error}" if error else ""))
        if not dlg.OK:
            return None
        previous = dict(info)
        error = _session_info_error(info["Mode"], info["Participant ID"])
        if error is None:
            break

    raw_participant_id = info["Participant ID"]
    test_mode = _resolve_test_mode(info["Mode"])
    participant_id = raw_participant_id.strip() or "anonymous"
    camera_index, camera_label = _resolve_camera_choice(camera_indices, camera_labels, info["Camera (webcam source only)"])
    gaze_backend = info["Gaze source"]
    if gaze_backend in NEON_BACKENDS:
        camera_index, camera_label = CAMERA_INDEX, gaze_backend
    # No dialog field any more: the baseline staircase's floor is the dashboard-configured default.
    contrast_floor_percent = default_contrast_floor_percent
    saccade_contrasts = resolve_saccade_contrasts(info["Saccade contrast min (%)"], info["Saccade contrast max (%)"])
    show_gaze_indicator = _resolve_show_gaze_indicator(info["Show gaze indicator"])
    run_tutorial = _resolve_run_tutorial(info["Run tutorial"])
    record_video = _resolve_record_video(info["Record video"])
    return (
        participant_id,
        gaze_backend,
        camera_index,
        camera_label,
        contrast_floor_percent,
        show_gaze_indicator,
        run_tutorial,
        record_video,
        test_mode,
        saccade_contrasts,
    )


def show_results_graph(win: visual.Window, results: list, output_path=None) -> None:
    png_path = build_comparison_graph(results) if output_path is None else build_comparison_graph(results, output_path=output_path)
    # The graph is now a 2x2 panel figure (aspect ~1.2, not the old 2-panel
    # 2:1) - sized/positioned to match so it isn't stretched.
    image = visual.ImageStim(win, image=str(png_path), size=(1.02, 0.85), pos=(0, 0.02))
    text = visual.TextStim(win, text="Press SPACE or ESCAPE to exit", pos=(0, -0.47), color="white", height=0.03)

    while True:
        keys = event.getKeys(keyList=["space", "escape"])
        if keys:
            return
        image.draw()
        text.draw()
        win.flip()


def show_done_screen(win: visual.Window, stimuli: dict, narrator: Narrator, folder) -> None:
    """Last screen of a run (also after Escape): where the results are; SPACE or ESCAPE closes. Spoken once, without the path."""
    caption = stimuli["caption"]
    caption.text = f"Done. Thank you.\nResults saved to {folder}\nPress SPACE to close."
    caption.opacity = 1.0
    narrator.speak("Done. Thank you. Results saved. Press SPACE to close.")
    event.getKeys()  # drop keys pressed before this screen appeared
    while not event.getKeys(keyList=["space", "escape"]):
        caption.draw()
        win.flip()


def main() -> None:
    # Let the Neon receiver thread get the interpreter lock within ~0.5 ms instead of the default
    # 5 ms while the render loop is busy: in real sessions gaze delivery steps up ~65 ms late and
    # drops ~19% of samples once drawing starts (see CLAUDE.md). A cheap first test of the cause.
    sys.setswitchinterval(0.0005)
    session_info = prompt_session_info()
    if session_info is None:
        return  # cancelled the participant-info dialog
    (
        participant_id,
        gaze_backend,
        camera_index,
        camera_label,
        contrast_floor_percent,
        show_gaze_indicator,
        run_tutorial,
        record_video,
        test_mode,
        saccade_contrasts,
    ) = session_info
    contrast_floor = contrast_floor_percent / 100

    logger = ResultLogger(
        participant_id, camera_label=camera_label, contrast_floor_percent=contrast_floor_percent, test_mode=test_mode
    )
    logger.set_saccade_contrasts(saccade_contrasts)
    win = build_window(fullscreen=True)
    logger.set_display_refresh_hz(measure_refresh_hz(win))
    stimuli = build_stimuli(win, show_hud=test_mode)
    print(f"Session folder: {logger.folder}")
    narrator = Narrator()

    try:
        precomputed_calibration_ratios = None
        precomputed_rt_samples_ms = None
        if run_tutorial:
            tutorial_session = run_tutorial_phase(
                win, stimuli, narrator, camera_index=camera_index, gaze_backend=gaze_backend
            )
            if tutorial_session is None:
                return  # quit early
            precomputed_calibration_ratios = tutorial_session.calibration_ratios
            precomputed_rt_samples_ms = tutorial_session.rt_samples_ms

        saccade_phase = partial(
            run_saccade_phase,
            narrator=narrator,
            camera_index=camera_index,
            gaze_backend=gaze_backend,
            contrast_floor=contrast_floor,
            show_gaze_indicator=show_gaze_indicator,
            test_mode=test_mode,
            precomputed_calibration_ratios=precomputed_calibration_ratios,
            skip_practice_trials=run_tutorial,
            precomputed_rt_samples_ms=precomputed_rt_samples_ms,
            record_video=record_video,
            saccade_contrasts=saccade_contrasts,
        )

        # Test runs go saccade-first (the phase under development); real runs are counterbalanced by participant ID.
        order = phase_order(participant_id, test_mode)
        print(f"Phase order: {' -> '.join(order)} (rows in the results CSV appear in this order, distinguished by the phase column)")

        sessions = {}
        calibration_ratios = precomputed_calibration_ratios  # known up front only if the tutorial ran
        for phase_name in order:
            if phase_name == PHASE_SACCADE:
                result = saccade_phase(win, stimuli, logger)
                if result is not None and result.calibration_ratios is not None:
                    calibration_ratios = result.calibration_ratios
            else:
                result = run_presaccade_phase(
                    win,
                    stimuli,
                    logger,
                    narrator,
                    contrast_floor=contrast_floor,
                    test_mode=test_mode,
                    gaze_backend=gaze_backend,
                    camera_index=camera_index,
                    calibration_ratios=calibration_ratios,
                )
                if result is not None and result.gated_trials:
                    print(f"Baseline fixation gating held {result.gated_trials} flash(es); {result.gate_timeouts} fired after timing out.")
            if result is None:
                return  # quit early
            sessions[phase_name] = result

        saccade_session = sessions.get(PHASE_SACCADE)
        if saccade_session is not None and saccade_session.calibration_ratios is not None:
            logger.set_calibration(*saccade_session.calibration_ratios)

        all_results = [r for name in (PHASE_BASELINE, PHASE_SACCADE) if name in sessions for r in sessions[name].results]
        if all_results:
            show_results_graph(win, all_results, logger.graph_path)
    finally:
        logger.close()
        if finish_run_summary(logger.folder) and sys.exc_info()[0] is None and logger.gaze_trace_path.exists():
            show_done_screen(win, stimuli, narrator, logger.folder)
        narrator.stop()
        win.close()


if __name__ == "__main__":
    main()
