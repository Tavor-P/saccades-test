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
    GRATING_SF_CYCLES_PER_HEIGHT_UNIT,
    GRATING_SIZE_HEIGHT_UNITS,
    HORIZONTAL_RESPONSE_KEYS,
    VERTICAL_RESPONSE_KEYS,
    WARNING_TONE_DURATION_S,
    WARNING_TONE_HZ,
)
from include.experiment.types import Orientation
from include.eye_tracking.constants import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH
from src.eye_tracking.camera import list_available_cameras
from src.eye_tracking.webcam_source import WebcamGazeSource
from src.experiment.logger import ResultLogger
from src.experiment.narrator import Narrator
from src.experiment.pausable_clock import PausableClock
from src.experiment.presaccade_session import PresaccadeSession
from src.experiment.results_graph import build_comparison_graph
from src.experiment.session import ExperimentSession
from src.experiment.settings import load_contrast_floor_percent, validate_contrast_floor_percent
from src.experiment.tutorial_session import TutorialSession

FADE_DURATION_S = 0.4  # dot/cross opacity fade; softens onset so it doesn't trigger a reflexive saccade

# Trial-start cue: a big peripheral green flash instead of text, since reading
# text means shifting gaze off the fixation target and ruining the trial.
START_FLASH_COLOR = [-1, 1, -1]  # pure green
START_FLASH_DURATION_FRAMES = 8

# Debug camera preview (top-left corner) showing the raw feed the tracker is
# seeing, for calibration/testing - not shown to real participants mid-run.
CAMERA_PREVIEW_HEIGHT = 0.22  # fraction of window height
CAMERA_PREVIEW_MARGIN = 0.02
CAMERA_PREVIEW_FRAME_ASPECT = FRAME_WIDTH / FRAME_HEIGHT


def camera_preview_enabled(trial_index: int) -> bool:
    """Single toggle point for the debug camera preview: on for trial 0 only
    right now. Change this condition (e.g. `return True` for every trial) to
    adjust when it's shown during future testing."""
    return trial_index == 0


# Orientation response keys, and the GratingStim `ori` degrees each orientation
# renders as (0=vertical stripes, 90=horizontal stripes).
_RESPONSE_KEY_ORIENTATION = {key: Orientation.VERTICAL for key in VERTICAL_RESPONSE_KEYS} | {
    key: Orientation.HORIZONTAL for key in HORIZONTAL_RESPONSE_KEYS
}
_ORIENTATION_DEGREES = {Orientation.VERTICAL: 0, Orientation.HORIZONTAL: 90}
RESPONSE_KEYS = list(_RESPONSE_KEY_ORIENTATION)

# Phase names (ExperimentSession.Phase.name / TutorialSession.Phase.name)
# whose entry is a target/cross appearing and should play the go-cue beep -
# see phase_just_became_active/build_go_cue_tone.
_GO_CUE_PHASES = frozenset({"TRIAL_ACTIVE", "RT_TEST_ACTIVE"})
_TUTORIAL_GO_CUE_PHASES = frozenset({"DRESS_ACTIVE", "RT_TEST_ACTIVE"})


def dispatch_response_keys(session, keys: list[str]) -> None:
    for key in keys:
        orientation = _RESPONSE_KEY_ORIENTATION.get(key)
        if orientation is not None:
            session.on_response_key(orientation)


def narrate_if_changed(narrator: Narrator, text: str, last_spoken: str | None) -> str:
    """Speaks `text` only if it differs from what was last spoken, returning
    the new last-spoken value for the caller to hold onto. Instructions text
    stays constant across many consecutive ticks/trials by design (see
    Session._instructions()), so speaking unconditionally every frame would
    queue the same phrase over and over - callers should call this once per
    frame and let it decide whether anything actually needs saying."""
    if text != last_spoken:
        narrator.speak(text)
    return text


class ClickPauseToggle:
    """Edge-detects mouse clicks to toggle a pause, so a participant can rest
    their eyes mid-session. Pausing/resuming the given PausableClock ensures
    whatever timer the session is mid-way through (foreperiod, response
    window, etc.) doesn't get corrupted by however long the pause lasted."""

    def __init__(self, win: visual.Window, clock: PausableClock) -> None:
        self._mouse = event.Mouse(win=win)
        self._was_pressed = False
        self._clock = clock
        self.paused = False

    def update(self) -> None:
        pressed = self._mouse.getPressed()[0] == 1
        if pressed and not self._was_pressed:
            self.paused = not self.paused
            if self.paused:
                self._clock.pause()
            else:
                self._clock.resume()
        self._was_pressed = pressed


def build_go_cue_tone() -> sound.Sound:
    """The go-cue beep, played the instant the target/cross appears (see
    _phase_just_became_active below) - anchors the reaction-time measurement
    itself, not a "get ready" warning before an already-visible target (see
    WARNING_TONE_HZ's docstring in constants.py for why this reuses that old,
    since-reverted tone with new semantics)."""
    return sound.Sound(value=WARNING_TONE_HZ, secs=WARNING_TONE_DURATION_S)


def phase_just_became_active(phase: str, last_phase: str | None, active_phases: frozenset[str]) -> bool:
    """True on the exact frame `phase` transitions into one of `active_phases`
    - the instant a target/cross appears, whether that's a real trial, a
    practice trial, or a reaction-time-test attempt. Used both to fire the
    go-cue beep and (in run_saccade_phase) the peripheral start flash."""
    return phase in active_phases and last_phase not in active_phases


class PauseMenu:
    """Two-button pause menu for the saccade phase - unlike the presaccade
    phase's anywhere-click ClickPauseToggle, this offers "Return" (resume as-
    is) and "Return and Recalibrate" (re-run eye-position calibration and the
    reaction-time test before resuming - see
    ExperimentSession.resume_from_pause), since recalibration only makes
    sense where there's gaze tracking to recalibrate. The first click (while
    not yet paused) just opens the menu, same edge-detected toggle-on
    behavior as ClickPauseToggle; once paused, only a click landing inside
    one of the two button rects does anything."""

    def __init__(
        self, win: visual.Window, clock: PausableClock, return_rect: visual.Rect, recalibrate_rect: visual.Rect
    ) -> None:
        self._mouse = event.Mouse(win=win)
        self._was_pressed = False
        self._clock = clock
        self._return_rect = return_rect
        self._recalibrate_rect = recalibrate_rect
        self.paused = False

    def update(self) -> str | None:
        """Call once per frame. Returns "return"/"recalibrate" on the frame
        the matching button is clicked (already resumed the clock by then),
        else None."""
        pressed = self._mouse.getPressed()[0] == 1
        clicked = pressed and not self._was_pressed
        self._was_pressed = pressed
        if not clicked:
            return None
        if not self.paused:
            self.paused = True
            self._clock.pause()
            return None
        pos = self._mouse.getPos()
        if self._return_rect.contains(pos):
            self.paused = False
            self._clock.resume()
            return "return"
        if self._recalibrate_rect.contains(pos):
            self.paused = False
            self._clock.resume()
            return "recalibrate"
        return None


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


def build_stimuli(win: visual.Window) -> dict:
    aspect = win.size[0] / win.size[1]
    return {
        "aspect": aspect,
        "start_flash": visual.Rect(
            win, width=aspect * 2 + 1, height=2, fillColor=START_FLASH_COLOR, lineColor=None, opacity=0
        ),
        # Tutorial-only correctness feedback (quiz misses/pass, dress-rehearsal
        # misses) - same full-screen-rect approach as start_flash, just with a
        # color chosen per-frame instead of fixed at construction.
        "feedback_flash": visual.Rect(win, width=aspect * 2 + 1, height=2, lineColor=None, opacity=0),
        # Tutorial-only on-screen instruction text (see TutorialSession -
        # every other session is speech-only by design, see narrator.py).
        # `pos` is overwritten per-frame based on render_state()'s requested
        # "center"/"bottom" position.
        "tutorial_text": visual.TextStim(win, text="", pos=(0, -0.4), color="white", height=0.04, opacity=0),
        "dot": visual.Circle(win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0),
        "cross": visual.Circle(win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0),
        "calibration_center": visual.Circle(
            win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0
        ),
        "gaze_indicator": visual.Circle(
            win, radius=GAZE_INDICATOR_RADIUS_RATIO, fillColor="white", lineColor=None, opacity=0
        ),
        # Diamond, Ross & Morrone (2000)'s probe: a horizontal sinusoidal
        # luminance grating (bars parallel to the saccade direction) windowed
        # by a Gaussian envelope - not a flat square.
        "grating": visual.GratingStim(
            win,
            tex="sin",
            mask="gauss",
            sf=GRATING_SF_CYCLES_PER_HEIGHT_UNIT,
            ori=90,
            size=GRATING_SIZE_HEIGHT_UNITS,
            color="white",
            contrast=0,
            opacity=0,
        ),
        # Presaccade phase only - anywhere-click toggle, no recalibration
        # (no gaze tracking there to recalibrate). See PauseMenu for the
        # saccade phase's two-button version below.
        "pause_text": visual.TextStim(
            win, text="Paused — click anywhere to resume", pos=(0, 0), color="white", height=0.045
        ),
        "pause_menu_label": visual.TextStim(win, text="Paused", pos=(0, 0.14), color="white", height=0.045),
        "pause_return_button": visual.Rect(
            win, width=0.34, height=0.1, pos=(-0.19, 0), fillColor=(-0.2, -0.2, -0.2), lineColor="white"
        ),
        "pause_return_text": visual.TextStim(win, text="Return", pos=(-0.19, 0), color="white", height=0.032),
        "pause_recalibrate_button": visual.Rect(
            win, width=0.34, height=0.1, pos=(0.19, 0), fillColor=(-0.2, -0.2, -0.2), lineColor="white"
        ),
        "pause_recalibrate_text": visual.TextStim(
            win, text="Return and\nRecalibrate", pos=(0.19, 0), color="white", height=0.026
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

    # calibration_center is only present in the saccade phase's render_state()
    # (the presaccade phase has no gaze calibration) - default it to hidden.
    for name in ("dot", "cross", "calibration_center"):
        symbol_state = state.get(name, {"visible": False, "x": 0.0, "y": 0.0})
        target_opacity = 1.0 if symbol_state["visible"] else 0.0
        fade_opacities[name] = _move_toward(fade_opacities[name], target_opacity, fade_step)
        stim = stimuli[name]
        stim.opacity = fade_opacities[name]
        stim.pos = ratio_to_pos(symbol_state["x"], symbol_state["y"], aspect)

    # Only the saccade phase's render_state() includes this - presaccade has
    # no gaze tracking at all. Snaps straight to its target (no fade): it's a
    # live cursor, so smoothing it would just make it lag behind real gaze.
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
        grating.pos = ratio_to_pos(state["grating"]["x"], state["grating"]["y"], aspect)
        # Window background is mid-grey at rgb 0 (see luminance_to_color), so the
        # grating's own contrast parameter is exactly the Michelson contrast of
        # the flash around that background - no extra luminance math needed.
        grating.contrast = state["grating"]["contrast"]
        orientation = state["grating"]["orientation"]
        if orientation is not None:
            grating.ori = _ORIENTATION_DEGREES[orientation]

    hud = state["hud"]
    stimuli["hud"].text = (
        f"phase: {hud['phase']} | trial: {hud['trial']} | gaze: {hud['gaze_zone']} | "
        f"face: {'yes' if hud['face_found'] else 'no'} | source: {'ok' if hud['source_available'] else 'unavailable'}"
    )

    # Tutorial-only - absent from the other two sessions' render_state(), so
    # this stays hidden for them.
    tutorial_text = state.get("tutorial_text")
    tutorial_text_stim = stimuli["tutorial_text"]
    if tutorial_text and tutorial_text["visible"]:
        tutorial_text_stim.text = tutorial_text["text"]
        tutorial_text_stim.pos = (0, 0) if tutorial_text["position"] == "center" else (0, -0.4)
        tutorial_text_stim.opacity = 1.0
    else:
        tutorial_text_stim.opacity = 0.0


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
    ):
        stimuli[name].draw()


def run_presaccade_phase(
    win: visual.Window,
    stimuli: dict,
    logger: ResultLogger,
    narrator: Narrator,
    contrast_floor: float | None = None,
    test_mode: bool = False,
) -> PresaccadeSession | None:
    """Phase 1: fixate center, detect flashes, no eye tracking. Returns the
    completed session, or None if the participant quit early."""
    clock = PausableClock()
    session = PresaccadeSession(logger=logger, clock=clock, contrast_floor=contrast_floor, test_mode=test_mode)
    pause_toggle = ClickPauseToggle(win, clock)
    fade_opacities = {"dot": 0.0, "cross": 0.0, "calibration_center": 0.0}
    frame_clock = core.Clock()
    last_spoken_instructions: str | None = None

    while True:
        keys = event.getKeys(keyList=["space", "escape", *RESPONSE_KEYS])
        if "escape" in keys:
            return None

        pause_toggle.update()
        if pause_toggle.paused:
            stimuli["pause_text"].draw()
            win.flip()
            frame_clock.reset()  # don't let a big paused dt snap the fade animation on resume
            continue

        for _ in range(keys.count("space")):  # no-op if already COMPLETE
            session.on_space()
        dispatch_response_keys(session, keys)

        session.tick()
        state = session.render_state()

        if state["hud"]["phase"] == "COMPLETE" and "space" in keys:
            return session  # a space press on the results screen moves on

        last_spoken_instructions = narrate_if_changed(narrator, state["instructions"], last_spoken_instructions)

        dt = frame_clock.getTime()
        frame_clock.reset()
        apply_render_state(state, stimuli, fade_opacities, dt)
        stimuli["start_flash"].opacity = 0.0  # unused in this phase

        draw_all(stimuli)
        win.flip()


def run_saccade_phase(
    win: visual.Window,
    stimuli: dict,
    logger: ResultLogger,
    narrator: Narrator,
    camera_index: int = CAMERA_INDEX,
    contrast_floor: float | None = None,
    show_gaze_indicator: bool = False,
    test_mode: bool = False,
    precomputed_calibration_ratios: tuple[float, float, float] | None = None,
    skip_practice_trials: bool = False,
) -> ExperimentSession | None:
    """Phase 2: the gaze-contingent saccade test. `precomputed_calibration_ratios`/
    `skip_practice_trials` come from a completed TutorialSession (see main())
    - when set, this phase reuses that calibration and its dress-rehearsal
    stage instead of repeating both. Returns the completed session, or None
    if the participant quit early."""
    gaze = WebcamGazeSource(camera_index)
    gaze.start()
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
    )
    pause_menu = PauseMenu(win, clock, stimuli["pause_return_button"], stimuli["pause_recalibrate_button"])
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

    fade_opacities = {"dot": 0.0, "cross": 0.0, "calibration_center": 0.0}
    frame_clock = core.Clock()
    last_phase = None
    last_spoken_instructions: str | None = None
    flash_frames_remaining = 0
    start_flash_shown = False

    try:
        while True:
            keys = event.getKeys(keyList=["space", "escape", *RESPONSE_KEYS])
            if "escape" in keys:
                return None

            pause_action = pause_menu.update()
            if pause_action == "return":
                session.resume_from_pause(recalibrate=False)
            elif pause_action == "recalibrate":
                session.resume_from_pause(recalibrate=True)
            if pause_menu.paused:
                stimuli["pause_menu_label"].draw()
                stimuli["pause_return_button"].draw()
                stimuli["pause_return_text"].draw()
                stimuli["pause_recalibrate_button"].draw()
                stimuli["pause_recalibrate_text"].draw()
                win.flip()
                frame_clock.reset()
                continue

            for _ in range(keys.count("space")):  # no-op if already COMPLETE
                session.on_space()
            dispatch_response_keys(session, keys)

            session.tick()
            state = session.render_state()
            phase = state["hud"]["phase"]

            if phase == "COMPLETE" and "space" in keys:
                return session  # a space press on the results screen moves on

            if phase == "FOREPERIOD" and last_phase != "FOREPERIOD" and not start_flash_shown:
                flash_frames_remaining = START_FLASH_DURATION_FRAMES
                start_flash_shown = True
            if phase_just_became_active(phase, last_phase, _GO_CUE_PHASES):
                go_cue_tone.play()
            last_phase = phase

            last_spoken_instructions = narrate_if_changed(narrator, state["instructions"], last_spoken_instructions)

            dt = frame_clock.getTime()
            frame_clock.reset()
            apply_render_state(state, stimuli, fade_opacities, dt)

            stimuli["start_flash"].opacity = 1.0 if flash_frames_remaining > 0 else 0.0
            flash_frames_remaining = max(0, flash_frames_remaining - 1)

            draw_all(stimuli)
            if camera_preview_enabled(state["hud"]["trial_index"]):
                frame = gaze.latest_frame()  # raw Mono8 (grayscale)
                if frame is not None:
                    camera_preview.image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
                camera_preview.draw()
            win.flip()
    finally:
        gaze.stop()


def run_tutorial_phase(
    win: visual.Window,
    stimuli: dict,
    narrator: Narrator,
    camera_index: int = CAMERA_INDEX,
) -> TutorialSession | None:
    """A one-time, narrated on-ramp for first-time participants, run before
    both real phases (see main()). Unlike the other two phases, no key or
    tick is forwarded to the session while narrator.is_speaking is true -
    the tutorial holds on the current screen until narration finishes,
    rather than letting a fast participant outrun instructions they haven't
    heard yet. Returns the completed session (its calibration_ratios feeds
    the real saccade phase, skipping its own calibration), or None if the
    participant quit early."""
    gaze = WebcamGazeSource(camera_index)
    gaze.start()
    clock = PausableClock()
    session = TutorialSession(gaze, clock)
    go_cue_tone = build_go_cue_tone()

    fade_opacities = {"dot": 0.0, "cross": 0.0, "calibration_center": 0.0}
    frame_clock = core.Clock()
    last_phase = None
    last_spoken_instructions: str | None = None
    last_feedback_token = 0
    feedback_frames_remaining = 0

    try:
        while True:
            keys = event.getKeys(keyList=["space", "escape", *RESPONSE_KEYS])
            if "escape" in keys:
                return None

            speaking = narrator.is_speaking
            if not speaking:
                for _ in range(keys.count("space")):
                    session.on_space()
                # At most one response key per frame - unlike the real
                # trial phases (where a repeat is harmless: a response,
                # once recorded, is ignored for the rest of that trial),
                # the tutorial's demo/quiz stages advance immediately on
                # any response key with no such guard, so an OS key-repeat
                # event landing in the same frame as the original press
                # could otherwise fire two stage transitions - or silently
                # cost a quiz streak - off a single keypress.
                for key in keys:
                    orientation = _RESPONSE_KEY_ORIENTATION.get(key)
                    if orientation is not None:
                        session.on_response_key(orientation)
                        break
                session.tick()

            state = session.render_state()
            phase = state["hud"]["phase"]

            if phase == "COMPLETE" and "space" in keys and not speaking:
                return session

            if phase_just_became_active(phase, last_phase, _TUTORIAL_GO_CUE_PHASES):
                go_cue_tone.play()
            last_phase = phase

            last_spoken_instructions = narrate_if_changed(narrator, state["instructions"], last_spoken_instructions)

            feedback_token = state["feedback_flash"]["token"]
            if feedback_token != last_feedback_token:
                last_feedback_token = feedback_token
                color = state["feedback_flash"]["color"]
                stimuli["feedback_flash"].fillColor = FEEDBACK_FLASH_GREEN if color == "green" else FEEDBACK_FLASH_RED
                feedback_frames_remaining = FEEDBACK_FLASH_DURATION_FRAMES

            dt = frame_clock.getTime()
            frame_clock.reset()
            apply_render_state(state, stimuli, fade_opacities, dt)
            stimuli["start_flash"].opacity = 0.0  # unused in this phase

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


def _resolve_contrast_floor_percent(raw: str, default_percent: float) -> float:
    """Falls back to the dashboard-configured default if the dialog's field
    was left blank/non-numeric, or set outside ZEST's own renderable range
    (see settings.validate_contrast_floor_percent) - a typo in a text field
    shouldn't be able to crash the whole session."""
    try:
        return validate_contrast_floor_percent(float(raw))
    except (TypeError, ValueError):
        return default_percent


def _resolve_show_gaze_indicator(selected_label: str) -> bool:
    return selected_label == "Yes"


def _resolve_test_mode(raw_participant_id: str) -> bool:
    """Leaving Participant ID blank means this is a quick test/smoke run
    (fewer trials, saccade phase first so you reach the thing you're testing
    faster); entering an actual ID means a real data-collection session."""
    return not raw_participant_id.strip()


def _resolve_run_tutorial(selected_label: str) -> bool:
    return selected_label == "Yes"


def prompt_session_info() -> tuple[str, int, str, float, bool, bool, bool] | None:
    """Standard PsychoPy participant-info dialog, shown before the window
    opens - also offers a camera picker (a dropdown of every camera index
    that actually opens) so switching between multiple connected cameras
    doesn't require hand-editing CAMERA_INDEX, and a contrast-floor field
    pre-filled with the dashboard-configured default (see
    src/experiment/settings.py and the dashboard's settings box) so a run can
    override it per-session without editing code. Also offers a "Show gaze
    indicator" toggle (default No) for the saccade phase's debug gaze dot -
    off by default since it's not something a real participant should see -
    and a "Run tutorial" toggle (default No) for the first-timer on-ramp (see
    TutorialSession), independent of test/real mode so it can be exercised
    from either. Whether this is a test run (see _resolve_test_mode) is
    derived from the Participant ID field itself, not a separate control.
    Returns None if the dialog was cancelled."""
    camera_indices = list_available_cameras() or [CAMERA_INDEX]
    camera_labels = _camera_labels(camera_indices)
    default_contrast_floor_percent = load_contrast_floor_percent()

    info = {
        "Participant ID": "",
        "Camera": camera_labels,
        "Contrast floor (%)": f"{default_contrast_floor_percent:.2f}",
        "Show gaze indicator": ["No", "Yes"],
        "Run tutorial": ["No", "Yes"],
    }
    dlg = gui.DlgFromDict(info, title="Saccade experiment")
    if not dlg.OK:
        return None

    raw_participant_id = info["Participant ID"]
    test_mode = _resolve_test_mode(raw_participant_id)
    participant_id = raw_participant_id.strip() or "anonymous"
    camera_index, camera_label = _resolve_camera_choice(camera_indices, camera_labels, info["Camera"])
    contrast_floor_percent = _resolve_contrast_floor_percent(info["Contrast floor (%)"], default_contrast_floor_percent)
    show_gaze_indicator = _resolve_show_gaze_indicator(info["Show gaze indicator"])
    run_tutorial = _resolve_run_tutorial(info["Run tutorial"])
    return (
        participant_id,
        camera_index,
        camera_label,
        contrast_floor_percent,
        show_gaze_indicator,
        run_tutorial,
        test_mode,
    )


def show_results_graph(win: visual.Window, results: list) -> None:
    png_path = build_comparison_graph(results)
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


def main() -> None:
    session_info = prompt_session_info()
    if session_info is None:
        return  # cancelled the participant-info dialog
    (
        participant_id,
        camera_index,
        camera_label,
        contrast_floor_percent,
        show_gaze_indicator,
        run_tutorial,
        test_mode,
    ) = session_info
    contrast_floor = contrast_floor_percent / 100

    logger = ResultLogger(
        participant_id, camera_label=camera_label, contrast_floor_percent=contrast_floor_percent, test_mode=test_mode
    )
    win = build_window(fullscreen=True)
    stimuli = build_stimuli(win)
    narrator = Narrator()

    try:
        precomputed_calibration_ratios = None
        if run_tutorial:
            tutorial_session = run_tutorial_phase(win, stimuli, narrator, camera_index=camera_index)
            if tutorial_session is None:
                return  # quit early
            precomputed_calibration_ratios = tutorial_session.calibration_ratios

        saccade_phase = partial(
            run_saccade_phase,
            narrator=narrator,
            camera_index=camera_index,
            contrast_floor=contrast_floor,
            show_gaze_indicator=show_gaze_indicator,
            test_mode=test_mode,
            precomputed_calibration_ratios=precomputed_calibration_ratios,
            skip_practice_trials=run_tutorial,
        )
        presaccade_phase = partial(
            run_presaccade_phase, narrator=narrator, contrast_floor=contrast_floor, test_mode=test_mode
        )

        # A blank Participant ID (test_mode) runs the saccade phase first -
        # that's the one under active development, so a smoke test shouldn't
        # have to sit through the baseline phase first just to reach it.
        phases = [saccade_phase, presaccade_phase] if test_mode else [presaccade_phase, saccade_phase]

        results_by_phase = {}
        for phase_fn in phases:
            result = phase_fn(win, stimuli, logger)
            if result is None:
                return  # quit early
            results_by_phase[phase_fn] = result

        saccade_session = results_by_phase[saccade_phase]
        if saccade_session.calibration_ratios is not None:
            logger.set_calibration(*saccade_session.calibration_ratios)

        all_results = results_by_phase[presaccade_phase].results + saccade_session.results
        show_results_graph(win, all_results)
    finally:
        narrator.stop()
        logger.close()
        win.close()


if __name__ == "__main__":
    main()
