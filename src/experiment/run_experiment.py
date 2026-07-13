from psychopy import core, event, gui, sound, visual

from include.experiment.constants import (
    BACKGROUND_LUMINANCE,
    CIRCLE_RADIUS_RATIO,
    GRATING_SF_CYCLES_PER_HEIGHT_UNIT,
    GRATING_SIZE_HEIGHT_UNITS,
    TEST_MODE,
)
from src.eye_tracking.webcam_source import WebcamGazeSource
from src.experiment.logger import ResultLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.presaccade_session import PresaccadeSession
from src.experiment.results_graph import build_comparison_graph
from src.experiment.session import ExperimentSession

FADE_DURATION_S = 0.4  # dot/cross opacity fade; softens onset so it doesn't trigger a reflexive saccade
WARNING_TONE_HZ = 880
WARNING_TONE_DURATION_S = 0.5  # matches Diamond, Ross & Morrone (2000)'s 500ms warning tone

# Trial-start cue: a big peripheral green flash instead of text, since reading
# text means shifting gaze off the fixation target and ruining the trial.
START_FLASH_COLOR = [-1, 1, -1]  # pure green
START_FLASH_DURATION_FRAMES = 8


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


def build_window(fullscreen: bool = True) -> visual.Window:
    kwargs = {"color": luminance_to_color(BACKGROUND_LUMINANCE), "units": "height", "allowGUI": False}
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
        "dot": visual.Circle(win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0),
        "cross": visual.Circle(win, radius=CIRCLE_RADIUS_RATIO, fillColor="white", lineColor="white", opacity=0),
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
        "instructions": visual.TextStim(win, text="", pos=(0, -0.4), color="white", height=0.035, wrapWidth=1.5),
        "pause_text": visual.TextStim(
            win, text="Paused — click anywhere to resume", pos=(0, 0), color="white", height=0.045
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

    for name in ("dot", "cross"):
        target_opacity = 1.0 if state[name]["visible"] else 0.0
        fade_opacities[name] = _move_toward(fade_opacities[name], target_opacity, fade_step)
        stim = stimuli[name]
        stim.opacity = fade_opacities[name]
        stim.pos = ratio_to_pos(state[name]["x"], state[name]["y"], aspect)

    grating = stimuli["grating"]
    grating.opacity = 1.0 if state["grating"]["visible"] else 0.0
    if state["grating"]["visible"]:
        grating.pos = ratio_to_pos(state["grating"]["x"], state["grating"]["y"], aspect)
        # Window background is mid-grey at rgb 0 (see luminance_to_color), so the
        # grating's own contrast parameter is exactly the Michelson contrast of
        # the flash around that background - no extra luminance math needed.
        grating.contrast = state["grating"]["contrast"]

    stimuli["instructions"].text = state["instructions"]
    hud = state["hud"]
    stimuli["hud"].text = (
        f"phase: {hud['phase']} | trial: {hud['trial']} | gaze: {hud['gaze_zone']} | "
        f"face: {'yes' if hud['face_found'] else 'no'} | source: {'ok' if hud['source_available'] else 'unavailable'}"
    )


def draw_all(stimuli: dict) -> None:
    # start_flash first so it sits behind the fixation stimuli, which stay
    # visible/fixatable on top of it during the flash.
    for name in ("start_flash", "dot", "cross", "grating", "instructions", "hud"):
        stimuli[name].draw()


def run_presaccade_phase(win: visual.Window, stimuli: dict, logger: ResultLogger) -> PresaccadeSession | None:
    """Phase 1: fixate center, detect flashes, no eye tracking. Returns the
    completed session, or None if the participant quit early."""
    clock = PausableClock()
    session = PresaccadeSession(logger=logger, clock=clock)
    pause_toggle = ClickPauseToggle(win, clock)
    fade_opacities = {"dot": 0.0, "cross": 0.0}
    frame_clock = core.Clock()

    while True:
        keys = event.getKeys(keyList=["space", "escape"])
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

        session.tick()
        state = session.render_state()

        if state["hud"]["phase"] == "COMPLETE" and "space" in keys:
            return session  # a space press on the results screen moves on

        dt = frame_clock.getTime()
        frame_clock.reset()
        apply_render_state(state, stimuli, fade_opacities, dt)
        stimuli["start_flash"].opacity = 0.0  # unused in this phase

        draw_all(stimuli)
        win.flip()


def run_saccade_phase(win: visual.Window, stimuli: dict, logger: ResultLogger) -> ExperimentSession | None:
    """Phase 2: the gaze-contingent saccade test. Returns the completed
    session, or None if the participant quit early."""
    gaze = WebcamGazeSource()
    gaze.start()
    clock = PausableClock()
    session = ExperimentSession(gaze, logger=logger, clock=clock)
    pause_toggle = ClickPauseToggle(win, clock)

    fade_opacities = {"dot": 0.0, "cross": 0.0}
    warning_tone = sound.Sound(value=WARNING_TONE_HZ, secs=WARNING_TONE_DURATION_S)
    frame_clock = core.Clock()
    last_phase = None
    flash_frames_remaining = 0
    start_flash_shown = False

    try:
        while True:
            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                return None

            pause_toggle.update()
            if pause_toggle.paused:
                stimuli["pause_text"].draw()
                win.flip()
                frame_clock.reset()
                continue

            for _ in range(keys.count("space")):  # no-op if already COMPLETE
                session.on_space()

            session.tick()
            state = session.render_state()
            phase = state["hud"]["phase"]

            if phase == "COMPLETE" and "space" in keys:
                return session  # a space press on the results screen moves on

            if phase == "FOREPERIOD" and last_phase != "FOREPERIOD" and not start_flash_shown:
                flash_frames_remaining = START_FLASH_DURATION_FRAMES
                start_flash_shown = True
            if phase == "TRIAL_ACTIVE" and last_phase != "TRIAL_ACTIVE":
                warning_tone.play()  # fires exactly when the target dot appears - the "move your eyes now" cue
            last_phase = phase

            dt = frame_clock.getTime()
            frame_clock.reset()
            apply_render_state(state, stimuli, fade_opacities, dt)

            stimuli["start_flash"].opacity = 1.0 if flash_frames_remaining > 0 else 0.0
            flash_frames_remaining = max(0, flash_frames_remaining - 1)

            draw_all(stimuli)
            win.flip()
    finally:
        gaze.stop()


def prompt_participant_id() -> str | None:
    """Standard PsychoPy participant-info dialog, shown before the window
    opens. Returns None if the participant cancelled the dialog."""
    info = {"Participant ID": ""}
    dlg = gui.DlgFromDict(info, title="Saccade experiment")
    if not dlg.OK:
        return None
    return info["Participant ID"].strip() or "anonymous"


def show_results_graph(win: visual.Window, results: list) -> None:
    png_path = build_comparison_graph(results)
    image = visual.ImageStim(win, image=str(png_path), size=(1.2, 0.9), pos=(0, 0.05))
    text = visual.TextStim(win, text="Press SPACE or ESCAPE to exit", pos=(0, -0.47), color="white", height=0.03)

    while True:
        keys = event.getKeys(keyList=["space", "escape"])
        if keys:
            return
        image.draw()
        text.draw()
        win.flip()


def main() -> None:
    participant_id = prompt_participant_id()
    if participant_id is None:
        return  # cancelled the participant-info dialog

    logger = ResultLogger(participant_id)
    win = build_window(fullscreen=True)
    stimuli = build_stimuli(win)

    try:
        # TEST_MODE runs the saccade phase first - that's the one under active
        # development, so a smoke test shouldn't have to sit through the
        # baseline phase first just to reach it.
        phases = [run_saccade_phase, run_presaccade_phase] if TEST_MODE else [run_presaccade_phase, run_saccade_phase]

        results_by_phase = {}
        for phase_fn in phases:
            result = phase_fn(win, stimuli, logger)
            if result is None:
                return  # quit early
            results_by_phase[phase_fn] = result

        saccade_session = results_by_phase[run_saccade_phase]
        if saccade_session.calibration_ratios is not None:
            logger.set_calibration(*saccade_session.calibration_ratios)

        all_results = results_by_phase[run_presaccade_phase].results + saccade_session.results
        show_results_graph(win, all_results)
    finally:
        logger.close()
        win.close()


if __name__ == "__main__":
    main()
