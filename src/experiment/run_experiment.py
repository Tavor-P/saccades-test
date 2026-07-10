from psychopy import core, event, sound, visual

from include.experiment.constants import BACKGROUND_LUMINANCE, CIRCLE_RADIUS_RATIO, SQUARE_SIZE_RATIO
from src.eye_tracking.webcam_source import WebcamGazeSource
from src.experiment.session import ExperimentSession

FADE_DURATION_S = 0.4  # dot/cross opacity fade; softens onset so it doesn't trigger a reflexive saccade
WARNING_TONE_HZ = 880
WARNING_TONE_DURATION_S = 0.5  # matches Diamond, Ross & Morrone (2000)'s 500ms warning tone

# Trial-start cue: a big peripheral green flash instead of text, since reading
# text means shifting gaze off the fixation target and ruining the trial.
START_FLASH_COLOR = [-1, 1, -1]  # pure green
START_FLASH_DURATION_FRAMES = 8


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
        "square": visual.Rect(
            win, width=SQUARE_SIZE_RATIO, height=SQUARE_SIZE_RATIO, fillColor="white", lineColor=None, opacity=0
        ),
        "instructions": visual.TextStim(win, text="", pos=(0, -0.4), color="white", height=0.035, wrapWidth=1.5),
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

    square = stimuli["square"]
    square.opacity = 1.0 if state["square"]["visible"] else 0.0
    if state["square"]["visible"]:
        square.pos = ratio_to_pos(state["square"]["x"], state["square"]["y"], aspect)
        contrast = state["square"]["contrast"]
        square.fillColor = luminance_to_color(BACKGROUND_LUMINANCE + contrast)

    stimuli["instructions"].text = state["instructions"]
    hud = state["hud"]
    stimuli["hud"].text = (
        f"phase: {hud['phase']} | trial: {hud['trial']} | gaze: {hud['gaze_zone']} | "
        f"face: {'yes' if hud['face_found'] else 'no'} | source: {'ok' if hud['source_available'] else 'unavailable'}"
    )


def draw_all(stimuli: dict) -> None:
    # start_flash first so it sits behind the fixation stimuli, which stay
    # visible/fixatable on top of it during the flash.
    for name in ("start_flash", "dot", "cross", "square", "instructions", "hud"):
        stimuli[name].draw()


def main() -> None:
    gaze = WebcamGazeSource()
    gaze.start()
    session = ExperimentSession(gaze)

    win = build_window(fullscreen=True)
    stimuli = build_stimuli(win)
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
                break
            if "space" in keys:
                session.on_space()

            session.tick()
            state = session.render_state()

            phase = state["hud"]["phase"]
            if phase == "WAITING_TO_START":
                start_flash_shown = False  # re-arm for the next run (after a restart)
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
        session.close()
        gaze.stop()
        win.close()


if __name__ == "__main__":
    main()
