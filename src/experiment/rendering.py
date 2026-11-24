"""PsychoPy rendering for run_experiment: the window, the stimuli, applying a session's render_state() to them (fades, line,
HUD, captions + narration), drawing a frame, and the glasses-video sync flashes. PsychoPy-only drawing code: no session logic
(the sessions are PsychoPy-free state machines) and no key handling (run_experiment's phase loops)."""

import pyglet
from psychopy import core, visual

from include.experiment import participant_text
from include.experiment.constants import (
    BACKGROUND_LUMINANCE,
    CIRCLE_RADIUS_RATIO,
    FADE_DURATION_S,
    GAZE_INDICATOR_OPACITY,
    GAZE_INDICATOR_RADIUS_RATIO,
    LINE_WIDTH_SCREENS,
    START_FLASH_COLOR,
    SYNC_LEAD_IN_FRAMES,
    WINDOWED_SIZE_PX,
)
from src import timebase
from src.experiment.geometry import Geometry
from src.experiment.glasses_video import sync_flash_frames
from src.experiment.logger import ResultLogger
from src.experiment.narrator import Narrator

SYNC_FLASH_ANNOUNCEMENT = participant_text.SYNC_FLASH  # the only spoken line of the sync step (the flashes themselves are silent)


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
        kwargs["size"] = WINDOWED_SIZE_PX
    return visual.Window(**kwargs)


def measure_refresh_hz(win: visual.Window) -> float | None:
    """The display's real refresh rate (None if PsychoPy cannot measure it): flash lengths are counted in frames from this."""
    try:
        return float(win.getActualFrameRate())
    except Exception:  # noqa: BLE001 - PsychoPy raises whatever the window backend (pyglet/GL) does; no rate only means the default frame counts are used
        return None


def build_stimuli(win: visual.Window, show_hud: bool = False, geometry: Geometry | None = None) -> dict:
    """`show_hud`: draw the top-left debug text (Test mode only; apply_render_state honours it). `geometry` sizes the line (0.15 deg thick) for the
    chosen screen height and viewing distance (default: the placeholder constants)."""
    geometry = geometry or Geometry.default()
    aspect = win.size[0] / win.size[1]
    return {
        "aspect": aspect,
        "show_hud": show_hud,
        "start_flash": visual.Rect(
            win, width=aspect * 2 + 1, height=2, fillColor=START_FLASH_COLOR, lineColor=None, opacity=0
        ),
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
        # GratingStim (PsychoPy's class) under the 'gauss' mask sized (very wide, 6 SDs tall). `contrast` scales the white field around the
        # mid-grey background, so the peak luminance step is contrast/2 (a light line).
        "line": visual.GratingStim(
            win,
            tex=None,
            mask="gauss",
            size=(LINE_WIDTH_SCREENS * aspect, geometry.line_height_height_units),
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

    line = stimuli["line"]
    line.opacity = 1.0 if state["line"]["visible"] else 0.0
    if state["line"]["visible"]:
        # Full screen width, so only the vertical position matters (x is just the target the line belongs to).
        line.pos = (0.0, ratio_to_pos(state["line"]["x"], state["line"]["y"], aspect)[1])
        # The window background is mid-grey, so the line's contrast is its peak contrast around it.
        line.contrast = state["line"]["contrast"]

    # Debug text: Test mode only (stimuli["show_hud"], set in build_stimuli); never drawn in a real run.
    if stimuli.get("show_hud"):
        hud = state["hud"]
        stimuli["hud"].text = (
            f"phase: {hud['phase']} | trial: {hud['trial']} | gaze: {hud['gaze_zone']} | "
            f"face: {'yes' if hud['face_found'] else 'no'} | source: {'ok' if hud['source_available'] else 'unavailable'}"
            f"{_calibration_wait_text(hud.get('calibration_wait_s'))}"
            f"{_head_drift_text(hud.get('head_drift_deg'))}"
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


def _head_drift_text(drift_deg: float | None) -> str:
    """HUD suffix: head rotation since calibration, from the Neon IMU (absent when the source has no head monitor)."""
    return "" if drift_deg is None else f" | head: {drift_deg:.1f} deg"


def _calibration_wait_text(waited_s: float | None) -> str:
    """HUD suffix while a calibration SPACE press is latched, waiting for the gaze window to fill."""
    if waited_s is None:
        return ""
    return f" | SPACE pressed, waiting for gaze samples {waited_s:.1f}s" + (" - TRACKING LOST?" if waited_s > 3.0 else "")


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


def draw_all(stimuli: dict) -> None:
    # start_flash first so it sits behind the fixation
    # stimuli, which stay visible/fixatable on top of it during a flash.
    for name in (
        "start_flash",
        "dot",
        "cross",
        "calibration_center",
        "gaze_indicator",
        "line",
        "hud",
        "tutorial_text",
        "caption",
    ):
        stimuli[name].draw()


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
