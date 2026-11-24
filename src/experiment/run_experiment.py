"""Entry point (`-m src.experiment.run_experiment`): startup dialog, the dialog -> session wiring, and the per-phase frame loops that call
each session's tick() + render_state() and draw the result (the drawing itself: rendering.py). All experiment logic lives in the session
classes, not here; this module and rendering.py are the only ones that need PsychoPy."""

import sys
from dataclasses import dataclass
from pathlib import Path

from psychopy import event, gui, sound, visual

from include.experiment.constants import (
    ABOVE_RESPONSE_KEYS,
    BELOW_RESPONSE_KEYS,
    FADE_DURATION_S,  # noqa: F401 (re-exported: tests read it here)
    WARNING_TONE_DURATION_S,
    WARNING_TONE_HZ,
    WARNING_TONE_VOLUME,
)
from include.experiment.types import Side
from src.experiment.geometry import Geometry
from src.experiment.glasses_video import END_FLASH_COUNT, SYNC_FLASH_COUNT
from src.experiment.logger import ResultLogger
from src.experiment.narrator import Narrator
from src.experiment.pausable_clock import PausableClock
from src.experiment.rendering import (  # noqa: F401 (re-exported: tests import the caption/render helpers from here)
    SYNC_FLASH_ANNOUNCEMENT,
    FramePresenter,
    _second_monitor_screen_index,
    apply_render_state,
    build_stimuli,
    build_window,
    draw_all,
    measure_refresh_hz,
    narrate_if_changed,
    render_caption,
    run_sync_flashes,
    update_caption,
)
from src.experiment.run_summary import finish_run_summary
from src.experiment.session import ExperimentSession
from src.experiment.session_config import SessionConfig, TutorialHandoff
from src.experiment.settings import saccade_contrast_levels
from src.experiment.space_input import NarrationKeyGate, deliver_space, leaves_complete_screen
from src.experiment.tutorial_session import TutorialSession
from src.eye_tracking.gaze_factory import BACKEND_NEON, build_gaze_source
from src.eye_tracking.neon_address import FAILURE_TITLE, Resolution, failure_message, network_summary, parse_address, resolve_neon_address
from src.eye_tracking.neon_source import NeonConnectionError  # no device library at import time (pupil_labs loads lazily in start())

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


def build_go_cue_tone() -> sound.Sound:
    """The go-cue beep, played the instant the target/cross appears: it anchors the reaction-time measurement (see WARNING_TONE_HZ)."""
    return sound.Sound(value=WARNING_TONE_HZ, secs=WARNING_TONE_DURATION_S, volume=WARNING_TONE_VOLUME)


def phase_just_became_active(phase: str, last_phase: str | None, active_phases: frozenset[str]) -> bool:
    """True on the frame `phase` enters `active_phases` (a target/cross appearing): fires the go-cue beep."""
    return phase in active_phases and last_phase not in active_phases


def warn_if_head_moved(session, narrator: Narrator) -> None:
    """Between trials, once the head has turned away from its calibration orientation: console + speech (no caption: real trials hide it)."""
    text = session.pop_head_warning()
    if text:
        print(f"WARNING: {text} (head rotated away from its orientation at calibration)", file=sys.stderr)
        narrator.speak(text)


def drain_stale_space(session) -> None:
    """Space pressed before the phase's first screen existed (during start-up) answers nothing on screen:
    discard it, but count it so it is visible in the console and the meta file."""
    session.space_tally.record("pressed before the phase started", len(event.getKeys(keyList=["space"])), phase="(start-up)")


def report_space_presses(session, label: str, logger: ResultLogger | None = None) -> None:
    print(session.space_tally.summary(label))
    if logger is not None and session.space_tally.total:
        logger.set_ignored_space_presses(label, session.space_tally.as_dict())


def deliver_keys(session, keys: list[str]) -> bool:
    """Forward SPACE and the answer keys to a session; True if a SPACE press was delivered."""
    space_delivered = deliver_space(session, keys.count("space"))  # no-op if already COMPLETE
    dispatch_response_keys(session, keys)
    return space_delivered


def _start_glasses_recording(gaze, logger: ResultLogger) -> bool:
    """The phone records the glasses' video itself; the green sync flashes (before any trial) let it be lined up with the PC clock.
    True if a recording started."""
    if not hasattr(gaze, "start_recording"):
        logger.update_glasses_video(recording="not available (gaze source cannot record)")
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


@dataclass(frozen=True)
class SaccadePhaseOptions:
    """What run_saccade_phase needs besides the window, stimuli, logger and narrator: the startup dialog's choices and the
    tutorial's hand-off. The defaults are a run without tutorial and the dialog's toggles off."""

    show_gaze_indicator: bool = False
    test_mode: bool = False
    # From a completed TutorialSession (see _run_phases): when set, the phase reuses that calibration and its practice
    # reaction times instead of calibrating, running its own RT test and practising again.
    precomputed_calibration_ratios: tuple[float, float, float] | None = None
    skip_practice_trials: bool = False
    precomputed_rt_samples_ms: list[float] | None = None
    saccade_contrasts: tuple[float, ...] | None = None
    geometry: Geometry | None = None
    flash_spread_sd_ms: float | None = None
    neon_address: tuple[str, int] | None = None
    return_jump: bool = False  # dialog "Return jump": the jump-back trials
    precomputed_return_rt_samples_ms: list[float] | None = None  # the tutorial's return latencies


def run_saccade_phase(
    win: visual.Window,
    stimuli: dict,
    logger: ResultLogger,
    narrator: Narrator,
    options: SaccadePhaseOptions | None = None,
) -> ExperimentSession | None:
    """The gaze-contingent saccade test, set up by `options` (default SaccadePhaseOptions()).
    Every trial waits for the participant's SPACE (the built-in break). Returns the completed session, or None if the participant quit early."""
    o = options or SaccadePhaseOptions()
    # Raw gaze is recorded alongside the results CSV so the session can be replayed offline.
    gaze = build_gaze_source(trace_path=logger.gaze_trace_path, **_address_kwargs(o.neon_address))
    gaze.start()
    recording_started = False
    try:  # a failure before the main loop must still stop the phone recording (gaze.stop() does)
        phone_ip = getattr(gaze, "phone_ip", None)
        logger.set_neon_phone_ip(phone_ip if isinstance(phone_ip, str) else None)
        recording_started = _start_glasses_recording(gaze, logger)
        clock = PausableClock()
        session = ExperimentSession(
            gaze,
            logger=logger,
            clock=clock,
            config=SessionConfig(
                test_mode=o.test_mode,
                show_gaze_indicator=o.show_gaze_indicator,
                skip_practice_trials=o.skip_practice_trials,
                fixed_contrasts=o.saccade_contrasts,
                geometry=o.geometry,
                flash_spread_sd_ms=o.flash_spread_sd_ms,
                return_jump=o.return_jump,
            ),
            handoff=TutorialHandoff(
                calibration_ratios=o.precomputed_calibration_ratios,
                rt_samples_ms=o.precomputed_rt_samples_ms,
                return_rt_samples_ms=o.precomputed_return_rt_samples_ms,
            ),
        )
        go_cue_tone = build_go_cue_tone()
        presenter = FramePresenter(stimuli, narrator)
        last_phase = None
    except BaseException:
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
            warn_if_head_moved(session, narrator)
            state = session.render_state()
            phase = state["hud"]["phase"]

            if leaves_complete_screen(last_phase, space_delivered):
                return session  # a space press on the results screen moves on

            if phase_just_became_active(phase, last_phase, _GO_CUE_PHASES):
                go_cue_tone.play()
            last_phase = phase

            presenter.update(state)
            draw_all(stimuli)
            win.flip()
    finally:
        # Every step is guarded on its own: one failing must never skip the rest, above all stopping the phone recording and gaze.stop().
        _best_effort("space-press report", lambda: report_space_presses(session, "saccade", logger))
        if hasattr(gaze, "quality_summary"):
            _best_effort("gaze-quality summary", lambda: logger.set_gaze_quality(gaze.quality_summary()))
        if hasattr(gaze, "head_motion_summary"):
            _best_effort("head-movement summary", lambda: _save_head_motion(gaze, session, logger))
        if recording_started:
            _best_effort("phone recording stop", lambda: _stop_glasses_recording(win, stimuli, logger, narrator, gaze))
        gaze.stop()


def _best_effort(label: str, step) -> None:
    """Runs one cleanup step; an Exception is printed as a WARNING instead of skipping the steps after it."""
    try:
        step()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not save the {label}: {type(exc).__name__}: {exc}", file=sys.stderr)


def _save_head_motion(gaze, session, logger: ResultLogger) -> None:
    logger.set_head_motion(gaze.head_motion_summary(session.head_flagged_trials))
    gaze.write_head_motion_csv(logger.head_motion_path)


def run_tutorial_phase(
    win: visual.Window,
    stimuli: dict,
    narrator: Narrator,
    geometry: Geometry | None = None,
    neon_address: tuple[str, int] | None = None,
    return_jump: bool = False,
) -> TutorialSession | None:
    """A one-time, narrated on-ramp for first-time participants, run before
    the saccade phase (see main()). Unlike that phase, no key or
    tick is forwarded to the session while narrator.is_speaking is true -
    the tutorial holds on the current screen until narration finishes,
    rather than letting a fast participant outrun instructions they haven't
    heard yet. Returns the completed session (its calibration_ratios feeds
    the real saccade phase, skipping its own calibration), or None if the
    participant quit early."""
    gaze = build_gaze_source(**_address_kwargs(neon_address))
    gaze.start()
    try:  # from here on gaze.stop() must run whatever raises (the Neon threads would leak otherwise)
        clock = PausableClock()
        session = TutorialSession(gaze, clock, geometry=geometry, return_jump=return_jump)
        go_cue_tone = build_go_cue_tone()

        presenter = FramePresenter(stimuli, narrator)
        last_phase = None

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

            draw_all(stimuli)
            win.flip()
    finally:
        gaze.stop()


def _address_kwargs(neon_address: tuple[str, int] | None) -> dict:
    return {} if neon_address is None else {"address": neon_address[0], "port": neon_address[1]}


def show_error_box(title: str, text: str) -> None:
    """A plain Qt message box (the startup dialog's QApplication already exists); never raises - the console has the same text."""
    try:
        from PyQt6 import QtWidgets

        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(title)
        box.setInformativeText(text)
        box.setStyleSheet("QLabel { font-family: Consolas, 'Courier New', monospace; }")
        box.exec()
    except Exception as exc:  # no display / no Qt: the printed message is enough
        print(f"(could not show the message box: {exc})", file=sys.stderr)


def find_neon_or_exit(address_text: str, resolve=None, network=None, show_box=None) -> Resolution:
    """The start-up check, run before any window or run folder exists: find the Neon phone (neon_address.resolve_neon_address) or print
    a plain explanation, show it in a message box and exit with status 2 (no traceback). The console text is printed first and stays
    readable while the box is open (the .bat launchers pause after exit)."""
    resolve = resolve or resolve_neon_address
    network = network or network_summary
    show_box = show_box or show_error_box
    address, port = parse_address(address_text)
    print("Looking for the Neon glasses...", flush=True)
    resolution = resolve(address, port, progress=lambda text: print(text, flush=True))
    if resolution.ok:
        print(f"Neon found at {resolution.address}:{resolution.port} ({resolution.source}).", flush=True)
        return resolution
    message = failure_message(resolution, network())
    print("", FAILURE_TITLE.upper(), message, "", sep="\n", file=sys.stderr, flush=True)
    show_box(FAILURE_TITLE, message)
    raise SystemExit(2)


def prompt_session_info() -> tuple[str, bool, bool, bool, tuple[float, ...], Geometry, float, str, bool] | None:
    """The startup dialog (src/experiment/startup_dialog.py): Participant ID, Mode, Run tutorial, screen height/distance and, under Advanced, the saccade
    phase's contrast range (min/max, see SACCADE_FIXED_CONTRASTS) and the Show gaze indicator toggle. Invalid input keeps the dialog open with a message.
    Returns (participant_id, show_gaze_indicator, run_tutorial, test_mode, saccade_contrasts, geometry, flash_spread_sd_ms, neon_address_text,
    return_jump),
    or None if cancelled."""
    from src.experiment.startup_dialog import prompt_startup_settings

    gui.ensureQtApp()  # PsychoPy's own QApplication (Fusion style), shared with the dialog so the PsychoPy window created afterwards keeps working
    chosen = prompt_startup_settings()
    if chosen is None:
        return None
    return (
        chosen.folder_id,
        chosen.show_gaze_indicator,
        chosen.run_tutorial,
        chosen.test_mode,
        saccade_contrast_levels(chosen.contrast_min_percent, chosen.contrast_max_percent),
        chosen.geometry,
        chosen.flash_spread_sd_ms,
        chosen.neon_address,
        chosen.return_jump,
    )


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
    # drops ~19% of samples once drawing starts (see CLAUDE.md). This is the confirmed fix for that
    # Oct 2026 ~95 ms delay: runs since show ~13-20 ms.
    sys.setswitchinterval(0.0005)
    session_info = prompt_session_info()
    if session_info is None:
        return  # cancelled the participant-info dialog
    (
        participant_id,
        show_gaze_indicator,
        run_tutorial,
        test_mode,
        saccade_contrasts,
        geometry,
        flash_spread_sd_ms,
        neon_address_text,
        return_jump,
    ) = session_info

    neon = find_neon_or_exit(neon_address_text)  # before the run folder and the fullscreen window exist
    neon_address = (neon.address, neon.port)
    logger = ResultLogger(participant_id, gaze_source_label=BACKEND_NEON, test_mode=test_mode)
    logger.set_neon_device(neon.label())
    logger.set_saccade_contrasts(saccade_contrasts)
    logger.set_geometry(geometry)
    logger.set_flash_spread_sd_ms(flash_spread_sd_ms)
    logger.set_return_jump(return_jump)
    win = build_window(fullscreen=True)
    logger.set_display_refresh_hz(measure_refresh_hz(win))
    stimuli = build_stimuli(win, show_hud=test_mode, geometry=geometry)
    print(f"Session folder: {logger.folder}")
    narrator = Narrator()

    lost: NeonConnectionError | None = None
    try:
        _run_phases(
            win, stimuli, logger, narrator, run_tutorial, show_gaze_indicator, test_mode, saccade_contrasts, geometry, flash_spread_sd_ms,
            neon_address, return_jump,
        )
    except NeonConnectionError as exc:  # e.g. the phone slept between the tutorial and the saccade phase: plain message, no traceback
        lost = exc
        print("", NEON_LOST_TITLE.upper(), _neon_lost_message(exc, logger.folder), "", sep="\n", file=sys.stderr, flush=True)
    finally:
        logger.close()
        if finish_run_summary(logger.folder) and sys.exc_info()[0] is None and lost is None and logger.gaze_trace_path.exists():
            show_done_screen(win, stimuli, narrator, logger.folder)
        narrator.stop()
        win.close()
    if lost is not None:  # the fullscreen window is closed now, so the box is visible
        show_error_box(NEON_LOST_TITLE, _neon_lost_message(lost, logger.folder))
        raise SystemExit(2)
    _auto_analyze(logger.folder)  # after the window closes: the participant never waits on it


def _auto_analyze(folder: Path) -> None:
    """Analyses the finished run into its analysis/ folder, as analyze_session does, and prints the report's top block.
    Never raises: the run is already saved, and a failure only means the analysis is rerun by hand."""
    from src.experiment import analyze_session  # lazy: keeps start-up fast

    print("Analysing this run...", flush=True)
    run_ref = f"{folder.parent.name}/{folder.name}"  # participant/folder: unambiguous even if another participant has a run with the same name
    try:
        result = analyze_session.analyze(str(folder), data_dir=folder.parent.parent)
    except Exception as exc:
        print(f"Automatic analysis skipped: {exc}. Rerun later with: scripts/py.ps1 -m src.experiment.analyze_session {run_ref}", flush=True)
        return
    top_block: list[str] = []
    for line in str(result["report"]).splitlines():
        if not line.strip():
            break
        top_block.append(line)
    print("\n".join(top_block))
    print(f"Analysis folder: {result['report_path'].parent}", flush=True)


NEON_LOST_TITLE = "Lost the Neon glasses"


def _neon_lost_message(exc: Exception, folder) -> str:
    return (
        f"{exc}\n\nThe run stopped. Whatever was recorded so far is saved in {folder}.\n"
        "Wake the phone, keep the Companion app open and the glasses plugged in, then start a new run."
    )


def _run_phases(  # noqa: PLR0913 - one arg per dialog setting
    win, stimuli, logger, narrator, run_tutorial, show_gaze_indicator, test_mode, saccade_contrasts, geometry, flash_spread_sd_ms,
    neon_address, return_jump=False,
) -> None:
    """Tutorial (optional) then the saccade phase; returns early if the participant quits."""
    precomputed_calibration_ratios = None
    precomputed_rt_samples_ms = None
    precomputed_return_rt_samples_ms = None
    if run_tutorial:
        tutorial_session = run_tutorial_phase(win, stimuli, narrator, geometry=geometry, neon_address=neon_address, return_jump=return_jump)
        if tutorial_session is None:
            return  # quit early
        precomputed_calibration_ratios = tutorial_session.calibration_ratios
        precomputed_rt_samples_ms = tutorial_session.rt_samples_ms
        precomputed_return_rt_samples_ms = tutorial_session.return_rt_samples_ms if return_jump else None

    options = SaccadePhaseOptions(
        show_gaze_indicator=show_gaze_indicator,
        test_mode=test_mode,
        precomputed_calibration_ratios=precomputed_calibration_ratios,
        skip_practice_trials=run_tutorial,
        precomputed_rt_samples_ms=precomputed_rt_samples_ms,
        saccade_contrasts=saccade_contrasts,
        geometry=geometry,
        flash_spread_sd_ms=flash_spread_sd_ms,
        neon_address=neon_address,
        return_jump=return_jump,
        precomputed_return_rt_samples_ms=precomputed_return_rt_samples_ms,
    )
    result = run_saccade_phase(win, stimuli, logger, narrator, options)
    if result is None:
        return  # quit early
    if result.calibration_ratios is not None:
        logger.set_calibration(*result.calibration_ratios)
    print("Run finished.")


if __name__ == "__main__":
    main()
