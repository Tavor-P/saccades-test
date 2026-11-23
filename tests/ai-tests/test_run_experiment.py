from unittest.mock import MagicMock, Mock, patch

from src.experiment.run_experiment import (
    _second_monitor_screen_index,
    narrate_if_changed,
    phase_just_became_active,
    render_caption,
    update_caption,
)


def _patch_screens(count: int):
    display = MagicMock()
    display.get_screens.return_value = [MagicMock() for _ in range(count)]
    return patch("src.experiment.run_experiment.pyglet.canvas.get_display", return_value=display)


def test_second_monitor_screen_index_picks_the_second_screen_or_falls_back():
    with _patch_screens(2):
        assert _second_monitor_screen_index() == 1
    with _patch_screens(1):
        assert _second_monitor_screen_index() == 0


def test_narrate_if_changed_speaks_only_new_non_blank_text():
    narrator = Mock()
    assert narrate_if_changed(narrator, "look at the dot", last_spoken=None) == "look at the dot"
    narrator.speak.assert_called_once_with("look at the dot")
    narrator.reset_mock()
    assert narrate_if_changed(narrator, "look at the dot", last_spoken="look at the dot") == "look at the dot"
    assert narrate_if_changed(narrator, "", last_spoken="look at the dot") == "look at the dot"
    assert narrate_if_changed(narrator, "", last_spoken=None) is None
    narrator.speak.assert_not_called()
    assert narrate_if_changed(narrator, "now look at the cross", last_spoken="look at the dot") == "now look at the cross"
    narrator.speak.assert_called_once_with("now look at the cross")


def test_phase_just_became_active_only_on_entry():
    active = frozenset({"TRIAL_ACTIVE", "RT_TEST_ACTIVE"})
    assert phase_just_became_active("TRIAL_ACTIVE", "FOREPERIOD", active) is True
    assert phase_just_became_active("RT_TEST_ACTIVE", "RT_TEST_FOREPERIOD", active) is True
    assert phase_just_became_active("RT_TEST_ACTIVE", None, active) is True  # the very first frame
    assert phase_just_became_active("TRIAL_ACTIVE", "TRIAL_ACTIVE", active) is False
    assert phase_just_became_active("FOREPERIOD", "WAITING_TO_START", active) is False


def test_caption_rendering_and_update_caption_show_and_speak_in_lockstep():
    stim = Mock()
    assert render_caption(stim, "look at the dot", last_caption="") == "look at the dot"
    assert (stim.text, stim.opacity) == ("look at the dot", 1.0)
    # "" means "don't re-announce", not "nothing to show": the last caption stays.
    stim = Mock()
    assert render_caption(stim, "", last_caption="look at the dot") == "look at the dot"
    assert (stim.text, stim.opacity) == ("look at the dot", 1.0)
    stim = Mock()
    assert render_caption(stim, "", last_caption="") == "" and stim.opacity == 0.0

    narrator, stim = Mock(), Mock()
    spoken, caption = update_caption(narrator, stim, {"instructions": "Press SPACE to begin"}, None, "")
    narrator.speak.assert_called_once_with("Press SPACE to begin")
    assert (spoken, caption, stim.text, stim.opacity) == ("Press SPACE to begin", "Press SPACE to begin", "Press SPACE to begin", 1.0)

    # Hidden: silent, and resets so a repeat of the same text is spoken again.
    narrator, stim = Mock(), Mock()
    text = "Press SPACE when you are ready"
    spoken, caption = update_caption(narrator, stim, {"instructions": text, "hide_caption": True}, text, text)
    narrator.speak.assert_not_called()
    assert (spoken, caption, stim.opacity) == (None, "", 0.0)
    update_caption(narrator, stim, {"instructions": text}, spoken, caption)
    narrator.speak.assert_called_once_with(text)


def _symbol_state(visible: bool) -> dict:
    return {"visible": visible, "x": 0.3, "y": 0.5}


def _render_state_for_fade_test(snap: bool, dot_visible: bool) -> dict:
    return {
        "dot": _symbol_state(dot_visible),
        "cross": _symbol_state(False),
        "calibration_center": _symbol_state(False),
        "grating": {"visible": False, "x": 0.5, "y": 0.5, "contrast": 0, "orientation": None},
        "hud": {"phase": "TRIAL_ACTIVE", "trial": "-", "gaze_zone": "left", "face_found": True, "source_available": True},
        "snap_symbols": snap,
    }


def _stimuli(**extra) -> dict:
    stimuli = {name: MagicMock() for name in ("dot", "cross", "calibration_center", "gaze_indicator", "grating", "hud", "tutorial_text")}
    stimuli["aspect"] = 1.78
    stimuli.update(extra)
    return stimuli


def test_apply_render_state_snaps_or_fades_symbols_and_draws_the_hud_only_in_test_mode():
    import pytest

    from src.experiment.run_experiment import FADE_DURATION_S, apply_render_state

    for snap, expected in ((True, 1.0), (False, pytest.approx(0.016 / FADE_DURATION_S))):
        fades = {"dot": 0.0, "cross": 0.0, "calibration_center": 0.0}
        apply_render_state(_render_state_for_fade_test(snap=snap, dot_visible=True), _stimuli(), fades, dt=0.016)
        assert fades["dot"] == expected, snap  # sharp cue: instant, not 0.016 / FADE_DURATION_S

    for extra, opacity in (({"show_hud": True}, 1.0), ({"show_hud": False}, 0.0), ({}, 0.0)):  # absent flag = hidden
        stimuli = _stimuli(**extra)
        fades = {"dot": 0.0, "cross": 0.0, "calibration_center": 0.0}
        apply_render_state(_render_state_for_fade_test(snap=False, dot_visible=False), stimuli, fades, dt=0.016)
        assert stimuli["hud"].opacity == opacity, extra


def test_a_failure_while_setting_up_the_saccade_phase_still_stops_the_gaze_source():
    """The phone recording is started before the main loop; if building the session (or the tone...) raises, the gaze
    source - whose stop() ends the recording - must still be stopped."""
    import pytest

    from src.experiment import run_experiment

    gaze = MagicMock()
    gaze.start_recording.return_value = {"recording": "started"}
    logger = MagicMock()
    with patch.object(run_experiment, "build_gaze_source", return_value=gaze), patch.object(
        run_experiment, "ExperimentSession", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError, match="boom"):
            run_experiment.run_saccade_phase(MagicMock(), {"aspect": 1.6}, logger, MagicMock())
    gaze.start_recording.assert_called_once()
    gaze.stop.assert_called_once()


# -- start-up check: the Neon is found (or the run stops plainly) before any window or run folder exists ------------


def _resolution(ok: bool):
    from src.eye_tracking.neon_address import SOURCE_GATEWAY, SOURCE_GIVEN, Attempt, Resolution

    tried = (Attempt(SOURCE_GIVEN, "192.168.0.60", 8080, False, "no answer within 2 s"),)
    if ok:
        return Resolution("192.168.43.1", 8080, SOURCE_GATEWAY, tried + (Attempt(SOURCE_GATEWAY, "192.168.43.1", 8080, True, "Neon answered"),))
    return Resolution(None, None, None, tried)


def test_find_neon_or_exit_prints_the_plain_message_shows_the_box_and_exits_2(capsys):
    from src.experiment.run_experiment import find_neon_or_exit

    boxes = []
    seen = {}

    def resolve(address, port, progress):
        seen["args"] = (address, port)
        return _resolution(False)

    with __import__("pytest").raises(SystemExit) as info:
        find_neon_or_exit(
            "192.168.0.60", resolve=resolve, network=lambda: {"wifi_name": "Pixel_1234", "ipv4": "192.168.43.57", "gateways": ["192.168.43.1"]},
            show_box=lambda title, text: boxes.append((title, text)),
        )
    assert info.value.code == 2 and seen["args"] == ("192.168.0.60", 8080)
    out = capsys.readouterr()
    assert "Looking for the Neon glasses" in out.out
    assert "NEON GLASSES NOT FOUND" in out.err and "192.168.0.60:8080" in out.err and "Pixel_1234" in out.err
    assert "Traceback" not in out.err
    assert boxes and boxes[0][0] == "Neon glasses not found" and boxes[0][1] in out.err


def test_find_neon_or_exit_returns_the_found_address():
    from src.experiment.run_experiment import find_neon_or_exit

    r = find_neon_or_exit("", resolve=lambda a, p, progress: _resolution(True), show_box=lambda *_: __import__("pytest").fail("no box"))
    assert (r.address, r.port) == ("192.168.43.1", 8080)


def test_main_stops_before_the_run_folder_and_window_when_the_neon_is_missing(monkeypatch):
    import pytest

    from src.experiment import run_experiment
    from src.experiment.geometry import Geometry

    monkeypatch.setattr(run_experiment.sys, "setswitchinterval", lambda *_: None)
    monkeypatch.setattr(run_experiment, "prompt_session_info", lambda: ("P01", False, True, False, (0.04,), Geometry.default(), 80.0, ""))
    monkeypatch.setattr(run_experiment, "resolve_neon_address", lambda a, p, progress: _resolution(False))
    monkeypatch.setattr(run_experiment, "network_summary", lambda: {})
    boxes = []
    monkeypatch.setattr(run_experiment, "show_error_box", lambda title, text: boxes.append(title))
    monkeypatch.setattr(run_experiment, "build_window", Mock(side_effect=AssertionError("window opened")))
    monkeypatch.setattr(run_experiment, "ResultLogger", Mock(side_effect=AssertionError("run folder created")))
    with pytest.raises(SystemExit) as info:
        run_experiment.main()
    assert info.value.code == 2 and boxes == ["Neon glasses not found"]


def test_main_passes_the_found_address_to_both_gaze_sources_and_session_json(monkeypatch):
    import pytest

    from src.experiment import run_experiment
    from src.experiment.geometry import Geometry

    monkeypatch.setattr(run_experiment.sys, "setswitchinterval", lambda *_: None)
    monkeypatch.setattr(run_experiment, "prompt_session_info", lambda: ("P01", False, True, False, (0.04,), Geometry.default(), 80.0, ""))
    monkeypatch.setattr(run_experiment, "resolve_neon_address", lambda a, p, progress: _resolution(True))
    logger = MagicMock()
    monkeypatch.setattr(run_experiment, "ResultLogger", Mock(return_value=logger))
    monkeypatch.setattr(run_experiment, "build_window", Mock())
    monkeypatch.setattr(run_experiment, "measure_refresh_hz", Mock(return_value=60.0))
    monkeypatch.setattr(run_experiment, "build_stimuli", Mock(return_value={}))
    monkeypatch.setattr(run_experiment, "Narrator", Mock())
    monkeypatch.setattr(run_experiment, "finish_run_summary", Mock(return_value=False))
    tutorial = Mock(return_value=None)  # participant quits during the tutorial
    monkeypatch.setattr(run_experiment, "run_tutorial_phase", tutorial)
    run_experiment.main()
    assert tutorial.call_args.kwargs["neon_address"] == ("192.168.43.1", 8080)
    logger.set_neon_device.assert_called_once_with("192.168.43.1:8080 (PC's gateway (phone hotspot))")
    assert run_experiment._address_kwargs(("192.168.43.1", 8080)) == {"address": "192.168.43.1", "port": 8080}
    assert run_experiment._address_kwargs(None) == {}
