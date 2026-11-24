from unittest.mock import MagicMock, Mock, patch

# run_experiment (PsychoPy, Qt, pandas) is imported inside the tests: every test worker collects this file, and the import costs seconds.


def _patch_screens(count: int):
    display = MagicMock()
    display.get_screens.return_value = [MagicMock() for _ in range(count)]
    return patch("src.experiment.rendering.pyglet.canvas.get_display", return_value=display)


def test_second_monitor_screen_index_picks_the_second_screen_or_falls_back():
    from src.experiment.run_experiment import _second_monitor_screen_index

    with _patch_screens(2):
        assert _second_monitor_screen_index() == 1
    with _patch_screens(1):
        assert _second_monitor_screen_index() == 0


def test_narrate_if_changed_speaks_only_new_non_blank_text():
    from src.experiment.run_experiment import narrate_if_changed

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
    from src.experiment.run_experiment import phase_just_became_active

    active = frozenset({"TRIAL_ACTIVE", "RT_TEST_ACTIVE"})
    assert phase_just_became_active("TRIAL_ACTIVE", "FOREPERIOD", active) is True
    assert phase_just_became_active("RT_TEST_ACTIVE", "RT_TEST_FOREPERIOD", active) is True
    assert phase_just_became_active("RT_TEST_ACTIVE", None, active) is True  # the very first frame
    assert phase_just_became_active("TRIAL_ACTIVE", "TRIAL_ACTIVE", active) is False
    assert phase_just_became_active("FOREPERIOD", "WAITING_TO_START", active) is False


def test_caption_rendering_and_update_caption_show_and_speak_in_lockstep():
    from src.experiment.run_experiment import render_caption, update_caption

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
        "line": {"visible": False, "x": 0.5, "y": 0.5, "contrast": 0, "side": None},
        "hud": {"phase": "TRIAL_ACTIVE", "trial": "-", "gaze_zone": "left", "face_found": True, "source_available": True},
        "snap_symbols": snap,
    }


def _stimuli(**extra) -> dict:
    stimuli = {name: MagicMock() for name in ("dot", "cross", "calibration_center", "gaze_indicator", "line", "hud", "tutorial_text")}
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
    monkeypatch.setattr(run_experiment, "prompt_session_info", lambda: ("P01", False, True, False, (0.04,), Geometry.default(), 80.0, "", True))
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
    from src.experiment import run_experiment
    from src.experiment.geometry import Geometry

    monkeypatch.setattr(run_experiment.sys, "setswitchinterval", lambda *_: None)
    monkeypatch.setattr(run_experiment, "prompt_session_info", lambda: ("P01", False, True, False, (0.04,), Geometry.default(), 80.0, "", True))
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
    auto = Mock()
    monkeypatch.setattr(run_experiment, "_auto_analyze", auto)
    run_experiment.main()
    auto.assert_called_once_with(logger.folder)  # a run that ends normally (even after quitting early) is analysed
    assert tutorial.call_args.kwargs["neon_address"] == ("192.168.43.1", 8080)
    assert tutorial.call_args.kwargs["return_jump"] is True
    logger.set_return_jump.assert_called_once_with(True)
    logger.set_neon_device.assert_called_once_with("192.168.43.1:8080 (PC's gateway (phone hotspot))")
    assert run_experiment._address_kwargs(("192.168.43.1", 8080)) == {"address": "192.168.43.1", "port": 8080}
    assert run_experiment._address_kwargs(None) == {}


# -- the Neon lost mid-run, and cleanup that must always reach gaze.stop() ------------------------------------------


def test_main_turns_a_lost_neon_into_a_plain_message_box_and_exit_2_after_closing_the_window(monkeypatch, capsys):
    import pytest

    from src.experiment import run_experiment
    from src.experiment.geometry import Geometry
    from src.eye_tracking.neon_source import NeonConnectionError

    monkeypatch.setattr(run_experiment.sys, "setswitchinterval", lambda *_: None)
    monkeypatch.setattr(run_experiment, "prompt_session_info", lambda: ("P01", False, False, False, (0.04,), Geometry.default(), 80.0, "", True))
    monkeypatch.setattr(run_experiment, "resolve_neon_address", lambda a, p, progress: _resolution(True))
    logger = MagicMock()
    monkeypatch.setattr(run_experiment, "ResultLogger", Mock(return_value=logger))
    win = Mock()
    monkeypatch.setattr(run_experiment, "build_window", Mock(return_value=win))
    monkeypatch.setattr(run_experiment, "measure_refresh_hz", Mock(return_value=60.0))
    monkeypatch.setattr(run_experiment, "build_stimuli", Mock(return_value={}))
    monkeypatch.setattr(run_experiment, "Narrator", Mock())
    monkeypatch.setattr(run_experiment, "finish_run_summary", Mock(return_value=True))
    done_screen = Mock()
    monkeypatch.setattr(run_experiment, "show_done_screen", done_screen)
    monkeypatch.setattr(run_experiment, "run_saccade_phase", Mock(side_effect=NeonConnectionError("The Neon phone at 192.168.43.1:8080 does not answer.")))
    boxes = []
    monkeypatch.setattr(run_experiment, "show_error_box", lambda title, text: boxes.append((title, text, win.close.called)))
    auto = Mock()
    monkeypatch.setattr(run_experiment, "_auto_analyze", auto)
    with pytest.raises(SystemExit) as info:
        run_experiment.main()
    assert info.value.code == 2
    assert boxes and boxes[0][0] == run_experiment.NEON_LOST_TITLE and "does not answer" in boxes[0][1]
    assert boxes[0][2], "the fullscreen window must be closed before the box is shown"
    logger.close.assert_called_once()
    done_screen.assert_not_called()
    auto.assert_not_called()  # a lost Neon leaves a partial run: no automatic analysis
    err = capsys.readouterr().err
    assert "does not answer" in err and "Traceback" not in err


# -- automatic analysis when a run ends ------------------------------------------------------------------------------


def _fake_analysis(report: str):
    from pathlib import Path

    def fake(timestamp, data_dir, **_):
        return {"report": report, "report_path": Path(data_dir) / "P01" / timestamp / "analysis" / "report.txt"}

    return fake


def test_auto_analyze_prints_the_report_top_block_and_the_analysis_folder(monkeypatch, tmp_path, capsys):
    from src.experiment import analyze_session, run_experiment

    report ="QUALITY  good\nRESULT   -12 %\nAMOUNT   3 flashes\n\nDetails below the blank line\nnot shown\n"
    monkeypatch.setattr(analyze_session, "analyze", _fake_analysis(report))
    folder = tmp_path / "P01" / "2026-11-24_10-00-00"
    run_experiment._auto_analyze(folder)
    out = capsys.readouterr().out
    assert "Analysing this run..." in out
    assert "QUALITY  good\nRESULT   -12 %\nAMOUNT   3 flashes" in out
    assert "Details below" not in out
    assert "Analysis folder:" in out and "analysis" in out


def test_auto_analyze_swallows_a_failed_analysis_and_prints_the_rerun_command(monkeypatch, tmp_path, capsys):
    from src.experiment import analyze_session, run_experiment

    def boom(*_args, **_kwargs):
        raise ValueError("no gaze trace in this run")

    monkeypatch.setattr(analyze_session, "analyze", boom)
    folder = tmp_path / "P01" / "2026-11-24_10-00-00"
    run_experiment._auto_analyze(folder)  # returns normally: the run is already saved
    out = capsys.readouterr().out
    assert "Automatic analysis skipped: no gaze trace in this run." in out
    assert "scripts/py.ps1 -m src.experiment.analyze_session 2026-11-24_10-00-00" in out


def _saccade_phase_with(monkeypatch, gaze, **patches):
    from src.experiment import run_experiment

    monkeypatch.setattr(run_experiment, "build_gaze_source", Mock(return_value=gaze))
    monkeypatch.setattr(run_experiment, "ExperimentSession", Mock(return_value=MagicMock()))
    monkeypatch.setattr(run_experiment, "build_go_cue_tone", Mock())
    monkeypatch.setattr(run_experiment, "run_sync_flashes", Mock())
    monkeypatch.setattr(run_experiment, "FramePresenter", Mock())
    monkeypatch.setattr(run_experiment, "drain_stale_space", Mock())
    monkeypatch.setattr(run_experiment.event, "getKeys", Mock(return_value=["escape"]))  # quit on the first frame
    for name, value in patches.items():
        monkeypatch.setattr(run_experiment, name, value)
    return run_experiment.run_saccade_phase(MagicMock(), {"aspect": 1.6}, MagicMock(), MagicMock())


def test_saccade_phase_cleanup_still_stops_the_recording_and_gaze_when_an_earlier_step_fails(monkeypatch, capsys):
    gaze = MagicMock()
    gaze.start_recording.return_value = {"recording": "started"}
    gaze.phone_ip = "192.168.43.1"
    gaze.quality_summary.side_effect = RuntimeError("summary broke")
    result = _saccade_phase_with(monkeypatch, gaze, report_space_presses=Mock(side_effect=RuntimeError("tally broke")))
    assert result is None
    gaze.stop_recording.assert_called_once()
    gaze.stop.assert_called_once()
    err = capsys.readouterr().err
    assert "tally broke" in err and "summary broke" in err


def test_tutorial_phase_stops_the_gaze_source_when_setting_up_the_session_fails(monkeypatch):
    import pytest

    from src.experiment import run_experiment

    gaze = MagicMock()
    monkeypatch.setattr(run_experiment, "build_gaze_source", Mock(return_value=gaze))
    for name in ("TutorialSession", "build_go_cue_tone"):
        with patch.object(run_experiment, name, Mock(side_effect=RuntimeError(name))):
            gaze.reset_mock()
            with pytest.raises(RuntimeError, match=name):
                run_experiment.run_tutorial_phase(MagicMock(), {"aspect": 1.6}, MagicMock())
            gaze.stop.assert_called_once()
