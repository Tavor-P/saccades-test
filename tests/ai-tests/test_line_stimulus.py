"""The line-stimulus branch: a thin horizontal line LINE_OFFSET_DEG (3 deg since 2026-11-27) above/below the saccade target (Braun, Schutz & Gegenfurtner 2021),
answered with UP / DOWN. Placement in each session, key mapping, the above/below quiz, scoring and old-CSV compatibility."""

import pytest
from synthetic_runs import write_old_column_run
from test_session import (
    _complete_calibration_and_rt_test,
    _enter_trial_active,
    _make_session,
    _set_single_trial,
)
from test_tutorial_session import _complete_calibration, _reach_practice, _start_attempt
from test_tutorial_session import _make_session as _make_tutorial

from include.experiment.constants import (
    CROSS_POSITION,
    DOT_POSITION,
    LINE_OFFSET_DEG,
    LINE_SD_DEG,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Side, Target
from src.dashboard import data_access
from src.experiment.geometry import Geometry
from src.experiment.logger import read_trials
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import score_outcome
from src.experiment.trial_mechanics import line_position, random_side
from src.experiment.tutorial_session import TutorialSession

pytestmark = pytest.mark.usefixtures("no_flash_jitter")



def test_side_enum_constants_and_line_position_shifted_up_or_down_with_the_geometry():
    assert {s.value for s in Side} == {"above", "below"}
    assert {random_side() for _ in range(200)} == {Side.ABOVE, Side.BELOW}
    assert LINE_OFFSET_DEG == 3.0 and LINE_SD_DEG == 0.15
    default = Geometry.default()
    offset = default.line_offset_height_units
    assert offset == pytest.approx(LINE_OFFSET_DEG / default.height_deg)
    assert default.line_sd_height_units == pytest.approx(LINE_SD_DEG / default.height_deg)

    x_a, y_a = line_position(DOT_POSITION, Side.ABOVE)
    x_b, y_b = line_position(DOT_POSITION, Side.BELOW)
    assert x_a == x_b == DOT_POSITION[0]
    assert y_a == pytest.approx(DOT_POSITION[1] - offset)  # y grows downwards: above = smaller y
    assert y_b == pytest.approx(DOT_POSITION[1] + offset)
    assert y_a < DOT_POSITION[1] < y_b

    g = Geometry(13, 40)
    assert line_position(DOT_POSITION, Side.ABOVE, g)[1] == pytest.approx(DOT_POSITION[1] - g.line_offset_height_units)
    assert line_position(DOT_POSITION, Side.BELOW) == line_position(DOT_POSITION, Side.BELOW, Geometry.default())


def _flashed_line(fake_gaze, fake_time, source, target, side, **kwargs):
    session = _make_session(fake_gaze, **kwargs)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session, source=source, target=target, line_side=side)
    _enter_trial_active(session, fake_time)
    fake_time(0.201)
    session.tick()  # the scheduled flash fires
    return session.render_state()["line"]


def test_saccade_phase_line_sits_next_to_the_target_not_the_source_at_the_session_geometry(fake_gaze, fake_time):
    for source, target, expected in [(Target.DOT, Target.CROSS, CROSS_POSITION), (Target.CROSS, Target.DOT, DOT_POSITION)]:
        for side in (Side.ABOVE, Side.BELOW):
            line = _flashed_line(fake_gaze, fake_time, source, target, side)
            assert line["visible"] is True
            assert (line["x"], line["y"]) == pytest.approx(line_position(expected, side)), (target, side)
            assert line["side"] is side
            assert (line["y"] < expected[1]) == (side is Side.ABOVE)

    geometry = Geometry(screen_height_cm=13.0, viewing_distance_cm=40.0)
    line = _flashed_line(fake_gaze, fake_time, Target.DOT, Target.CROSS, Side.ABOVE, geometry=geometry)
    assert line["y"] == pytest.approx(CROSS_POSITION[1] - LINE_OFFSET_DEG / geometry.height_deg)
    assert line["y"] != pytest.approx(CROSS_POSITION[1] - Geometry.default().line_offset_height_units)


def test_answer_keys_scoring_and_instructions_are_up_down_only(fake_gaze, fake_time):
    assert score_outcome(True, Side.ABOVE, Side.ABOVE) == "correct"
    assert score_outcome(True, Side.BELOW, Side.ABOVE) == "incorrect"
    assert score_outcome(False, None, Side.BELOW) == "miss"

    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    session.on_space()
    text = session.render_state()["instructions"]
    assert "UP" in text and "DOWN" in text
    assert "LEFT" not in text and "RIGHT" not in text and "grating" not in text.lower()
    assert len(text.split()) <= 15

    # Last: skipped where run_experiment (PsychoPy) cannot be imported.
    run_experiment = pytest.importorskip("src.experiment.run_experiment")
    assert sorted(run_experiment.RESPONSE_KEYS) == ["down", "up"]
    seen = []

    class _Session:
        def on_response_key(self, side):
            seen.append(side)

    run_experiment.dispatch_response_keys(_Session(), ["up", "left", "right", "down", "x"])
    assert seen == [Side.ABOVE, Side.BELOW]


def test_tutorial_quiz_line_is_above_or_below_the_dot_at_the_geometry_and_practice_line_follows_the_target(fake_gaze, fake_time):
    session = _make_tutorial(fake_gaze)
    _complete_calibration(session)
    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        state = session.render_state()
        assert state["hud"]["phase"] == "QUIZ"
        assert state["dot"]["visible"] and not state["cross"]["visible"]
        line = state["line"]
        assert line["visible"] and line["side"] in (Side.ABOVE, Side.BELOW)
        assert (line["x"], line["y"]) == pytest.approx(line_position(DOT_POSITION, line["side"]))
        assert (line["y"] < DOT_POSITION[1]) == (line["side"] is Side.ABOVE)
        assert len(state["instructions"].split()) <= 15
        session.on_response_key(line["side"])
    assert session.render_state()["hud"]["phase"] == "PRACTICE_INTRO"

    geometry = Geometry(screen_height_cm=13.0, viewing_distance_cm=40.0)
    session = TutorialSession(fake_gaze, PausableClock(), geometry=geometry)
    _complete_calibration(session)
    line = session.render_state()["line"]
    assert abs(line["y"] - DOT_POSITION[1]) == pytest.approx(LINE_OFFSET_DEG / geometry.height_deg)

    session = _make_tutorial(fake_gaze)
    _reach_practice(session)
    _start_attempt(session, fake_time)
    fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
    session.tick()
    line = session.render_state()["line"]
    target = CROSS_POSITION if session._target is Target.CROSS else DOT_POSITION
    assert line["visible"]
    assert (line["x"], line["y"]) == pytest.approx(line_position(target, session._line_side))


def test_old_trials_csv_with_vertical_horizontal_still_loads_and_new_values_load(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    fields = [
        "trial_index", "phase", "source", "target", "saccade_duration_ms", "grating_shown", "contrast",
        "orientation", "responded", "response_orientation", "response_time_ms", "outcome",
    ]
    base = dict(phase="saccade", source="", target="", saccade_duration_ms="", grating_shown="True", contrast="0.1",
                responded="True", response_time_ms="300")
    rows = [
        dict(base, trial_index=0, orientation="vertical", response_orientation="vertical", outcome="correct"),
        dict(base, trial_index=1, orientation="horizontal", response_orientation="vertical", outcome="incorrect"),
        dict(base, trial_index=2, orientation="above", response_orientation="below", outcome="incorrect"),
        dict(base, trial_index=3, grating_shown="False", contrast="", orientation="", response_orientation="", outcome="correct_rejection"),
    ]
    write_old_column_run(tmp_path, 9, rows, fields)
    trials = read_trials(data_access.csv_path_for("9"))
    assert [t.line_side for t in trials] == [Side.ABOVE, Side.BELOW, Side.ABOVE, None]
    assert [t.answer_side for t in trials] == [Side.ABOVE, Side.ABOVE, Side.BELOW, None]
