"""The line-stimulus branch: a thin horizontal line 2 deg above/below the saccade target (Braun, Schutz & Gegenfurtner 2021),
answered with UP / DOWN. Placement in each session, key mapping, the above/below quiz, scoring and old-CSV compatibility."""

import csv
import json

import pytest

from include.experiment.constants import (
    CENTER_POSITION,
    CROSS_POSITION,
    DOT_POSITION,
    LINE_OFFSET_DEG,
    LINE_OFFSET_HEIGHT_UNITS,
    LINE_SD_DEG,
    LINE_SD_HEIGHT_UNITS,
    SCREEN_HEIGHT_DEG,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import FlashTrialSpec, Side, Target
from src.dashboard import data_access
from src.experiment.scoring import score_outcome
from src.experiment.trial_mechanics import line_position, random_orientation
from test_presaccade_session import _enter_flash_window
from test_presaccade_session import _make_session as _make_presaccade
from test_session import (
    _complete_calibration_and_rt_test,
    _enter_trial_active,
    _make_session,
    _set_single_trial,
)
from test_tutorial_session import _complete_calibration
from test_tutorial_session import _make_session as _make_tutorial
from test_tutorial_session import _reach_practice, _start_attempt

pytestmark = pytest.mark.usefixtures("scheduled_flash_mode")


def test_side_enum_is_above_and_below_only():
    assert {s.value for s in Side} == {"above", "below"}
    assert {random_orientation() for _ in range(200)} == {Side.ABOVE, Side.BELOW}


def test_offset_and_sd_are_converted_with_the_screen_geometry():
    assert LINE_OFFSET_DEG == 2.0 and LINE_SD_DEG == 0.15
    assert LINE_OFFSET_HEIGHT_UNITS == pytest.approx(LINE_OFFSET_DEG / SCREEN_HEIGHT_DEG)
    assert LINE_SD_HEIGHT_UNITS == pytest.approx(LINE_SD_DEG / SCREEN_HEIGHT_DEG)


def test_line_position_is_shifted_up_or_down_from_the_reference():
    x_a, y_a = line_position(DOT_POSITION, Side.ABOVE)
    x_b, y_b = line_position(DOT_POSITION, Side.BELOW)
    assert x_a == x_b == DOT_POSITION[0]
    assert y_a == pytest.approx(DOT_POSITION[1] - LINE_OFFSET_HEIGHT_UNITS)  # y grows downwards: above = smaller y
    assert y_b == pytest.approx(DOT_POSITION[1] + LINE_OFFSET_HEIGHT_UNITS)
    assert y_a < DOT_POSITION[1] < y_b


@pytest.mark.parametrize(
    "source,target,expected", [(Target.DOT, Target.CROSS, CROSS_POSITION), (Target.CROSS, Target.DOT, DOT_POSITION)]
)
@pytest.mark.parametrize("side", [Side.ABOVE, Side.BELOW])
def test_saccade_phase_line_sits_next_to_the_target_not_the_source(fake_gaze, fake_time, source, target, expected, side):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session, source=source, target=target, orientation=side)
    _enter_trial_active(session, fake_time)
    fake_time(0.201)
    session.tick()  # the scheduled flash fires
    line = session.render_state()["grating"]
    assert line["visible"] is True
    assert (line["x"], line["y"]) == pytest.approx(line_position(expected, side))
    assert line["orientation"] is side
    assert (line["y"] < expected[1]) == (side is Side.ABOVE)


def test_baseline_line_sits_next_to_the_screen_centre(fake_time):
    for side in (Side.ABOVE, Side.BELOW):
        session = _make_presaccade()
        session._trials = [FlashTrialSpec(index=0, orientation=side)]
        _enter_flash_window(session, fake_time)
        line = session.render_state()["grating"]
        assert line["visible"] is True
        assert (line["x"], line["y"]) == pytest.approx(line_position(CENTER_POSITION, side))


def test_answer_keys_are_up_and_down_only():
    run_experiment = pytest.importorskip("src.experiment.run_experiment")
    assert sorted(run_experiment.RESPONSE_KEYS) == ["down", "up"]
    seen = []

    class _Session:
        def on_response_key(self, side):
            seen.append(side)

    run_experiment.dispatch_response_keys(_Session(), ["up", "left", "right", "down", "x"])
    assert seen == [Side.ABOVE, Side.BELOW]


def test_scoring_above_below():
    assert score_outcome(True, Side.ABOVE, Side.ABOVE) == "correct"
    assert score_outcome(True, Side.BELOW, Side.ABOVE) == "incorrect"
    assert score_outcome(False, None, Side.BELOW) == "miss"


def test_instructions_name_up_and_down_and_no_other_keys(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    session.on_space()
    text = session.render_state()["instructions"]
    assert "UP" in text and "DOWN" in text
    pre = _make_presaccade()
    pre.on_space()
    for t in (text, pre.render_state()["instructions"]):
        assert "LEFT" not in t and "RIGHT" not in t and "grating" not in t.lower()
    assert len(text.split()) <= 15


def test_tutorial_quiz_shows_a_line_above_or_below_the_dot_and_needs_a_streak(fake_gaze):
    session = _make_tutorial(fake_gaze)
    _complete_calibration(session)
    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        state = session.render_state()
        assert state["hud"]["phase"] == "QUIZ"
        assert state["dot"]["visible"] and not state["cross"]["visible"]
        line = state["grating"]
        assert line["visible"] and line["orientation"] in (Side.ABOVE, Side.BELOW)
        assert (line["x"], line["y"]) == pytest.approx(line_position(DOT_POSITION, line["orientation"]))
        assert (line["y"] < DOT_POSITION[1]) == (line["orientation"] is Side.ABOVE)
        assert len(state["instructions"].split()) <= 15
        session.on_response_key(line["orientation"])
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"


def test_tutorial_practice_line_follows_the_target(fake_gaze, fake_time):
    session = _make_tutorial(fake_gaze)
    _reach_practice(session)
    _start_attempt(session, fake_time)
    fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
    session.tick()
    line = session.render_state()["grating"]
    target = CROSS_POSITION if session._target is Target.CROSS else DOT_POSITION
    assert line["visible"]
    assert (line["x"], line["y"]) == pytest.approx(line_position(target, session._orientation))


def test_old_csv_with_vertical_horizontal_still_loads_and_new_values_load(tmp_path, monkeypatch):
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
    with (tmp_path / "results_9.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "results_9_meta.json").write_text(json.dumps({}))
    trials = data_access.load_session_trials("9")
    assert [t.orientation for t in trials] == [Side.ABOVE, Side.BELOW, Side.ABOVE, None]
    assert [t.response_orientation for t in trials] == [Side.ABOVE, Side.ABOVE, Side.BELOW, None]
