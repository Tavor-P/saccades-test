"""Everything the participant reads or hears: short, plain, positional ("the dot on the left"), feedback only in the tutorial."""

import re
from unittest.mock import MagicMock

import pytest

from include.experiment import participant_text as words
from include.experiment.constants import SACCADE_FIXED_TRIALS_TEST
from include.experiment.types import Side
from src.experiment import run_experiment
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession, Phase
from src.experiment.tutorial_session import TutorialSession

_FEEDBACK = {*words.FEEDBACK_CORRECT, *words.FEEDBACK_WRONG, *words.FEEDBACK_MISSED}


class _Logger:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _all_strings() -> list[str]:
    out = []
    for name, value in vars(words).items():
        if name.startswith("_"):
            continue
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, tuple):
            out.extend(v if isinstance(v, str) else v[0] for v in value)
    return out


def test_every_string_is_short_and_uses_plain_words():
    strings = _all_strings()
    assert len(strings) >= 20
    for text in strings:
        assert len(text.split()) <= 15, text
        assert not re.search(r"other dot|cross", text, re.IGNORECASE), text


def test_the_sync_step_line_says_what_is_coming():
    assert run_experiment.SYNC_FLASH_ANNOUNCEMENT == words.SYNC_FLASH
    assert "green" in words.SYNC_FLASH and "normal" in words.SYNC_FLASH


def test_dots_are_named_by_position():
    assert "left" in words.CALIBRATE_LEFT and "right" in words.CALIBRATE_RIGHT and "middle" in words.CALIBRATE_CENTER
    assert "left" in words.RT_TEST_INTRO


def test_feedback_variants_differ_so_an_equal_verdict_is_still_spoken():
    for variants in (words.FEEDBACK_CORRECT, words.FEEDBACK_WRONG, words.FEEDBACK_MISSED):
        assert len(set(variants)) == len(variants)
    assert not set(words.FEEDBACK_CORRECT) & set(words.FEEDBACK_WRONG)


def test_the_saccade_phase_never_shows_the_feedback_cue(fake_gaze):
    session = ExperimentSession(fake_gaze, logger=_Logger(), clock=PausableClock(), test_mode=True)
    texts = {session.render_state()["instructions"]}
    for _ in range(30):
        session.on_space()
        session.on_response_key(Side.ABOVE)
        session.tick()
        texts.add(session.render_state()["instructions"])
    assert not texts & _FEEDBACK


def test_in_the_tutorial_the_cue_only_appears_in_quiz_and_practice(fake_gaze):
    session = TutorialSession(fake_gaze, PausableClock())
    phases_with_cue = set()
    for _ in range(60):
        state = session.render_state()
        if state["instructions"] in _FEEDBACK:
            phases_with_cue.add(state["hud"]["phase"])
        session.on_space()
        session.on_response_key(Side.ABOVE)
    assert phases_with_cue  # the quiz was reached and answered
    assert phases_with_cue <= {"QUIZ", "PRACTICE_AWAITING_SPACE", "PRACTICE_FOREPERIOD", "PRACTICE_ACTIVE"}


def _session_waiting_before_trial(fake_gaze, done: int) -> ExperimentSession:
    session = ExperimentSession(
        fake_gaze,
        logger=_Logger(),
        clock=PausableClock(),
        test_mode=True,
        precomputed_calibration_ratios=(0.2, 0.5, 0.8),
        precomputed_rt_samples_ms=[250.0, 260.0, 255.0],
    )
    session.on_space()  # straight to the first trial (no practice trials after a tutorial)
    session._main_trial_counter = done + 1
    session._phase = Phase.AWAITING_SPACE
    return session


def test_milestone_lines_come_before_the_trial_after_25_50_75_percent(fake_gaze):
    n = SACCADE_FIXED_TRIALS_TEST
    said = {}
    for done in range(1, n):
        line = _session_waiting_before_trial(fake_gaze, done).render_state()["milestone_line"]
        if line:
            said[done] = line
    assert set(said.values()) == {line for line, _ in words.MILESTONES}
    assert said[round(n * 0.5)] == "Halfway there."
    assert _session_waiting_before_trial(fake_gaze, 0).render_state()["milestone_line"] == ""


def test_milestone_speech_is_spoken_once_and_never_drawn():
    narrator, stim = MagicMock(), MagicMock()
    state = {"instructions": "x", "hide_caption": True, "milestone_line": "Halfway there."}
    spoken, caption = run_experiment.update_caption(narrator, stim, state, None, "")
    spoken, caption = run_experiment.update_caption(narrator, stim, state, spoken, caption)
    narrator.speak.assert_called_once_with("Halfway there.")
    assert stim.text == "" and stim.opacity == 0.0 and caption == ""
    state = {"instructions": "x", "hide_caption": True, "milestone_line": ""}
    assert run_experiment.update_caption(narrator, stim, state, spoken, caption) == (None, "")
