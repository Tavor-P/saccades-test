"""Everything the participant reads or hears: short, plain, positional ("the dot on the left")."""

import re

from include.experiment import participant_text as words
from src.experiment import run_experiment


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
    assert len(strings) >= 15
    for text in strings:
        assert len(text.split()) <= 15, text
        assert not re.search(r"other dot|cross", text, re.IGNORECASE), text


def test_the_sync_step_line_says_what_is_coming():
    assert run_experiment.SYNC_FLASH_ANNOUNCEMENT == words.SYNC_FLASH
    assert "green" in words.SYNC_FLASH and "normal" in words.SYNC_FLASH


def test_dots_are_named_by_position():
    assert "left" in words.CALIBRATE_LEFT and "right" in words.CALIBRATE_RIGHT and "middle" in words.CALIBRATE_CENTER
    assert "left" in words.RT_TEST_INTRO
