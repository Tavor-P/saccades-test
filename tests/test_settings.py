import json

import pytest

from src.experiment import settings as settings_module
from src.experiment.settings import (
    DEFAULT_CONTRAST_FLOOR_PERCENT,
    MAX_CONTRAST_FLOOR_PERCENT,
    MIN_CONTRAST_FLOOR_PERCENT,
    load_contrast_floor_percent,
    save_contrast_floor_percent,
    validate_contrast_floor_percent,
)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_module, "DATA_DIR", tmp_path)


def test_load_falls_back_to_default_when_no_settings_file():
    assert load_contrast_floor_percent() == DEFAULT_CONTRAST_FLOOR_PERCENT


def test_save_then_load_round_trips():
    save_contrast_floor_percent(3.5)
    assert load_contrast_floor_percent() == 3.5


def test_save_preserves_other_keys_in_the_file(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"unrelated_key": "keep me"}))
    save_contrast_floor_percent(2.0)
    data = json.loads((tmp_path / "settings.json").read_text())
    assert data == {"unrelated_key": "keep me", "contrast_floor_percent": 2.0}


def test_validate_rejects_below_the_hardware_floor():
    with pytest.raises(ValueError):
        validate_contrast_floor_percent(MIN_CONTRAST_FLOOR_PERCENT / 2)


def test_validate_rejects_at_or_above_the_zest_ceiling():
    with pytest.raises(ValueError):
        validate_contrast_floor_percent(MAX_CONTRAST_FLOOR_PERCENT)


def test_validate_accepts_the_hardware_floor_itself():
    assert validate_contrast_floor_percent(MIN_CONTRAST_FLOOR_PERCENT) == MIN_CONTRAST_FLOOR_PERCENT


def test_save_rejects_out_of_range_value_without_writing_the_file(tmp_path):
    with pytest.raises(ValueError):
        save_contrast_floor_percent(-1)
    assert not (tmp_path / "settings.json").exists()
