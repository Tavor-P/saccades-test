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


def test_load_defaults_then_save_round_trips_and_preserves_other_keys(tmp_path):
    assert load_contrast_floor_percent() == DEFAULT_CONTRAST_FLOOR_PERCENT  # no settings file
    save_contrast_floor_percent(3.5)
    assert load_contrast_floor_percent() == 3.5
    (tmp_path / "settings.json").write_text(json.dumps({"unrelated_key": "keep me"}))
    save_contrast_floor_percent(2.0)
    assert json.loads((tmp_path / "settings.json").read_text()) == {"unrelated_key": "keep me", "contrast_floor_percent": 2.0}


def test_validate_bounds_and_save_rejects_out_of_range_without_writing(tmp_path):
    with pytest.raises(ValueError):
        validate_contrast_floor_percent(MIN_CONTRAST_FLOOR_PERCENT / 2)  # below the hardware floor
    with pytest.raises(ValueError):
        validate_contrast_floor_percent(MAX_CONTRAST_FLOOR_PERCENT)  # at the ZEST ceiling
    assert validate_contrast_floor_percent(MIN_CONTRAST_FLOOR_PERCENT) == MIN_CONTRAST_FLOOR_PERCENT
    with pytest.raises(ValueError):
        save_contrast_floor_percent(-1)
    assert not (tmp_path / "settings.json").exists()
