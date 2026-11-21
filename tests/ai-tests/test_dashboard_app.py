import csv
import json

import pytest

from src.dashboard import app as app_module
from src.dashboard import data_access
from src.experiment import settings as settings_module
from src.experiment.settings import DEFAULT_CONTRAST_FLOOR_PERCENT, MIN_CONTRAST_FLOOR_PERCENT

_FIELDNAMES = [
    "trial_index",
    "phase",
    "source",
    "target",
    "saccade_duration_ms",
    "grating_shown",
    "contrast",
    "responded",
    "response_time_ms",
    "outcome",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # app.py only ever reaches data_access's DATA_DIR through
    # data_access's own functions (delete_session, ...),
    # so patching them here is enough - app.py holds no separate binding of
    # its own to go stale.
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings_module, "DATA_DIR", tmp_path)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _write_session(tmp_path, timestamp, meta=None):
    csv_path = tmp_path / f"results_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()
    if meta is not None:
        (tmp_path / f"results_{timestamp}_meta.json").write_text(json.dumps(meta))



def test_index_lists_sessions_or_says_none_and_shows_the_default_contrast_floor(client, tmp_path):
    response = client.get("/")
    assert response.status_code == 200
    assert b"No sessions logged yet" in response.data
    assert f'value="{DEFAULT_CONTRAST_FLOOR_PERCENT:.2f}"'.encode() in response.data
    _write_session(tmp_path, "123", meta={"participant_id": "p9"})
    response = client.get("/")
    assert response.status_code == 200
    assert b"p9" in response.data


def test_update_session_persists_fields_rejects_a_bad_age_and_404s_for_unknown(client, tmp_path):
    _write_session(tmp_path, "456")
    response = client.post("/sessions/456", json={"name": "Sam", "gender": "man", "age": "31"})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    meta = json.loads((tmp_path / "results_456_meta.json").read_text())
    assert meta["name"] == "Sam" and meta["age"] == 31
    assert client.post("/sessions/456", json={"name": "", "gender": "", "age": "nope"}).status_code == 400
    assert client.post("/sessions/000000", json={"name": "", "gender": "", "age": ""}).status_code == 404


def test_delete_session_removes_it_from_disk_and_the_index_and_404s_for_unknown(client, tmp_path):
    _write_session(tmp_path, "654", meta={"participant_id": "gone"})
    response = client.delete("/sessions/654")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert not (tmp_path / "results_654.csv").exists()
    assert b"gone" not in client.get("/").data
    assert client.delete("/sessions/000000").status_code == 404


def test_update_settings_persists_the_contrast_floor_and_rejects_bad_values(client, tmp_path):
    response = client.post("/settings", json={"contrast_floor_percent": MIN_CONTRAST_FLOOR_PERCENT / 2})
    assert response.status_code == 400  # below the hardware floor
    assert not (tmp_path / "settings.json").exists()
    assert client.post("/settings", json={"contrast_floor_percent": "not a number"}).status_code == 400

    response = client.post("/settings", json={"contrast_floor_percent": 4.5})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert json.loads((tmp_path / "settings.json").read_text()) == {"contrast_floor_percent": 4.5}
    assert b'value="4.50"' in client.get("/").data  # takes effect on the next page load without restarting anything
