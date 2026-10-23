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
    # app.py only ever reaches data_access's DATA_DIR/GRAPH_CACHE_DIR through
    # data_access's own functions (graph_cache_path_for, delete_session, ...),
    # so patching them here is enough - app.py holds no separate binding of
    # its own to go stale.
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_access, "GRAPH_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_module, "DATA_DIR", tmp_path)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _write_session(tmp_path, timestamp, meta=None):
    csv_path = tmp_path / f"results_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()
    if meta is not None:
        (tmp_path / f"results_{timestamp}_meta.json").write_text(json.dumps(meta))


def test_index_lists_sessions(client, tmp_path):
    _write_session(tmp_path, "123", meta={"participant_id": "p9"})
    response = client.get("/")
    assert response.status_code == 200
    assert b"p9" in response.data


def test_index_handles_no_sessions(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"No sessions logged yet" in response.data


def test_update_session_persists_fields(client, tmp_path):
    _write_session(tmp_path, "456")
    response = client.post("/sessions/456", json={"name": "Sam", "gender": "man", "age": "31"})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    meta = json.loads((tmp_path / "results_456_meta.json").read_text())
    assert meta["name"] == "Sam"
    assert meta["age"] == 31


def test_update_session_rejects_bad_age(client, tmp_path):
    _write_session(tmp_path, "789")
    response = client.post("/sessions/789", json={"name": "", "gender": "", "age": "nope"})
    assert response.status_code == 400


def test_update_session_404s_for_unknown_timestamp(client):
    response = client.post("/sessions/000000", json={"name": "", "gender": "", "age": ""})
    assert response.status_code == 404


def test_session_graph_renders_a_png(client, tmp_path):
    _write_session(tmp_path, "321")
    response = client.get("/sessions/321/graph.png")
    assert response.status_code == 200
    assert response.content_type == "image/png"
    assert len(response.data) > 0


def test_session_graph_404s_for_unknown_timestamp(client):
    response = client.get("/sessions/000000/graph.png")
    assert response.status_code == 404


def test_delete_session_removes_the_session(client, tmp_path):
    _write_session(tmp_path, "654")
    response = client.delete("/sessions/654")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert not (tmp_path / "results_654.csv").exists()


def test_delete_session_404s_for_unknown_timestamp(client):
    response = client.delete("/sessions/000000")
    assert response.status_code == 404


def test_deleted_session_no_longer_appears_in_the_index(client, tmp_path):
    _write_session(tmp_path, "111", meta={"participant_id": "gone"})
    client.delete("/sessions/111")
    response = client.get("/")
    assert b"gone" not in response.data


def test_index_shows_the_default_contrast_floor(client):
    response = client.get("/")
    assert f'value="{DEFAULT_CONTRAST_FLOOR_PERCENT:.2f}"'.encode() in response.data


def test_update_settings_persists_the_contrast_floor(client, tmp_path):
    response = client.post("/settings", json={"contrast_floor_percent": 4.5})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert json.loads((tmp_path / "settings.json").read_text()) == {"contrast_floor_percent": 4.5}

    # Takes effect on the next page load without restarting anything.
    response = client.get("/")
    assert b'value="4.50"' in response.data


def test_update_settings_rejects_below_the_hardware_floor(client, tmp_path):
    response = client.post("/settings", json={"contrast_floor_percent": MIN_CONTRAST_FLOOR_PERCENT / 2})
    assert response.status_code == 400
    assert not (tmp_path / "settings.json").exists()


def test_update_settings_rejects_non_numeric_value(client):
    response = client.post("/settings", json={"contrast_floor_percent": "not a number"})
    assert response.status_code == 400
