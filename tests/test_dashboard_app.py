import csv
import json

import pytest

from src.dashboard import app as app_module
from src.dashboard import data_access

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
    # data_access's own functions read DATA_DIR from their own module globals,
    # so patching it there is enough for them - but app.py did `from ... import
    # GRAPH_CACHE_DIR`, a separate name binding, so that one needs patching
    # where app.py actually looks it up.
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module, "GRAPH_CACHE_DIR", tmp_path / "cache")
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
