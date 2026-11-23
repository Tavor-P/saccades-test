import csv
import json

import pytest

from src.dashboard import app as app_module
from src.dashboard import data_access
from src.experiment import settings as settings_module

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



def test_index_lists_sessions_or_says_none(client, tmp_path):
    response = client.get("/")
    assert response.status_code == 200
    assert b"No sessions logged yet" in response.data
    _write_session(tmp_path, "123", meta={"participant_id": "p9"})
    response = client.get("/")
    assert response.status_code == 200
    assert b"p9" in response.data


def test_update_session_persists_fields_rejects_a_bad_age_and_404s_for_unknown(client, tmp_path):
    _write_session(tmp_path, "456")
    response = client.post("/sessions/456", json={"gender": "man", "age": "31"})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    meta = json.loads((tmp_path / "results_456_meta.json").read_text())
    assert meta["gender"] == "man" and meta["age"] == 31 and "name" not in meta
    assert client.post("/sessions/456", json={"gender": "", "age": "nope"}).status_code == 400
    assert client.post("/sessions/000000", json={"gender": "", "age": ""}).status_code == 404


def test_old_session_with_a_name_lists_fine_and_page_has_no_name_field(client, tmp_path):
    _write_session(tmp_path, "321", meta={"participant_id": "p5", "name": "Zelda Quux", "gender": "nonbinary", "age": 47})
    response = client.get("/")
    assert response.status_code == 200
    assert b"p5" in response.data
    for leaked in (b"Zelda", b"Quux", b"Name", b"filter-name", b'class="field name"'):
        assert leaked not in response.data
    assert b"Gender" in response.data and b"Age" in response.data


def test_delete_session_removes_it_from_disk_and_the_index_and_404s_for_unknown(client, tmp_path):
    _write_session(tmp_path, "654", meta={"participant_id": "gone"})
    response = client.delete("/sessions/654")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert not (tmp_path / "results_654.csv").exists()
    assert b"gone" not in client.get("/").data
    assert client.delete("/sessions/000000").status_code == 404
