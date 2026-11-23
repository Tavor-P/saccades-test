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


def test_old_session_with_personal_fields_lists_fine_and_page_never_shows_them(client, tmp_path):
    _write_session(tmp_path, "321", meta={"participant_id": "p5", "name": "Zelda Quux", "gender": "nonbinary", "age": 47})
    response = client.get("/")
    assert response.status_code == 200
    assert b"p5" in response.data
    for leaked in (b"Zelda", b"Quux", b"nonbinary", b"47", b"Gender", b"Age", b"Name contains"):
        assert leaked not in response.data
    assert client.post("/sessions/321", json={"name": "Sam"}).status_code == 405  # the save route is gone


def test_delete_session_removes_it_from_disk_and_the_index_and_404s_for_unknown(client, tmp_path):
    _write_session(tmp_path, "654", meta={"participant_id": "gone"})
    response = client.delete("/sessions/654")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert not (tmp_path / "results_654.csv").exists()
    assert b"gone" not in client.get("/").data
    assert client.delete("/sessions/000000").status_code == 404
