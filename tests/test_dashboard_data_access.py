import csv
import json

import pytest

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


def _write_session(tmp_path, timestamp, rows=None, meta=None):
    csv_path = tmp_path / f"results_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows or []:
            writer.writerow(row)
    if meta is not None:
        (tmp_path / f"results_{timestamp}_meta.json").write_text(json.dumps(meta))


def _row(phase="presaccade", **overrides):
    row = {
        "trial_index": "0",
        "phase": phase,
        "source": "",
        "target": "",
        "saccade_duration_ms": "",
        "grating_shown": "True",
        "contrast": "0.05",
        "responded": "True",
        "response_time_ms": "250.0",
        "outcome": "hit",
    }
    row.update(overrides)
    return row


def test_load_session_trials_round_trips_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "111", rows=[_row()])

    trials = data_access.load_session_trials("111")

    assert len(trials) == 1
    trial = trials[0]
    assert trial.grating_shown is True
    assert trial.contrast == 0.05
    assert trial.source is None
    assert trial.outcome == "hit"


def test_load_metadata_defaults_when_no_meta_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "222", rows=[])

    meta = data_access.load_metadata("222")

    assert meta["name"] is None
    assert meta["participant_id"] is None


def test_save_participant_info_merges_without_clobbering_other_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(
        tmp_path,
        "333",
        rows=[],
        meta={"participant_id": "p1", "calibration": {"left_ratio": 0.4, "right_ratio": 0.6}},
    )

    data_access.save_participant_info("333", name="Jamie", gender="woman", age="29")

    meta = data_access.load_metadata("333")
    assert meta["name"] == "Jamie"
    assert meta["gender"] == "woman"
    assert meta["age"] == 29
    assert meta["participant_id"] == "p1"
    assert meta["calibration"] == {"left_ratio": 0.4, "right_ratio": 0.6}


def test_save_participant_info_blank_fields_clear_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "444", rows=[], meta={"name": "Old Name"})

    data_access.save_participant_info("444", name="", gender="", age="")

    meta = data_access.load_metadata("444")
    assert meta["name"] is None
    assert meta["gender"] is None
    assert meta["age"] is None


def test_save_participant_info_rejects_non_numeric_age(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "555", rows=[])

    with pytest.raises(ValueError):
        data_access.save_participant_info("555", name="X", gender="Y", age="not-a-number")


def test_discover_sessions_counts_trials_per_phase_and_sorts_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "100", rows=[_row("presaccade"), _row("presaccade"), _row("saccade")])
    _write_session(tmp_path, "200", rows=[])

    sessions = data_access.discover_sessions()

    assert [s.timestamp for s in sessions] == ["200", "100"]  # newest first
    by_timestamp = {s.timestamp: s for s in sessions}
    assert by_timestamp["100"].trial_counts == {"presaccade": 2, "saccade": 1}
    assert by_timestamp["200"].trial_counts == {}
