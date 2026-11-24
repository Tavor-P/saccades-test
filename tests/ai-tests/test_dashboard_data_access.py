import csv
import json

import pytest

from src.dashboard import data_access
from src.experiment.logger import read_trials

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


def test_trials_csv_round_trips_rows_and_reads_missing_newer_columns_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "111", rows=[_row()])  # _FIELDNAMES omits newer columns: a session logged before they existed
    (trial,) = read_trials(data_access.csv_path_for("111"))
    assert trial.flash_shown is True and trial.contrast == 0.05 and trial.source is None and trial.outcome == "hit"
    assert trial.flash_during_saccade is None and trial.timing_offset_ms is None

    fieldnames = _FIELDNAMES + ["flash_during_saccade", "timing_offset_ms"]
    with (tmp_path / "results_222.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_row(flash_during_saccade="True", timing_offset_ms="-40"))
        writer.writerow(_row(trial_index="1", flash_during_saccade="False", timing_offset_ms="40"))
        writer.writerow(_row(trial_index="2", flash_during_saccade="", timing_offset_ms=""))
    trials = read_trials(data_access.csv_path_for("222"))
    assert [t.flash_during_saccade for t in trials] == [True, False, None]
    assert [t.timing_offset_ms for t in trials] == [-40, 40, None]


def test_discover_sessions_counts_trials_per_phase_sorts_newest_first_and_survives_damage(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "100", rows=[_row("presaccade"), _row("rt_test"), _row("rt_test"), _row("saccade")])
    _write_session(tmp_path, "200", rows=[])
    sessions = data_access.discover_sessions()
    assert [s.timestamp for s in sessions] == ["200", "100"]  # newest first
    by_timestamp = {s.timestamp: s for s in sessions}
    assert by_timestamp["100"].trial_counts == {"presaccade": 1, "rt_test": 2, "saccade": 1}
    assert by_timestamp["200"].trial_counts == {}

    _write_session(tmp_path, "1700000200", rows=[_row(phase="presaccade", outcome="correct"), _row(phase="saccade", trial_index="oops")])
    (tmp_path / "results_1700000200_meta.json").write_text('{"participant_id": "p1", "start')  # cut off mid-write
    damaged = {s.timestamp: s for s in data_access.discover_sessions()}["1700000200"]
    assert damaged.metadata["participant_id"] is None
    assert damaged.trial_counts == {}


def test_metadata_defaults_and_participant_info_merges_clears_and_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "222", rows=[])
    meta = data_access.load_metadata("222")
    assert "name" not in meta and meta["participant_id"] is None  # no meta file
    with pytest.raises(ValueError):
        data_access.save_participant_info("222", gender="Y", age="not-a-number")

    _write_session(tmp_path, "333", rows=[], meta={"participant_id": "p1", "calibration": {"left_ratio": 0.4, "right_ratio": 0.6}})
    data_access.save_participant_info("333", gender="woman", age="29")
    meta = data_access.load_metadata("333")
    assert (meta["gender"], meta["age"], meta["participant_id"]) == ("woman", 29, "p1") and "name" not in meta
    assert meta["calibration"] == {"left_ratio": 0.4, "right_ratio": 0.6}  # other fields not clobbered

    data_access.save_participant_info("333", gender="", age="")
    meta = data_access.load_metadata("333")
    assert meta["gender"] is None and meta["age"] is None


def test_delete_session_removes_csv_and_meta_tolerates_missing_meta_and_drops_it_from_the_list(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    _write_session(tmp_path, "666", rows=[], meta={"name": "Old Name"})
    _write_session(tmp_path, "777", rows=[])  # no meta.json
    _write_session(tmp_path, "999", rows=[])
    data_access.delete_session("666")
    assert not data_access.csv_path_for("666").exists() and not data_access.meta_path_for("666").exists()
    data_access.delete_session("777")  # should not raise
    assert not data_access.csv_path_for("777").exists()
    assert [s.timestamp for s in data_access.discover_sessions()] == ["999"]


def test_a_path_is_never_a_session_key(tmp_path, monkeypatch):
    data = tmp_path / "data"
    outside = tmp_path / "outside"
    for folder in (data / "alice" / "2026-10-25_10-00-00", outside):
        folder.mkdir(parents=True)
        (folder / "trials.csv").write_text("phase,trial\n")
    monkeypatch.setattr(data_access, "DATA_DIR", data)
    for key in ["..", ".", "../outside", r"..\outside", "alice/..", str(outside), "alice"]:
        assert not data_access.csv_path_for(key).exists(), key
        data_access.delete_session(key)
        assert (outside / "trials.csv").exists() and (data / "alice" / "2026-10-25_10-00-00" / "trials.csv").exists(), key
