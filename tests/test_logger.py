import json

from include.experiment.types import TrialResult
from src.experiment import logger as logger_module
from src.experiment.logger import FIELDNAMES, ResultLogger


def _make_logger(tmp_path, monkeypatch, participant_id="p1"):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    return ResultLogger(participant_id)


def test_writes_csv_header(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch)
    log.close()
    csv_files = list(tmp_path.glob("results_*.csv"))
    assert len(csv_files) == 1
    header = csv_files[0].read_text().splitlines()[0]
    assert header.split(",") == FIELDNAMES


def test_writes_metadata_sidecar(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch, participant_id="alice")
    log.close()
    meta_files = list(tmp_path.glob("results_*_meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text())
    assert meta["participant_id"] == "alice"
    assert meta["calibration"] is None
    assert "viewing_distance_cm" in meta


def test_metadata_records_which_camera_was_used(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger("p1", camera_label="Camera 2")
    log.close()
    meta_files = list(tmp_path.glob("results_*_meta.json"))
    meta = json.loads(meta_files[0].read_text())
    assert meta["camera"] == "Camera 2"


def test_metadata_camera_defaults_to_none(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch)
    log.close()
    meta_files = list(tmp_path.glob("results_*_meta.json"))
    meta = json.loads(meta_files[0].read_text())
    assert meta["camera"] is None


def test_set_calibration_updates_metadata(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch)
    log.set_calibration(0.3, 0.5, 0.7)
    log.close()
    meta_files = list(tmp_path.glob("results_*_meta.json"))
    meta = json.loads(meta_files[0].read_text())
    assert meta["calibration"] == {"left_ratio": 0.3, "center_ratio": 0.5, "right_ratio": 0.7}


def test_log_writes_a_row(tmp_path, monkeypatch):
    log = _make_logger(tmp_path, monkeypatch)
    log.log(
        TrialResult(
            index=0,
            phase="presaccade",
            source=None,
            target=None,
            saccade_duration_ms=None,
            grating_shown=True,
            contrast=0.05,
            responded=True,
            response_time_ms=250.0,
            outcome="hit",
        )
    )
    log.close()
    csv_files = list(tmp_path.glob("results_*.csv"))
    lines = csv_files[0].read_text().splitlines()
    assert len(lines) == 2  # header + one row
    assert "hit" in lines[1]
