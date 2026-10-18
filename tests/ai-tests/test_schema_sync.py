"""TrialResult <-> CSV <-> dashboard loader must stay in sync.
Bug class: adding a TrialResult field but forgetting the logger column or the dashboard parser
(the data silently disappears, or old CSVs stop loading)."""

import dataclasses

import pytest

from include.experiment.types import Orientation, Target, TrialResult
from src.dashboard import data_access
from src.experiment import logger as logger_module
from src.experiment.logger import FIELDNAMES, ResultLogger

# TrialResult field name -> CSV column name, where they differ.
RENAMED = {"index": "trial_index"}


def _full_result() -> TrialResult:
    """Every field populated with a non-default value, so a dropped field shows up as a diff."""
    return TrialResult(
        index=7,
        phase="saccade",
        source=Target.DOT,
        target=Target.CROSS,
        saccade_duration_ms=61.25,
        grating_shown=True,
        contrast=0.1234,
        orientation=Orientation.HORIZONTAL,
        responded=True,
        response_orientation=Orientation.VERTICAL,
        response_time_ms=432.5,
        outcome="incorrect",
        reaction_latency_ms=251.5,
        onset_detection_lag_ms=16.5,
        flash_during_saccade=False,
        timing_offset_ms=-20,
        grating_shown_at_unix_ms=1_700_000_000_123.5,
        tracking_ok=False,
    )


def test_every_trialresult_field_has_a_csv_column_and_vice_versa():
    fields = {RENAMED.get(f.name, f.name) for f in dataclasses.fields(TrialResult)}
    assert fields == set(FIELDNAMES), f"only in dataclass: {fields - set(FIELDNAMES)}; only in CSV: {set(FIELDNAMES) - fields}"


def test_full_result_round_trips_through_csv_and_dashboard_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    original = _full_result()
    log = ResultLogger("p")
    log.log(original)
    log.close()
    timestamp = next(tmp_path.glob("results_*.csv")).stem.removeprefix("results_")
    (loaded,) = data_access.load_session_trials(timestamp)
    for f in dataclasses.fields(TrialResult):
        assert getattr(loaded, f.name) == pytest.approx(getattr(original, f.name)) if isinstance(
            getattr(original, f.name), float
        ) else getattr(loaded, f.name) == getattr(original, f.name), f"field {f.name!r} lost in the CSV round trip"


@pytest.mark.parametrize("column", [f.name for f in dataclasses.fields(TrialResult) if f.default is None])
def test_csvs_written_before_an_optional_column_existed_still_load(tmp_path, monkeypatch, column):
    """Old sessions (no such column) must load with that field = None, never crash."""
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    log = ResultLogger("p")
    log.log(_full_result())
    log.close()
    csv_path = next(tmp_path.glob("results_*.csv"))
    header, row = [line.split(",") for line in csv_path.read_text().splitlines()[:2]]
    drop = header.index(RENAMED.get(column, column))
    del header[drop], row[drop]
    csv_path.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    (loaded,) = data_access.load_session_trials(csv_path.stem.removeprefix("results_"))
    assert getattr(loaded, column) is None
