"""TrialResult <-> trials.csv (TRIAL_COLUMNS) <-> dashboard loader must stay in sync.
Bug class: adding a TrialResult field but forgetting the logger column or the loader (the data silently disappears, or old CSVs stop loading).
Every TrialResult field is either a TRIAL_COLUMNS column or listed in DROPPED_FROM_CSV with a reason."""

import dataclasses

import pytest

from include.experiment.types import Orientation, Target, TrialResult
from src.dashboard import data_access
from src.experiment import logger as logger_module
from src.experiment.logger import DROPPED_FROM_CSV, FIELDNAMES, TRIAL_COLUMNS, ResultLogger


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
        flash_relative_to_onset_ms=12.5,
        avg_rt_used_ms=230.0,
        scheduled_delay_ms=210.0,
        gaze_latency_ms=16.0,
    )


def test_every_trialresult_field_is_a_column_or_explicitly_dropped():
    fields = {f.name for f in dataclasses.fields(TrialResult)}
    in_csv = {c.field for c in TRIAL_COLUMNS if c.field is not None}
    assert in_csv | set(DROPPED_FROM_CSV) == fields, f"unaccounted: {fields - in_csv - set(DROPPED_FROM_CSV)}; unknown: {(in_csv | set(DROPPED_FROM_CSV)) - fields}"
    assert not in_csv & set(DROPPED_FROM_CSV)
    assert all(DROPPED_FROM_CSV.values()), "every dropped column needs a reason"
    assert len(set(FIELDNAMES)) == len(FIELDNAMES)


def test_full_result_round_trips_through_trials_csv_and_dashboard_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    original = _full_result()
    log = ResultLogger("p")
    log.log(original)
    log.close()
    (loaded,) = data_access.load_session_trials(str(log._timestamp))
    for f in dataclasses.fields(TrialResult):
        want = None if f.name in DROPPED_FROM_CSV else getattr(original, f.name)
        if isinstance(want, float):
            want = pytest.approx(want, abs=0.1) if f.name != "contrast" else pytest.approx(want)
        assert getattr(loaded, f.name) == want, f"field {f.name!r} lost in the CSV round trip"


_OPTIONAL = {f.name for f in dataclasses.fields(TrialResult) if f.default is None}


@pytest.mark.parametrize("column", [c for c in TRIAL_COLUMNS if c.field in _OPTIONAL], ids=lambda c: c.name)
def test_a_file_without_an_optional_column_still_loads(tmp_path, monkeypatch, column):
    """A trials.csv missing an optional column (older/other writer) loads with that field = None, never crashes."""
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    log = ResultLogger("p")
    log.log(_full_result())
    log.close()
    csv_path = next(tmp_path.glob("*/*/trials.csv"))
    header, row = [line.split(",") for line in csv_path.read_text().splitlines()[:2]]
    drop = header.index(column.name)
    del header[drop], row[drop]
    csv_path.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    (loaded,) = data_access.load_session_trials(str(log._timestamp))
    assert getattr(loaded, column.field) is None
