"""Group labels (src/experiment/participants.py): round trip, atomic write, damaged-file tolerance, set selection."""

import json

from src.experiment import participants as labels
from src.experiment.participants import list_labels, participants_in_group, resolve_set, set_label


def test_label_round_trip_sanitizes_ids_and_matches_groups_case_insensitively(tmp_path):
    assert list_labels(tmp_path) == {}  # no file yet
    set_label(" P01 ", "TBI", "first pilot", data_dir=tmp_path)
    set_label("P02", "control", data_dir=tmp_path)
    set_label("P03", "tbi", data_dir=tmp_path)
    set_label("P01", "TBI", data_dir=tmp_path)  # note=None keeps the note
    assert list_labels(tmp_path)["P01"] == {"group": "TBI", "note": "first pilot"}
    assert participants_in_group("TBI", tmp_path) == ["P01", "P03"]
    assert resolve_set("group:control, P09, P02", tmp_path) == ["P02", "P09"]
    on_disk = json.loads(labels.labels_path(tmp_path).read_text())
    assert set(on_disk) == {"P01", "P02", "P03"} and not list(tmp_path.glob(".participants-*"))  # no temp file left
    set_label("P02", None, None, data_dir=tmp_path)
    assert "P02" not in list_labels(tmp_path)


def test_damaged_label_file_reads_as_empty_and_is_kept_aside_on_the_next_write(tmp_path, capsys):
    path = labels.labels_path(tmp_path)
    path.write_text('{"P01": {"group": "TBI"', encoding="utf-8")
    assert list_labels(tmp_path) == {}
    assert "damaged" in capsys.readouterr().err
    set_label("P02", "control", data_dir=tmp_path)
    assert list_labels(tmp_path) == {"P02": {"group": "control", "note": ""}}
    (backup,) = tmp_path.glob("participants.json.damaged-*")
    assert backup.read_text(encoding="utf-8") == '{"P01": {"group": "TBI"'
    path.write_text('{"P01": "not an entry", "P02": {"group": "control"}}', encoding="utf-8")
    assert list_labels(tmp_path) == {"P02": {"group": "control", "note": ""}}  # bad entries skipped, good ones kept


def test_cli_set_and_list(tmp_path, capsys):
    assert labels.main(["--data-dir", str(tmp_path), "set", "P01", "TBI", "--note", "x"]) == 0
    assert labels.main(["--data-dir", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "P01" in out and "TBI" in out and "(no run folders)" in out
