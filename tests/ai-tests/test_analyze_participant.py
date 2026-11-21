"""The per-participant pooled analysis (src/experiment/analyze_participant.py) on synthetic anonymous run folders
(tests/ai-tests/synthetic_runs.py) built in pytest tmp folders - never under data/."""

import csv
import json

import pytest

import synthetic_runs as sr
from src.experiment import analyze_participant as ap
from src.experiment import analyze_session as an

DAY = 86_400
START = 1_790_000_000
PID = "SYN-01"


@pytest.fixture(scope="module")
def pooled(tmp_path_factory):
    """Three compatible real runs, one with other contrasts, one with another screen geometry, one test run."""
    data = tmp_path_factory.mktemp("participants")
    for i in range(3):
        sr.write_run(data, PID, START + i * DAY, n_trials=50, seed=i + 1)
    sr.write_run(data, PID, START + 3 * DAY, n_trials=30, seed=7, contrasts_pct=(2.0, 4.0, 6.0))
    sr.write_run(data, PID, START + 4 * DAY, n_trials=30, seed=8, geometry={"viewing_distance_cm": 60.0})
    sr.write_run(data, PID, START + 5 * DAY, n_trials=20, seed=9, test_mode=True)
    sr.write_run(data, "SYN-02", START, n_trials=20, seed=10)  # someone else: never pooled in
    return data, ap.analyze_participant(PID, data, n_boot=300)


def test_pooling_compatible_runs_sums_the_per_run_flashes(pooled):
    data, result = pooled
    s = result.summary
    assert [r["run"] for r in s["runs_used"]] == sorted(r["run"] for r in s["runs_used"]) and len(s["runs_used"]) == 3
    for cond in ("saccade", "fixation", "excluded"):
        assert s["n_flashes"][cond] == sum(r[f"n_{cond}"] for r in s["runs_used"])
    assert s["n_flashes"]["total"] == len(result.placed) == 150
    single = max(r["n_saccade"] for r in s["runs_used"])
    assert s["n_flashes"]["saccade"] > single
    # The pooled index is the same function analyze_session uses, on the concatenated per-run gates.
    direct = an.suppression_index(result.placed, gates=result.gates, n_boot=300)
    assert (s["index"], s["method"]) == (direct["index"], direct["method"])
    assert s["method"] == "psychometric" and 0.2 < s["index"] < 0.8  # planted: 1 - 0.04/0.08 = 0.5
    assert s["slope"] in an._SLOPES


def test_runs_with_other_contrasts_or_geometry_are_excluded_with_the_reason(pooled):
    _, result = pooled
    reasons = {r["run"]: r["reason"] for r in result.summary["runs_excluded"]}
    assert len(reasons) == 2
    assert any("contrasts 2/4/6 % differ" in why for why in reasons.values())
    assert any("screen geometry" in why and "60 cm" in why for why in reasons.values())
    report = result.report
    assert report.index("NOT POOLED: 2 runs left out") < report.index("Runs pooled:")  # visible right under the headline


def test_test_runs_are_skipped_unless_asked_for(tmp_path, pooled):
    _, result = pooled
    assert all(r["mode"] == "real" for r in result.summary["runs_used"] + result.summary["runs_excluded"])
    assert result.summary["analysis"]["test_runs_skipped"] == 1
    assert "1 test run skipped" in result.report

    for i in range(2):
        sr.write_run(tmp_path, "T", START + i * DAY, n_trials=20, seed=i, test_mode=True)
    with pytest.raises(ap.NoRunsError, match=r"analyze_participant.bat T --include-test"):
        ap.analyze_participant("T", tmp_path, write=False)
    with_tests = ap.analyze_participant("T", tmp_path, include_test=True, write=False, n_boot=50)
    assert [r["mode"] for r in with_tests.summary["runs_used"]] == ["test", "test"]


def test_a_participant_without_runs_gets_a_clear_error(tmp_path, capsys):
    sr.write_run(tmp_path, "SOMEONE", START, n_trials=10)
    with pytest.raises(ap.NoRunsError, match="No runs for participant 'NOBODY'.*SOMEONE"):
        ap.analyze_participant("NOBODY", tmp_path)
    assert ap.main(["NOBODY", "--data-dir", str(tmp_path)]) == 1
    assert capsys.readouterr().out.startswith("Error: No runs for participant 'NOBODY'")
    with pytest.raises(ap.NoRunsError):
        ap.analyze_participant("X", tmp_path / "missing")
    assert not (tmp_path / "NOBODY").exists()  # nothing written for a missing participant


def test_outputs_are_written_in_the_participant_analysis_folder(pooled):
    data, result = pooled
    out = data / PID / "analysis"
    assert result.paths["summary"] == out / "summary.json"
    report = (out / "report.txt").read_text(encoding="utf-8")
    assert report.splitlines()[2].startswith("Suppression index 0.") and "from 3 runs" in report.splitlines()[2]
    assert (out / "plot.png").stat().st_size > 10_000
    saved = json.loads((out / "summary.json").read_text())
    rows = list(csv.DictReader((out / saved["flashes_csv"]).open()))  # flashes_csv is relative to summary.json
    assert len(rows) == saved["n_flashes"]["total"]
    assert {r["run"] for r in rows} == {r["run"] for r in saved["runs_used"]}
    assert sum(r["condition"] == "saccade" for r in rows) == saved["n_flashes"]["saccade"]
    # The new analysis folder is not mistaken for a run on the next call.
    again = ap.analyze_participant(PID, data, write=False, n_boot=50)
    assert len(again.summary["runs_used"]) == 3


def test_summary_json_schema_is_stable(pooled):
    """summary.json is the input of the comparison module: changing these keys needs a schema_version bump."""
    data, _ = pooled
    s = json.loads((data / PID / "analysis" / "summary.json").read_text())
    assert list(s) == [
        "schema_version", "participant", "generated_at", "include_test",
        "index", "ci", "method", "label", "threshold_saccade", "threshold_fixation", "slope", "notes",
        "n_flashes", "correct", "by_contrast", "exclusion_reasons",
        "contrasts_pct", "geometry", "runs_used", "runs_excluded", "flashes_csv", "analysis",
    ]
    assert s["schema_version"] == 1 and s["participant"] == PID
    assert set(s["n_flashes"]) == {"saccade", "fixation", "excluded", "total"}
    assert set(s["correct"]) == {"saccade", "fixation", "excluded"}
    assert len(s["ci"]) == 2 and s["ci"][0] < s["index"] < s["ci"][1]
    assert s["contrasts_pct"] == [4.0, 6.0, 8.0]
    assert {(r["contrast_pct"], r["condition"]) for r in s["by_contrast"]} == {(c, k) for c in (4.0, 6.0, 8.0) for k in ("saccade", "fixation", "excluded")}
    assert all(set(r) == {"contrast_pct", "condition", "n", "correct"} for r in s["by_contrast"])
    run_keys = {"run", "run_key", "date", "mode", "n_saccade", "n_fixation", "n_excluded", "index", "ci", "method", "notes",
                "contrasts_pct", "contrasts_from", "geometry", "gaze_latency_ms_median", "longest_gap_ms", "gaps_over_threshold"}
    assert all(set(r) == run_keys for r in s["runs_used"])
    assert all(set(r) == run_keys | {"reason"} for r in s["runs_excluded"])
    assert s["runs_used"][0]["gaze_latency_ms_median"] == 17.5


def test_a_damaged_run_is_listed_not_fatal_and_old_runs_use_the_shown_contrasts(tmp_path):
    sr.write_run(tmp_path, PID, START, n_trials=30, seed=1)
    sr.write_run(tmp_path, PID, START + DAY, n_trials=30, seed=2, record_contrasts=False)  # older run: no saccade_contrasts_pct
    broken = sr.write_run(tmp_path, PID, START + 2 * DAY, n_trials=30, seed=3)
    (broken / "gaze_trace.csv").unlink()
    no_offset = sr.write_run(tmp_path, PID, START + 3 * DAY, n_trials=30, seed=4)
    meta = json.loads((no_offset / "session.json").read_text())
    meta["gaze_quality"] = None
    (no_offset / "session.json").write_text(json.dumps(meta))

    s = ap.analyze_participant(PID, tmp_path, write=False, n_boot=50).summary
    assert len(s["runs_used"]) == 2 and {r["contrasts_from"] for r in s["runs_used"]} == {"settings", "trials"}
    reasons = sorted(r["reason"] for r in s["runs_excluded"])
    assert reasons[0].startswith("could not be analysed (ValueError: this session has no clock_offset_ms")
    assert reasons[1] == "no gaze_trace.csv in the run folder"


def test_the_participant_id_is_matched_as_its_folder_name(tmp_path):
    sr.write_run(tmp_path, "pilot 01", START, n_trials=30, seed=1)  # folder: pilot_01
    result = ap.analyze_participant("pilot 01", tmp_path, write=False, n_boot=50)
    assert result.participant == "pilot_01" and len(result.summary["runs_used"]) == 1


def test_headline_covers_every_outcome():
    base = {"runs_used": [{}], "n_flashes": {"saccade": 5, "fixation": 30}, "ci": None, "notes": []}
    assert ap.headline({**base, "method": "psychometric", "index": 0.5, "ci": [-0.1, 0.8]}) == (
        "Suppression index 0.50 (95 % CI -0.10 to 0.80) from 1 run, 5 saccade and 30 fixation flashes.")
    acc = ap.headline({**base, "method": "accuracy", "index": 0.3, "notes": ["psychometric index not computed: fewer than 12 flashes per condition (5 saccade, 30 fixation)"]})
    assert acc.startswith("Accuracy-based suppression index 0.30 (95 % CI not available)") and "not a threshold ratio" in acc and "fewer than 12" in acc
    assert "0.5 = guessing" in ap.meaning({"method": "accuracy"}) and "1 = the line is never seen" in ap.meaning({"method": "psychometric"})
    assert ap.meaning({"method": None}) is None
    none = ap.headline({**base, "method": None, "index": None, "notes": ["cannot compute: 0 saccade flashes, 30 fixation flashes"]})
    assert none.startswith("Suppression index cannot be computed yet from 1 run") and none.endswith("0 saccade flashes, 30 fixation flashes.")
