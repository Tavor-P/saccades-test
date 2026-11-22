"""Comparison engine (src/experiment/compare.py) and study summary, checked against planted truth.

Statistics tests use in-memory people (`compare_people`) whose flash counts are drawn from a known 2AFC Weibull, so they
take milliseconds; two end-to-end tests run the real pipeline on synthetic run folders in tmp_path (never data/)."""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_runs import write_run  # noqa: E402

from src.experiment import compare as cmp  # noqa: E402
from src.experiment import study_summary  # noqa: E402
from src.experiment.analyze_session import _weibull  # noqa: E402
from src.experiment.flash_gates import FIXATION, SACCADE  # noqa: E402
from src.experiment.participants import participants_in_group, set_label  # noqa: E402

CONTRASTS = (4.0, 6.0, 8.0)
GEOMETRY = {"screen_height_cm": 26.0, "viewing_distance_cm": 50.0, "line_offset_deg": 2.0, "line_sd_deg": 0.15}


def person(pid, side, ts=0.08, tf=0.04, n_cell=40, seed=0, index=None, method="psychometric", geometry=None, contrasts=CONTRASTS):
    """A Person with by_contrast counts drawn from the Weibull (index = the refit), or with a given index only."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in contrasts:
        for cond, t in ((SACCADE, ts), (FIXATION, tf)):
            p = float(_weibull(np.asarray(c / 100), np.asarray(t), np.asarray(3.0)))
            rows.append({"contrast_pct": c, "condition": cond, "n": n_cell, "correct": int(rng.binomial(n_cell, p))})
    n = n_cell * len(contrasts)
    k_s = sum(r["correct"] for r in rows if r["condition"] == SACCADE)
    k_f = sum(r["correct"] for r in rows if r["condition"] == FIXATION)
    if index is None:
        index = index_from_counts(rows) if method == "psychometric" else k_f / n - k_s / n
    return cmp.Person(pid, side=side, index=index, ci=None if index is None else [index - 0.2, index + 0.2], method=method, n_saccade=n, n_fixation=n, n_total=2 * n,
                      correct_saccade=k_s, correct_fixation=k_f, contrasts_pct=list(contrasts), geometry=geometry or dict(GEOMETRY), by_contrast=rows)


def index_from_counts(by_contrast: list) -> float | None:
    """The psychometric index refitted from by_contrast counts with compare's own cells and fit; None if unidentified."""
    cs, ns, ks = cmp._cells(by_contrast, SACCADE)
    cf, nf, kf = cmp._cells(by_contrast, FIXATION)
    if len(cs) < 2 or len(cf) < 2:
        return None
    ts, tf, edge = cmp._fit_pair(cs, ns, ks, cf, nf, kf)
    return None if edge else 1 - tf / ts


def true_index(ts, tf):
    return 1 - tf / ts


# -- 1 vs 1 --------------------------------------------------------------------------------------


def test_one_vs_one_detects_a_planted_difference():
    """True indices 0.67 (A) vs 0.07 (B), 240 flashes per condition each: the CI of A - B excludes 0 in most seeds."""
    detected = 0
    for seed in range(5):
        a, b = person("A1", "A", ts=0.12, tf=0.04, n_cell=80, seed=seed), person("B1", "B", ts=0.043, tf=0.04, n_cell=80, seed=100 + seed)
        r = cmp.compare_people([a, b], n_boot=250, seed=seed)
        assert r.mode == cmp.ONE_VS_ONE and r.method == "psychometric"
        assert r.stats["difference"] == pytest.approx(a.index - b.index)
        detected += bool(r.stats["ci_excludes_zero"])
    assert detected >= 4
    assert "A1 shows more suppression" in r.headline and "exploratory" in r.headline


def test_one_vs_one_difference_ci_covers_the_truth_in_most_seeds():
    """Planted difference 0.27; among seeds where both bootstraps are identified the 95 % CI should contain it ~95 % of the time."""
    truth = true_index(0.10, 0.04) - true_index(0.06, 0.04)
    cis = []
    for seed in range(8):
        r = cmp.compare_people([person("A1", "A", 0.10, 0.04, n_cell=80, seed=10 + seed), person("B1", "B", 0.06, 0.04, n_cell=80, seed=50 + seed)], n_boot=250, seed=seed)
        if r.stats["difference_ci"] is not None:
            cis.append(r.stats["difference_ci"])
    assert len(cis) >= 5
    assert sum(lo <= truth <= hi for lo, hi in cis) >= len(cis) - 1


def test_one_vs_one_accuracy_fallback_is_compared_and_labelled():
    a = person("A1", "A", ts=0.12, tf=0.04, seed=3, method="accuracy")
    b = person("B1", "B", ts=0.05, tf=0.04, seed=4, method="accuracy")
    r = cmp.compare_people([a, b], n_boot=400)
    assert r.comparable and r.method == "accuracy"
    assert "accuracy-based" in r.headline and "fallback" in r.headline
    assert r.stats["difference_ci"][0] < r.stats["difference"] < r.stats["difference_ci"][1]


def test_one_vs_one_with_different_methods_is_not_comparable():
    r = cmp.compare_people([person("A1", "A", seed=5), person("B1", "B", seed=6, method="accuracy")], n_boot=50)
    assert not r.comparable and r.mode is None and not r.stats
    assert "not comparable" in r.headline and "different scales" in r.headline


def test_one_vs_one_with_different_geometry_is_not_comparable():
    other = {**GEOMETRY, "viewing_distance_cm": 70.0}
    r = cmp.compare_people([person("A1", "A", seed=5), person("B1", "B", seed=6, geometry=other)], n_boot=50)
    assert not r.comparable and all(not p.included for p in r.people)
    assert "different settings" in r.headline and "70 cm" in r.headline


# -- group vs group ------------------------------------------------------------------------------


def _group(prefix, side, values, **kw):
    return [person(f"{prefix}{i}", side, index=v, seed=i, **kw) for i, v in enumerate(values)]


def test_separated_groups_give_a_small_p_and_identical_groups_do_not():
    sep = cmp.compare_people(_group("T", "A", [0.6, 0.65, 0.7, 0.55, 0.62, 0.68]) + _group("C", "B", [0.1, 0.2, 0.15, 0.05, 0.22, 0.12]), n_boot=500)
    assert sep.mode == cmp.GROUP_VS_GROUP and sep.stats["p_value"] < 0.01
    assert sep.stats["rank_biserial"] == 1.0 and sep.stats["prob_a_higher"] == 1.0
    assert sep.stats["median_difference_ci"][0] > 0
    same_vals = [0.3, 0.5, 0.4, 0.6, 0.35]
    same = cmp.compare_people(_group("T", "A", same_vals) + _group("C", "B", [v + 0.01 for v in same_vals[::-1]]), n_boot=500)
    assert same.stats["p_value"] > 0.3
    assert same.stats["median_difference_ci"][0] < 0 < same.stats["median_difference_ci"][1]
    assert "Mann-Whitney p" in same.headline and "exploratory, small groups" in same.headline


def test_mann_whitney_exact_matches_scipy_without_ties_and_handles_ties():
    a, b = [0.61, 0.42, 0.55, 0.7], [0.31, 0.45, 0.2, 0.12, 0.5]
    ours = cmp.mann_whitney(a, b)
    ref = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    assert ours["p"] == pytest.approx(ref.pvalue) and ours["u_a"] == ref.statistic
    tied = cmp.mann_whitney([0.5, 0.5, 0.6], [0.5, 0.4, 0.4])
    assert 0 < tied["p"] <= 1 and "ties handled" in tied["test"]
    assert cmp.rank_biserial(cmp.mann_whitney([1, 2], [3, 4])["u_a"], 2, 2) == -1.0


def test_group_mode_excludes_minority_method_and_other_contrasts_and_warns():
    people = _group("T", "A", [0.6, 0.7, 0.65]) + _group("C", "B", [0.2, 0.3])
    people.append(person("CX", "B", index=0.25, method="accuracy"))
    people.append(person("CY", "B", index=0.25, contrasts=(2.0, 4.0, 6.0)))
    people[1].n_saccade = 8
    r = cmp.compare_people(people, n_boot=200)
    reasons = {p.participant: p.reason for p in r.excluded}
    assert set(reasons) == {"CX", "CY"}
    assert "accuracy method" in reasons["CX"] and "2/4/6 %" in reasons["CY"]
    assert r.stats["n_b"] == 2 and r.exploratory
    assert any("only 2 people" in w for w in r.warnings) and any(w.startswith("T1: only 8 saccade") for w in r.warnings)


def test_group_side_shrunk_to_one_person_becomes_one_vs_group():
    people = _group("T", "A", [0.6, 0.7]) + _group("C", "B", [0.2, 0.3, 0.1])
    people[1].method, people[1].index = None, None
    people[1].included, people[1].reason = False, "suppression index cannot be computed: no flashes"
    r = cmp.compare_people(people, n_boot=100)
    assert r.requested_mode == cmp.GROUP_VS_GROUP and r.mode == cmp.ONE_VS_GROUP
    assert any("after exclusions" in d for d in r.details)


# -- 1 vs group ----------------------------------------------------------------------------------


def test_one_vs_group_reports_rank_and_percentile_without_p_value():
    people = [person("P7", "A", index=0.2)] + _group("C", "B", [0.1, 0.3, 0.4, 0.5, 0.6])
    r = cmp.compare_people(people)
    assert r.mode == cmp.ONE_VS_GROUP and r.comparable
    s = r.stats
    assert (s["members_above"], s["members_below"], s["percentile"]) == (4, 1, 20.0)
    assert s["group_median"] == 0.4 and s["group_range"] == [0.1, 0.6]
    assert "p_value" not in s and "Mann-Whitney" not in r.headline
    assert "below 4 of 5" in r.headline and "20th percentile" in r.headline
    assert any(d.startswith("No p-value") for d in r.details)
    # the other way round, and ties count half
    r2 = cmp.compare_people(_group("C", "A", [0.1, 0.2, 0.3]) + [person("P7", "B", index=0.2)])
    assert r2.stats["single"] == "B" and r2.stats["percentile"] == 50.0 and r2.stats["members_tied"] == 1


def test_overlapping_or_empty_sets_are_refused(tmp_path):
    with pytest.raises(ValueError, match="both sets"):
        cmp.compare(["P01", "P02"], ["P02"], tmp_path)
    with pytest.raises(ValueError):
        cmp.compare([], ["P01"], tmp_path)


# -- end to end on synthetic run folders ---------------------------------------------------------


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    d = tmp_path_factory.mktemp("study")
    for i, (pid, ts) in enumerate((("S01", 0.11), ("S02", 0.045))):
        for r in range(2):
            write_run(d, pid, 1790000000 + 86400 * r + 3600 * i, n_trials=90, threshold_saccade=ts, threshold_fixation=0.04, seed=10 * i + r)
    write_run(d, "S03", 1790500000, n_trials=90, geometry={"viewing_distance_cm": 70.0}, seed=7)
    set_label("S01", "TBI", data_dir=d)
    set_label("S02", "control", note="pilot", data_dir=d)
    set_label("S09", "control", data_dir=d)  # labelled, but no data
    return d


def test_end_to_end_one_vs_one_analyses_once_reuses_then_refreshes_when_a_run_changes(study, tmp_path):
    r = cmp.compare(["S01"], ["S02"], study, n_boot=200)
    assert all(p.analysis_source.startswith("re-analysed (no saved analysis") for p in r.people)
    saved = json.loads((study / "S01" / "analysis" / "summary.json").read_text())
    assert r.people[0].index == saved["index"]
    assert index_from_counts(saved["by_contrast"]) == pytest.approx(saved["index"], abs=1e-9)  # the refit is the same fit
    assert r.comparable and r.stats["difference"] > 0 and r.name_a == "A (TBI)"

    again = cmp.compare(["S01"], ["S02"], study, n_boot=200)
    assert [p.analysis_source for p in again.people] == ["saved", "saved"]
    run = sorted((study / "S02").glob("2026-*"))[0] / "trials.csv"
    later = (study / "S02" / "analysis" / "summary.json").stat().st_mtime + 5
    os.utime(run, (later, later))
    third = cmp.compare(["S01"], ["S02"], study, n_boot=200)
    assert third.people[0].analysis_source == "saved" and "changed since" in third.people[1].analysis_source

    paths = cmp.write_comparison(third, tmp_path / "out")
    assert {p.name for p in paths.values()} == {"comparison.json", "comparison.csv", "report.txt", "plot.png"}
    data = json.loads(paths["json"].read_text())
    assert data["mode"] == cmp.ONE_VS_ONE and data["people"][0]["participant"] == "S01" and "flashes" not in data["people"][0]
    with paths["csv"].open() as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == cmp.CSV_COLUMNS and [r["included"] for r in rows] == ["yes", "yes"]
    assert "EXPLORATORY" in paths["report"].read_text()


def test_end_to_end_incompatible_geometry_and_missing_data_are_listed(study):
    r = cmp.compare(participants_in_group("control", study) + ["S03"], ["S01"], study, n_boot=100)
    reasons = {p.participant: p.reason for p in r.excluded}
    assert "No runs" in reasons["S09"]
    assert "70 cm" in reasons["S03"]
    assert r.mode == cmp.ONE_VS_ONE and r.requested_mode == cmp.ONE_VS_GROUP


def test_study_summary_csv_has_one_row_per_participant_with_the_documented_columns(study, tmp_path):
    out = study_summary.write_study_summary(study, out_dir=tmp_path)
    with out["csv"].open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert tuple(reader.fieldnames) == study_summary.COLUMNS
    by_id = {r["participant"]: r for r in rows}
    assert set(by_id) == {"S01", "S02", "S03", "S09"}
    assert by_id["S02"]["group"] == "control" and by_id["S02"]["note"] == "pilot" and by_id["S02"]["status"] == "ok"
    assert by_id["S01"]["method"] == "psychometric" and float(by_id["S01"]["ci_low"]) < float(by_id["S01"]["index"])
    assert by_id["S01"]["runs_used"] == "2" and int(by_id["S01"]["n_saccade"]) > 0
    assert by_id["S09"]["status"].startswith("No runs") and by_id["S09"]["index"] == ""
    assert "Group control" in out["overview"] and "EXPLORATORY" in out["overview"]
