"""Dashboard one-click analysis (src/dashboard/results.py + the /runs/... and /participants/... routes) through Flask's
test client, on synthetic anonymous run folders (tests/ai-tests/synthetic_runs.py) in pytest tmp folders - never data/."""

import json
import os
import shutil

import pytest
import synthetic_runs as sr

from src.dashboard import app as app_module
from src.dashboard import data_access, results
from src.experiment import analyze_participant as ap
from src.experiment import analyze_session as an
from src.experiment import settings as settings_module

DAY = 86_400
START = 1_790_000_000
PID = "SYN-01"
TEST_ONLY = START + 2 * DAY  # every run has its own start time: the dashboard addresses runs by it
NO_TRACE = START + 3 * DAY


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """Built once: two real runs of SYN-01, a test-only participant, and a run without a gaze trace."""
    data = tmp_path_factory.mktemp("dash_template")
    sr.write_run(data, PID, START, n_trials=40, seed=1)
    sr.write_run(data, PID, START + DAY, n_trials=40, seed=2)
    sr.write_run(data, "SYN-TEST", TEST_ONLY, n_trials=20, seed=3, test_mode=True)
    no_trace = sr.write_run(data, "SYN-NOTRACE", NO_TRACE, n_trials=20, seed=4)
    (no_trace / "gaze_trace.csv").unlink()
    return data


@pytest.fixture
def data(template, tmp_path, monkeypatch):
    target = tmp_path / "data"
    shutil.copytree(template, target)
    monkeypatch.setattr(data_access, "DATA_DIR", target)
    monkeypatch.setattr(settings_module, "DATA_DIR", target)
    monkeypatch.setattr(results, "N_BOOT", 150)
    return target


@pytest.fixture
def client(data):
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _run_folder(data, pid=PID, i=0):
    return sorted(p for p in (data / pid).iterdir() if p.is_dir() and p.name != "analysis")[i]


def _html(response):
    return response.get_data(as_text=True)


def test_index_has_an_analyse_link_per_run_and_a_participant_row(client):
    page = _html(client.get("/"))
    assert f"/runs/{START}/results" in page
    assert f'data-participant="{PID}"' in page and f"/participants/{PID}/results" in page
    assert 'name="include_test"' in page
    assert "only test runs" in page  # SYN-TEST


def test_run_results_analyses_on_first_open_and_shows_headline_quality_and_plot(client, data):
    response = client.get(f"/runs/{START + DAY}/results")
    page = _html(response)
    assert response.status_code == 200
    folder = _run_folder(data, i=1)
    suppression = json.loads((folder / "analysis" / "suppression.json").read_text())
    expected = ap.headline(results._run_headline_summary(suppression))
    assert expected in page and "from 1 run" in expected
    if suppression["method"]:
        assert ap.meaning(results._run_headline_summary(suppression)) in page
    assert "Data quality" in page and "Gaze delay" in page and "Tracking gaps" in page and "Flashes per condition" in page
    assert "18 ms" in page  # synthetic latency_ms_median 17.5
    assert "Head movement" in page and "not recorded" in page  # synthetic runs have no head_motion section
    assert f"at least {an.MIN_FIT_TRIALS} saccade" in page  # the rule is spelled out
    assert f'src="/runs/{START + DAY}/files/plot"' in page
    assert "Analysed at" in page and "Re-analyse" in page


def test_fresh_run_analysis_is_not_rerun_but_stale_one_is(client, data, monkeypatch):
    assert client.get(f"/runs/{START}/results").status_code == 200
    calls = []
    monkeypatch.setattr(an, "analyze", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("re-ran")))
    assert client.get(f"/runs/{START}/results").status_code == 200 and calls == []
    trials = _run_folder(data) / "trials.csv"
    later = os.stat(_run_folder(data) / "analysis" / "suppression.json").st_mtime + 10
    os.utime(trials, (later, later))
    client.get(f"/runs/{START}/results")
    assert len(calls) == 1


def test_reanalyse_post_reruns_and_redirects(client, data):
    client.get(f"/runs/{START}/results")
    out = _run_folder(data) / "analysis" / "suppression.json"
    os.utime(out, (1, 1))
    response = client.post(f"/runs/{START}/analyse")
    assert response.status_code == 302 and response.headers["Location"].endswith(f"/runs/{START}/results")
    assert out.stat().st_mtime > 1


def test_run_downloads(client):
    client.get(f"/runs/{START}/results")
    for kind, marker in (("flashes", b"trial,contrast"), ("suppression", b'"method"'), ("report", b"Saccade-phase analysis")):
        response = client.get(f"/runs/{START}/files/{kind}")
        assert response.status_code == 200 and marker in response.data
        assert "attachment" in response.headers["Content-Disposition"]
    plot = client.get(f"/runs/{START}/files/plot")
    assert plot.status_code == 200 and plot.data[:4] == b"\x89PNG"
    assert client.get(f"/runs/{START}/files/session").status_code == 404  # only the analysis files
    assert client.get("/runs/999/files/report").status_code == 404


def test_participant_results_pools_runs_with_table_quality_and_downloads(client, data):
    response = client.get(f"/participants/{PID}/results")
    page = _html(response)
    assert response.status_code == 200
    summary = json.loads((data / PID / "analysis" / "summary.json").read_text())
    assert ap.headline(summary) in page and "from 2 runs" in page
    assert len(summary["runs_used"]) == 2
    for record in summary["runs_used"]:
        assert record["run"] in page
    assert "Runs pooled" in page and "all 2 analysed run(s) pooled" in page
    for kind, marker in (("flashes", b"run,run_date,mode"), ("summary", b'"schema_version"'), ("report", b"pooled saccadic-suppression")):
        download = client.get(f"/participants/{PID}/files/{kind}")
        assert download.status_code == 200 and marker in download.data
    assert client.get(f"/participants/{PID}/files/plot").data[:4] == b"\x89PNG"
    assert client.get("/participants/..%2F..%2Fetc/files/report").status_code == 404


def test_participant_analysis_reused_when_fresh_and_rerun_when_include_test_flips(client, monkeypatch):
    client.get(f"/participants/{PID}/results")
    calls = []
    real = ap.analyze_participant
    monkeypatch.setattr(ap, "analyze_participant", lambda *a, **k: calls.append(k.get("include_test")) or real(*a, **k))
    client.get(f"/participants/{PID}/results")
    assert calls == []
    client.get(f"/participants/{PID}/results?include_test=1")
    assert calls == [True]


def test_participant_analyse_post_runs_and_redirects_keeping_include_test(client, data):
    response = client.post(f"/participants/{PID}/analyse", data={"include_test": "1"})
    assert response.status_code == 302
    assert response.headers["Location"].endswith(f"/participants/{PID}/results?include_test=1")
    assert json.loads((data / PID / "analysis" / "summary.json").read_text())["include_test"] is True


def test_analyse_posts_with_problems_render_the_friendly_page(client):
    response = client.post("/participants/SYN-TEST/analyse")
    page = _html(response)
    assert response.status_code == 422 and "only test runs" in page and "Analyse including test runs" in page and "Traceback" not in page
    response = client.post(f"/runs/{NO_TRACE}/analyse")
    assert response.status_code == 422 and "no gaze recording" in _html(response)


def test_participant_with_only_test_runs_gets_a_friendly_message_then_include_test_works(client):
    response = client.get("/participants/SYN-TEST/results")
    page = _html(response)
    assert response.status_code == 422
    assert "only test runs" in page and "Include test runs" in page and "Traceback" not in page
    assert "Analyse including test runs" in page
    ok = client.get("/participants/SYN-TEST/results?include_test=1")
    assert ok.status_code == 200 and "suppression index" in _html(ok).lower() and "from 1 run" in _html(ok)


def test_unknown_participant_and_run_are_friendly_404s(client):
    for url in ("/participants/NOBODY/results", f"/runs/{START + 5}/results", "/runs/not-a-key/results"):
        response = client.get(url)
        assert response.status_code == 404
        assert "There is no" in _html(response) and "Traceback" not in _html(response)


def test_run_without_gaze_trace_is_a_friendly_message(client):
    response = client.get(f"/runs/{NO_TRACE}/results")
    page = _html(response)
    assert response.status_code == 422
    assert "no gaze recording (gaze_trace.csv)" in page and "Traceback" not in page


def test_participant_whose_only_run_has_no_gaze_trace_gets_no_stack_trace(client):
    response = client.get("/participants/SYN-NOTRACE/results")
    page = _html(response)
    assert response.status_code == 200 and "Traceback" not in page  # analyze_participant lists the run as left out
    assert "no gaze_trace.csv" in page and "no run could be pooled" in page and "cannot be computed" in page


def test_missing_clock_offset_is_a_friendly_message(client, data):
    meta_path = _run_folder(data) / "session.json"
    meta = json.loads(meta_path.read_text())
    del meta["gaze_quality"]["clock_offset_ms"]
    meta_path.write_text(json.dumps(meta))
    response = client.get(f"/runs/{START}/results")
    assert response.status_code == 422
    assert "no phone-to-computer clock offset" in _html(response) and "Traceback" not in _html(response)


def test_unexpected_error_is_a_sentence_not_a_stack_trace(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(an, "analyze", boom)
    response = client.get(f"/runs/{START + DAY}/results")
    assert response.status_code == 500
    page = _html(response)
    assert "RuntimeError: disk on fire" in page and "console window" in page and "Traceback" not in page


def test_deleting_a_run_removes_the_stale_participant_analysis_but_keeps_the_folder(client, data):
    client.get(f"/participants/{PID}/results")
    assert (data / PID / "analysis" / "summary.json").exists()
    assert client.delete(f"/sessions/{START + DAY}").status_code == 200
    assert not (data / PID / "analysis").exists()
    assert (data / PID).is_dir() and len([p for p in (data / PID).iterdir()]) == 1
    page = _html(client.get(f"/participants/{PID}/results"))
    assert "from 1 run" in page  # re-analysed on open, without the deleted run


def test_deleting_the_last_run_removes_the_participant_folder_with_its_analysis(client, data):
    client.get(f"/participants/{PID}/results")
    client.delete(f"/sessions/{START + DAY}")
    client.get(f"/participants/{PID}/results")  # writes analysis/ again
    assert (data / PID / "analysis").exists()
    assert client.delete(f"/sessions/{START}").status_code == 200
    assert not (data / PID).exists()
    assert f'data-participant="{PID}"' not in _html(client.get("/"))
    assert client.get(f"/participants/{PID}/results").status_code == 404


def test_quality_rules_traffic_lights():
    meta = {"gaze_quality": {"latency_ms_median": 200.0, "longest_gap_ms": 20.0, "gaps_over_threshold": 0},
            "head_motion": {"status": "ok", "flagged_trials": 2, "max_drift_deg": 1.5}}
    q = results.run_quality(meta, {"n_saccade": 8, "n_fixation": 40, "method": "accuracy", "notes": []})
    lights = {r["name"]: r["light"] for r in q["rules"]}
    assert lights == {"Gaze delay": "amber", "Tracking gaps": "green", "Head movement": "amber", "Flashes per condition": "amber", "Result": "amber"}
    assert q["overall"] == "amber"
    q = results.run_quality({}, {"n_saccade": 3, "n_fixation": 40, "method": None, "notes": ["cannot compute"]})
    lights = {r["name"]: r["light"] for r in q["rules"]}
    assert lights["Flashes per condition"] == "red" and lights["Result"] == "red" and lights["Head movement"] == "grey"
    assert q["overall"] == "red"
    meta = {"gaze_quality": {"latency_ms_median": 90.0, "longest_gap_ms": 40.0}, "head_motion": {"status": "ok", "flagged_trials": 0}}
    q = results.run_quality(meta, {"n_saccade": 30, "n_fixation": 50, "method": "psychometric", "notes": []})
    assert q["overall"] == "green"
    assert results.run_quality({"head_motion": {"status": "unavailable (no IMU)"}}, {})["rules"][2]["light"] == "amber"


def test_download_labels_and_the_bootstrap_count_are_defined_once():
    import inspect

    from src.dashboard import comparison
    from src.experiment import analyze_participant as ap_module

    assert comparison.DOWNLOAD_LABELS is results.DOWNLOAD_LABELS
    for kind in ("flashes", "suppression", "summary", "report", "plot", "csv", "json"):
        assert results.DOWNLOAD_LABELS[kind]
    assert results.N_BOOT == ap_module.N_BOOT == inspect.signature(ap_module.analyze_participant).parameters["n_boot"].default


def test_run_page_shows_the_plot_the_gaze_timeline_and_the_companion_numbers(client, data):
    page = _html(client.get(f"/runs/{START}/results"))
    assert f'src="/runs/{START}/files/plot"' in page and f'src="/runs/{START}/files/gaze"' in page
    assert "Eye position and flashes" in page and "gaze_timeline.png" in page  # listed in the downloads too
    assert "Companion numbers" in page and "Accuracy, fixation minus saccade" in page
    saved = json.loads((_run_folder(data) / "analysis" / "suppression.json").read_text())
    assert saved["companion"] and saved["timecourse"]
    gaze = client.get(f"/runs/{START}/files/gaze")
    assert gaze.status_code == 200 and gaze.data[:4] == b"\x89PNG" and "attachment" not in gaze.headers.get("Content-Disposition", "")


def test_an_old_run_analysis_without_the_gaze_timeline_is_rerun_and_one_without_companion_numbers_renders(client, data):
    client.get(f"/runs/{START}/results")
    analysis = _run_folder(data) / "analysis"
    (analysis / "gaze_timeline.png").unlink()  # an analysis folder from before the timeline existed
    page = client.get(f"/runs/{START}/results")
    assert page.status_code == 200 and (analysis / "gaze_timeline.png").exists()  # stale, so analysed again
    saved = json.loads((analysis / "suppression.json").read_text())
    saved.pop("companion"), saved.pop("timecourse")  # ...and one from before the companion numbers
    (analysis / "suppression.json").write_text(json.dumps(saved), encoding="utf-8")
    later = os.stat(_run_folder(data) / "trials.csv").st_mtime + 100
    for f in analysis.iterdir():
        os.utime(f, (later, later))
    page = client.get(f"/runs/{START}/results")
    assert page.status_code == 200 and "Companion numbers" not in _html(page)


def test_participant_page_shows_the_companion_numbers_and_reanalyses_a_schema_3_summary(client, data):
    assert "Companion numbers" in _html(client.get(f"/participants/{PID}/results"))
    path = data / PID / "analysis" / "summary.json"
    summary = json.loads(path.read_text())
    assert summary["schema_version"] == 4
    summary.pop("companion"), summary.pop("timecourse")
    summary["schema_version"] = 3
    path.write_text(json.dumps(summary), encoding="utf-8")
    page = client.get(f"/participants/{PID}/results")
    assert page.status_code == 200 and "Companion numbers" in _html(page)  # re-analysed, now schema 4
    assert json.loads(path.read_text())["schema_version"] == 4
