"""Dashboard comparison, group labels and study summary (src/dashboard/comparison.py + /compare, /study,
/participants/<id>/label) through Flask's test client, on synthetic anonymous run folders in pytest tmp folders - never data/."""

import functools
import json
import shutil

import pytest
import synthetic_runs as sr
from markupsafe import escape

from src.dashboard import app as app_module
from src.dashboard import comparison, data_access, results
from src.experiment import analyze_participant as ap
from src.experiment import compare as cmp
from src.experiment import participants
from src.experiment import settings as settings_module

DAY = 86_400
START = 1_791_000_000
PEOPLE = {"T01": "TBI", "T02": "TBI", "C01": "control", "C02": "control"}


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """Built once: two TBI-labelled and two control-labelled people (pre-analysed with a small bootstrap, so the copies'
    summaries are fresh), a person whose run has no gaze trace and a person with only a test run."""
    data = tmp_path_factory.mktemp("compare_template")
    for i, (pid, group) in enumerate(PEOPLE.items()):
        strong = group == "TBI"
        sr.write_run(data, pid, START + i * DAY, n_trials=100, seed=10 + i, threshold_saccade=0.12 if strong else 0.05)
        participants.set_label(pid, group, data_dir=data)
        assert ap.analyze_participant(pid, data, n_boot=50).summary["method"] == "psychometric", pid  # else the modes shift
    (sr.write_run(data, "N01", START + 10 * DAY, n_trials=20, seed=30) / "gaze_trace.csv").unlink()
    sr.write_run(data, "X01", START + 11 * DAY, n_trials=20, seed=31, test_mode=True)
    return data


@pytest.fixture
def data(template, tmp_path, monkeypatch):
    target = tmp_path / "data"
    shutil.copytree(template, target)
    monkeypatch.setattr(data_access, "DATA_DIR", target)
    monkeypatch.setattr(settings_module, "DATA_DIR", target)
    monkeypatch.setattr(results, "N_BOOT", 50)
    monkeypatch.setattr(comparison, "N_BOOT", 100)
    monkeypatch.setattr(cmp, "analyze_participant", functools.partial(ap.analyze_participant, n_boot=50))
    return target


@pytest.fixture
def client(data):
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _html(response):
    return response.get_data(as_text=True)


def _written(data):
    folders = list((data / cmp.COMPARISONS_DIR / comparison.DASHBOARD_DIR).iterdir())
    assert len(folders) == 1
    return folders[0], json.loads((folders[0] / "comparison.json").read_text(encoding="utf-8"))


def test_label_edit_round_trip_and_index_page(client, data):
    page = _html(client.get("/"))
    assert 'href="/compare"' in page and 'href="/study"' in page
    assert 'value="TBI"' in page and 'list="group-suggestions"' in page
    response = client.post("/participants/T01/label", json={"group": " control ", "note": "pilot <b>1</b>"})
    assert response.status_code == 200 and response.get_json() == {"ok": True, "group": "control", "note": "pilot <b>1</b>"}
    assert participants.list_labels(data).get("T01") == {"group": "control", "note": "pilot <b>1</b>"}
    page = _html(client.get("/"))
    assert "pilot &lt;b&gt;1&lt;/b&gt;" in page and "<b>1</b>" not in page  # escaped
    assert client.post("/participants/T01/label", json={"group": "", "note": ""}).status_code == 200
    assert "T01" not in participants.list_labels(data)
    assert client.post("/participants/NOBODY/label", json={"group": "TBI"}).status_code == 404
    assert client.post("/participants/T02/label", data='{"group": "x"}', content_type="text/plain").status_code == 400  # no cross-site form posts
    bad = client.post("/participants/T02/label", json={"group": "group:x"})
    assert bad.status_code == 400 and "cannot contain" in bad.get_json()["error"]


def test_compare_page_without_a_selection_shows_pickers_and_quick_picks(client, data):
    page = _html(client.get("/compare"))
    assert 'data-group="TBI"' in page and 'data-group="control"' in page and "all in group TBI" in page
    for pid in PEOPLE:
        assert f'value="{pid}"' in page
    assert "Result" not in page and not (data / cmp.COMPARISONS_DIR).exists()  # nothing run yet


@pytest.mark.parametrize("query, mode", [
    ("a=T01&b=C01", "1 vs 1"),
    ("a=group:tbi&b=group:Control", "group vs group"),  # group tokens, case-insensitive
    ("a=T01&b=C01,C02", "1 vs group"),
])
def test_compare_each_mode_renders_headline_mode_and_table(client, data, query, mode):
    response = client.get(f"/compare?{query}")
    page = _html(response)
    assert response.status_code == 200, page
    _, saved = _written(data)
    assert saved["headline"] and str(escape(saved["headline"])) in page
    assert f"<strong>{mode}</strong>" in page and "How to read this" in page and "Exploratory" in page
    expected = {"a=T01&b=C01": ["T01", "C01"], "a=T01&b=C01,C02": ["T01", "C01", "C02"]}.get(query, list(PEOPLE))
    for pid in expected:
        assert f'class="person-row" data-participant="{pid}"' in page
    if mode == "1 vs group":
        assert "no p-value" in page.lower()


def test_repeated_query_values_and_group_quick_pick_resolve_alike(client):
    assert comparison.tokens(["T01,T02", "T02", " "]) == ["T01", "T02"]
    with app_module.app.test_request_context():
        assert comparison.resolve("A", ["group:TBI"]) == ["T01", "T02"]
        assert comparison.resolve("B", ["all"]) == sorted([*PEOPLE, "N01", "X01"])
        everyone = sorted([*PEOPLE, "N01", "X01"])
        assert comparison.resolve_pair(["T01"], ["all"]) == (["T01"], [p for p in everyone if p != "T01"])  # "all" = everyone else, like the CLI
        tbi = comparison.resolve("A", ["group:TBI"])
        assert comparison.resolve_pair(["all"], ["group:TBI"]) == ([p for p in everyone if p not in tbi], tbi)
        for a, b in ((["all"], ["all"]), (["T01"], ["T01", "C01"])):  # both "all", or one person picked twice: still an error
            with pytest.raises(comparison.FriendlyError, match="in both set A and set B"):
                comparison.resolve_pair(a, b)


def test_selection_errors_are_friendly(client):
    cases = [
        ("a=group:TBI&b=T01", 400, "in both set A and set B"),
        ("a=T01", 400, "Set B is empty"),
        ("a=T01&b=NOBODY", 404, "no participant &#39;NOBODY&#39;"),
        ("a=T01&b=group:nope", 404, "there is no group &#39;nope&#39;"),
        ("a=..%2F..%2Fetc&b=C01", 404, "no participant"),
    ]
    for query, status, text in cases:
        response = client.get(f"/compare?{query}")
        page = _html(response)
        assert response.status_code == status and text in page and "Traceback" not in page, query


def test_everyone_excluded_is_explained_not_an_error_page(client):
    page = _html(client.get("/compare?a=N01&b=X01"))
    assert "Everyone was left out" in page and "Nothing to compare" in page
    assert page.count("left out:") >= 2 and "Traceback" not in page


def test_analysis_failure_is_a_sentence(client, monkeypatch):
    monkeypatch.setattr(cmp, "compare", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    response = client.get("/compare?a=T01&b=C01")
    assert response.status_code == 500 and "RuntimeError: disk on fire" in _html(response) and "Traceback" not in _html(response)


def test_downloads_are_served_by_whitelisted_kind_only(client, data):
    query = "a=T01&b=C01,C02"
    assert client.get(f"/compare/files/csv?{query}").status_code == 404  # not computed yet
    page = _html(client.get(f"/compare?{query}"))
    assert 'src="/compare/files/plot?a=T01&amp;b=C01' in page
    for kind, marker in (("csv", b"participant,group,set"), ("json", b'"headline"'), ("report", b"Comparison (1 vs group)")):
        response = client.get(f"/compare/files/{kind}?{query}")
        assert response.status_code == 200 and marker in response.data and "attachment" in response.headers["Content-Disposition"]
    assert client.get(f"/compare/files/plot?{query}").data[:4] == b"\x89PNG"
    assert client.get(f"/compare/files/png?{query}").status_code == 404
    assert client.get(f"/compare/files/csv?{query}&include_test=1").status_code == 404  # another selection
    assert client.get("/compare/files/csv?a=..&b=C01").status_code == 404
    client.get(f"/compare?{query}")
    _written(data)  # re-opening the same selection overwrote the same folder


def test_study_page_lists_people_and_serves_the_csv(client, data):
    page = _html(client.get("/study"))
    for pid in PEOPLE:
        assert f'data-participant="{pid}"' in page and f'href="/participants/{pid}/results"' in page
    assert "TBI" in page and "control" in page and "Exploratory" in page
    csv_response = client.get("/study/study_summary.csv")
    assert csv_response.status_code == 200 and csv_response.data.startswith(b"participant,group,note,status")
    assert (data / "study_summary.csv").exists()


def test_single_vs_return_jump_mismatch_is_shown_on_the_compare_page(client, data):
    sr.write_run(data, "R01", START + 20 * DAY, n_trials=60, seed=40, return_jump=True)
    page = _html(client.get("/compare?a=T01&b=R01"))
    assert "different settings" in page and "run kind return_jump/" in page and "run kind single" in page and "Traceback" not in page


def test_old_schema_3_summaries_still_render_on_the_compare_and_study_pages(client, data):
    """Saved pooled analyses from before the companion numbers (schema 3) are stale: the pages re-analyse them and render."""
    for pid in PEOPLE:
        path = data / pid / "analysis" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["schema_version"] == 5 and "companion" in summary and "timecourse" in summary
        summary.pop("companion"), summary.pop("timecourse"), summary.pop("secondary_onset_window")
        summary["schema_version"] = 3
        path.write_text(json.dumps(summary), encoding="utf-8")
    page = _html(client.get("/compare?a=group:tbi&b=group:control"))
    assert "group vs group" in page and "Traceback" not in page
    assert all(json.loads((data / pid / "analysis" / "summary.json").read_text())["schema_version"] == 5 for pid in PEOPLE)
    study = client.get("/study")
    assert study.status_code == 200 and all(f'data-participant="{pid}"' in _html(study) for pid in PEOPLE)
