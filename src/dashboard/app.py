"""Flask dashboard (`-m src.dashboard.app`, localhost:5000) for browsing sessions in data/. Separate process from the
experiment; it only reads/writes files via data_access.py."""

from datetime import datetime

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from src.dashboard import comparison, results
from src.dashboard.data_access import (
    csv_path_for,
    delete_session,
    discover_sessions,
    save_participant_info,
)

app = Flask(__name__)


@app.template_filter("session_datetime")
def _session_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M")


@app.get("/")
def index():
    return render_template(
        "index.html",
        sessions=discover_sessions(),
        participants=comparison.overview_with_labels(),
        groups=comparison.group_names(),
    )


@app.post("/sessions/<timestamp>")
def update_session(timestamp: str):
    if not csv_path_for(timestamp).exists():
        abort(404)
    body = request.get_json(force=True)
    try:
        save_participant_info(
            timestamp,
            name=body.get("name", ""),
            gender=body.get("gender", ""),
            age=body.get("age", ""),
        )
    except ValueError:
        return jsonify({"ok": False, "error": "age must be a whole number"}), 400
    return jsonify({"ok": True})


@app.delete("/sessions/<timestamp>")
def remove_session(timestamp: str):
    if not csv_path_for(timestamp).exists():
        abort(404)
    delete_session(timestamp)
    return jsonify({"ok": True})


# -- results (one-click analysis; every number comes from analyze_session / analyze_participant via results.py) -------


def _include_test() -> bool:
    return (request.values.get("include_test") or "").lower() in ("1", "true", "on", "yes")


def _results_page(load, **context):
    try:
        page = load()
    except results.FriendlyError as error:
        return render_template("results.html", page=None, error=error, labels=results.DOWNLOAD_LABELS, **context), error.status
    return render_template("results.html", page=page, error=None, labels=results.DOWNLOAD_LABELS, **context)


@app.get("/runs/<timestamp>/results")
def run_results(timestamp: str):
    """The results page for one run; analyses it first when there is no analysis newer than the run's files."""
    return _results_page(lambda: results.run_page(timestamp), kind="run", key=timestamp, include_test=None)


@app.post("/runs/<timestamp>/analyse")
def run_analyse(timestamp: str):
    try:
        results.analyse_run(timestamp)
    except results.FriendlyError as error:
        return render_template("results.html", page=None, error=error, labels=results.DOWNLOAD_LABELS, kind="run", key=timestamp,
                               include_test=None), error.status
    return redirect(url_for("run_results", timestamp=timestamp))


@app.get("/runs/<timestamp>/files/<kind>")
def run_file(timestamp: str, kind: str):
    if kind not in results.RUN_FILES:
        abort(404)
    try:
        path = results.run_file(results.run_ref(timestamp), kind)
    except results.FriendlyError:
        abort(404)
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=kind != "plot", download_name=path.name)


@app.get("/participants/<pid>/results")
def participant_results(pid: str):
    """The pooled results page for one participant (?include_test=1 adds test runs); analyses first when stale."""
    include_test = _include_test()
    return _results_page(lambda: results.participant_page(pid, include_test), kind="participant", key=pid, include_test=include_test)


@app.post("/participants/<pid>/analyse")
def participant_analyse(pid: str):
    include_test = _include_test()
    try:
        results.analyse_participant(pid, include_test)
    except results.FriendlyError as error:
        return render_template("results.html", page=None, error=error, labels=results.DOWNLOAD_LABELS, kind="participant", key=pid,
                               include_test=include_test), error.status
    return redirect(url_for("participant_results", pid=pid, include_test=1 if include_test else None))


@app.get("/participants/<pid>/files/<kind>")
def participant_file(pid: str, kind: str):
    if kind not in results.PARTICIPANT_FILES or pid not in results.participant_ids():
        abort(404)
    path = results.participant_file(pid, kind)
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=kind != "plot", download_name=f"{pid}_{path.name}")


# -- group labels, comparison, study summary (numbers from compare.py / study_summary.py via comparison.py) ------------


@app.post("/participants/<pid>/label")
def participant_label(pid: str):
    body = request.get_json(silent=True)  # no force: only a real application/json request (another website cannot send one without CORS)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "expected a JSON body with group and note"}), 400
    try:
        entry = comparison.save_label(pid, body.get("group"), body.get("note"))
    except results.FriendlyError as error:
        return jsonify({"ok": False, "error": error.message}), error.status
    return jsonify({"ok": True, "group": entry.get("group", ""), "note": entry.get("note", "")})


def _selection() -> tuple[list[str], list[str]]:
    return comparison.tokens(request.args.getlist("a")), comparison.tokens(request.args.getlist("b"))


@app.get("/compare")
def compare_page():
    """Pickers for set A and set B; with ?a=..&b=.. (participant IDs, group:<label>, all) it also runs the comparison."""
    a_tokens, b_tokens = _selection()
    include_test = _include_test()
    result, error = None, None
    if a_tokens or b_tokens:
        try:
            result = comparison.run_comparison(a_tokens, b_tokens, include_test)
        except results.FriendlyError as exc:
            error = exc
    context = dict(people=comparison.picker_rows(), groups=comparison.group_names(), a_tokens=a_tokens, b_tokens=b_tokens,
                   include_test=include_test, result=result, error=error, labels=comparison.DOWNLOAD_LABELS)
    return render_template("compare.html", **context), (error.status if error else 200)


@app.get("/compare/files/<kind>")
def compare_file(kind: str):
    """One file of the comparison named by the same query (?a=..&b=..&include_test=..); only the four whitelisted kinds."""
    a_tokens, b_tokens = _selection()
    path = comparison.output_file(a_tokens, b_tokens, _include_test(), kind)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=kind != "plot", download_name=path.name)


@app.get("/study")
def study_page():
    include_test = _include_test()
    try:
        page, error = comparison.study_page(include_test), None
    except results.FriendlyError as exc:
        page, error = None, exc
    return render_template("study.html", page=page, error=error, include_test=include_test), (error.status if error else 200)


@app.get("/study/study_summary.csv")
def study_csv():
    path = comparison.study_csv()
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


def main() -> None:
    app.run(host="127.0.0.1", port=5000)  # this computer only: the dashboard shows and deletes participant data


if __name__ == "__main__":
    main()
