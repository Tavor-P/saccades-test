from datetime import datetime

from flask import Flask, abort, jsonify, render_template, request, send_file

from src.dashboard.data_access import (
    csv_path_for,
    delete_session,
    discover_sessions,
    graph_cache_path_for,
    load_session_trials,
    save_participant_info,
)
from src.experiment.results_graph import build_comparison_graph

app = Flask(__name__)


@app.template_filter("session_datetime")
def _session_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M")


@app.get("/")
def index():
    return render_template("index.html", sessions=discover_sessions())


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


@app.get("/sessions/<timestamp>/graph.png")
def session_graph(timestamp: str):
    csv_path = csv_path_for(timestamp)
    if not csv_path.exists():
        abort(404)
    output_path = graph_cache_path_for(timestamp)
    output_path.parent.mkdir(exist_ok=True)
    build_comparison_graph(load_session_trials(timestamp), output_path=output_path)
    return send_file(output_path, mimetype="image/png")


def main() -> None:
    app.run(port=5000)


if __name__ == "__main__":
    main()
