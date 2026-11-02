"""Flask dashboard (`-m src.dashboard.app`, localhost:5000) for browsing sessions in data/. Separate process from the
experiment; it only reads/writes files via data_access.py."""

from datetime import datetime

from flask import Flask, abort, jsonify, render_template, request

from src.dashboard.data_access import (
    csv_path_for,
    delete_session,
    discover_sessions,
    save_participant_info,
)
from src.experiment.settings import (
    MAX_CONTRAST_FLOOR_PERCENT,
    MIN_CONTRAST_FLOOR_PERCENT,
    load_contrast_floor_percent,
    save_contrast_floor_percent,
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
        contrast_floor_percent=load_contrast_floor_percent(),
        contrast_floor_min=MIN_CONTRAST_FLOOR_PERCENT,
        contrast_floor_max=MAX_CONTRAST_FLOOR_PERCENT,
    )


@app.post("/settings")
def update_settings():
    body = request.get_json(force=True)
    try:
        save_contrast_floor_percent(float(body.get("contrast_floor_percent")))
    except (TypeError, ValueError) as error:
        return jsonify({"ok": False, "error": str(error) or "contrast floor must be a number"}), 400
    return jsonify({"ok": True})


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


def main() -> None:
    app.run(port=5000)


if __name__ == "__main__":
    main()
