r"""Golden (snapshot) test of the whole analysis pipeline on two fixed synthetic runs.

Cases: `single` (plain saccade run) and `return_jump` (out + return legs). Each is written with `synthetic_runs.write_run`
(fixed seed, tmp_path only), analysed with `analyze_session.analyze` and pooled with `analyze_participant.analyze_participant`.
The files report.txt, suppression.json, flashes.csv, the pooled summary.json and the pooled report (participant_report.txt) are compared with
tests/fixtures/golden/<case>/ (PNGs are skipped). Volatile content is normalised first: run-folder names / timestamps become
`<RUN>` (also the local-time `date` fields), `generated_at` and absolute paths are blanked, and every float is rounded to 6 significant digits.

After an INTENDED output change, regenerate the golden files and review the git diff:

    $env:GOLDEN_UPDATE=1; .\scripts\py.ps1 -m pytest -q tests/ai-tests/test_golden_analysis.py
"""

import csv
import difflib
import io
import json
import os
import re
from pathlib import Path

import pytest
import synthetic_runs as sr

from src.experiment import analyze_participant as ap
from src.experiment import analyze_session as an

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
START = 1_790_000_000
CASES = {"single": {}, "return_jump": {"return_jump": True}}
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}[_ ]\d{2}[-:]\d{2}([-:]\d{2})?(_test)?")
_FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _sig(x: float) -> str:
    return "0" if x == 0 else f"{x:.6g}"


def _norm_text(text: str, tmp: Path) -> str:
    text = text.replace(str(tmp), "<TMP>").replace(tmp.as_posix(), "<TMP>").replace("\r\n", "\n")
    text = _STAMP.sub("<RUN>", text)
    return _FLOAT.sub(lambda m: _sig(float(m.group())), text)


def _norm_json(obj, tmp: Path):
    if isinstance(obj, float):
        return float(_sig(obj))
    if isinstance(obj, dict):
        return {_norm_text(k, tmp): ("" if k == "generated_at" else _norm_json(v, tmp)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm_json(v, tmp) for v in obj]
    if isinstance(obj, str):
        return _norm_text(obj, tmp)
    return obj


def _snapshot(path: Path, tmp: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.dumps(_norm_json(json.loads(text), tmp), indent=2, sort_keys=False) + "\n"
    if path.suffix == ".csv":
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        for row in csv.reader(io.StringIO(text)):
            writer.writerow([_norm_text(cell, tmp) for cell in row])
        return out.getvalue()
    return _norm_text(text, tmp)


@pytest.mark.parametrize("case", CASES)
def test_analysis_output_matches_the_golden_files(case, tmp_path):
    sr.write_run(tmp_path, "GOLD", START, n_trials=60, seed=5, **CASES[case])
    sr.write_run(tmp_path, "GOLD", START + 86_400, n_trials=60, seed=6, **CASES[case])
    run = an.analyze(None, tmp_path)  # newest run
    pooled = ap.analyze_participant("GOLD", tmp_path, n_boot=100)
    files = {
        "report.txt": run["report_path"],
        "suppression.json": run["suppression_json"],
        "flashes.csv": run["csv"],
        "summary.json": pooled.paths["summary"],
        "participant_report.txt": pooled.paths["report"],
    }
    folder = GOLDEN / case
    update = os.environ.get("GOLDEN_UPDATE") == "1"
    if update:
        folder.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        got = _snapshot(path, tmp_path)
        golden = folder / name
        if update:
            golden.write_bytes(got.encode("utf-8"))
            continue
        assert golden.exists(), f"missing {golden}; run with GOLDEN_UPDATE=1"
        want = golden.read_bytes().decode("utf-8").replace("\r\n", "\n")
        if got != want:
            diff = "".join(difflib.unified_diff(want.splitlines(True), got.splitlines(True), f"golden/{case}/{name}", f"current/{name}", n=2))
            pytest.fail(f"{case}/{name} differs from the golden file (GOLDEN_UPDATE=1 regenerates):\n{diff[:6000]}")
