import json
import subprocess
import sys
from pathlib import Path

import pytest
import sim_tools

from include.experiment import constants as exp_constants

HERE = Path(__file__).resolve().parent


def test_override_constants_patches_every_importer_restores_rejects_typos_and_changes_behaviour():
    original = exp_constants.SACCADE_ONSET_STABILITY_MS
    from src.experiment import trial_mechanics

    with sim_tools.override_constants({"SACCADE_ONSET_STABILITY_MS": 5}):
        assert exp_constants.SACCADE_ONSET_STABILITY_MS == 5
        assert trial_mechanics.SACCADE_ONSET_STABILITY_MS == 5  # module that copied it by name
    assert exp_constants.SACCADE_ONSET_STABILITY_MS == original
    assert trial_mechanics.SACCADE_ONSET_STABILITY_MS == original
    with pytest.raises(KeyError):
        with sim_tools.override_constants({"SACCADE_ONSET_STABILTY_MS": 5}):  # typo
            pass
    _, fast = sim_tools.run(trials=10, seed=1, constants={"SACCADE_ONSET_STABILITY_MS": 5})
    _, slow = sim_tools.run(trials=10, seed=1, constants={"SACCADE_ONSET_STABILITY_MS": 80})
    assert slow["mean_onset_detection_lag_ms"] > fast["mean_onset_detection_lag_ms"] + 30


def test_profiles_summary_and_sweep():
    with pytest.raises(KeyError):
        sim_tools.make_profile("nope")
    with pytest.raises(KeyError):
        sim_tools.make_profile("default", not_a_field=1)
    _, summary = sim_tools.run(trials=15, seed=2)
    json.dumps(summary)  # JSON-serialisable
    assert summary["trials"] == 15
    assert summary["flashes_truly_in_saccade"] <= summary["flashes"]  # ground truth recorded
    assert summary["valid_flashes"] <= summary["flashes"]
    rows = sim_tools.sweep("SACCADE_ONSET_STABILITY_MS", [10, 40], trials=8, seeds=2)
    assert [r["SACCADE_ONSET_STABILITY_MS"] for r in rows] == [10, 40]


def test_cli_json_and_list_work_from_the_project_root():
    root = HERE.parents[1]
    out = subprocess.run(
        [sys.executable, str(HERE / "run_sim.py"), "--trials", "5", "--json"], cwd=root, capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout)["trials"] == 5
    listing = subprocess.run([sys.executable, str(HERE / "run_sim.py"), "--list"], cwd=root, capture_output=True, text=True, check=True)
    assert "slow_reactor" in listing.stdout
