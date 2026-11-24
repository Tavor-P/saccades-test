import pytest
import sim_tools

from include.experiment import constants as exp_constants


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
