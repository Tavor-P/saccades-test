from include.experiment.types import Orientation, TrialResult
from src.experiment.results_graph import _exclude_flashes_not_during_saccade, _replay_zest, build_comparison_graph


def _trial(
    phase, index, contrast, correct, grating_shown=True, orientation=Orientation.VERTICAL, flash_during_saccade=None
):
    return TrialResult(
        index=index,
        phase=phase,
        source=None,
        target=None,
        saccade_duration_ms=None,
        grating_shown=grating_shown,
        contrast=contrast,
        orientation=orientation if grating_shown else None,
        responded=correct,
        response_orientation=(orientation if correct else None) if grating_shown else None,
        response_time_ms=250.0 if correct else None,
        outcome="correct" if correct else "miss",
        flash_during_saccade=flash_during_saccade,
    )


def test_replay_zest_ignores_other_phases_and_catch_trials():
    results = [
        _trial("presaccade", 0, 0.05, True),
        _trial("saccade", 1, 0.2, True),
        _trial("presaccade", 2, None, False, grating_shown=False),
    ]
    replay = _replay_zest(results, "presaccade")
    assert replay is not None
    _, trials = replay
    assert len(trials) == 1
    assert trials[0].phase == "presaccade"


def test_replay_zest_returns_none_when_no_matching_trials():
    assert _replay_zest([], "presaccade") is None


def test_exclude_flashes_not_during_saccade_drops_only_explicit_false():
    results = [
        _trial("saccade", 0, 0.1, True, flash_during_saccade=True),
        _trial("saccade", 1, 0.1, True, flash_during_saccade=False),  # flash landed after arrival - dropped
        _trial("saccade", 2, 0.1, True, flash_during_saccade=None),  # unknown (legacy session) - kept
        _trial("presaccade", 3, 0.05, True, flash_during_saccade=None),  # no saccade concept at all - kept
        _trial("saccade", 4, None, False, grating_shown=False),  # catch trial, no flash at all - kept
    ]
    kept = _exclude_flashes_not_during_saccade(results)
    assert [r.index for r in kept] == [0, 2, 3, 4]


def test_replay_zest_via_build_comparison_graph_excludes_false_flash_during_saccade(tmp_path):
    # The excluded trial's contrast (0.9) is nowhere near the others (0.1) -
    # if it leaked into the replay, the fitted threshold would be dragged way
    # off from what the "during saccade" trials alone actually show.
    saccade_trials = [_trial("saccade", i, 0.1, i % 2 == 0, flash_during_saccade=True) for i in range(10)]
    saccade_trials.append(_trial("saccade", 10, 0.9, False, flash_during_saccade=False))
    replay = _replay_zest(_exclude_flashes_not_during_saccade(saccade_trials), "saccade")
    assert replay is not None
    _, trials = replay
    assert all(t.flash_during_saccade is not False for t in trials)
    assert 10 not in [t.index for t in trials]


def test_build_comparison_graph_writes_a_png(tmp_path):
    results = [_trial("presaccade", i, 0.05, i % 2 == 0) for i in range(10)]
    out = tmp_path / "graph.png"
    path = build_comparison_graph(results, output_path=out)
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_comparison_graph_handles_no_results_at_all(tmp_path):
    out = tmp_path / "empty.png"
    build_comparison_graph([], output_path=out)
    assert out.exists()


def test_build_comparison_graph_handles_only_one_phase(tmp_path):
    results = [_trial("presaccade", i, 0.05, i % 2 == 0) for i in range(10)]
    out = tmp_path / "one_phase.png"
    build_comparison_graph(results, output_path=out)
    assert out.exists()


def _false_alarm_trial(phase, index):
    return TrialResult(
        index=index,
        phase=phase,
        source=None,
        target=None,
        saccade_duration_ms=None,
        grating_shown=False,
        contrast=None,
        orientation=None,
        responded=True,
        response_orientation=Orientation.VERTICAL,
        response_time_ms=300.0,
        outcome="false_alarm",
    )


def test_build_comparison_graph_handles_false_alarms(tmp_path):
    # Regression test: false alarms have no real contrast (grating_shown=False,
    # contrast=None), so they need their own display path rather than being
    # scattered on the log-scaled contrast axis alongside real correct/miss trials.
    results = [_trial("presaccade", i, 0.05, i % 2 == 0) for i in range(10)]
    results.append(_false_alarm_trial("presaccade", 10))
    out = tmp_path / "false_alarms.png"
    build_comparison_graph(results, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_comparison_graph_handles_flash_not_during_saccade(tmp_path):
    results = [_trial("saccade", i, 0.1, i % 2 == 0, flash_during_saccade=True) for i in range(10)]
    results.append(_trial("saccade", 10, 0.9, False, flash_during_saccade=False))
    out = tmp_path / "flash_not_during_saccade.png"
    build_comparison_graph(results, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_comparison_graph_handles_mixed_orientations(tmp_path):
    # Regression test for the orientation-accuracy panels: vertical and
    # horizontal trials mixed together shouldn't crash the new bar charts,
    # including the per-phase breakdown across both phases.
    results = [
        _trial("presaccade", 0, 0.05, True, orientation=Orientation.VERTICAL),
        _trial("presaccade", 1, 0.05, False, orientation=Orientation.HORIZONTAL),
        _trial("saccade", 2, 0.1, True, orientation=Orientation.HORIZONTAL),
        _trial("saccade", 3, 0.1, False, orientation=Orientation.VERTICAL),
    ]
    out = tmp_path / "orientations.png"
    build_comparison_graph(results, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
