from include.experiment.types import Orientation, TrialResult
from src.experiment.results_graph import _replay_zest, build_comparison_graph


def _trial(phase, index, contrast, correct, grating_shown=True, orientation=Orientation.VERTICAL):
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
