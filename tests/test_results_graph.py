from include.experiment.types import TrialResult
from src.experiment.results_graph import _replay_zest, build_comparison_graph


def _trial(phase, index, contrast, hit, grating_shown=True):
    return TrialResult(
        index=index,
        phase=phase,
        source=None,
        target=None,
        saccade_duration_ms=None,
        grating_shown=grating_shown,
        contrast=contrast,
        responded=hit,
        response_time_ms=250.0 if hit else None,
        outcome="hit" if hit else "miss",
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
