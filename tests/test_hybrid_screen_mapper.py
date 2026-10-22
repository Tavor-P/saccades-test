"""HybridScreenMapper: tags preferred, outline as fallback; backend registration."""
from types import SimpleNamespace

import pytest

from src.eye_tracking.hybrid_screen_mapper import HybridScreenMapper
from src.eye_tracking.screen_gaze_mapper import ScreenGaze


class _Fake:
    def __init__(self, result, located=True, detected=4):
        self.result, self.located, self.markers_detected, self.frames = result, located, detected, 0

    def process_scene(self, frame):
        self.frames += 1

    def map_gaze(self, gaze):
        return self.result


TAG, OUTLINE = ScreenGaze(100.0, 100.0, 0.0), ScreenGaze(110.0, 95.0, 0.0)


def test_both_locators_see_every_frame():
    tags, outline = _Fake(TAG), _Fake(OUTLINE)
    HybridScreenMapper(tags, outline).process_scene(object())
    assert tags.frames == outline.frames == 1


def test_tags_win_when_located():
    mapper = HybridScreenMapper(_Fake(TAG), _Fake(OUTLINE))
    assert mapper.map_gaze(None) is TAG
    assert mapper.sample_counts == {"tags": 1, "outline": 0}


def test_outline_is_the_fallback_when_tags_are_lost():
    mapper = HybridScreenMapper(_Fake(None, located=False, detected=0), _Fake(OUTLINE))
    assert mapper.map_gaze(None) is OUTLINE
    assert mapper.located and mapper.markers_detected == 4
    assert mapper.sample_counts == {"tags": 0, "outline": 1}


def test_unlocated_when_neither_finds_the_screen():
    mapper = HybridScreenMapper(_Fake(None, located=False, detected=0), _Fake(None, located=False, detected=0))
    assert mapper.map_gaze(None) is None and not mapper.located


def test_backend_is_registered_and_built():
    from src.eye_tracking.gaze_factory import BACKEND_NEON_HYBRID, BACKENDS, NEON_BACKENDS, build_gaze_source
    from src.eye_tracking.neon_hybrid_source import NeonHybridGazeSource

    assert BACKEND_NEON_HYBRID in BACKENDS and BACKEND_NEON_HYBRID in NEON_BACKENDS
    assert isinstance(build_gaze_source(BACKEND_NEON_HYBRID, screen_size=(1920, 1080)), NeonHybridGazeSource)
    with pytest.raises(ValueError):
        build_gaze_source(BACKEND_NEON_HYBRID)


def test_quality_summary_reports_which_locator_mapped_samples():
    from src.eye_tracking.neon_hybrid_source import NeonHybridGazeSource

    source = NeonHybridGazeSource((1920, 1080), device=SimpleNamespace(), screen_mapper=HybridScreenMapper(_Fake(TAG), _Fake(OUTLINE)))
    source._screen_mapper.map_gaze(None)
    summary = source.quality_summary()
    assert summary["source"] == "pupil_labs_neon_screen_hybrid"
    assert summary["samples_mapped_by_tags"] == 1 and summary["samples_mapped_by_outline"] == 0
