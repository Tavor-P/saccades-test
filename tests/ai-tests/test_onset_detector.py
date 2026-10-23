"""OnsetDetector: the displacement rule (gaze moved SACCADE_ONSET_DISPLACEMENT away from its fixation baseline at the source)."""

import pytest

from include.eye_tracking.types import GazeSample, GazeZone
from include.experiment.constants import ONSET_BASELINE_WINDOW_MS, SACCADE_ONSET_DISPLACEMENT, SACCADE_ONSET_STABILITY_MS
from src.eye_tracking.gaze_mapping import GazeMapper
from src.experiment.trial_mechanics import OnsetDetector

DT = 0.005  # 200 Hz


def sample(position, t=0.0, face=True):
    if not face or position is None:
        return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=t)
    return GazeSample(zone=GazeMapper.classify(position), ratio=0.5, face_found=True, timestamp=t, position=position)


def feed(detector, positions, start=0.0):
    """Feeds positions at 200 Hz; returns the capture time of the sample that confirmed, or None."""
    for i, p in enumerate(positions):
        t = start + i * DT
        if detector.update(sample(p, t), t):
            return t
    return None


def test_fixation_noise_never_fires():
    d = OnsetDetector(GazeZone.LEFT)
    noise = [0.02 * (-1) ** i for i in range(400)]  # +-0.02 jitter for 2 s
    assert feed(d, [0.05 + n for n in noise]) is None
    assert d.since is None and not d.confirmed


def test_fires_when_position_moves_away_and_reports_the_first_sample_beyond_the_threshold():
    d = OnsetDetector(GazeZone.LEFT)
    fixation = [0.02] * 40
    move = [0.02 + SACCADE_ONSET_DISPLACEMENT + 0.01 + 0.1 * k for k in range(10)]
    confirmed_at = feed(d, fixation + move)
    first_beyond = 40 * DT
    assert d.since == pytest.approx(first_beyond)
    assert confirmed_at == pytest.approx(first_beyond + SACCADE_ONSET_STABILITY_MS / 1000, abs=DT / 2)
    assert d.confirmed


def test_a_fixation_offset_from_the_symbol_is_not_an_onset():
    """Calibration offsets of +-0.1 around the dot are normal; the baseline is the fixation itself, not the nominal 0."""
    for offset in (0.10, -0.12):
        d = OnsetDetector(GazeZone.LEFT)
        assert feed(d, [offset] * 100) is None
        assert feed(d, [offset + SACCADE_ONSET_DISPLACEMENT + 0.05] * 10, start=0.5) is not None


def test_right_hand_source_fires_on_movement_to_the_left_only():
    d = OnsetDetector(GazeZone.RIGHT)
    assert feed(d, [0.98] * 40 + [1.1] * 20) is None  # further right of the cross is not leaving it
    assert feed(d, [0.98 - SACCADE_ONSET_DISPLACEMENT - 0.02] * 10, start=1.0) is not None


def test_moving_past_the_source_away_from_the_target_does_not_fire():
    d = OnsetDetector(GazeZone.LEFT)
    assert feed(d, [0.0] * 40 + [-0.3] * 20) is None


def test_a_one_sample_spike_is_debounced():
    d = OnsetDetector(GazeZone.LEFT)
    assert feed(d, [0.02] * 40 + [0.9] + [0.02] * 40) is None
    assert d.since is None


def test_lost_tracking_resets_the_candidate_and_the_baseline():
    d = OnsetDetector(GazeZone.LEFT)
    d.update(sample(0.02, 0.0), 0.0)
    d.update(sample(0.3, DT), DT)  # beyond the threshold: candidate onset
    assert d.since == DT
    d.update(sample(None, 2 * DT, face=False), 2 * DT)
    assert d.since is None and not d.confirmed
    # after the blink the baseline restarts from the first near-source sample
    assert feed(d, [0.12] * 40, start=1.0) is None


def test_slow_drift_does_not_fire_but_a_saccade_starting_from_the_drifted_position_does():
    d = OnsetDetector(GazeZone.LEFT)
    drift = [0.1 * (i * DT) for i in range(400)]  # 0.1 position units per second for 2 s
    assert feed(d, drift) is None
    end = drift[-1]
    assert feed(d, [end + 0.2] * 10, start=2.0) is not None


def test_the_baseline_window_forgets_old_fixation():
    d = OnsetDetector(GazeZone.LEFT)
    # fixation at 0.0 then (slowly, 0.01 per 100 ms) at 0.2: baseline follows the new level and never fires
    slow = [min(0.2, 0.0001 * i) for i in range(2000)]
    assert feed(d, slow) is None


def test_starting_far_from_the_source_counts_as_already_left():
    d = OnsetDetector(GazeZone.LEFT)
    assert feed(d, [0.6] * 10) is not None  # e.g. the participant looked away before the cue
    assert d.since == 0.0


def test_samples_without_a_position_are_treated_as_lost():
    d = OnsetDetector(GazeZone.LEFT)
    s = GazeSample(zone=GazeZone.RIGHT, ratio=None, face_found=True, timestamp=0.0)
    assert not d.update(s, 0.0) and d.since is None


def test_triggers_before_the_old_zone_rule_would_on_a_realistic_saccade():
    """A smooth 30 ms saccade from the dot to the cross: the zone rule (outside LEFT zone => position >= 0.4) sees it
    after ~40% of the travel; the displacement rule after SACCADE_ONSET_DISPLACEMENT."""
    n = 6
    saccade = []
    for k in range(1, n + 1):
        u = k / n
        saccade.append(u * u * (3 - 2 * u))  # smoothstep 0 -> 1 in 6 samples (30 ms)
    d = OnsetDetector(GazeZone.LEFT)
    feed(d, [0.01] * 40 + saccade + [1.0] * 10)
    onset_index = round((d.since - 0.0) / DT) - 40
    first_outside_zone = next(i for i, p in enumerate(saccade) if GazeMapper.classify(p) is not GazeZone.LEFT)
    assert d.confirmed and onset_index <= first_outside_zone


def test_the_baseline_window_constant_is_a_few_samples_wide():
    assert ONSET_BASELINE_WINDOW_MS / 1000 > 3 * DT
