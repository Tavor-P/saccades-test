"""What ExperimentSession is built from, besides its gaze source, logger and clock: the run's settings (`SessionConfig`,
mostly from the startup dialog) and what a completed TutorialSession measured (`TutorialHandoff`). Plain values; None =
the constants.py default, resolved by the session."""

from dataclasses import dataclass

from src.experiment.geometry import Geometry


@dataclass(frozen=True)
class SessionConfig:
    test_mode: bool = False  # SACCADE_FIXED_TRIALS_TEST and the short RT test / practice counts
    show_gaze_indicator: bool = False
    skip_practice_trials: bool = False
    gaze_latency_ms: float | None = None  # fallback when the source reports no latency of its own
    fixed_contrasts: tuple[float, ...] | None = None  # None = SACCADE_FIXED_CONTRASTS
    geometry: Geometry | None = None  # None = Geometry.default()
    flash_spread_sd_ms: float | None = None  # None = FLASH_SPREAD_SD_MS
    return_jump: bool = False
    onset_trigger: bool = False  # ONSET_TRIGGER_FRACTION of the bell-curve trials flash at the live saccade onset


@dataclass(frozen=True)
class TutorialHandoff:
    """A completed tutorial's results: the session then skips its own calibration, RT test and practice trials. An empty
    list means "measured, but nothing usable" (-> DEFAULT_REACTION_TIME_MS); None means "not measured"."""

    calibration_ratios: tuple[float, float, float] | None = None
    rt_samples_ms: list[float] | None = None
    return_rt_samples_ms: list[float] | None = None
