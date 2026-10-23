"""Tunable constants for the experiment: stimulus geometry, trial timing, staircase (ZEST) settings, trial counts. Pure data - no logic. Read a
constant's comment before retuning it. Constants are copied by name into importing modules (`from ... import X`), so to change one temporarily in a
test/simulation use tests/ai-tests/sim_tools.override_constants()."""

import math

DOT_POSITION = (0.3, 0.5)  # (xRatio, yRatio), 0-1 fraction of screen
CROSS_POSITION = (0.7, 0.5)
CENTER_POSITION = (0.5, 0.5)  # presaccade-phase fixation point
GRATING_POSITION = (
    (DOT_POSITION[0] + CROSS_POSITION[0]) / 2,
    (DOT_POSITION[1] + CROSS_POSITION[1]) / 2,
)

# Sizes are a fraction of window height (resolution-independent).
CIRCLE_RADIUS_RATIO = 0.03

BACKGROUND_LUMINANCE = 0.5  # 0=black, 1=white; mid-grey so the flash reads as a small delta

# Diamond, Ross & Morrone (2000, J Neurosci 20:3449-3455) used a 35x26 cm monitor at 50 cm (38.6x29.1 deg). There is no physical monitor size at
# runtime, so these are the paper's assumptions - edit them to match your setup.
VIEWING_DISTANCE_CM = 50.0
SCREEN_HEIGHT_CM = 26.0
SCREEN_HEIGHT_DEG = 2 * math.degrees(math.atan((SCREEN_HEIGHT_CM / 2) / VIEWING_DISTANCE_CM))

# The probe: a horizontal sinusoidal grating (bars parallel to the saccade, so its motion doesn't smear the pattern) in a Gaussian envelope,
# flashed for one frame. The paper's 0.04 cpd gives under 1.2 cycles in the patch (cycles = cpd * sigma_deg * 6), i.e. a single gradient that is
# visible peripherally even at low contrast. 0.12 cpd fits ~3.5 cycles, which peripheral vision blurs toward grey; 0.35 cpd (~10 cycles) was
# invisible even when fixated. A deviation from the paper that depends on VIEWING_DISTANCE_CM/SCREEN_HEIGHT_CM being right.
GRATING_SPATIAL_FREQUENCY_CPD = 0.12  # cycles/degree
GRATING_ENVELOPE_SIGMA_DEG = 4.8  # Gaussian envelope space constant
GRATING_DURATION_FRAMES = 1  # one frame as in the paper (~8 ms on their 120 Hz CRT, ~16.7 ms at 60 Hz)

GRATING_SF_CYCLES_PER_HEIGHT_UNIT = GRATING_SPATIAL_FREQUENCY_CPD * SCREEN_HEIGHT_DEG
GRATING_ENVELOPE_SIGMA_HEIGHT_UNITS = GRATING_ENVELOPE_SIGMA_DEG / SCREEN_HEIGHT_DEG
# PsychoPy's 'gauss' mask has a standard deviation of size/6, so size = 6 * sigma reproduces the paper's space constant.
GRATING_SIZE_HEIGHT_UNITS = GRATING_ENVELOPE_SIGMA_HEIGHT_UNITS * 6

# Test vs real session is a runtime choice (blank Participant ID = test; see run_experiment._resolve_test_mode), not a constant. These pairs are
# what it picks between. Fixed trial count for PresaccadeSession only: the saccade block is open-ended (ZEST_* / MAX_SACCADE_TRIALS_* below).
NUM_TRIALS_PER_PHASE_TEST = 25
NUM_TRIALS_PER_PHASE_REAL = 25
CATCH_TRIAL_FRACTION = 0.2  # the rest drive the adaptive staircase

# Throwaway trials before each phase's real block, so the response mapping isn't learned on counted data. Fixed high contrast; not logged, not fed to ZEST.
NUM_PRACTICE_TRIALS_TEST = 2
NUM_PRACTICE_TRIALS_REAL = 5
PRACTICE_CONTRAST = 0.25

# Optional live gaze cursor (startup dialog toggle, off by default): a small faint circle, dimmer than the symbols so it never reads as a third target.
GAZE_INDICATOR_RADIUS_RATIO = CIRCLE_RADIUS_RATIO * 0.6
GAZE_INDICATOR_OPACITY = 0.35

# ZEST adaptive contrast staircase (King-Smith et al. 1994), the paper's contrast-selection method. Beta/lapse are standard Weibull assumptions
# (Watson & Pelli 1983); the paper doesn't publish them.
#
# Contrast c is a peak luminance deviation of c/2 around the grey background. On an 8-bit panel anything below c = 2/255 cannot change a pixel, so the
# floor sits one step above that. Its own constant so a configurable floor (src/experiment/settings.py) can still enforce this bound.
HARDWARE_CONTRAST_FLOOR = 2 / 255
ZEST_LOG_CONTRAST_MIN = math.log10(HARDWARE_CONTRAST_FLOOR) + 0.05  # ~0.0088 (0.88%)
ZEST_LOG_CONTRAST_MAX = math.log10(0.5)
ZEST_GRID_SIZE = 120
ZEST_BETA = 3.5
# 2AFC (vertical vs horizontal), so chance is 50%; a lower guess rate would bias the threshold. This only holds if participants guess when unsure:
# logged sessions showed almost no wrong guesses ("incorrect") against "miss", which read to ZEST as "still can't see it" and drove contrast past the
# ~82%-correct criterion. The on-screen instructions now tell participants to guess.
ZEST_GUESS_RATE = 0.5
ZEST_LAPSE_RATE = 0.02

# Respond with an arrow key for the grating's orientation (not yes/no), so always answering "seen" can't inflate accuracy.
VERTICAL_RESPONSE_KEYS = ("up", "down")
HORIZONTAL_RESPONSE_KEYS = ("left", "right")

RESPONSE_WINDOW_MS = 1000

# Paper: a foreperiod randomized 800-1200 ms of fixation before the target (go-cue) appears.
FOREPERIOD_MIN_MS = 800
FOREPERIOD_MAX_MS = 1200

# Paper's attention check: false-alarm rate under 1/200. Needs a minimum number of catch trials, or 0 false alarms could be luck.
FALSE_ALARM_RATE_THRESHOLD = 1 / 200
MIN_CATCH_TRIALS_FOR_RELIABILITY = 8  # each phase has 20 catch trials

SACCADE_TIMEOUT_MS = 15_000  # force-advance a trial if gaze never lands (broken tracking)
# A flash trial is tracking_ok=False (excluded from ZEST and the graph, like flash_during_saccade != True) if gaze tracking was lost within this window
# either side of the flash: onset/landing times can't be trusted across a dropout.
TRACKING_LOSS_WINDOW_MS = 150
GAZE_LANDING_STABILITY_MS = 150  # gaze must hold in the target zone this long to count as "landed"

# Saccade onset (OnsetDetector): gaze has left the source once GazeSample.position (0=dot, 1=cross) is more than SACCADE_ONSET_DISPLACEMENT from its
# fixation baseline there; the onset is the first sample beyond it. (The old zone-crossing rule noticed only after ~40% of the travel, and a saccade
# lasts 30-35 ms.) Lower = earlier but more jitter false triggers (Neon noise is a few hundredths; screen-mapped backends add spikes). Chosen from
# recorded traces with tests/ai-tests/analyze_onset_threshold.py; sweep with `run_sim.py --sweep SACCADE_ONSET_DISPLACEMENT=0.03,0.05,0.08,0.12`.
SACCADE_ONSET_DISPLACEMENT = 0.08
# Baseline = median of the last ONSET_BASELINE_WINDOW_MS of samples within ONSET_BASELINE_NEAR_SOURCE of the source and not yet past the threshold
# (the movement never drags it along); until the first such sample, the nominal source position (0 or 1).
ONSET_BASELINE_WINDOW_MS = 150
ONSET_BASELINE_NEAR_SOURCE = 0.25
# Gaze must stay beyond the threshold this long (sample capture time) to count: rejects one- and two-sample tracker glitches (10 ms = 3 samples at
# Neon's 200 Hz; it was 40 ms for the ~120 fps webcam). `since` is still the first sample beyond the threshold, so this delays confirmation, not
# the reported onset. Onset detection only measures reaction latency and the flash_during_saccade window (flash timing is open-loop); shorter biases
# latency down, longer up. The robust RT average (RT_* below) is the main defence against blips. If average reaction time comes out well under
# ~100 ms in real sessions, raise this or the threshold. Verify with neon_fixation_test.
SACCADE_ONSET_STABILITY_MS = 10

# 3-point calibration runs this many times; round 1 is discarded as the least reliable, the rest averaged (src/experiment/calibration.py). Used by
# ExperimentSession and the tutorial, so a session that skips the tutorial calibrates the same way.
CALIBRATION_ROUNDS = 3
# Pause-menu "Return and Recalibrate": a faster 2 rounds, both averaged (the participant is already warmed up).
RECALIBRATION_ROUNDS = 2

# Open-loop flash timing: each flash fires at avg_reaction_time_ms + one of these offsets from target onset, independent of the real-time gaze
# classifier, so detection lag doesn't limit precision. Some flashes miss the saccade entirely (flash_during_saccade, types.py); only valid trials
# feed ZEST.
TIMING_OFFSETS_MS = (-40, -20, 0, 20, 40)

# Upfront reaction-time test: beep + target appears -> saccade onset (no grating, no response), averaged into the initial avg_reaction_time_ms.
# Re-run by "Return and Recalibrate". Reaction times are noisy; 3 attempts gave unreliable averages, so 6 is the floor even for a test run.
NUM_RT_TEST_TRIALS_TEST = 6
NUM_RT_TEST_TRIALS_REAL = 10

# Used if every RT-test attempt times out: a typical simple saccadic reaction time. If it's used often, fix tracking/calibration, not this number.
DEFAULT_REACTION_TIME_MS = 225.0

# avg_reaction_time_ms is recomputed from the last RT_AVERAGE_ROLLING_WINDOW detected reaction times every RT_AVERAGE_RECOMPUTE_EVERY completed
# trials (a counter never reset, even by recalibration). Separate names so window size and update rate can be tuned apart. 5 rather than 10 so
# the schedule follows fatigue/practice drift.
RT_AVERAGE_ROLLING_WINDOW = 5
RT_AVERAGE_RECOMPUTE_EVERY = 5

# Robust average (trial_mechanics.robust_average_ms): latencies under RT_MIN_PLAUSIBLE_MS are anticipations/noise and dropped; values more than
# RT_OUTLIER_MAD_K scaled MADs above the median (MAD floored at RT_OUTLIER_MAD_FLOOR_MS) are dropped as lapses/blinks; the result is the median of the rest.
RT_MIN_PLAUSIBLE_MS = 100.0
RT_OUTLIER_MAD_K = 3.0
RT_OUTLIER_MAD_FLOOR_MS = 25.0

# A source's reported gaze latency is clamped to [0, this] before being used to shift sample times, so a bad clock-offset estimate can't move
# onset/landing by seconds.
GAZE_LATENCY_MAX_MS = 150.0

# A final ZEST estimate (or 68% credible interval) within this many log10 units of the contrast grid's ends is flagged in _meta.json: the real
# threshold is probably outside the searchable range.
ZEST_GRID_BOUND_MARGIN_LOG = 0.05

# Stopping rule for the saccade block: stop once the 68% credible interval is narrower than ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH in log10 contrast
# (credible_interval returns linear bounds, so compare log10(hi) - log10(lo)) AND at least *_MIN_VALID_TRIALS valid (flash_during_saccade True)
# trials exist, or when the MAX_SACCADE_TRIALS cap is hit. Open-loop misses are structural, so the cap needs real headroom. First guesses, not yet
# validated on real sessions.
ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH = 0.1  # ~26% linear ratio between the bounds
ZEST_MIN_VALID_TRIALS_TEST = 6
ZEST_MIN_VALID_TRIALS_REAL = 25
MAX_SACCADE_TRIALS_TEST = 30
MAX_SACCADE_TRIALS_REAL = 200

# Go-cue beep, played the instant the target appears (run_experiment.py): it anchors the reaction-time measurement. Same tone the paper used as a
# pre-target warning, with different meaning here.
WARNING_TONE_HZ = 880
WARNING_TONE_DURATION_S = 0.5

# Tutorial (src/experiment/tutorial_session.py) - a fixed-size demo in both test and real mode, run once before a real saccade-phase session; its
# calibration is handed to the saccade phase.
TUTORIAL_RT_TEST_ATTEMPTS = 3
TUTORIAL_DEMO_CONTRAST = 1.0  # key-mapping demo screens teach, not measure: make the grating easy
TUTORIAL_QUIZ_CONTRAST = 0.5  # mixed-orientation quiz: fixed and comfortably above threshold
TUTORIAL_QUIZ_STREAK_TARGET = 5  # consecutive correct needed to pass the quiz; a miss resets the streak
TUTORIAL_GAZE_PRACTICE_ATTEMPTS = 5  # fixed count, not gated by success - tracking is noisy this early
TUTORIAL_DRESS_REHEARSAL_ATTEMPTS = 5  # fixed count; a miss resets contrast to the top of the staircase's range

# Correctness-feedback flash for the tutorial's quiz and dress rehearsal.
FEEDBACK_FLASH_RED = [1, -1, -1]
FEEDBACK_FLASH_GREEN = [-1, 1, -1]
FEEDBACK_FLASH_DURATION_FRAMES = 8
