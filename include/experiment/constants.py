import math

DOT_POSITION = (0.3, 0.5)  # (xRatio, yRatio), 0-1 fraction of screen, framework-agnostic
CROSS_POSITION = (0.7, 0.5)
CENTER_POSITION = (0.5, 0.5)  # presaccade-phase fixation point
GRATING_POSITION = (
    (DOT_POSITION[0] + CROSS_POSITION[0]) / 2,
    (DOT_POSITION[1] + CROSS_POSITION[1]) / 2,
)

# Sizes as a fraction of window height (resolution-independent); the PsychoPy
# runner converts these + the positions above into its 'height' unit system.
CIRCLE_RADIUS_RATIO = 0.03

BACKGROUND_LUMINANCE = 0.5  # 0=black, 1=white; mid-grey so the flash reads as a small delta, not a stark contrast

# Diamond, Ross & Morrone (2000, J Neurosci 20:3449-3455) ran on a 35x26cm
# monitor at 50cm viewing distance (38.6x29.1 deg field of view). We don't
# have a physical monitor size at runtime, so these are the same assumptions
# the paper used - edit them to match your actual setup for a more precise
# replication.
VIEWING_DISTANCE_CM = 50.0
SCREEN_HEIGHT_CM = 26.0
SCREEN_HEIGHT_DEG = 2 * math.degrees(math.atan((SCREEN_HEIGHT_CM / 2) / VIEWING_DISTANCE_CM))

# The paper's probe: a horizontal sinusoidal luminance grating (bars parallel
# to the saccade direction, so the saccade's own motion doesn't smear the
# pattern into a spurious signal) windowed by a Gaussian envelope, flashed for
# a single frame.
GRATING_SPATIAL_FREQUENCY_CPD = 0.04  # cycles/degree
GRATING_ENVELOPE_SIGMA_DEG = 4.8  # Gaussian envelope space constant
GRATING_DURATION_FRAMES = 1  # a single frame, exactly as in the paper (their 120Hz CRT -> ~8ms;
# on a typical 60Hz display this is ~16.7ms instead - still "one frame", just a longer one)

GRATING_SF_CYCLES_PER_HEIGHT_UNIT = GRATING_SPATIAL_FREQUENCY_CPD * SCREEN_HEIGHT_DEG
GRATING_ENVELOPE_SIGMA_HEIGHT_UNITS = GRATING_ENVELOPE_SIGMA_DEG / SCREEN_HEIGHT_DEG
# PsychoPy's 'gauss' mask tapers to ~0 by the edge of `size`, at a standard
# deviation of size/6 - so size = 6 * sigma reproduces the paper's space constant.
GRATING_SIZE_HEIGHT_UNITS = GRATING_ENVELOPE_SIGMA_HEIGHT_UNITS * 6

# Flip on for a quick smoke test: fewer trials per phase, saccade phase first
# (so you don't have to sit through the baseline to check the thing you're
# actually changing). Flip back to False for a real data-collection run.
TEST_MODE = True

# Fixed trial counts shared by both phases (presaccade baseline and saccade
# test), so the two conditions can be directly compared afterward. Contrast
# itself is no longer one of a fixed set of levels - see ZEST_* below.
NUM_TRIALS_PER_PHASE = 25 if TEST_MODE else 100
CATCH_TRIAL_FRACTION = 0.2
CATCH_TRIAL_COUNT = round(NUM_TRIALS_PER_PHASE * CATCH_TRIAL_FRACTION)  # rest drive/query the adaptive staircase

# ZEST (Zippy Estimation by Sequential Testing, King-Smith et al. 1994)
# adaptive contrast staircase - the paper's actual contrast-selection method:
# "their contrast varying according to a ZEST procedure, which independently
# estimated the most informative contrast at which to present the next
# stimulus." Beta/guess/lapse follow the standard Weibull psychometric-function
# assumptions used for staircases on simple detection tasks (Watson & Pelli,
# 1983); the paper doesn't publish these fitting constants itself.
ZEST_LOG_CONTRAST_MIN = -3.0  # 0.001
ZEST_LOG_CONTRAST_MAX = math.log10(0.5)
ZEST_GRID_SIZE = 120
ZEST_BETA = 3.5
ZEST_GUESS_RATE = 0.02
ZEST_LAPSE_RATE = 0.02

RESPONSE_WINDOW_MS = 1000

# Diamond, Ross & Morrone (2000, J Neurosci 20:3449-3455): each trial begins
# with a 500ms warning tone, then a foreperiod randomized 800-1200ms during
# which the observer fixates, before the target (go-cue) appears.
FOREPERIOD_MIN_MS = 800
FOREPERIOD_MAX_MS = 1200

# Same paper's data-quality check: they report false-alarm rates <1/200 as
# evidence the observer was reliably attentive, not guessing/spamming responses.
# That bar assumes a much larger sample than a short session gets, so require a
# minimum number of catch trials before claiming "reliable" at all - otherwise
# a small catch-trial count could show 0 false alarms by luck, not evidence.
FALSE_ALARM_RATE_THRESHOLD = 1 / 200
MIN_CATCH_TRIALS_FOR_RELIABILITY = 8  # each phase has 20 catch trials, comfortably above this floor

SACCADE_TIMEOUT_MS = 15_000  # force-advance a trial if gaze never lands (broken tracking)
GAZE_LANDING_STABILITY_MS = 150  # gaze must hold in the target zone this long to count as "landed"
SACCADE_ONSET_STABILITY_MS = 60  # gaze must hold outside the source zone this long before it counts as a real saccade onset (debounces classifier noise so the flash doesn't fire on a single jittery frame)
