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
#
# At the paper's literal 0.04 cpd, fewer than 1.2 full cycles fit inside the
# Gaussian-windowed patch (cycles = GRATING_SPATIAL_FREQUENCY_CPD *
# GRATING_ENVELOPE_SIGMA_DEG * 6, independent of screen size) - so instead of
# a grating, the patch looks like a single lopsided light/dark gradient, which
# is trivially visible in peripheral vision even at low contrast (peripheral
# acuity is far coarser than foveal, but a single gradient has no fine detail
# to lose). Bumped up so multiple cycles fit in the patch instead - fine
# enough that peripheral vision blurs it toward flat gray, resolvable mainly
# with foveal acuity - but pulled back from an earlier, more aggressive value
# (0.35 cpd, ~10 cycles) that made the ZEST-driven low-contrast trials come
# back essentially invisible even to a directly-fixating eye. This is still a
# deviation from the paper's own value, and its correctness depends on
# VIEWING_DISTANCE_CM/SCREEN_HEIGHT_CM above actually matching your physical
# setup - adjust further if it's still too faint or too easy.
GRATING_SPATIAL_FREQUENCY_CPD = 0.12  # cycles/degree, ~3.5 cycles across the patch
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

# A few throwaway trials before each phase's real block, so a participant
# doesn't learn the response mapping on data that counts. Fixed at a high,
# easily-visible contrast (not drawn from the staircase) and excluded from
# both the ZEST update and the logged CSV/results.
NUM_PRACTICE_TRIALS = 2 if TEST_MODE else 5
PRACTICE_CONTRAST = 0.25

# ZEST (Zippy Estimation by Sequential Testing, King-Smith et al. 1994)
# adaptive contrast staircase - the paper's actual contrast-selection method:
# "their contrast varying according to a ZEST procedure, which independently
# estimated the most informative contrast at which to present the next
# stimulus." Beta/lapse follow the standard Weibull psychometric-function
# assumptions used for staircases on simple detection tasks (Watson & Pelli,
# 1983); the paper doesn't publish these fitting constants itself.

# A contrast of `c` here becomes a peak luminance deviation of `c/2` around the
# mid-grey background (see luminance_to_color / apply_render_state in
# run_experiment.py). On a standard 8-bit panel (256 levels), one quantization
# step is 1/255 of luminance, so anything below `c = 2/255` can't move a pixel
# value at all - it's bit-identical to the background regardless of the
# observer's actual threshold. Floor the staircase one step above that (not
# exactly at it) so the lowest testable contrast is still guaranteed
# renderable rather than sitting right on the rounding boundary. Exposed as
# its own constant (not just inlined below) so anything letting the floor be
# configured - see src/experiment/settings.py - can still enforce this
# physical lower bound no matter what's typed in.
HARDWARE_CONTRAST_FLOOR = 2 / 255
ZEST_LOG_CONTRAST_MIN = math.log10(HARDWARE_CONTRAST_FLOOR) + 0.05  # ~0.0088 (0.88%)
ZEST_LOG_CONTRAST_MAX = math.log10(0.5)
ZEST_GRID_SIZE = 120
ZEST_BETA = 3.5
# The task is a 2-alternative forced choice (report vertical vs horizontal),
# not plain yes/no detection - so chance performance is 50%, not a small
# guess-rate floor. Using anything lower than 0.5 here would systematically
# bias the fitted threshold, since the model would wrongly attribute
# above-floor "guessing" accuracy to genuine detection.
#
# This 0.5 floor is only valid if participants actually guess when unsure
# instead of withholding a response - logged sessions showed "incorrect"
# (a confident wrong guess) is almost never chosen versus "miss" (no
# response), which means real behavior was closer to a 0%, not 50%,
# guess floor. That undershoot reads to ZEST as much stronger evidence of
# "still can't see it" than a true coin-flip would, so it kept pushing
# contrast up past the intended ~82%-correct criterion - hence sessions
# converging to contrasts people then aced. The on-screen instructions now
# explicitly tell participants to guess if unsure, to match this assumption.
ZEST_GUESS_RATE = 0.5
ZEST_LAPSE_RATE = 0.02

# Trial responses: report the grating's orientation via arrow key rather than
# a plain yes/no SPACE press, so accuracy can't be inflated by just always
# pressing "I saw it".
VERTICAL_RESPONSE_KEYS = ("up", "down")
HORIZONTAL_RESPONSE_KEYS = ("left", "right")

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
# Gaze must hold outside the source zone this long before it counts as a real
# saccade onset (debounces classifier noise so the flash doesn't fire on a
# single jittery frame). At the dot/cross separation this experiment uses
# (~20 degrees), a real saccade's flight time is only ~65ms (main-sequence
# estimate), so this value eats directly into how much of that flight time is
# still ahead of the flash - the previous 60ms, then 30ms, both left the flash
# firing well after the eye was already most of the way to the target.
# Dropped further now that the tracker itself runs much faster (~115-145fps,
# up from ~19fps - see the camera/gaze_tracker perf fix) and onset_detection_lag_ms
# in real sessions clusters right at the debounce floor rather than scattering
# across tens of ms, meaning the classifier has a fresh sample well within this
# window rather than being the limiting factor. Still not 0: some minimum
# guards against a single misclassified frame (blink, momentary tracking
# glitch) firing the flash on pure noise. reaction_latency_ms/onset_detection_lag_ms
# in the logged CSV (see TrialResult) let you check empirically whether this
# is catching real saccade onsets or still firing late/early - if it's still
# too late, GAZE_DEAD_ZONE (in eye_tracking/constants.py) is the other lever:
# it gates how far off-center gaze must drift before onset detection even
# starts counting, which isn't visible in onset_detection_lag_ms at all.
SACCADE_ONSET_STABILITY_MS = 10
