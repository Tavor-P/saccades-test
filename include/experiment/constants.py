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

# Whether a session is a quick test/smoke run (fewer trials, saccade phase
# first so you reach the thing you're testing faster) or a real
# data-collection session is a runtime decision, not a constant to edit here
# - see run_experiment.py's _resolve_test_mode(): leaving the startup
# dialog's Participant ID blank means test, entering one means real. The two
# trial-count pairs below are what that choice picks between.
#
# Fixed trial count for PresaccadeSession (baseline/fixation phase) only now
# - ExperimentSession's real saccade block no longer uses this at all, since
# its trial count is open-ended (see ZEST_MIN_VALID_TRIALS_*/MAX_SACCADE_TRIALS_*
# below). 25 either way (not 100 for "real") - that's plenty of data for the
# baseline condition and a shorter baseline keeps total session length down.
NUM_TRIALS_PER_PHASE_TEST = 25
NUM_TRIALS_PER_PHASE_REAL = 25
CATCH_TRIAL_FRACTION = 0.2  # rest drive/query the adaptive staircase, in either mode

# A few throwaway trials before each phase's real block, so a participant
# doesn't learn the response mapping on data that counts. Fixed at a high,
# easily-visible contrast (not drawn from the staircase) and excluded from
# both the ZEST update and the logged CSV/results.
NUM_PRACTICE_TRIALS_TEST = 2
NUM_PRACTICE_TRIALS_REAL = 5
PRACTICE_CONTRAST = 0.25

# Live gaze cursor, opt-in via the startup dialog's "Show gaze indicator"
# toggle (default off - not something a real participant should see) - a
# small faint circle at the participant's current classified gaze zone, for
# checking tracking quality at any point in the saccade phase. Smaller and
# dimmer than the fixation symbols so it never reads as a third target.
GAZE_INDICATOR_RADIUS_RATIO = CIRCLE_RADIUS_RATIO * 0.6
GAZE_INDICATOR_OPACITY = 0.35

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
# saccade onset (debounces classifier noise so a single jittery/misclassified
# frame doesn't get read as onset). This used to be tuned down to 10ms because
# onset detection also *triggered the flash* in the old gaze-contingent
# design, and this value ate directly into how much of a real saccade's short
# (~65ms main-sequence) flight time was still ahead of the flash. That
# tradeoff is gone now that flash timing is open-loop (scheduled off
# avg_reaction_time_ms, independent of real-time detection speed - see
# ExperimentSession) - onset detection is now purely a *measurement*
# (reaction_latency_ms, and via that the reaction-time test's average and the
# in-session rolling average), so accuracy matters more than latency. 10ms
# was too short for that: a brief misclassified-zone blip (lighting flicker,
# tracking glitch) reads as a real saccade onset just as easily as an actual
# one, which is why the reaction-time test could measure something like 40ms
# average against clearly-real onsets over 90ms - not a participant being
# fast, just noise dragging the average down. Raised well above a single
# frame's worth of jitter at this tracker's ~115-145fps (~7-9ms/frame).
# reaction_latency_ms/onset_detection_lag_ms in the logged CSV (see
# TrialResult) let you check empirically whether this is still catching false
# positives - if avg_reaction_time_ms in real sessions is still
# implausibly fast (well under ~100ms), raise this further.
SACCADE_ONSET_STABILITY_MS = 40

# 3-point calibration is run this many times (round 1 discarded, rounds 2-3
# averaged - see src/experiment/calibration.py) on the assumption that a
# participant's first attempt is their least reliable. Used both by
# ExperimentSession's own self-calibration and by the tutorial's calibration
# stage (see src/experiment/tutorial_session.py), so a session that skips the
# tutorial still gets the same improved calibration.
CALIBRATION_ROUNDS = 3

# Pause-menu "Return and Recalibrate" (see ExperimentSession.resume_from_pause)
# runs a faster 2-round recalibration instead of the full CALIBRATION_ROUNDS.
# Unlike the initial calibration, there's no "first attempt is least
# reliable" concern mid-session - the participant is already warmed up - so
# average_calibration_rounds() is called on both rounds (its [-2:] indexing
# already handles any round count, not just 3), with none discarded.
RECALIBRATION_ROUNDS = 2

# Reaction-time-based, open-loop saccade timing (replaces gaze-contingent
# flash triggering): each participant's own saccadic reaction time is
# measured up front (see the RT_TEST_* constants below), then each trial's
# flash is scheduled at a fixed delay from target/beep onset -
# avg_reaction_time_ms + one of these offsets - instead of firing the moment
# the real-time gaze classifier detects onset. This removes real-time
# detection lag as a constraint on flash timing precision, at the cost of
# some trials' flashes landing outside the actual saccade window entirely
# (see flash_during_saccade in include/experiment/types.py) - expected and
# handled by only feeding valid trials to ZEST (see ExperimentSession).
TIMING_OFFSETS_MS = (-40, -20, 0, 20, 40)

# Upfront reaction-time test: repeated beep+target-appear -> saccade-onset
# attempts (no grating, no response), averaged into the initial
# avg_reaction_time_ms. Also re-run (this same attempt count) by "Return and
# Recalibrate". Real saccadic reaction time is genuinely variable trial to
# trial, so a handful of samples is not enough to average out that noise -
# 3 (the old NUM_RT_TEST_TRIALS_TEST) produced unreliable averages in
# practice; 6 is a better floor even for a quick test/smoke run.
NUM_RT_TEST_TRIALS_TEST = 6
NUM_RT_TEST_TRIALS_REAL = 10

# Fallback average reaction time if every RT-test attempt times out (broken
# tracking, participant confusion) - a literature-typical simple saccadic
# reaction time, just enough to let the session proceed rather than divide by
# zero or hang. Should be rare in practice; if this constant is actually
# getting used often, something upstream (tracking quality, calibration) is
# broken and needs investigating, not this number tuned.
DEFAULT_REACTION_TIME_MS = 225.0

# The rolling average of avg_reaction_time_ms recomputes from the most recent
# RT_AVERAGE_ROLLING_WINDOW completed trials' actual detected reaction times,
# every RT_AVERAGE_RECOMPUTE_EVERY completed trials - a continuously running
# counter, never reset (including across a mid-session recalibration, which
# only overwrites the average value itself - see resume_from_pause). Kept as
# two separate names even though equal today, since "how many samples feed
# the average" and "how often it's recomputed" are conceptually distinct
# knobs a future tweak might decouple. Lowered from 10 to 5 - reaction time
# can drift (fatigue, practice effects) meaningfully within 10 trials, so
# updating twice as often keeps the schedule closer to the participant's
# actual current reaction time.
RT_AVERAGE_ROLLING_WINDOW = 5
RT_AVERAGE_RECOMPUTE_EVERY = 5

# Stopping criterion for the main saccade block, replacing a fixed trial
# count now that contrast stays ZEST-adaptive rather than a discrete level
# grid (an efficient, well-estimated threshold is the goal - not full-curve
# coverage). Runs until ZestStaircase.credible_interval(0.68) is narrow
# enough (in log-contrast space - credible_interval returns *linear*
# bounds, so compare log10(hi) - log10(lo), not a raw linear difference) AND
# at least this many valid (flash_during_saccade is True) trials have been
# collected, so a narrow interval isn't trusted before enough data has
# actually shaped it - OR the max-trial safety cap is hit regardless (open-
# loop misses are structural now, not rare lag noise, so this needs real
# headroom above the min-valid floor to guarantee termination without cutting
# a normally-behaving session short).
#
# All three are first guesses, not validated against real session data yet -
# tune once results from an actual run are in.
ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH = 0.1  # ~26% linear ratio between the interval's bounds
ZEST_MIN_VALID_TRIALS_TEST = 6
ZEST_MIN_VALID_TRIALS_REAL = 25
MAX_SACCADE_TRIALS_TEST = 30
MAX_SACCADE_TRIALS_REAL = 200

# Go-cue beep, played the instant the target appears (see run_experiment.py) -
# same tone this project's trial-start warning tone once used (see the
# now-reverted commit 638138f) before being removed as a departure from
# Diamond, Ross & Morrone (2000)'s methodology. Repurposed here with new
# semantics: not a "get ready" warning before an already-visible target, but
# the actual go-cue that anchors the reaction-time measurement itself (the
# target appears and the beep plays at the same instant - see
# NUM_RT_TEST_TRIALS_REAL above).
WARNING_TONE_HZ = 880
WARNING_TONE_DURATION_S = 0.5

# Tutorial's own scoped-down RT-test stage (see TutorialSession), before its
# dress rehearsal. Fixed, not a _TEST/_REAL pair - the whole tutorial is
# already a fixed-size demo regardless of test/real session mode.
TUTORIAL_RT_TEST_ATTEMPTS = 3

# First-timer tutorial (see src/experiment/tutorial_session.py), offered as a
# yes/no dialog toggle before a real saccade-phase session. Runs once, ahead
# of both phases, and hands its calibration to the saccade phase so it isn't
# repeated.

# "Highest" contrast for the two single-orientation key-mapping demo screens
# - these are teaching which key means which orientation, not measuring a
# threshold, so there's no reason to make the grating hard to see.
TUTORIAL_DEMO_CONTRAST = 1.0
# Fixed (not staircased) contrast for the mixed-orientation quiz - comfortably
# above threshold so a participant who understands the key mapping can
# reliably clear the streak.
TUTORIAL_QUIZ_CONTRAST = 0.5
TUTORIAL_QUIZ_STREAK_TARGET = 5  # consecutive correct needed to pass the quiz; a miss resets the streak
TUTORIAL_GAZE_PRACTICE_ATTEMPTS = 5  # fixed count, not gated by success - tracking accuracy is noisy this early
TUTORIAL_DRESS_REHEARSAL_ATTEMPTS = 5  # fixed count; a miss resets contrast back to the top of the staircase's range

# Correctness-feedback flash for the tutorial's quiz/dress-rehearsal stages -
# same style as START_FLASH_COLOR above, just red/green instead of the
# trial-start green.
FEEDBACK_FLASH_RED = [1, -1, -1]
FEEDBACK_FLASH_GREEN = [-1, 1, -1]
FEEDBACK_FLASH_DURATION_FRAMES = 8
