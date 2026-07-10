NUM_TRIALS = 20

DOT_POSITION = (0.3, 0.5)  # (xRatio, yRatio), 0-1 fraction of screen, framework-agnostic
CROSS_POSITION = (0.7, 0.5)
SQUARE_POSITION = (
    (DOT_POSITION[0] + CROSS_POSITION[0]) / 2,
    (DOT_POSITION[1] + CROSS_POSITION[1]) / 2,
)

# Sizes as a fraction of window height (resolution-independent); the PsychoPy
# runner converts these + the positions above into its 'height' unit system.
CIRCLE_RADIUS_RATIO = 0.03
SQUARE_SIZE_RATIO = 0.02  # small: detection thresholds drop fast with target size (spatial summation)

BACKGROUND_LUMINANCE = 0.5  # 0=black, 1=white; mid-grey so the flash reads as a small delta, not a stark contrast

SQUARE_FLASH_PROBABILITY = 0.5  # remainder are zero-contrast catch trials
CONTRAST_LEVELS = (0.01, 0.02, 0.03, 0.05, 0.08)  # luminance increment ABOVE the background, near-threshold
SQUARE_DURATION_FRAMES = 1  # a single display refresh, matching the literature's single-frame flash
RESPONSE_WINDOW_MS = 1000

# Jittered foreperiod before each trial's target (go-cue) appears, during which
# only the current fixation symbol is shown. Diamond, Ross & Morrone (2000,
# J Neurosci 20:3449-3455) randomize this foreperiod (they use 800-1200ms) so
# participants can't anticipate the saccade cue.
FOREPERIOD_MIN_MS = 1000
FOREPERIOD_MAX_MS = 2000

SACCADE_TIMEOUT_MS = 15_000  # force-advance a trial if gaze never lands (broken tracking)
GAZE_LANDING_STABILITY_MS = 150  # gaze must hold in the target zone this long to count as "landed"
SACCADE_ONSET_STABILITY_MS = 60  # gaze must hold outside the source zone this long before it counts as a real saccade onset (debounces classifier noise so the flash doesn't fire on a single jittery frame)
