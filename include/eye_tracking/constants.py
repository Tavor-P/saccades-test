"""Constants for gaze tracking: Blackfly camera settings, MediaPipe, calibration/zone tuning (GAZE_DEAD_ZONE), and the Pupil Labs Neon
settings (device address, scene size, stale/gap thresholds). Pure data - no logic."""

CAMERA_INDEX = 0  # fallback used if camera probing at startup finds nothing
CAMERA_PROBE_LIMIT = 5  # how many device indices to check when listing available cameras
# Region of interest (found via SpinView) framing a face on the Blackfly S at desk distance; full sensor resolution would only waste bandwidth.
FRAME_WIDTH = 1196
FRAME_HEIGHT = 874
FRAME_OFFSET_X = 132
FRAME_OFFSET_Y = 112
CAMERA_EXPOSURE_TIME_US = 3000.0  # ExposureTime (microseconds), fixed (ExposureAuto = Off)
# Fixed gain (dB), like exposure, so brightness does not hunt mid-experiment. A conservative guess for active IR illumination, NOT validated: check a
# real frame against your illuminator (higher if dim, lower if washed out/noisy) before relying on it.
CAMERA_GAIN_DB = 6.0
CAMERA_FRAME_RATE = 200.0  # target AcquisitionFrameRate (fps); clamped to the camera's actual max
# Optional session video (src/eye_tracking/video_recorder.py), deliberately far below CAMERA_FRAME_RATE. Each written frame gets its own wall-clock
# timestamp in a companion CSV, so playback aligns with the results CSV's grating_shown_at_unix_ms without assuming a nominal frame rate.
VIDEO_RECORDING_FPS = 30.0
# MediaPipe inference time scales with pixel count, so frames are downscaled to this width before tracking to keep pace with the camera.
TRACKING_FRAME_WIDTH = 640
MODEL_PATH = "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Canonical MediaPipe FaceLandmarker indices (478-point mesh, iris-refined).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

# Live classification uses the raw, unsmoothed ratio so onset/landing are not lagged; noise is rejected by the stability windows in the session.
# This window only averages a steady fixation for calibration.
CALIBRATION_SAMPLE_WINDOW = 30
# +/- fraction around the calibrated midpoint treated as CENTER (GazeSample.zone). Zones decide where gaze has LANDED (the far target zone), not
# saccade onset - that is a position displacement (SACCADE_ONSET_DISPLACEMENT). A wider band lands sooner but accepts less precise fixations; too
# narrow makes landing fragile to calibration offset.
GAZE_DEAD_ZONE = 0.10

# Smoothing for GazeSample.smoothed_position, which only drives the gaze indicator; onset/landing use the raw position/zone so they are not lagged.
POSITION_MEDIAN_WINDOW = 5  # median of this many recent raw readings rejects a single stray misdetection outright
POSITION_SMOOTHING = 0.35  # EMA weight given to each new (post-median) reading; lower = smoother but laggier

# --- Pupil Labs Neon glasses (src/eye_tracking/neon_source.py) ---
# Gaze arrives over the local network (Pupil Labs Real-Time API); phone and PC must share a network. None = auto-discover the first Neon.
NEON_DEVICE_ADDRESS: str | None = "192.168.0.60"
NEON_DEVICE_PORT = 8080
NEON_DISCOVERY_TIMEOUT_S = 10.0
# Scene camera size; gaze x / width is the 0-1 "ratio" the calibration fits. Only horizontal gaze is used.
NEON_SCENE_WIDTH = 1600
NEON_SCENE_HEIGHT = 1200
# Samples further apart than this count as a "gap" in the quality summary (~6 missed samples at Neon's 200 Hz).
NEON_GAP_THRESHOLD_MS = 30.0
# No sample for this long = dead stream: latest_sample() reports "no face" instead of repeating the last reading (which looks like a still eye).
NEON_STALE_TIMEOUT_S = 0.5
# Samples are stamped with their capture time (phone clock mapped onto the PC timeline, NeonGazeSource._stamp). A capture time implying more latency
# than this is taken as a bad clock offset and the sample falls back to its arrival time.
NEON_MAX_PLAUSIBLE_LATENCY_MS = 150.0
# Samples kept for samples_since() (~2 s at 200 Hz) and the window of the rolling median behind gaze_latency_ms.
NEON_HISTORY_SAMPLES = 400
NEON_LATENCY_WINDOW_SAMPLES = 200
# The phone-vs-PC clock offset is re-estimated this often (the clocks drift); an estimate noisier than the max std is ignored, a good one blended 50/50.
NEON_CLOCK_REESTIMATE_S = 60.0
NEON_CLOCK_REESTIMATE_MAX_STD_MS = 10.0
# Replayed traces (gaze_trace.py) have no clock offset: capture times come from the device timestamps, assuming this median latency (Neon, Oct 2026).
NEON_REPLAY_LATENCY_MS = 16.0

# --- Neon screen-marker mapping (src/eye_tracking/screen_markers.py, neon_screen_gaze.py) ---
# AprilTags in the four screen corners locate the screen every frame, so gaze is in screen coordinates that do not change with head movement. Sized
# physically and drawn at low contrast with the background's mean luminance, so each corner is a faint texture. Synthetic-render detection rates
# (tag36h11, 300 renders; nominal = 12.6 scene px/cm, 20 deg tilt; harsh = 10 px/cm, 40 deg tilt):
#   old 100 px, black/white tags:  nominal 100%, harsh 0%
#   216 px, contrast 110 (this):   nominal 100%, harsh 94%
#   216 px, contrast 80 / 130:     nominal 100%, harsh 73% / 96%
#   200 px, contrast 110:          nominal 100%, harsh 85%
# Simulated only. If detection is unreliable on real glasses, raise SCREEN_MARKER_CONTRAST first, then SCREEN_MARKER_WIDTH_CM.
SCREEN_MARKER_WIDTH_CM = 6.0  # physical tag width
SCREEN_MARKER_SCREEN_WIDTH_CM = 53.1  # physical width of the display (24in 16:9); None -> use SCREEN_MARKER_PX
SCREEN_MARKER_PX = 216  # fallback tag width in screen px when the physical width is unknown (multiple of 8)
SCREEN_MARKER_MARGIN_CELLS = 1  # quiet zone width in tag cells (tag = 8x8 cells), drawn at the background grey
SCREEN_MARKER_CONTRAST = 110  # 0-255 gap between a tag's white and black cells; their mean is held at the background grey
# Detector downscale (1.0 = full resolution); the library default 2.0 misses small tags.
NEON_TAG_QUAD_DECIMATE = 1.0
# How long the last good screen location is kept after the markers are lost before gaze is reported lost.
NEON_MARKER_HOLD_S = 0.2
# Head rotation faster than this (gyro magnitude, deg/s) counts as "head moving" in the quality summary.
NEON_HEAD_SPEED_LIMIT_DPS = 30.0
NEON_HEAD_SPEED_WINDOW_S = 0.1  # reported head speed = peak over this window

# --- Neon marker-free screen detection (src/eye_tracking/screen_outline_mapper.py) ---
# The monitor is found in each scene frame as a uniform quadrilateral, so no tags are drawn.
NEON_OUTLINE_WORK_SCALE = 0.5  # contours are found on a frame this fraction of full size (speed); edges are line-fitted for sub-pixel corners
NEON_OUTLINE_MIN_AREA_FRACTION = 0.03  # the screen must cover at least this share of the scene frame
NEON_OUTLINE_MIN_FILL = 0.96  # blob area / fitted-quad area; lower = a corner is occluded or the blob is not a rectangle
NEON_OUTLINE_MIN_CONTRAST = 12.0  # grey-level gap between the screen and the ring just outside it
NEON_OUTLINE_BORDER_MARGIN_PX = 2  # a blob touching the frame edge is a screen partly out of view: not usable
# The blob's convex hull must reduce to 4 corners within this fraction of its perimeter; a bigger bite (a hand, a monitor stand) is rejected.
NEON_OUTLINE_MAX_APPROX_FRACTION = 0.03
NEON_OUTLINE_MAX_SIDE_RATIO = 3.5  # opposite quad sides may differ at most this much (perspective sanity)
NEON_OUTLINE_HOLD_S = 0.2  # keep the last good outline this long when detection fails
# Corner smoothing: the per-frame blend grows from MIN_ALPHA (corners still: heavy smoothing) to 1 (moved >= JUMP_PX: real head movement, no lag).
NEON_OUTLINE_SMOOTH_MIN_ALPHA = 0.25
NEON_OUTLINE_SMOOTH_JUMP_PX = 6.0
