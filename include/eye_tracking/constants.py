"""Constants for gaze tracking: Blackfly camera settings, MediaPipe, calibration/zone tuning (GAZE_DEAD_ZONE), and the Pupil Labs Neon
settings (device address, scene size, stale/gap thresholds). Pure data - no logic."""

CAMERA_INDEX = 0  # fallback used if camera probing at startup finds nothing
CAMERA_PROBE_LIMIT = 5  # how many device indices to check when listing available cameras
# ROI (region of interest) found via SpinView to frame a face well on the
# Blackfly S at typical desk distance - full sensor res isn't needed and just
# wastes bandwidth/CPU on pixels outside the face.
FRAME_WIDTH = 1196
FRAME_HEIGHT = 874
FRAME_OFFSET_X = 132
FRAME_OFFSET_Y = 112
CAMERA_EXPOSURE_TIME_US = 3000.0  # ExposureTime (microseconds), fixed (ExposureAuto = Off)
# Fixed gain (dB), for the same reason exposure is fixed: no auto-gain hunting
# mid-experiment (brightness shifting as the participant moves would corrupt
# both detection stability and psychometric contrast). Set up for active IR
# illumination rather than ambient room light - this starting value is a
# conservative guess, not a validated one: tune it against your actual
# illuminator via diagnose_camera.py (higher if frames look dim/underexposed,
# lower if they look washed out/noisy) before relying on it.
CAMERA_GAIN_DB = 6.0
CAMERA_FRAME_RATE = 200.0  # target AcquisitionFrameRate (fps); clamped to the camera's actual max
# Optional session video recording (see src/eye_tracking/video_recorder.py) -
# deliberately much lower than CAMERA_FRAME_RATE: a video doesn't need every
# frame the tracker sees to be watchable, and polling at the full acquisition
# rate would mean encoding/writing ~200 frames/sec for no real benefit. Each
# written frame gets its own wall-clock (time.time()) timestamp in a
# companion CSV, so playback can be aligned exactly against the results
# CSV's grating_shown_at_unix_ms - no dead-reckoning from a nominal frame
# rate needed, which is what would actually drift over a long recording.
VIDEO_RECORDING_FPS = 30.0
# MediaPipe FaceLandmarker inference time scales with input pixel count, and
# face/pupil detection doesn't need full sensor resolution - frames are
# downscaled to this width (preserving aspect ratio) before tracking, so
# inference can actually keep pace with the camera's frame rate instead of
# being the pipeline's bottleneck. 640 matches the resolution this project
# was originally tuned/validated at.
TRACKING_FRAME_WIDTH = 640
MODEL_PATH = "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Canonical MediaPipe FaceLandmarker indices (478-point mesh, iris-refined).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

# Live per-frame classification uses the raw ratio (no smoothing) so onset/landing
# detection isn't lagged behind the actual eye movement; noise is instead rejected
# downstream by the onset/landing stability windows in the experiment session.
# This window is only used to average a steady fixation into a calibration baseline.
CALIBRATION_SAMPLE_WINDOW = 30
# +/- fraction around the calibrated midpoint treated as CENTER. This gates
# onset detection just as much as SACCADE_ONSET_STABILITY_MS does, but
# earlier and invisibly - the eye has to travel out of this whole band before
# _tick_trial_active even starts counting the onset-stability window, and
# that travel time doesn't show up in onset_detection_lag_ms at all (it's
# already elapsed by the time that timer starts). Narrowed from 0.15 so the
# eye doesn't have to travel as far before onset detection starts noticing.
# Trade-off: too narrow makes fixational jitter near center misfire as a
# false onset - watch the false_alarm rate (correct_rejection vs false_alarm
# in the CSV) after tightening this, and widen it back if that climbs.
GAZE_DEAD_ZONE = 0.10

# Smoothing for GazeSample.smoothed_position (display only - the gaze
# indicator - never used for onset/landing scoring, which stays on the raw,
# unsmoothed position/zone so it isn't lagged behind real eye movement).
# MediaPipe's per-frame iris landmark reading is noisy frame-to-frame even
# with the eye still, so a faster camera alone doesn't fix a jittery-looking
# indicator - it just samples that noise more often. Values match
# eye_position_manual_test.py's already-validated live gaze-dot demo.
POSITION_MEDIAN_WINDOW = 5  # median of this many recent raw readings rejects a single stray misdetection outright
POSITION_SMOOTHING = 0.35  # EMA weight given to each new (post-median) reading; lower = smoother but laggier

# --- Pupil Labs Neon glasses (src/eye_tracking/neon_source.py) ---
# Neon streams gaze over the local network via the Pupil Labs Real-Time API
# (companion phone + PC must share a network/hotspot). None = auto-discover
# the first Neon on the network; set to e.g. "192.168.1.50" to skip discovery.
NEON_DEVICE_ADDRESS: str | None = "192.168.0.60"
NEON_DEVICE_PORT = 8080
NEON_DISCOVERY_TIMEOUT_S = 10.0
# Scene camera resolution - gaze x is divided by this to give a 0-1 "ratio"
# comparable to the webcam's iris ratio (calibration then fits ratio->position
# exactly as for the webcam). Only horizontal gaze is used by the experiment.
NEON_SCENE_WIDTH = 1600
NEON_SCENE_HEIGHT = 1200
# Gaze samples arriving further apart than this count as a "gap" in the
# session quality summary (Neon streams ~200 Hz, i.e. 5 ms apart, so this is
# roughly 6 missed samples - enough to matter for onset detection).
NEON_GAP_THRESHOLD_MS = 30.0
# If no gaze sample has arrived for this long, the stream is treated as dead:
# latest_sample() reports "no face" instead of repeating the last reading
# forever (which would look like a perfectly still eye).
NEON_STALE_TIMEOUT_S = 0.5
# Each sample is stamped with when the eye was where it says (the phone's capture time
# moved onto the PC's perf_counter timeline), not when it arrived over Wi-Fi - see
# NeonGazeSource._stamp. A capture time implying more latency than this is treated as a bad
# clock offset and the sample falls back to its arrival time.
NEON_MAX_PLAUSIBLE_LATENCY_MS = 150.0
# Recent samples kept for samples_since() (~2 s at 200 Hz) and the window of the rolling median behind gaze_latency_ms.
NEON_HISTORY_SAMPLES = 400
NEON_LATENCY_WINDOW_SAMPLES = 200
# The phone-vs-PC clock offset is re-estimated this often in the background (phone and PC clocks drift apart);
# an estimate noisier than the max std is ignored and a good one is blended 50/50 with the previous one.
NEON_CLOCK_REESTIMATE_S = 60.0
NEON_CLOCK_REESTIMATE_MAX_STD_MS = 10.0
# Replaying a recorded trace (gaze_trace.py) has no clock offset; capture times come from the trace's device
# timestamps, assuming this typical end-to-end latency (measured median on Neon, Oct 2026).
NEON_REPLAY_LATENCY_MS = 16.0

# --- Neon screen-marker mapping (src/eye_tracking/screen_markers.py, neon_screen_gaze.py) ---
# AprilTags in the four screen corners let the scene camera locate the screen
# every frame, so gaze is reported in screen coordinates that don't change when
# the head moves. Sized physically (so they look the same on any monitor) and
# drawn at low contrast with a mean luminance equal to the background, so each
# corner is a faint texture rather than a bright patch. Synthetic-render study
# (pupil_apriltags tag36h11, quad_decimate 1.0, 300 renders per row; "nominal" =
# 12.6 scene px/cm, blur 0.9, noise 3, 20 deg tilt; "harsh" = 10 px/cm, blur 1.3,
# noise 5, 40 deg tilt). Detection rate:
#   old 100 px, black 0 / white 255, light-grey quiet zone: nominal 100%, harsh 0%
#   216 px, contrast 110 (this setting):                    nominal 100%, harsh 94%
#   216 px, contrast 80 / 130:                              nominal 100%, harsh 73% / 96%
#   200 px, contrast 110:                                   nominal 100%, harsh 85%
# Simulated only - not confirmed on real glasses. If detection gets unreliable
# there, raise SCREEN_MARKER_CONTRAST first, then SCREEN_MARKER_WIDTH_CM.
SCREEN_MARKER_WIDTH_CM = 6.0  # physical tag width
SCREEN_MARKER_SCREEN_WIDTH_CM = 53.1  # physical width of the display (24in 16:9); None -> use SCREEN_MARKER_PX
SCREEN_MARKER_PX = 216  # fallback tag width in screen px when the physical width is unknown (multiple of 8)
SCREEN_MARKER_MARGIN_CELLS = 1  # quiet zone width in tag cells (tag = 8x8 cells), drawn at the background grey
SCREEN_MARKER_CONTRAST = 110  # 0-255 gap between a tag's white and black cells; their mean is held at the background grey
# Detector downscale (1.0 = full resolution). The library default of 2.0 is
# fast but misses small tags; we use small tags, so full resolution.
NEON_TAG_QUAD_DECIMATE = 1.0
# How long the last good screen location keeps being used after the markers
# are lost (a missed detection frame or two) before gaze is reported as lost.
NEON_MARKER_HOLD_S = 0.2
# Head rotation faster than this (gyro magnitude, deg/s) counts as "head
# moving" in the session quality summary / GazeSample.head_speed_dps.
NEON_HEAD_SPEED_LIMIT_DPS = 30.0
NEON_HEAD_SPEED_WINDOW_S = 0.1  # reported head speed = peak over this window

# --- Neon marker-free screen detection (src/eye_tracking/screen_outline_mapper.py) ---
# The monitor itself is found in each scene frame as a bright/dark quadrilateral
# (the experiment screen is a uniform mid-grey rectangle), so no tags are drawn.
NEON_OUTLINE_WORK_SCALE = 0.5  # contours are found on a frame this fraction of full size (speed); edges are line-fitted for sub-pixel corners
NEON_OUTLINE_MIN_AREA_FRACTION = 0.03  # the screen must cover at least this share of the scene frame
NEON_OUTLINE_MIN_FILL = 0.96  # blob area / fitted-quad area; lower = a corner is occluded or the blob is not a rectangle
NEON_OUTLINE_MIN_CONTRAST = 12.0  # grey-level gap between the screen and the ring just outside it
NEON_OUTLINE_BORDER_MARGIN_PX = 2  # a blob touching the frame edge is a screen partly out of view: not usable
# The blob's convex hull must reduce to 4 corners within this tolerance (fraction of its
# perimeter); a bigger bite out of a corner (a hand, a monitor stand) is rejected, not
# silently turned into a quad with a cut corner.
NEON_OUTLINE_MAX_APPROX_FRACTION = 0.03
NEON_OUTLINE_MAX_SIDE_RATIO = 3.5  # opposite quad sides may differ at most this much (perspective sanity)
NEON_OUTLINE_HOLD_S = 0.2  # keep the last good outline this long when detection fails
# Corner smoothing: per-frame blend factor grows from MIN_ALPHA (corners still:
# heavy smoothing, so detection noise does not move the mapping) to 1 (corners
# moved >= JUMP_PX: a real head movement, follow it with no lag).
NEON_OUTLINE_SMOOTH_MIN_ALPHA = 0.25
NEON_OUTLINE_SMOOTH_JUMP_PX = 6.0
