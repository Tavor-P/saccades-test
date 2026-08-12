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
