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
CAMERA_FRAME_RATE = 200.0  # target AcquisitionFrameRate (fps); clamped to the camera's actual max
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
GAZE_DEAD_ZONE = 0.15  # +/- fraction around the calibrated midpoint treated as CENTER

# Smoothing for GazeSample.smoothed_position (display only - the gaze
# indicator - never used for onset/landing scoring, which stays on the raw,
# unsmoothed position/zone so it isn't lagged behind real eye movement).
# MediaPipe's per-frame iris landmark reading is noisy frame-to-frame even
# with the eye still, so a faster camera alone doesn't fix a jittery-looking
# indicator - it just samples that noise more often. Values match
# eye_position_manual_test.py's already-validated live gaze-dot demo.
POSITION_MEDIAN_WINDOW = 5  # median of this many recent raw readings rejects a single stray misdetection outright
POSITION_SMOOTHING = 0.35  # EMA weight given to each new (post-median) reading; lower = smoother but laggier
