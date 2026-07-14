CAMERA_INDEX = 0  # fallback used if camera probing at startup finds nothing
CAMERA_PROBE_LIMIT = 5  # how many device indices to check when listing available cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
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
