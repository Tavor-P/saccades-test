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
# +/- fraction around the calibrated midpoint treated as CENTER (the GazeSample.zone
# classification). Zones decide where gaze has LANDED (GAZE_LANDING_STABILITY_MS, the
# target zone) and the gaze indicator / calibration checks. They no longer decide
# saccade ONSET: that is a displacement of GazeSample.position from the fixation baseline
# (SACCADE_ONSET_DISPLACEMENT in include/experiment/constants.py). Under the old zone-
# crossing onset rule CENTER counted as "left the source", so a narrower band fired
# earlier - that no longer applies. Landing needs gaze inside the far target zone, so a
# wider band lands sooner but accepts less precise fixations; too narrow makes landing
# fragile to calibration offset (the target zone is the 0.4 of travel beyond the band).
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
# Also keep: traces of old sessions recorded with the removed screen-mapped backends store screen-ratio x * this width
# in `x_px`, and analyze_session / ReplayGazeSource still divide by it to read them.
NEON_SCENE_WIDTH = 1600
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
