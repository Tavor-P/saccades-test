"""Constants for gaze tracking: calibration/zone tuning (GAZE_DEAD_ZONE), position smoothing, the head monitor and the Pupil Labs Neon
settings (device address, scene size, stale/gap thresholds). Pure data - no logic."""

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
# GazeSample.position scale: 0 = the dot (left target), 1 = the cross (right target); calibration maps the ratio onto it.
LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION = 0.0, 0.5, 1.0

# Smoothing for GazeSample.smoothed_position (display only - the gaze
# indicator - never used for onset/landing scoring, which stays on the raw,
# unsmoothed position/zone so it isn't lagged behind real eye movement).
# The raw per-sample position is noisy frame-to-frame even with the eye still
# (values carried over from the earlier webcam tracker's validated gaze-dot demo).
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
# that the 3-point calibration fits to a 0=dot..1=cross position. Only horizontal gaze is used by the experiment.
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
# -- Head-movement monitor (Neon IMU; src/eye_tracking/head_monitor.py). It only FLAGS head movement, it never corrects gaze.
# ALL of these values are untested defaults, not validated on glasses: check head_motion.csv / session.json "head_motion" after the first run.
# IMU is polled at this rate on its own thread (the device keeps only the newest ~110 Hz sample, so this is a subsample).
HEAD_POLL_HZ = 30.0
# Seconds of IMU history kept for moved_since()/max_speed_between() (a trial is well under this).
HEAD_BUFFER_S = 10.0
# A flash whose +-TRACKING_LOSS_WINDOW_MS window contains head angular speed above this (deg/s) gets tracking_ok=False.
# A still head on a chin rest reads well under 5 deg/s (gyro noise ~0.1-1); 25 only trips on a real nudge or turn.
HEAD_MAX_SPEED_DPS = 25.0
# Head rotation (deg, total angle) away from the orientation at calibration that triggers a console/spoken "keep your head still".
# Roughly the gaze error a rotation of this size adds to the scene-camera gaze x; the dot-cross span is ~41 deg.
HEAD_DRIFT_WARN_DEG = 3.0
# At most one warning per this many seconds, and only between trials.
HEAD_WARN_MIN_INTERVAL_S = 120.0
HEAD_WARN_TEXT = "Please keep your head still."
# No IMU datum within this many seconds of starting the poll: the monitor is switched off ("unavailable") with one warning.
HEAD_NO_DATA_TIMEOUT_S = 5.0
# Consecutive receive_imu_datum exceptions before the monitor switches itself off.
HEAD_MAX_CONSECUTIVE_ERRORS = 5
# Cap on the samples kept for head_motion.csv (30 Hz x 2 h).
HEAD_MAX_LOGGED_SAMPLES = 216_000
