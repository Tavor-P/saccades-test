"""Thresholds of the post-hoc analysis (analyze_session, flash_gates, the per-leg gating and the suppression index), in one place.
Pure data - no logic. Units: gaze x is a 0-1 ratio of the scene width, so speeds are ratio/s and amplitudes ratio. Every value is
written into each report header and into summary.json `analysis_thresholds` (analysis_report.thresholds_record), so a result
always says which rules made it. Changing one changes every re-analysed result: rerun the analysis of every run afterwards."""

# -- saccade detection (saccade_detection.detect_saccades) -------------------------------------------------------------------
SPEED_HIGH = 2.0  # a movement counts once its 3-sample-smoothed speed exceeds this (ratio/s)
SPEED_LOW = 0.7  # ...and its onset/offset are where the speed falls back under this (ratio/s)
MIN_AMPLITUDE = 0.12  # only movements of at least this (ratio of the width) are saccades; the CLI's --min-amplitude overrides it

# -- time windows around onset (the report's near/beyond/far rows; the index's fallback when no gates are given) -------------
NEAR_ONSET_MS = 50.0  # "near onset" = flash within this many ms of a saccade onset
FAR_MS = 300.0  # flashes at least this far from every onset are the fixation control
BIN_EDGES_MS = (-200, -150, -100, -50, 0, 50, 100, 150, 200)  # accuracy-vs-time bins of the report and plot (ms from onset)

# -- pre-specified flash gates (flash_gates.py, after Braun, Schutz & Gegenfurtner 2021) -------------------------------------
FIXATION_EXCLUSION_MS = 100.0  # a fixation flash must be this far from every saccade (Braun et al.: +-100 ms)
TRACKING_GAP_MS = 100.0  # no gaze samples for this long around a flash = blink / tracking loss
LATENCY_RANGE_MS = (100.0, 500.0)  # saccade latency from the go-cue; faster = anticipation, slower = lapse
AMPLITUDE_RANGE_REL = (0.5, 1.5)  # saccade amplitude relative to the session median (Braun: 7.5-15 deg for 10)
MIN_FLASH_SPEED = 2.0  # ratio-of-width/s at the flash: it must have been drawn during fast movement
MAX_FLASH_SPEED = 25.0  # above this the trace is a spike, not an eye (Braun: 500 deg/s)
RETURN_LATENCY_MAX_MS = 500.0  # return onset at most this long after the jump; no lower bound: predictive returns (< 0) are kept

# -- return-jump runs: finding each trial's two saccades (leg_gating.find_trial_saccades) ------------------------------------
OUT_SEARCH_S = (-0.2, 1.5)  # the out saccade: the first one with onset in [go cue - 0.2 s, go cue + 1.5 s]
RETURN_SEARCH_AFTER_JUMP_S = 1.0  # the return saccade starts at most this long after the jump (and before the next go cue)
RETURN_MIN_AMPLITUDE_REL = 0.5  # ...and moves back toward the start by >= this x the out amplitude (an overshoot correction is not a return)

# -- suppression index (suppression_stats.suppression_index) -----------------------------------------------------------------
MIN_FIT_TRIALS = 12  # fewer flashes than this in a condition -> no threshold fit (the dashboard's quality box reads it)
MIN_ACCURACY_TRIALS = 5  # ... and fewer than this -> not even the accuracy fallback
CHANCE = 0.5  # 2AFC guess rate
LAPSE = 0.02  # small lapse rate in the psychometric function
THRESHOLD_EXTRAPOLATION = 2.0  # a fitted threshold more than this factor outside the tested contrasts counts as not identified
BOOT_MIN_IDENTIFIED_SHARE = 0.75  # the bootstrap CI needs at least this share of the resamples to give an identified fit...
BOOT_MIN_IDENTIFIED = 200  # ...and at least this many of them
ACCURACY_FLOOR_MARGIN = 0.1  # accuracy fallback refused when fixation accuracy is below CHANCE + this (flashes not visible at all)
ACCURACY_CEILING = 0.95  # both conditions at or above this accuracy -> noted: the accuracy index cannot show suppression

# -- secondary suppression index (flash_gates.onset_window_gates, suppression_stats.onset_window_index) ----------------------
# Pre-specified beside the headline (saccade-gate) index: flashes within +-this of their OWN leg's saccade onset (inclusive), moving
# or not (suppression starts ~50 ms before onset: Diamond, Ross & Morrone 2000), against the same fixation-gate flashes. Every other
# saccade gate applies (tracking gap, own leg, forced jump, latency, amplitude, MAX_FLASH_SPEED); only MIN_FLASH_SPEED is dropped.
# Must stay below FIXATION_EXCLUSION_MS, so no flash can be in both conditions.
ONSET_WINDOW_MS = 50.0

# -- time-course figure (timecourse.py, analysis_output.plot_timecourse_figure) ----------------------------------------------
TIMECOURSE_SIGMA_MS = 20.0  # Gaussian kernel SD of the accuracy-vs-time curve (Nadaraya-Watson); 15 / 20 / 25 were compared in the simulator
TIMECOURSE_STEP_MS = 5.0  # the curve is evaluated every this many ms...
TIMECOURSE_RANGE_MS = (-150.0, 200.0)  # ...from the first to the last of these (ms from the flash's own saccade onset)
TIMECOURSE_MIN_EFFECTIVE_N_DRAWN = 10.0  # kernel effective n (Kish) below this: the curve is not drawn there
TIMECOURSE_MIN_EFFECTIVE_N_SOLID = 20.0  # from the line above up to this: drawn grey and dashed ("low"); this or more: solid

# -- companion numbers next to the index (relative_suppression.py) -----------------------------------------------------------
STRATUM_MIN_SACCADE_FLASHES = 5  # a contrast enters the stratified accuracies only with at least this many saccade-gate flashes
RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES = 15  # fewer saccade-gate flashes in total: no relative suppression S, counts only
RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY = 0.65  # stratified fixation accuracy below this: no S (the line is not clearly visible even without a saccade)

# -- gaze timeline figure (gaze_timeline.py, gaze_timeline.png next to plot.png) -------------------------------------------------
GAZE_TIMELINE_BIN_MS = 10.0  # the gaze trace is drawn as the mean x of every bin of this width
GAZE_TIMELINE_ROW_S = 20.0  # each row of the stacked figure covers this many seconds
GAZE_TIMELINE_MAX_ROWS = 80  # a longer run shows only its first rows (the figure keeps readable rows instead of squashing them)
