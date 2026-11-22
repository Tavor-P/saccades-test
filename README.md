# saccades-test

A PsychoPy experiment that measures **saccadic suppression**: around the time of a rapid eye movement (a *saccade*) we become much less sensitive to brief, faint flashes.
The participant looks at a dot, moves their gaze to a second dot when it swaps, and says whether a briefly flashed thin line appeared above or below that target.
Gaze comes from Pupil Labs Neon glasses. This is `main` (line flash); the `grating` branch flashes a grating instead. Developer docs: [CLAUDE.md](CLAUDE.md).

**Status.** A *feasibility pilot (n = 1)* toward a grant comparing saccadic suppression in people with and without traumatic brain injury (TBI). The pipeline is built; no group
result exists. The line stimulus, contrasts, tutorial, results layout, phone video and head monitor have been run **only against simulators, fake devices and tests, not yet on real glasses**
(see [Known limitations](#known-limitations)).

## First-time setup

Once per computer.

1. **Hardware:** a Windows PC, a monitor (60 Hz), a chin rest (the Neon scene camera moves with the head, so calibration assumes a still head), Neon glasses, and the Neon Companion phone on the same
   Wi-Fi/hotspot as the PC.
2. Install [PsychoPy Standalone](https://www.psychopy.org/download.html) (it bundles its own Python). Run Python only through the wrapper `.\scripts\py.ps1` (PowerShell) or `scripts/py.sh` (Git Bash).
3. Install dependencies:
   ```
   .\scripts\py.ps1 -m pip install -r requirements.txt
   ```
4. Start the Neon Companion app on the phone. The experiment looks for the Neon at `NEON_DEVICE_ADDRESS` in `include/eye_tracking/constants.py` (set it to your phone's IP, or `None` to auto-discover).
5. **Measure your screen** (once per monitor and seat position): the visible picture's height (cm) and the eye-to-screen distance (cm). You type them into the startup dialog; the line's offset
   (2 deg) and thickness, and the saccade size in degrees, follow them. Until you do, the dialog warns that the values are placeholders (26 cm / 50 cm).

You can double-click `run_experiment.bat`, `analyze_latest.bat` or `run_tests.bat`. Typing `scripts\py.ps1` into Command Prompt or Explorer opens it in an editor instead; from any shell use
`powershell -ExecutionPolicy Bypass -File scripts\py.ps1 -m src.experiment.run_experiment`.

## Run day

Printable version: [docs/experimenter_checklist.md](docs/experimenter_checklist.md).

1. Glasses on the participant (clean lenses), Neon Companion app open (same Wi-Fi as the PC, a wearer and workspace selected), head on the chin rest.
2. Double-click `run_experiment.bat`.
3. In the startup dialog enter **screen height** and **viewing distance**, Mode **Real**, a **Participant ID** (a study code such as `P01`, never a name), and **Run tutorial** = Yes for a first-timer
   (suggested automatically for an ID with no folder under `data/`). Click OK.
4. Read the participant script below, then follow the narrated tutorial (SPACE advances; Esc quits and keeps the data).
5. At the end the console and `RUN_SUMMARY.txt` in the run folder show what happened and the next commands; SPACE closes the window.
6. Export the scene video from the phone and attach it (see [Glasses video](#glasses-video)).
7. Analyse: dashboard **Analyse** button, or `analyze_latest.bat`; after several runs of one person, `analyze_participant.bat`.

> **Participant script.** "Look at the dot and press SPACE when you are ready. After a moment, with a beep, it jumps to the other side of the screen: look at it as fast as you can.
> Around then a very faint thin line flashes for an instant above or below it. Press UP if it was above, DOWN if below, and guess if you are not sure.
> You can rest between trials: nothing happens until you press SPACE."

### The startup dialog

| Field | Meaning |
|---|---|
| Participant ID | required for Real; blank in Test becomes `anonymous` |
| Mode | **Real** (150 trials) or **Test** (30 trials, debug text top-left) |
| Run tutorial | first-timer on-ramp; defaults to Yes for an ID without a folder under `data/` |
| Screen height (cm) / Viewing distance (cm) | measure them (5-100 / 20-300; decimal comma accepted); saved in `session.json` |
| *Advanced*: Saccade contrast min / max (%) | the fixed contrasts, spaced between these (defaults 4 and 8: levels 4, 6, 8) |
| *Advanced*: Flash timing spread (ms) | width (SD) of the bell curve of flash times around the average reaction time; default 80 (10-250) |
| *Advanced*: Show gaze indicator | debug aid |

Click a field's name or its **?** for a help bubble. The last values are remembered; an invalid entry shows a message and keeps what you typed.

### What happens in a run

1. **Tutorial** (optional, narrated and captioned): intro, calibration, a how-to-answer quiz (line above/below a dot, 3 correct in a row) and 8 practice trials that seed the reaction-time average. No feedback on answers.
2. **Calibration**: look steadily at each of 3 targets, 3 rounds (without the tutorial, a short reaction-time test and practice follow).
3. **Green sync flashes**: the screen flashes green three times before the first trial and once at the end (announced aloud; normal). They align the phone's video to the PC clock.
4. **Trials** (150 Real, 30 Test). The participant presses **SPACE** to start each trial (the wait is their break). 0.5-1 s later the dot swaps to the cross with a beep and they look to it as fast as they can.
   A thin line flashes for one frame 2 deg above or below the target, at a random time on a bell curve centred on the average reaction time (0-500 ms after the swap), plus some control flashes
   (about 2 in 11) 400-500 ms after it, when the eye has already landed. They press **Up** or **Down** within 1 s; a missed flash is still guessed. Every trial flashes.
5. **End.** A plain-language summary and the `analyze_session` command are printed. Fixation data come from the control flashes and far-from-saccade flashes of the same trials, sorted after the fact.

The phone records the scene video during the trials; a failure there never stops the session. `Esc` quits at once.

### Head movement

The Neon's IMU is polled at ~30 Hz while trials run. It **flags, it does not correct** (the chin rest remains the plan):
- a flash within `TRACKING_LOSS_WINDOW_MS` of a head-speed spike (> `HEAD_MAX_SPEED_DPS`) gets `tracking_ok = no` and is excluded;
- if the head turns more than `HEAD_DRIFT_WARN_DEG` from the calibration orientation, "Please keep your head still." is spoken between trials (at most once per `HEAD_WARN_MIN_INTERVAL_S`);
- `session.json` gets a `head_motion` section and the run folder a downsampled `head_motion.csv`.

No IMU or an error switches it off with one console warning. Thresholds, rate and clock alignment are **not validated on glasses** yet.

## Output

Everything is under `data/` (git-ignored): one folder per participant, one subfolder per run.

```
data/<participant id>/<YYYY-MM-DD_HH-MM-SS>[_test]/
  trials.csv           one row per trial
  session.json         metadata
  gaze_trace.csv       raw Neon gaze with the phone's capture times
  head_motion.csv      IMU head speed and drift (only if the IMU worked)
  RUN_SUMMARY.txt      plain-language end-of-run summary
  analysis/            report.txt, plot.png, flashes.csv, suppression.json (analyze_session)
  glasses/             the phone's video + video_sync.json + flash_video_times.csv (import_glasses_video)
data/<participant id>/analysis/   report.txt, plot.png, flashes.csv, summary.json, all runs pooled (analyze_participant)
data/participants.json            group labels
data/study_summary.csv + .txt    one row per participant (study_summary)
data/comparisons/<date-time>_<A>_vs_<B>/   report.txt, comparison.csv, comparison.json, plot.png (compare; the dashboard uses data/comparisons/dashboard/)
```

Test-mode runs (`_test`) are left out of every pooled analysis, comparison and study summary unless you ask (`--include-test`, or **Include test runs** in the dashboard).

`trials.csv` columns (booleans `yes`/`no`; practice trials not written): `phase`, `trial`, `contrast_pct`, `flash_shown`, `flash_during_saccade` (live, unreliable), `answered`, `correct`,
`outcome` (correct / wrong / no_answer / detected), `line_side` (above / below), `answer_side`, `flash_ms_after_go_cue`, `reaction_ms`, `answer_ms`, `flash_unix_ms`, `tracking_ok`.
`session.json` sections: `participant`, `run`, `settings`, `hardware`, `results`, `gaze_quality` (rate, gaps, latency, clock offset), `head_motion`, `ignored_space_presses`, `glasses_video`.

Older sessions (loose `data/results_<timestamp>.csv`, `_meta.json`, `vertical`/`horizontal` sides from the grating version) are never rewritten and are still read.

## Analysing your data in 3 clicks

For anyone who does not program. The numbers are exploratory (feasibility pilot).

1. **Start the dashboard.** In a terminal in the project folder run `.\scripts\py.ps1 -m src.dashboard.app`, then open <http://localhost:5000>. It runs only on this computer (Ctrl+C closes it).
2. **Click Analyse.** Each run has an **Analyse** button; the **Participants** table has one per person, giving the **pooled result** of all their runs (one run has too few flashes for a stable index).
   Tick **Include test runs** only if you want test-mode runs counted. The page stays blank for a few seconds.
3. **Read the page.** The result in one sentence, the plot, then the **data-quality box** (a green / amber / red light per check, rule written next to it). `flashes.csv`, `summary.json` and `report.txt` can be downloaded.

**Comparing people** (two or more participants):
1. In the **Participants** table type a **Group** for each person (e.g. `TBI`, `control`); it saves when you click out.
2. Open **Compare**, tick people in **Set A** and **Set B** (or use the quick picks), press **Compare**. The mode (1 vs 1, 1 vs group, group vs group) follows from who is ticked.
3. Open **Study summary** for one row per participant and a `study_summary.csv` download for Excel, SPSS or R.

**Without the dashboard**, double-click:

| Launcher | Does | Output |
|---|---|---|
| `analyze_latest.bat` | analyses the newest run | `analysis/` in that run folder |
| `analyze_participant.bat` | asks for an ID, pools all their runs | `data/<participant id>/analysis/` |
| `compare.bat` | asks for Set A and Set B (e.g. `P01` and `group:control`; set labels first with `participants set`) | `data/comparisons/...` |
| `study_summary.bat` | the one-CSV overview | `data/study_summary.csv` |

### How to read the numbers

- **Suppression index (0 to 1)** = `1 - threshold(fixation) / threshold(saccade)`. 0 = seen as well during saccades as during fixation; 1 = never seen during saccades; negative = better during saccades. Per person.
- **95 % CI**: the range the true number plausibly lies in (resampling flashes; for groups, people). Wide = uncertain. For a difference, a CI that excludes 0 means the data show a difference; one that includes 0 means they cannot tell.
- **p-value** (group vs group only, Mann-Whitney): the chance of a difference this large if the groups did not differ. Below 0.05 is the usual "evidence of a difference"; a large p does not show the groups are the same.
  **Every comparison with fewer than 10 people per side is marked "exploratory"**: treat it as a first look.
- **1 vs group has no p-value**: one person gives no estimate of within-group spread. You get the person's percentile in the group and whether the group's median lies inside the person's CI.
- **"Accuracy-based (fallback)"**: with fewer than 12 flashes in a condition (or too few contrasts) the index is replaced by the accuracy difference (fixation minus saccade). Different scale: never compared with a threshold index.
- **Quality lights.** Green = passes. **Amber** = usable but weaker: runs left out of pooling, fallback method, 5-11 flashes in the smaller condition, long gaze delay or tracking pause, head movement flagged.
  **Red** = no number: fewer than 5 flashes in a condition, or no run could be pooled. Grey = not recorded.

## Analysing from the command line

```
.\scripts\py.ps1 -m src.experiment.analyze_session [session] [--data-dir DIR]
.\scripts\py.ps1 -m src.experiment.analyze_participant <participant id> [--data-dir DIR] [--include-test]
.\scripts\py.ps1 -m src.experiment.participants set P01 TBI        # also: set P07 control --note "pilot" | remove P01 | list
.\scripts\py.ps1 -m src.experiment.compare "group:TBI" "group:control" [--include-test]    # sets: IDs, group:<label>, all
.\scripts\py.ps1 -m src.experiment.study_summary [--include-test]
```

**`analyze_session`**: `session` defaults to the newest; it can be a unix timestamp, a run folder name, `participant/folder` or a path. It finds saccades in `gaze_trace.csv` using the phone's capture times (so the
late arrival at the PC does not matter), places each flash against the nearest saccade onset, and labels it with pre-specified gates (`src/experiment/flash_gates.py`): **saccade** (a saccade within +-100 ms, the eye
moving fast at the flash, latency from the go cue 100-500 ms, amplitude 0.5-1.5x the session median), **fixation** (at least 100 ms from any saccade, no tracking gap) or **excluded** (reason in `flashes.csv`).
The index is a Weibull psychometric fit per condition (guess rate 0.5, shared slope) with a bootstrap CI; with too few flashes it falls back to the accuracy difference, or reports "cannot be computed" with the
reason. Other options: `--offset-ms`, `--flash-delay-ms`, `--min-amplitude`, `--index-near-ms`, `--index-far-ms` (see `--help`).

**`analyze_participant`**: one run gives only ~19 saccade flashes, so each person does several runs that are pooled. Test runs are skipped unless `--include-test`. Only runs with the **same saccade contrasts and screen
geometry** are pooled (the largest such set; others are listed at the top of the report with the reason).

**`compare`**: uses everyone's pooled analysis (redone first if missing or stale). People are left out, with the reason, when their index cannot be computed, their runs used other contrasts/geometry than the majority,
or (groups) their index came from a different method. Warnings appear for groups under 3 people or under 12 flashes per condition. Group labels are free text, **no clinical details** (those need IRB approval).

**`study_summary`**: `data/study_summary.csv` (group, runs used/left out, index, CI, method, flashes per condition, gaze delay and tracking gaps, flags) and `study_summary.txt`. The "gaze delay > 50 ms" flag is the
known delay problem; it affects only live classification, not these post-hoc numbers.

## Glasses video

The Neon phone records the scene video itself; the recording starts with the trials and is saved when they end (`glasses_video.recording` in `session.json`: `started`/`saved`/`failed (reason)`). The PC logs the wall-clock time of each green flash. Afterwards:
1. Export the scene video from the phone (Companion app, a USB copy of the recording, or Pupil Cloud; an `.mp4`).
2. Attach it:
   ```
   .\scripts\py.ps1 -m src.experiment.import_glasses_video data\<participant>\<run folder> C:\path\to\scene_video.mp4
   ```
   This copies it to `glasses/`, finds the green flashes, and writes `video_sync.json` (clock offset, residual, `ok` or `check manually`) and `flash_video_times.csv`. PC time `T` (unix ms) is at
   `T / 1000 - offset_s` seconds into the video.

Not verified on real footage: whether the scene camera sees the green flash, and the exported video's frame-rate/clock behaviour.

## Dashboard

`.\scripts\py.ps1 -m src.dashboard.app`, then <http://localhost:5000>. It lists past sessions (both layouts): label, filter, delete (a deleted run takes its folder, including the glasses video, and that
participant's pooled analysis with it).

- **Results pages** (**Analyse** on a run or participant): same analysis as the CLI, roughly 7-10 s on synthetic data; a saved analysis is shown again ("Analysed at ...") while newer than every run file, else
  or after **Re-analyse** it reruns. The quality box lights: gaze delay and tracking gaps (green up to 150 ms; the usual ~95 ms is fine; never red), head movement, flashes per condition
  (green 12+, amber 5-11, red under 5) with the method used, and for a participant the runs left out. Problems (no gaze recording, only test runs) appear as a sentence.
- **Group labels**: the **Participants** table's **Group** and **Note** boxes write `data/participants.json`, the same file the CLI uses. Empty both to remove a label.
- **Compare** (`/compare`): Set A / Set B columns, then the headline, the mode with a "how to read this" line, the plot, a per-person table, amber warnings and downloads. The address bar holds only the selection
  (`/compare?a=P01,P02&b=group:control`), so a bookmark reopens it; a `group:` pick follows the group's members at open time.
- **Study summary** (`/study`): the table with links to each person's results and a `study_summary.csv` download (opening the page rewrites it).

## Data and privacy

- **Stored**: `data/<participant id>/<run>/` holds answers and timing, settings and git commit, raw gaze, analyses and the phone's scene video (it shows the room and anyone in it: personal data too).
  `data/` is git-ignored. `tests/fixtures/` holds only anonymous test-mode recordings.
- **Participant IDs**: a study code (`P01`), never a name. The program turns the ID into a safe folder name (`sanitize_participant_id`); that is the only processing.
- **Network**: the dashboard listens only on this computer. Do not expose it: it lists and deletes participant data. The Neon phone's Real-Time API has no password, so use a private hotspot or your own router.
- **Not handled here**: consent, ethics approval, de-identification of the video, encryption and backups. Recommended: BitLocker, a backup plan for `data/`, and the study's IRB protocol.
- **Before pushing or sharing code**: `git status` shows nothing under `data/`; never copy real sessions into `tests/fixtures/`; delete the phone recording after export if the protocol requires it.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run from the project folder, or through `scripts/py.ps1` |
| `No module named 'flask'` | The install did not take (often non-writable `Program Files`): rerun the terminal as Administrator and reinstall |
| `_ARRAY_API not found` / `numpy.core.multiarray failed to import` | numpy/opencv drifted: reinstall `requirements.txt` (it pins working versions) |
| Neon not found | Phone and PC on one network, Companion app running; set `NEON_DEVICE_ADDRESS` to the phone's IP |
| Eye tracking seems inaccurate | Redo calibration, looking steadily at each target; keep the head still afterwards |
| SPACE seems to do nothing | Presses during a trial's wait are ignored by design; they are counted in the console summary and `ignored_space_presses` |
| Glasses recording `failed` in `session.json` | The recording may need the Companion app open with a wearer and workspace selected (not verified); other data is unaffected |

## Research background

- **Diamond, Ross & Morrone (2000)**, *J Neurosci* 20:3449-3455: the original design (fixate, jittered wait, saccade target, one-frame flash, detection during saccades and fixation). The paper was not checked for this
  README. Here the wait is 500-1000 ms after the participant's own SPACE, the tone is a go cue at target onset, contrasts are fixed (no staircase), and the fixation condition comes from the saccade phase's own flashes.
- **Braun, Schutz & Gegenfurtner (2021)**, "Age effects on saccadic suppression of luminance and color", *J Vis* 21(6):11, <https://pmc.ncbi.nlm.nih.gov/articles/PMC8237129/> (abstract and Methods read): the line stimulus
  (full-width horizontal line, vertical Gaussian SD 0.15 deg, one refresh, 2 deg above/below the target), the above/below task, the saccade/fixation gates, and the index `1 - CS_saccade / CS_fixation` (CS = 1 / threshold).

| | Braun et al. 2021 | This experiment |
|---|---|---|
| Eye tracker | EyeLink 1000, 1000 Hz | Pupil Labs Neon, ~160-200 Hz, gaze reaches the PC late |
| Flash trigger | live, 15 ms after online onset detection | scheduled on a bell curve around the person's average reaction time, judged **after the fact** from the raw trace |
| Display | 120 Hz, 8 ms flash | 60 Hz, one frame (~17 ms) |
| Saccade size | 10 deg | ~30-40 deg on a large screen (dot at 15 %, cross at 85 % of the width); depends on geometry |
| Thresholds | staircase, then psychometric fit | fixed contrasts (default 4/6/8 %), then a Weibull fit |
| Viewing | 90 cm, lab | chin rest, geometry from the startup dialog |

## Known limitations

- **Untested on glasses:** line rendering at 60 Hz, tutorial and captions, results layout, phone recording and sync flashes, head monitor, every contrast level, and the dashboard pages in a real browser.
- **Contrasts are a guess.** 4/6/8 % is only ~5-10 grey steps on an 8-bit display, so levels are coarse. Re-pilot and adjust in the startup dialog.
- **Screen geometry is only as good as what you type in**; the placeholders (26 cm at 50 cm) make degrees approximate. The geometry used is saved in `session.json` and printed in the analysis report.
- **Gaze reaches the PC ~90-100 ms late** (about 16 ms in standalone scripts), so the live `flash_during_saccade` flag is unreliable. Post-hoc judging from the phone's capture times is unaffected. Cause not found.
- **Timing is open-loop:** reaction times vary (SD ~60-80 ms), so only a minority of flashes land near onset (simulator: ~43 of 150 within +-50 ms, ~31 pass the saccade gate, ~55 fixation flashes). People with TBI may vary more.
- **One participant so far:** the group tools are built and tested on synthetic data only.

## For developers

[CLAUDE.md](CLAUDE.md) has the architecture, repo map and gotchas. Everything runs without hardware:

```
.\scripts\py.ps1 -m pytest -q                 # 275 tests pass, about 4 min, no glasses or window needed
.\scripts\py.ps1 -m src.experiment.simulation # headless simulated trial; see tests/ai-tests/README.md for run_sim.py
```
