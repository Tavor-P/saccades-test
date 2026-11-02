# Experimenter checklist (print this page)

Commands run from the project folder in PowerShell. Full details: [README.md](../README.md).

## Before the session
- [ ] Phone: Neon Companion app open, glasses connected, a wearer and workspace selected, same Wi-Fi/hotspot as the PC, battery OK.
- [ ] Glasses on the participant (clean lenses), head on the chin rest, seat height so the screen centre is at eye level.
- [ ] Measured once per setup: screen height (cm) and eye-to-screen distance (cm), set as `SCREEN_HEIGHT_CM` / `VIEWING_DISTANCE_CM` in `include/experiment/constants.py`.
- [ ] Room dim, screen brightness the same as last time, notifications off, nothing else running.
- [ ] Start: `powershell -ExecutionPolicy Bypass -File scripts\py.ps1 -m src.experiment.run_experiment`
- [ ] Dialog: Mode **Real**, a **Participant ID**, Gaze source Pupil Labs Neon, **Run tutorial = Yes** (it is "No" by default), Record video as is. OK.
- [ ] Read the participant script (README, Quick start) before pressing OK.

## During
- [ ] Tutorial runs on its own (narrated): calibration, how to answer, 8 practice trials. SPACE advances.
- [ ] Calibration: the participant looks steadily at each target, head still.
- [ ] Green screen flashes three times before the first trial (phone sync): normal, nothing to do.
- [ ] Each trial waits for the participant's SPACE; they may rest as long as they like between trials.
- [ ] Watch the participant, not the screen: head off the chin rest or glasses slipping means stop and recalibrate (Esc, new run).
- [ ] Esc quits at once; the data so far are saved.

## After
- [ ] Read the summary printed on the console (also saved as `RUN_SUMMARY.txt` in the run folder); press SPACE to close the window.
- [ ] Phone: confirm the recording is saved; export the scene video (mp4) to the PC.
- [ ] `.\scripts\py.ps1 -m src.experiment.import_glasses_video data\<participant>\<run folder> C:\path\to\scene_video.mp4`
- [ ] `.\scripts\py.ps1 -m src.experiment.analyze_session data\<participant>\<run folder>`
- [ ] Reprint a past summary any time: `.\scripts\py.ps1 -m src.experiment.run_summary` (newest run).

## Troubleshooting
- Neon not found: phone and PC on the same network, Companion app running; else set `NEON_DEVICE_ADDRESS` in `include/eye_tracking/constants.py`.
- Gaze looks off: redo calibration, keep the head still; check the glasses sit tightly.
- Summary says tracking lost / low gaze rate: check Wi-Fi and the glasses cable/phone, then rerun.
- Recording `failed` in the summary: open the Companion app, select a wearer and workspace; the rest of the data is fine.
- SPACE "does nothing": presses during a trial's wait are ignored by design (counted in the summary).
- `No module named ...`: run through `scripts\py.ps1` from the project folder (never plain `python`).
