"""Offline study of the saccade-onset rule on recorded Neon gaze traces (the data behind SACCADE_ONSET_DISPLACEMENT).

For each trace it finds the saccades between the two targets (position crossing 0.5 from one side to the other without a
tracking dropout), replays the pre-saccade fixation + movement through the real OnsetDetector at several thresholds, and
compares with the old zone rule (gaze outside the source zone, CENTER included, for the debounce). It reports

  lead     = ms the trigger (`since`, the first sample beyond the rule) comes BEFORE the position crosses the midline (0.5);
             conf lead = the same for the moment the debounce confirms it (since + SACCADE_ONSET_STABILITY_MS at best)
  delay    = ms the trigger comes AFTER the physical onset (last sample still within noise of the baseline before the move)
  trig pos = how far along the path (0 = source, 1 = target) the gaze was at the trigger
  FA       = false triggers when the detector is started on fresh 600 ms windows (a trial's foreperiod + reaction time)
             of quiet source fixation (near a target, peak-to-peak position range <= --quiet, default 0.12): % of windows and per minute

Time axis: the phone's device_unix_s (5 ms spacing, no Wi-Fi jitter; the recorded arrival column is quantised to 15.6 ms).
Calibration comes from the matching *_meta.json when present, else from the two position clusters of the trace itself.

Usage: scripts/py.sh tests/ai-tests/analyze_onset_threshold.py [trace.csv ...] [--thresholds 0.03,0.05,...] [--stability 0,5,10]
(default trace: tests/fixtures/neon_trace_sample.csv)."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.experiment.trial_mechanics as tm  # noqa: E402
from include.eye_tracking.constants import NEON_SCENE_WIDTH  # noqa: E402
from include.eye_tracking.types import GazeSample, GazeZone  # noqa: E402
from src.eye_tracking.gaze_mapping import GazeMapper  # noqa: E402
from src.eye_tracking.gaze_trace import load_trace  # noqa: E402

DEFAULT_TRACE = ROOT / "tests" / "fixtures" / "neon_trace_sample.csv"
WINDOW_S = 0.6
PRE_S = 0.4  # fixation fed to the detector before the saccade


def load_positions(path: Path):
    """(device times s, calibrated positions with NaN for lost samples, how the calibration was obtained)"""
    rows = load_trace(path)
    t = np.array([r.device_unix_s for r in rows])
    ratio = np.array(
        [r.x_px / NEON_SCENE_WIDTH if r.worn and r.x_px is not None and r.x_px == r.x_px else np.nan for r in rows]
    )
    meta_path = path.with_name(path.name.replace("_gaze_trace.csv", "_meta.json"))
    cal = json.loads(meta_path.read_text()).get("calibration") if meta_path != path and meta_path.exists() else None
    if cal:
        left, centre, right = cal["left_ratio"], cal["center_ratio"], cal["right_ratio"]
        slope, intercept = np.polyfit([left, centre, right], [0, 0.5, 1], 1)
        how = "session calibration"
    else:
        valid = ratio[~np.isnan(ratio)]
        mid = (np.percentile(valid, 2) + np.percentile(valid, 98)) / 2
        left, right = np.median(valid[valid < mid]), np.median(valid[valid >= mid])
        slope, intercept = 1 / (right - left), -left / (right - left)
        how = "two-cluster fit"
    return t, slope * ratio + intercept, how


def make_sample(pos, t):
    if pos != pos:
        return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=t)
    return GazeSample(zone=GazeMapper.classify(pos), ratio=0.5, face_found=True, timestamp=t, position=pos)


class ZoneRule:
    """The rule being replaced: outside the source zone (CENTER counts) for `stability` ms."""

    def __init__(self, avoid, stability_ms):
        self.avoid, self.stab, self.since, self.confirmed = avoid, stability_ms / 1000, None, False

    def update(self, s, now):
        if s.face_found and s.zone not in (self.avoid, GazeZone.UNKNOWN):
            if self.since is None:
                self.since = now
            elif now - self.since >= self.stab:
                self.confirmed = True
                return True
        else:
            self.since = None
        return False


def run(detector, ts, pos, lo, hi):
    """(onset estimate `since`, time the detector confirmed it) or None if it never fired in [lo, hi)."""
    for i in range(lo, hi):
        if detector.update(make_sample(pos[i], ts[i]), ts[i]):
            return detector.since, ts[i]
    return None


def find_saccades(ts, pos):
    """(index of the last sample on the source side, index of the first sample on the other side, source side 0/1)
    for clean crossings: under 150 ms in transit, no tracking dropout in between."""
    out = []
    side, side_end = None, 0
    for i, p in enumerate(pos):
        if p != p:
            continue
        s = 0 if p < 0.3 else 1 if p > 0.7 else None
        if s is None:
            continue
        if side is not None and s != side and ts[i] - ts[side_end] < 0.15:
            seg = pos[side_end : i + 1]
            if not np.isnan(seg).any() and np.all(np.diff(ts[side_end : i + 1]) < 0.03):
                out.append((side_end, i, side))
        side, side_end = s, i
    return out


def physical_onset(ts, pos, a, b, side):
    """Last index still within noise of the pre-move baseline, walking back from the first sample 0.3 away."""
    sign = 1 if side == 0 else -1
    base = np.nanmedian(pos[max(a - 20, 0) : a + 1])
    k = a
    while k < b and sign * (pos[k] - base) < 0.3:
        k += 1
    while k > a and sign * (pos[k] - base) > 0.02:
        k -= 1
    return k


def fixation_windows(ts, pos, quiet):
    """(lo, hi, side) index windows of WINDOW_S with no tracking dropout, near a target (median within 0.3 of it) and
    quiet (peak-to-peak within `quiet`): fixation noise, drift and microsaccades but no real saccade."""
    wins, k, n = [], 0, len(ts)
    while k < n:
        hi = int(np.searchsorted(ts, ts[k] + WINDOW_S))
        if hi >= n:
            break
        seg = pos[k:hi]
        if np.isnan(seg).any() or np.any(np.diff(ts[k:hi]) > 0.03) or np.ptp(seg) > quiet:
            k += 20
            continue
        med = float(np.median(seg))
        if min(abs(med), abs(med - 1)) < 0.3:
            wins.append((k, hi, 0 if med < 0.5 else 1))
        k = hi
    return wins


def analyse(path, thresholds, stabilities, quiet):
    ts, pos, how = load_positions(path)
    ts = ts - ts[0]
    sacc = find_saccades(ts, pos)
    wins = fixation_windows(ts, pos, quiet)
    fix_minutes = len(wins) * WINDOW_S / 60
    valid = ~np.isnan(pos)
    print(f"\n== {path.name}: {len(ts)} rows ({valid.sum()} valid), {ts[-1]:.0f} s, calibration: {how}")
    print(f"   {len(sacc)} clean saccades, {len(wins)} fixation windows ({fix_minutes:.2f} min)")
    dur = []
    for a, b, side in sacc:
        seg = np.abs(pos[a : b + 1] - side)
        if np.any(seg > 0.1) and np.any(seg > 0.9 * 1.0):
            dur.append(ts[a + int(np.argmax(seg > 0.9))] - ts[a + int(np.argmax(seg > 0.1))])
    if dur:
        print(f"   10%->90% travel time of the clean crossings: median {1000 * statistics.median(dur):.0f} ms")
    print(f"   {'rule':<26}{'lead med':>9}{'lead p10':>9}{'conf lead':>10}{'delay med':>10}{'trig pos':>9}{'missed':>7}{'FA win%':>8}{'FA/min':>8}")
    rules = [("old zone rule", lambda side, p: ZoneRule(GazeZone.LEFT if side == 0 else GazeZone.RIGHT, p), 10)]
    for stab in stabilities:
        for thr in thresholds:
            rules.append((f"displace {thr:.2f} / {stab:g} ms", None, (thr, stab)))
    for name, factory, param in rules:
        leads, conf_leads, delays, trig_pos, missed = [], [], [], [], 0
        saved = (tm.SACCADE_ONSET_DISPLACEMENT, tm.SACCADE_ONSET_STABILITY_MS)

        def make(side):
            if factory is not None:
                return factory(side, param)
            tm.SACCADE_ONSET_DISPLACEMENT, tm.SACCADE_ONSET_STABILITY_MS = param
            return tm.OnsetDetector(GazeZone.LEFT if side == 0 else GazeZone.RIGHT)

        try:
            for a, b, side in sacc:
                start = int(np.searchsorted(ts, ts[a] - PRE_S))
                fired = run(make(side), ts, pos, start, min(b + 40, len(ts)))
                if fired is None:
                    missed += 1
                    continue
                since, confirmed_at = fired
                leads.append((ts[b] - since) * 1000)
                conf_leads.append((ts[b] - confirmed_at) * 1000)
                delays.append((since - ts[physical_onset(ts, pos, a, b, side)]) * 1000)
                trig_pos.append(abs(float(pos[int(np.searchsorted(ts, since))]) - side))
            fa = sum(run(make(side), ts, pos, lo, hi) is not None for lo, hi, side in wins)
        finally:
            tm.SACCADE_ONSET_DISPLACEMENT, tm.SACCADE_ONSET_STABILITY_MS = saved
        med = statistics.median
        print(
            f"   {name:<26}{(f'{med(leads):.1f}' if leads else '-'):>9}"
            f"{(f'{np.percentile(leads, 10):.1f}' if leads else '-'):>9}{(f'{med(conf_leads):.1f}' if conf_leads else '-'):>10}"
            f"{(f'{med(delays):.1f}' if delays else '-'):>10}{(f'{med(trig_pos):.2f}' if trig_pos else '-'):>9}{missed:>7}"
            f"{100 * fa / max(len(wins), 1):>8.1f}{(fa / fix_minutes if fix_minutes else float('nan')):>8.1f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", type=Path, default=[DEFAULT_TRACE])
    ap.add_argument("--thresholds", default="0.03,0.04,0.05,0.06,0.08,0.10,0.15")
    ap.add_argument("--stability", default="10")
    ap.add_argument("--quiet", type=float, default=0.12, help="max peak-to-peak position range of a fixation window")
    args = ap.parse_args()
    for trace in args.traces:
        analyse(trace, [float(x) for x in args.thresholds.split(",")], [float(x) for x in args.stability.split(",")], args.quiet)
