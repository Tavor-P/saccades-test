"""30-second fixation noise check for the Neon: look at one fixed point and
this reports how far gaze strays from its median, and for how long - i.e. how
short SACCADE_ONSET_STABILITY_MS can safely be before fixational jitter would
be misread as a saccade onset. Run with the glasses on:

    python -m src.eye_tracking.neon_fixation_test [seconds]
"""

import math
import statistics as st
import sys
import time

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from src.eye_tracking.neon_source import _connect
from src import timebase

# Neon scene camera is ~103 deg wide horizontally; approximate, good enough to
# put the jitter in rough degrees.
DEG_PER_PX = 103.0 / NEON_SCENE_WIDTH
THRESHOLDS_DEG = (0.5, 1.0, 2.0, 3.0)


def longest_excursion_ms(samples: list[tuple[float, float]], median_x: float, threshold_px: float) -> float:
    """Longest run of consecutive samples more than threshold_px from the median, in ms."""
    longest, run_start, run_end = 0.0, None, None
    for t, x in samples:
        if abs(x - median_x) > threshold_px:
            if run_start is None:
                run_start = t
            run_end = t
            longest = max(longest, (run_end - run_start) * 1000)
        else:
            run_start = None
    return longest


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    device = _connect()
    if device is None:
        sys.exit("No Neon found.")
    print("Put the glasses on and stare at ONE fixed point (e.g. the middle of the screen).")
    for n in (3, 2, 1):
        print(f"Starting in {n}...", flush=True)
        time.sleep(1)
    print(f"Recording {seconds:.0f}s - keep looking at the point, blink normally.", flush=True)

    samples: list[tuple[float, float]] = []  # (device timestamp s, x px)
    bad = 0
    end = timebase.now() + seconds
    while timebase.now() < end:
        g = device.receive_gaze_datum(timeout_seconds=1)
        if g is None:
            continue
        if not g.worn or g.x is None or math.isnan(g.x):
            bad += 1
            continue
        samples.append((g.timestamp_unix_seconds, g.x))
    device.close()

    if len(samples) < 100:
        sys.exit(f"Only {len(samples)} usable samples - is the Neon worn and streaming?")
    xs = [x for _, x in samples]
    med = st.median(xs)
    print(f"\nsamples: {len(samples)} usable, {bad} not-worn/NaN")
    print(f"median x: {med:.0f}px   std: {st.pstdev(xs) * DEG_PER_PX:.2f} deg   "
          f"range: {(max(xs) - min(xs)) * DEG_PER_PX:.1f} deg")
    print("\nlongest stretch gaze stayed further than X from the median (ms):")
    for deg in THRESHOLDS_DEG:
        px = deg / DEG_PER_PX
        far = sum(abs(x - med) > px for x in xs)
        print(f"  > {deg:3.1f} deg: longest {longest_excursion_ms(samples, med, px):5.0f} ms   "
              f"({100 * far / len(xs):.2f}% of samples)")
    # Short-timescale jitter: deviation from a rolling median over the
    # surrounding ~250 ms, which cancels slow drift / head movement and leaves
    # only what could actually be misread as a saccade onset.
    half = max(1, round(len(samples) / seconds * 0.125))
    local = sorted(abs(xs[i] - st.median(xs[max(0, i - half) : i + half + 1])) * DEG_PER_PX for i in range(len(xs)))
    steps = sorted(abs(xs[i] - xs[i - 1]) * DEG_PER_PX for i in range(1, len(xs)))
    print("\nshort-timescale jitter (drift removed):")
    print(
        f"  deviation from local median: median {st.median(local):.2f} deg, "
        f"p99 {local[int(.99 * len(local))]:.2f} deg, max {local[-1]:.2f} deg"
    )
    print(
        f"  sample-to-sample step:       median {st.median(steps):.2f} deg, "
        f"p99 {steps[int(.99 * len(steps))]:.2f} deg, max {steps[-1]:.2f} deg"
    )
    print(f"  effective sample rate: {len(samples) / seconds:.0f} Hz")
    print("\nIf the p99 jitter is well under your dead-zone size, a short (~10 ms) onset debounce is safe.")


if __name__ == "__main__":
    main()
