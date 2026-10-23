"""The experiment's one time source: time.perf_counter() (100 ns resolution), because on Windows time.monotonic() and time.time() tick only every
15.6 ms - about half a saccade. Everything that stamps or compares event times reads `now()`; wall-clock (unix) values are derived from perf_counter via an
anchor, never from time.time() per event. This is the only module allowed to call time.monotonic/time.time/time.perf_counter (tests/ai-tests/test_architecture_rules.py)."""

import time
from typing import NamedTuple

# Tests patch this (see tests/conftest.py fake_time) rather than the global time module, so other threads/libraries are unaffected.
_clock = time.perf_counter
_wall_clock = time.time

_ANCHOR_EDGE_TIMEOUT_S = 0.1


class Anchor(NamedTuple):
    """One instant read on both clocks: unix_s = the system clock, perf_s = now()."""

    unix_s: float
    perf_s: float

    def to_unix(self, perf_s: float) -> float:
        return self.unix_s + (perf_s - self.perf_s)

    def to_perf(self, unix_s: float) -> float:
        return self.perf_s + (unix_s - self.unix_s)


def now() -> float:
    """Seconds on the high-resolution experiment timeline."""
    return _clock()


def measure_anchor() -> Anchor:
    """Reads (system clock, now()) as close to the same instant as possible. A coarse system clock (15.6 ms ticks) is
    only accurate right at a tick, so spin until it changes and read now() immediately - otherwise the pair carries up to
    a tick of constant bias. Costs up to one tick (~16 ms) of busy waiting; with a fine clock it returns at once."""
    first = _wall_clock()
    deadline = _clock() + _ANCHOR_EDGE_TIMEOUT_S
    while True:
        unix_s = _wall_clock()
        perf_s = _clock()
        if unix_s != first or perf_s > deadline:
            return Anchor(unix_s, perf_s)


_anchor = measure_anchor()


def reanchor() -> None:
    """Re-pair the clocks (they drift apart by tens of ppm); for long-lived sessions."""
    global _anchor
    _anchor = measure_anchor()


def perf_to_unix(perf_s: float) -> float:
    return _anchor.to_unix(perf_s)


def unix_to_perf(unix_s: float) -> float:
    return _anchor.to_perf(unix_s)


def unix_now() -> float:
    """Wall-clock seconds since the epoch, with now()'s resolution."""
    return _anchor.to_unix(_clock())


def unix_ms() -> float:
    return unix_now() * 1000
