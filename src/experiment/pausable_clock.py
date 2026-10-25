"""PausableClock: the injected monotonic clock all session timers read instead of the raw timebase (the name is historical: pausing was removed). Tests/simulation substitute a
controlled clock (tests' fake_time fixture, simulation.SimulatedClock)."""

from src import timebase


class PausableClock:
    """Monotonic clock for the session classes; built on timebase.now() (time.perf_counter, 100 ns). Tests substitute a controlled clock."""

    def now(self) -> float:
        return timebase.now()

    def from_perf_counter(self, timestamp: float) -> float:
        """Converts a timebase.now() timestamp (what gaze sources stamp their samples with) onto this clock's timebase (the identity)."""
        return timestamp

    def unix_ms(self) -> float:
        """Wall-clock milliseconds since the epoch right now (sub-ms resolution, derived from the perf counter)."""
        return timebase.unix_ms()
