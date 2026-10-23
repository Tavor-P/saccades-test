"""PausableClock: a monotonic clock that freezes while paused. All session timers read it instead of the raw timebase. Tests/simulation substitute a
controlled clock (tests' fake_time fixture, simulation.SimulatedClock)."""

from src import timebase


class PausableClock:
    """A monotonic clock whose `.now()` freezes while paused and resumes
    counting from where it left off once resumed. Session classes read all
    their timestamps through this instead of the raw timebase directly, so
    a click-to-pause (to rest your eyes) doesn't corrupt in-flight timers -
    a foreperiod or response window that was 400ms into its duration when
    paused is still exactly 400ms in when it resumes, no matter how long the
    pause lasted. Built on timebase.now() (time.perf_counter, 100 ns).
    """

    def __init__(self) -> None:
        self._paused_at: float | None = None
        self._total_paused = 0.0

    def now(self) -> float:
        if self._paused_at is not None:
            return self._paused_at - self._total_paused
        return timebase.now() - self._total_paused

    def from_perf_counter(self, timestamp: float) -> float:
        """Converts a timebase.now() timestamp (what gaze sources stamp
        their samples with) onto this clock's timebase, i.e. the value now()
        had at that instant - valid for instants since the last pause ended."""
        return timestamp - self._total_paused

    def unix_ms(self) -> float:
        """Wall-clock milliseconds since the epoch right now (sub-ms resolution, derived from the perf counter)."""
        return timebase.unix_ms()

    @property
    def is_paused(self) -> bool:
        return self._paused_at is not None

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = timebase.now()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._total_paused += timebase.now() - self._paused_at
            self._paused_at = None
