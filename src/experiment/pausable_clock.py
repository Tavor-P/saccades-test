import time


class PausableClock:
    """A monotonic clock whose `.now()` freezes while paused and resumes
    counting from where it left off once resumed. Session classes read all
    their timestamps through this instead of `time.monotonic()` directly, so
    a click-to-pause (to rest your eyes) doesn't corrupt in-flight timers -
    a foreperiod or response window that was 400ms into its duration when
    paused is still exactly 400ms in when it resumes, no matter how long the
    pause lasted.
    """

    def __init__(self) -> None:
        self._paused_at: float | None = None
        self._total_paused = 0.0

    def now(self) -> float:
        if self._paused_at is not None:
            return self._paused_at - self._total_paused
        return time.monotonic() - self._total_paused

    @property
    def is_paused(self) -> bool:
        return self._paused_at is not None

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._total_paused += time.monotonic() - self._paused_at
            self._paused_at = None
