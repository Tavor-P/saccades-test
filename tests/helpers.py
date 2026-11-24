"""Small test doubles shared by several test files (tests/ is on sys.path, see tests/ai-tests/conftest.py)."""


import time


def wait_until(predicate, timeout_s: float = 3.0, poll_s: float = 0.005) -> bool:
    """Poll `predicate` until it is truthy or `timeout_s` of real time passes; returns the last truth value.
    For tests that wait on a background thread (never on session logic, which runs on the injected clock)."""
    deadline = time.perf_counter() + timeout_s
    while True:
        if predicate():
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(poll_s)


class NullLogger:
    """Swallows .log() calls - tests check session.results instead of a real CSV, so they need no filesystem isolation."""

    def log(self, result) -> None:
        pass
