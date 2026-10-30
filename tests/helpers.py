"""Small test doubles shared by several test files (tests/ is on sys.path, see tests/ai-tests/conftest.py)."""


class NullLogger:
    """Swallows .log() calls - tests check session.results instead of a real CSV, so they need no filesystem isolation."""

    def log(self, result) -> None:
        pass
