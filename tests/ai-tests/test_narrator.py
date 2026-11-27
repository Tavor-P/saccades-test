import threading
from unittest.mock import Mock

from helpers import wait_until

from src.experiment.narrator import Narrator


def _wait_until(predicate, timeout: float = 2.0) -> None:
    assert wait_until(predicate, timeout), "condition never became true within timeout"


class _FakeVoice:
    """SpVoice stand-in: a line "plays" until `finished` is set. Clearing `gate` holds the narrator thread inside
    WaitUntilDone (it sets `held` there), so a test can queue several lines before the thread looks again."""

    def __init__(self):
        self.spoken: list[tuple[str, int]] = []  # (text, flags)
        self.finished = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.held = threading.Event()

    def Speak(self, text, flags):
        self.spoken.append((text, flags))

    def WaitUntilDone(self, timeout_ms):
        if not self.gate.is_set():
            self.held.set()
            self.gate.wait(2.0)
        return self.finished.wait(timeout_ms * 0.001)


def test_a_new_line_replaces_older_speech_on_the_background_thread_and_stop_ends_it(monkeypatch):
    """Regression: screens skipped quickly left the narrator reading the old screens' lines one after another."""
    voice = _FakeVoice()
    monkeypatch.setattr("win32com.client.Dispatch", Mock(return_value=voice))
    monkeypatch.setattr("pythoncom.CoInitialize", Mock())
    monkeypatch.setattr("pythoncom.CoUninitialize", Mock())
    async_purge, async_only = 3, 1

    narrator = Narrator()
    try:
        voice.gate.clear()
        narrator.speak("first")
        assert voice.held.wait(2.0)  # "first" is playing
        narrator.speak("second")
        narrator.speak("third")
        assert narrator.is_speaking
        voice.gate.set()
        _wait_until(lambda: len(voice.spoken) >= 2)
        assert voice.spoken == [("first", async_purge), ("third", async_purge)]  # "second" was never started

        narrator.speak("warning", interrupt=False)  # waits for "third" instead of cutting it off
        assert voice.spoken[-1][0] == "third"
        voice.finished.set()
        _wait_until(lambda: not narrator.is_speaking)
        assert voice.spoken[-1] == ("warning", async_only)

        voice.finished.clear()
        narrator.speak("fourth")
        narrator.silence()  # purges, and does not wait for any audio
        _wait_until(lambda: not narrator.is_speaking)
        assert voice.spoken[-1] == ("", async_purge)
    finally:
        voice.finished.set()
        narrator.stop()
    assert not narrator._thread.is_alive()
