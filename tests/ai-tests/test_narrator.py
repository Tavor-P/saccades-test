import time
from unittest.mock import Mock

from src.experiment.narrator import Narrator


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true within timeout")


def test_speak_reaches_sapi_in_order_on_the_background_thread_and_stop_ends_it(monkeypatch):
    mock_voice = Mock()
    monkeypatch.setattr("win32com.client.Dispatch", Mock(return_value=mock_voice))
    monkeypatch.setattr("pythoncom.CoInitialize", Mock())
    monkeypatch.setattr("pythoncom.CoUninitialize", Mock())

    narrator = Narrator()
    try:
        narrator.speak("first")
        narrator.speak("second")
        _wait_until(lambda: mock_voice.Speak.call_count >= 2)
        assert [call.args[0] for call in mock_voice.Speak.call_args_list] == ["first", "second"]
    finally:
        narrator.stop()
    assert not narrator._thread.is_alive()
