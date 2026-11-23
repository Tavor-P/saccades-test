"""The standalone connection check (src/eye_tracking/neon_check.py). Everything is faked: no test contacts the real Neon,
and the report is written only to pytest's tmp_path."""

import sys
from types import SimpleNamespace

import pytest

from src.eye_tracking import neon_check
from src.eye_tracking.neon_address import SOURCE_DISCOVERY, SOURCE_GATEWAY, SOURCE_GIVEN, Attempt, Resolution, failure_message

STATUS = {
    "message": "ok",
    "result": [
        {"model": "Phone", "data": {"battery_level": 81, "battery_state": "OK", "memory_state": "OK", "device_name": "Neon Companion"}},
        {"model": "Hardware", "data": {"glasses_serial": "-1", "module_serial": "123456", "version": "2.0"}},
        {"model": "Sensor", "data": {"sensor": "gaze", "conn_type": "DIRECT", "connected": True}},
        {"model": "Sensor", "data": {"sensor": "world", "conn_type": "DIRECT", "connected": False}},
    ],
}
ADAPTERS = [{"name": "WLAN", "ipv4": "192.168.43.57", "gateway": "192.168.43.1", "network": "Pixel_1234", "category": "Public"}]


def found_resolution(given, port, progress=None):
    attempts = (
        Attempt(SOURCE_GIVEN, given, port, False, "no answer within 2 s"),
        Attempt(SOURCE_DISCOVERY, None, None, False, "no Neon announced itself"),
        Attempt(SOURCE_GATEWAY, "192.168.43.1", 8080, True, "Neon answered"),
    )
    if progress:
        progress("  trying ...")
    return Resolution("192.168.43.1", 8080, SOURCE_GATEWAY, attempts)


def missing_resolution(given, port, progress=None):
    return Resolution(None, None, None, (Attempt(SOURCE_GIVEN, given, port, False, "no answer within 2 s"),))


def test_connected_report_has_network_attempts_discovery_status_and_gaze_rate():
    ok, text = neon_check.build_report(
        "192.168.0.60", resolve=found_resolution, adapters=lambda: ADAPTERS, network=lambda: pytest.fail("not needed"),
        status=lambda a, p: STATUS, gaze_rate=lambda a, p: (585, 3.0), stamp="2026-11-23T10:00:00+01:00",
    )
    assert ok
    for needle in ("2026-11-23T10:00:00+01:00", "192.168.43.57", "gw 192.168.43.1", "Pixel_1234", "Public",
                   "FAIL 192.168.0.60:8080", "OK   192.168.43.1:8080", "Auto-discovery found a Neon: no", "CONNECTED to 192.168.43.1:8080",
                   "Phone battery: 81 %", "Glasses connected: yes", "gaze(DIRECT)=on", "world(DIRECT)=off",
                   "585 samples in 3.0 s = 195 per second (OK", "type 192.168.43.1 into 'Neon address'"):
        assert needle in text, needle


def test_not_connected_report_lists_the_checks_and_never_touches_the_device():
    fail = lambda *a: pytest.fail("no device calls when nothing answered")
    ok, text = neon_check.build_report(
        "192.168.0.60", resolve=missing_resolution, adapters=lambda: [], network=lambda: {"wifi_name": "Guest", "ipv4": "10.1.2.3"},
        status=fail, gaze_rate=fail, stamp="t",
    )
    assert not ok
    for needle in ("NOT CONNECTED", "no answer within 2 s", "adapter list unavailable", "Guest", "10.1.2.3", "same network", "2.4 GHz"):
        assert needle in text, needle


def test_status_or_gaze_failures_become_report_lines_not_crashes():
    def boom(*_):
        raise OSError("stream died")

    ok, text = neon_check.build_report(
        "192.168.43.1", resolve=found_resolution, adapters=boom, network=lambda: {}, status=boom, gaze_rate=boom, stamp="t",
    )
    assert ok and "could not list adapters" in text and "could not read the status (OSError" in text
    assert "Gaze stream: failed (OSError: stream died)" in text


def test_summaries_of_odd_replies():
    assert "not understood" in neon_check.summarize_status({"message": "x"})[0]
    lines = neon_check.summarize_status({"result": [{"model": "Hardware", "data": {"glasses_serial": "-1", "module_serial": ""}}]})
    assert any("NO - plug the glasses" in line for line in lines)
    assert neon_check.parse_adapters('{"name": "WLAN"}') == [{"name": "WLAN"}]
    assert neon_check.parse_adapters('[{"name": "A"}, {"name": "B"}]')[1]["name"] == "B"
    assert neon_check.parse_adapters("not json") == [] and neon_check.parse_adapters("") == []


def test_gaze_rate_counts_samples_from_a_fake_device_and_closes_it(monkeypatch):
    closed = []

    class FakeDevice:
        def __init__(self, address, port):
            pass

        def receive_gaze_datum(self, timeout_seconds):
            return object()

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "pupil_labs.realtime_api.simple", SimpleNamespace(Device=FakeDevice))
    count, seconds = neon_check.measure_gaze_rate("192.168.43.1", 8080, seconds=0.05)
    assert count > 0 and seconds >= 0.05 and closed == [True]


def test_main_prints_and_saves_the_report_and_exits_1_when_not_found(monkeypatch, tmp_path, capsys):
    report = tmp_path / "neon_check_report.txt"
    monkeypatch.setattr(neon_check, "build_report", lambda address, progress=None: (False, "RESULT: NOT CONNECTED\n"))
    assert neon_check.main(["192.168.0.60"], report_path=report) == 1
    assert report.read_text(encoding="utf-8") == "RESULT: NOT CONNECTED\n"
    out = capsys.readouterr().out
    assert "NOT CONNECTED" in out and str(report) in out
    monkeypatch.setattr(neon_check, "build_report", lambda address, progress=None: (True, "RESULT: CONNECTED\n"))
    assert neon_check.main(["192.168.43.1:8080", "--report", str(report)]) == 0
    assert neon_check.main(["1.2.3"], report_path=report) == 2  # bad address: message, no search


def test_report_path_is_git_ignored_and_the_failure_message_names_the_command():
    root = neon_check.REPORT_PATH.parent
    assert "neon_check_report.txt" in (root / ".gitignore").read_text(encoding="utf8").split()
    assert (root / "neon_check.bat").is_file()
    assert "neon_check" in failure_message(missing_resolution("1.2.3.4", 8080), {})
