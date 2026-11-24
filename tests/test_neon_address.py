"""Finding the Neon phone before the window opens (src/eye_tracking/neon_address.py) and connecting without the aiohttp traceback
(neon_source._connect). Everything is faked: no test contacts the real Neon. The probe test talks only to a throwaway HTTP server
on 127.0.0.1."""

import gc
import http.server
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from src.eye_tracking import neon_address, neon_source
from src.eye_tracking.neon_address import (
    SOURCE_DISCOVERY,
    SOURCE_GATEWAY,
    SOURCE_GIVEN,
    failure_message,
    parse_address,
    parse_default_gateways,
    parse_linux_gateways,
    parse_wifi_name,
    probe_status,
    resolve_neon_address,
)
from src.eye_tracking.neon_source import NeonConnectionError, NeonGazeSource

ROUTE_PRINT = """\
===========================================================================
Liste der Schnittstellen
 12...ac 12 03 4f 9a 2b ......Intel(R) Wi-Fi 6 AX201 160MHz
===========================================================================

IPv4-Routentabelle
===========================================================================
Aktive Routen:
     Netzwerkziel    Netzwerkmaske          Gateway    Schnittstelle Metrik
          0.0.0.0          0.0.0.0     192.168.43.1   192.168.43.57     55
          0.0.0.0          0.0.0.0       10.8.0.1       10.8.0.6     25
          0.0.0.0          0.0.0.0     192.168.43.1   192.168.43.57     60
        127.0.0.0        255.0.0.0   Auf Verbindung         127.0.0.1    331
     192.168.43.0    255.255.255.0   Auf Verbindung     192.168.43.57    311
===========================================================================
Ständige Routen:
  Netzwerkadresse          Netzmaske  Gatewayadresse  Metrik
          0.0.0.0          0.0.0.0     Auf Verbindung  Standard
===========================================================================
"""


def fake_probe(answering: set[tuple[str, int]], calls: list):
    def probe(address, port):
        calls.append((address, port))
        return None if (address, port) in answering else "no answer within 2 s"

    return probe


# -- parsing ----------------------------------------------------------------------------------------


def test_parse_address_blank_ip_ip_port_and_errors():
    assert parse_address("") == (None, 8080) and parse_address(None) == (None, 8080)
    assert parse_address(" 192.168.43.1 ") == ("192.168.43.1", 8080)
    assert parse_address("192.168.43.1:9000") == ("192.168.43.1", 9000)
    assert parse_address("neon.local") == ("neon.local", 8080)
    for bad in ("1.2.3", "1.2.3.256", "1.2.3.4:", "1.2.3.4:0", "1.2.3.4:65536", "x y", "ftp://1.2.3.4"):
        with pytest.raises(ValueError, match="Neon address"):
            parse_address(bad)


def test_parse_address_accepts_pasted_urls():
    for text in ("http://192.168.43.1:8080", "http://192.168.43.1:8080/", "192.168.43.1:8080/", "HTTPS://192.168.43.1:8080/"):
        assert parse_address(text) == ("192.168.43.1", 8080), text
    assert parse_address("http://192.168.43.1/") == ("192.168.43.1", 8080)
    assert parse_address("http://neon.local:9000/") == ("neon.local", 9000)


def test_gateways_are_read_from_the_numeric_columns_of_a_non_english_route_print():
    assert parse_default_gateways(ROUTE_PRINT) == ["192.168.43.1", "10.8.0.1"]  # deduplicated, 'On-link' rows skipped
    assert parse_default_gateways("") == []
    assert parse_linux_gateways("default via 192.168.1.1 dev wlan0\n10.0.0.0/8 dev eth0\ndefault via 192.168.1.1 dev eth0\n") == ["192.168.1.1"]
    assert neon_address.default_gateways(run=lambda cmd: ROUTE_PRINT if sys.platform == "win32" else "default via 192.168.43.1 dev x\n")[0] == "192.168.43.1"


def test_wifi_name_is_the_ssid_line_not_bssid():
    out = "    Name                   : WLAN\n    BSSID                  : aa:bb:cc:dd:ee:ff\n    SSID                   : Pixel_1234\n"
    assert parse_wifi_name(out) == "Pixel_1234"
    assert parse_wifi_name("Es gibt keine Drahtlosschnittstelle im System.") is None


# -- resolution order -------------------------------------------------------------------------------


def test_a_working_given_address_is_used_without_searching():
    calls = []
    def discover():
        return pytest.fail("discovery must not run")
    r = resolve_neon_address("192.168.0.60", 8080, probe=fake_probe({("192.168.0.60", 8080)}, calls), discover=discover, gateways=lambda: [])
    assert r.ok and (r.address, r.port, r.source) == ("192.168.0.60", 8080, SOURCE_GIVEN)
    assert calls == [("192.168.0.60", 8080)] and r.label() == "192.168.0.60:8080 (address in the dialog)"


def test_a_dead_given_address_falls_back_to_discovery():
    calls = []
    r = resolve_neon_address(
        "192.168.0.60", 8080, probe=fake_probe({("10.0.0.7", 8080)}, calls),
        discover=lambda: [("10.0.0.7", 8080)], gateways=lambda: pytest.fail("gateway must not be needed"),
    )
    assert (r.address, r.source) == ("10.0.0.7", SOURCE_DISCOVERY)
    assert [a.ok for a in r.attempts] == [False, True] and r.attempts[0].detail == "no answer within 2 s"


def test_hotspot_case_falls_back_to_the_gateway_and_skips_addresses_already_tried():
    calls = []
    progress = []
    r = resolve_neon_address(
        "192.168.0.60", 8080, probe=fake_probe({("192.168.43.1", 8080)}, calls),
        discover=lambda: [], gateways=lambda: ["192.168.0.60", "192.168.43.1"], progress=progress.append,
    )
    assert (r.address, r.port, r.source) == ("192.168.43.1", 8080, SOURCE_GATEWAY)
    assert calls == [("192.168.0.60", 8080), ("192.168.43.1", 8080)]  # the given address is not probed twice
    assert [(a.source, a.ok) for a in r.attempts] == [(SOURCE_GIVEN, False), (SOURCE_DISCOVERY, False), (SOURCE_GATEWAY, True)]
    assert any("searching" in p for p in progress) and any("192.168.43.1" in p for p in progress)


def test_blank_address_starts_with_discovery_and_a_missing_library_or_failed_search_is_reported_not_raised():
    def no_library():
        raise ImportError("No module named 'pupil_labs'")

    def broken():
        raise OSError("zeroconf exploded")

    for discover, expected in ((no_library, "not installed"), (broken, "zeroconf exploded")):
        calls = []
        r = resolve_neon_address(None, 8080, probe=fake_probe(set(), calls), discover=discover, gateways=lambda: ["192.168.1.1"])
        assert not r.ok and r.label() == "not found"
        assert calls == [("192.168.1.1", 8080)]
        assert expected in r.attempts[0].detail and r.attempts[0].source == SOURCE_DISCOVERY


def test_everything_failing_lists_each_attempt_and_the_message_has_the_checks():
    r = resolve_neon_address(
        "192.168.0.60", 8080, probe=fake_probe(set(), []), discover=lambda: [], gateways=lambda: [],
    )
    assert not r.ok and len(r.attempts) == 3
    text = failure_message(r, {"wifi_name": "Guest-WiFi", "ipv4": "172.20.1.5", "gateways": ["172.20.0.1"]})
    for needle in ("192.168.0.60:8080", "no answer within 2 s", "auto-discovery", "no default gateway", "Guest-WiFi", "172.20.1.5",
                   "172.20.0.1", "same network", "Guest", "hotspot", "Companion app", "2.4 GHz", "Neon address"):
        assert needle in text, needle
    assert "Traceback" not in text
    assert "unknown" in failure_message(r, {})


# -- the real probe against a local fake server ---------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    reply: tuple[int, bytes] = (200, b"{}")

    def do_GET(self):
        code, body = self.reply
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_probe_accepts_only_a_status_reply_and_explains_everything_else(local_server):
    port = local_server.server_address[1]
    _Handler.reply = (200, json.dumps({"message": "ok", "result": []}).encode())
    assert probe_status("127.0.0.1", port, timeout_s=2) is None
    _Handler.reply = (404, b"nope")
    assert "not the Neon Companion app" in probe_status("127.0.0.1", port, timeout_s=2)
    _Handler.reply = (200, b"<html>router login</html>")
    assert "not the Neon Companion app" in probe_status("127.0.0.1", port, timeout_s=2)
    _Handler.reply = (200, b'{"other": 1}')
    assert "not the Neon Companion app" in probe_status("127.0.0.1", port, timeout_s=2)
    local_server.shutdown()
    local_server.server_close()
    reason = probe_status("127.0.0.1", port, timeout_s=0.3)
    assert reason is not None and "Neon Companion" not in reason  # refused or (Windows retries a refused connect) timed out


def _status(*sensors):
    return json.dumps({"message": "Success", "result": [{"model": "Phone", "data": {"ip": "127.0.0.1"}},
                                                          *({"model": "Sensor", "data": d} for d in sensors)]}).encode()


def test_probe_requires_a_connected_gaze_sensor_but_accepts_a_reply_without_sensor_info(local_server):
    port = local_server.server_address[1]
    gaze = {"sensor": "gaze", "conn_type": "DIRECT", "connected": True, "ip": "127.0.0.1", "port": 8086}
    _Handler.reply = (200, _status(gaze, {"sensor": "world", "conn_type": "DIRECT", "connected": True}))
    assert probe_status("127.0.0.1", port, timeout_s=2) is None
    assert neon_address.probe_status_note("127.0.0.1", port, timeout_s=2) == (None, None)
    _Handler.reply = (200, _status({**gaze, "connected": False}, {"sensor": "world", "conn_type": "DIRECT", "connected": True}))
    assert probe_status("127.0.0.1", port, timeout_s=2) == "Companion app found, but the glasses are not connected to the phone"
    _Handler.reply = (200, _status())  # no Sensor entries at all: as before
    assert probe_status("127.0.0.1", port, timeout_s=2) is None
    _Handler.reply = (200, _status({**gaze, "ip": "10.1.2.3"}))
    reason, note = neon_address.probe_status_note("127.0.0.1", port, timeout_s=2)
    assert reason is None and "10.1.2.3" in note and "127.0.0.1" in note


def test_probe_turns_a_malformed_http_reply_into_a_reason_not_a_crash():
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        conn.recv(4096)
        conn.sendall(b"this is not http" + bytes([13, 10, 13, 10]))
        conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        reason = probe_status("127.0.0.1", server.getsockname()[1], timeout_s=2)
    finally:
        server.close()
    assert reason is not None and "not the Neon Companion app" in reason


def test_an_unexpected_probe_exception_is_that_candidates_reason_and_a_note_becomes_the_warning():
    def probe(address, port):
        if address == "10.0.0.1":
            raise RuntimeError("boom")
        return None, "the gaze stream will come from 9.9.9.9, not 10.0.0.2"

    r = resolve_neon_address("10.0.0.1", 8080, probe=probe, discover=lambda: [("10.0.0.2", 8080)], gateways=lambda: [])
    assert r.ok and r.address == "10.0.0.2" and "9.9.9.9" in r.warning
    assert not r.attempts[0].ok and "RuntimeError: boom" in r.attempts[0].detail
    assert "9.9.9.9" in r.attempts[1].detail


# -- NeonGazeSource never leaks the aiohttp chain or the half-built Device's __del__ noise ---------------------------


class HalfBuiltDevice:
    """Like pupil_labs' simple Device when its status request fails: __init__ raises before _event_manager is set and __del__
    (close()) then reads it."""

    constructed = 0

    def __init__(self, address, port):
        HalfBuiltDevice.constructed += 1
        raise OSError("Cannot connect to host 192.168.0.60:8080 ssl:default [The semaphore timeout period has expired]")

    def __del__(self):
        self._event_manager  # noqa: B018 - attribute access is the point (raises AttributeError like the real library)


@pytest.fixture
def fake_library(monkeypatch):
    module = SimpleNamespace(Device=HalfBuiltDevice, discover_one_device=lambda **_: None)
    monkeypatch.setitem(sys.modules, "pupil_labs.realtime_api.simple", module)
    HalfBuiltDevice.constructed = 0
    return module


def test_dead_address_fails_with_one_plain_error_without_building_a_device(fake_library, monkeypatch):
    monkeypatch.setattr(neon_address, "probe_status", lambda a, p: "no answer within 2 s")
    source = NeonGazeSource(address="192.168.0.60", port=8080)
    with pytest.raises(NeonConnectionError, match=r"192\.168\.0\.60:8080 does not answer \(no answer within 2 s\)") as info:
        source.start()
    assert HalfBuiltDevice.constructed == 0
    assert info.value.__cause__ is None


def test_a_device_failing_after_a_good_probe_raises_plainly_and_its_del_noise_is_swallowed(fake_library, monkeypatch):
    monkeypatch.setattr(neon_address, "probe_status", lambda a, p: None)
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    with pytest.raises(NeonConnectionError, match="Could not connect to the Neon phone at 192.168.0.60:8080") as info:
        neon_source._connect("192.168.0.60", 8080)
    assert info.value.__cause__ is None and info.value.__context__ is None  # no aiohttp chain under it
    del info
    gc.collect()
    assert HalfBuiltDevice.constructed == 1
    assert seen == []  # the AttributeError from __del__ never reached the hook
    assert sys.unraisablehook == seen.append  # and the hook is restored


def test_other_unraisable_errors_still_reach_the_normal_hook(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)

    class Noisy:
        def __del__(self):
            raise ValueError("unrelated")

    def make(address, port):
        Noisy()
        raise OSError("down")

    with pytest.raises(NeonConnectionError):
        neon_source._build_device(make, "1.2.3.4", 8080)
    assert len(seen) == 1 and isinstance(seen[0].exc_value, ValueError)


def test_discovery_errors_become_one_plain_error(fake_library, monkeypatch):
    def boom(**_):
        raise RuntimeError("zeroconf")

    fake_library.discover_one_device = boom
    with pytest.raises(NeonConnectionError, match="Searching for the Neon phone failed"):
        NeonGazeSource(address=None).start()
    fake_library.discover_one_device = lambda **_: None
    with pytest.raises(NeonConnectionError, match="No Pupil Labs Neon found"):
        NeonGazeSource(address=None).start()
