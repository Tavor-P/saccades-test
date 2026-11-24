"""Find the Neon phone before the experiment window opens (run_experiment.main) and explain plainly when it can't be found.

Candidates, in order, each checked with one Real-Time API status request (GET /api/status, short timeout):
(a) the address typed in the startup dialog, (b) auto-discovery (mDNS, short search), (c) the PC's default IPv4 gateway(s) on the
Neon port - on a phone hotspot the phone IS the gateway. Stdlib only; `pupil_labs` is imported lazily inside discovery, so the
simulator and tests never need it. Every network call is injectable for tests."""

import http.client
import ipaddress
import json
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from include.eye_tracking.constants import NEON_DEVICE_PORT, NEON_PREFLIGHT_DISCOVERY_S, NEON_PROBE_TIMEOUT_S
from src import timebase

SOURCE_GIVEN = "address in the dialog"
SOURCE_DISCOVERY = "auto-discovery"
SOURCE_GATEWAY = "PC's gateway (phone hotspot)"
_COMMAND_TIMEOUT_S = 3.0
_HOSTNAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)*$")


@dataclass(frozen=True)
class Attempt:
    """One thing tried: where the candidate came from, the address (None for a search that found nothing) and what happened."""

    source: str
    address: str | None
    port: int | None
    ok: bool
    detail: str

    def line(self) -> str:
        where = f"{self.address}:{self.port}" if self.address else "(no address)"
        return f"{'OK  ' if self.ok else 'FAIL'} {where:<22} {self.source}: {self.detail}"


@dataclass(frozen=True)
class Resolution:
    address: str | None
    port: int | None
    source: str | None
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)
    warning: str | None = None  # the Neon answered, but something is odd (e.g. gaze stream comes from another address)

    @property
    def ok(self) -> bool:
        return self.address is not None

    def label(self) -> str:
        """For session.json hardware.neon_device."""
        return f"{self.address}:{self.port} ({self.source})" if self.ok else "not found"


def parse_address(text: str | None, default_port: int = NEON_DEVICE_PORT) -> tuple[str | None, int]:
    """'192.168.43.1' or '192.168.43.1:8080' (or a host name; a leading http(s):// and trailing '/' are ignored) -> (host, port); blank -> (None, default_port). Raises ValueError."""
    text = (text or "").strip()
    text = re.sub(r"^https?://", "", text, flags=re.IGNORECASE).rstrip("/").strip()  # pasted 'http://192.168.43.1:8080/'
    if not text:
        return None, default_port
    host, port = text, default_port
    if text.count(":") == 1:
        host, port_text = text.split(":")
        if not port_text.strip().isdigit():
            raise ValueError("Neon address: the port after ':' must be a number")
        port = int(port_text)
        host = host.strip()
    if not 1 <= port <= 65535:
        raise ValueError("Neon address: port must be between 1 and 65535")
    if re.fullmatch(r"[\d.]+", host):
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            raise ValueError("Neon address: not a valid IP address (example: 192.168.43.1)") from None
    elif not _HOSTNAME.match(host):
        raise ValueError("Neon address: enter an IP like 192.168.43.1 or 192.168.43.1:8080, or leave blank")
    return host, port


# -- the network calls (each replaceable in tests) -----------------------------------------------


def _gaze_problem(body: dict, address: str) -> tuple[str | None, str | None]:
    """(reason, note) from a status reply's component list. Sensor entries are {'model': 'Sensor', 'data': {sensor, connected, ip, ...}}.
    No Sensor entries at all -> accepted (not stricter than the reply format); Sensor entries but no connected gaze sensor -> reason;
    the gaze sensor's ip differing from the probed address -> note only."""
    result = body.get("result")
    if not isinstance(result, list):
        return None, None
    sensors = []
    for component in result:
        if isinstance(component, dict) and component.get("model") == "Sensor" and isinstance(component.get("data"), dict):
            sensors.append(component["data"])
    if not sensors:
        return None, None
    gaze = [d for d in sensors if d.get("sensor") == "gaze" and d.get("connected")]
    if not gaze:
        return "Companion app found, but the glasses are not connected to the phone", None
    ip = gaze[0].get("ip")
    if ip and str(ip) != address:
        return None, f"the gaze stream will come from {ip}, not {address}"
    return None, None


def probe_status_note(address: str, port: int, timeout_s: float = NEON_PROBE_TIMEOUT_S) -> tuple[str | None, str | None]:  # noqa: PLR0911 - search order, one return per outcome
    """(reason, note): reason is None if a Neon Companion app with connected glasses answers GET /api/status at address:port, else a
    short reason; note is a warning for an answering Neon (or None). Bypasses any system proxy (a campus proxy must not be asked
    for a 192.168.x.x address)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = timebase.now()
    try:
        with opener.open(f"http://{address}:{port}/api/status", timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except TimeoutError:
        return f"no answer within {timeout_s:g} s", None
    except urllib.error.HTTPError as exc:
        return f"something answered, but not the Neon Companion app (HTTP {exc.code})", None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return f"no answer within {timeout_s:g} s", None
        if isinstance(reason, ConnectionRefusedError):
            return "reachable, but nothing listens on this port (Companion app closed?)", None
        return f"cannot connect ({reason})", None
    except http.client.HTTPException:
        return "something answered, but not the Neon Companion app (not an HTTP status reply)", None
    except ValueError:
        return "something answered, but not the Neon Companion app (not its status reply)", None
    except OSError as exc:
        waited = timebase.now() - started
        return f"cannot connect after {waited:.1f} s ({exc})", None
    if not (isinstance(body, dict) and "result" in body):
        return "something answered, but not the Neon Companion app (unexpected reply)", None
    return _gaze_problem(body, address)


def probe_status(address: str, port: int, timeout_s: float = NEON_PROBE_TIMEOUT_S) -> str | None:
    """None if a Neon Companion app with connected glasses answers at address:port; otherwise a short reason."""
    return probe_status_note(address, port, timeout_s)[0]


def discover_addresses(timeout_s: float = NEON_PREFLIGHT_DISCOVERY_S) -> list[tuple[str, int]]:
    """Addresses of the first Neon found by mDNS within timeout_s ([] if none). Raises if the library is missing or the search fails."""
    import asyncio

    from pupil_labs.realtime_api.discovery import Network

    async def _find():
        async with Network() as network:
            return await network.wait_for_new_device(timeout_s)

    info = asyncio.run(_find())
    if info is None:
        return []
    return [(str(a), int(info.port)) for a in info.addresses]


def _run(command: list[str], timeout_s: float = _COMMAND_TIMEOUT_S) -> str:
    """A console command's output ('' on any failure), without flashing a console window."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(command, capture_output=True, timeout=timeout_s, creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.decode(errors="replace")


def parse_default_gateways(route_print: str) -> list[str]:
    """IPv4 default gateways from `route print -4` (Windows). Only the numeric columns are read, so a non-English Windows works too:
    rows '0.0.0.0  0.0.0.0  <gateway>  <interface>  <metric>'; 'On-link' rows are skipped."""
    found: list[str] = []
    for line in route_print.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
            try:
                gateway = str(ipaddress.IPv4Address(parts[2]))
            except ValueError:
                continue
            if gateway != "0.0.0.0" and gateway not in found:
                found.append(gateway)
    return found


def parse_linux_gateways(ip_route: str) -> list[str]:
    return list(dict.fromkeys(m.group(1) for m in re.finditer(r"^default via (\d+\.\d+\.\d+\.\d+)", ip_route, re.MULTILINE)))


def default_gateways(run=_run) -> list[str]:
    if sys.platform == "win32":
        return parse_default_gateways(run(["route", "print", "-4"]))
    return parse_linux_gateways(run(["ip", "-4", "route"]))


def local_ipv4() -> str | None:
    """The PC's IPv4 address on the default route (a UDP 'connect' sends nothing)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def parse_wifi_name(netsh_output: str) -> str | None:
    """The SSID from `netsh wlan show interfaces` (the 'SSID' label is not translated); None if not on Wi-Fi."""
    for line in netsh_output.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "SSID" and value.strip():
            return value.strip()
    return None


def network_summary(run=_run) -> dict:
    """What the PC is connected to, best effort: {'wifi_name', 'ipv4', 'gateways'} (values None/[] when unknown)."""
    wifi = parse_wifi_name(run(["netsh", "wlan", "show", "interfaces"])) if sys.platform == "win32" else None
    return {"wifi_name": wifi, "ipv4": local_ipv4(), "gateways": default_gateways(run)}


# -- resolution -----------------------------------------------------------------------------------


def resolve_neon_address(  # noqa: C901 - address search order, split later
    given: str | None,
    port: int = NEON_DEVICE_PORT,
    *,
    probe=probe_status_note,
    discover=discover_addresses,
    gateways=default_gateways,
    progress=None,
) -> Resolution:
    """Try the given address, then discovery, then the default gateway(s); return the first that answers plus everything tried.
    `probe` returns a reason (None = ok) or a (reason, note) pair; a note ends up in Resolution.warning."""
    attempts: list[Attempt] = []
    tried: set[tuple[str, int]] = set()

    def say(text: str) -> None:
        if progress is not None:
            progress(text)

    def check(source: str, address: str, port_: int) -> Resolution | None:
        if (address, port_) in tried:
            return None
        tried.add((address, port_))
        say(f"  trying {address}:{port_} ({source}) ...")
        try:
            result = probe(address, port_)
            reason, note = result if isinstance(result, tuple) else (result, None)
        except Exception as exc:  # one bad probe must never stop the search
            reason, note = f"check failed ({type(exc).__name__}: {exc})", None
        detail = reason if reason is not None else "Neon answered" + (f" ({note})" if note else "")
        attempts.append(Attempt(source, address, port_, reason is None, detail))
        return Resolution(address, port_, source, tuple(attempts), note) if reason is None else None

    if given:
        if found := check(SOURCE_GIVEN, given, port):
            return found

    say("  searching the network (auto-discovery) ...")
    try:
        discovered = discover()
    except ImportError:
        discovered, problem = [], "pupil-labs-realtime-api is not installed"
    except Exception as exc:  # zeroconf/OS errors: report and move on to the gateway
        discovered, problem = [], f"search failed ({type(exc).__name__}: {exc})"
    else:
        problem = "no Neon announced itself (often blocked on Wi-Fi marked 'Public' or on guest networks)"
    if not discovered:
        attempts.append(Attempt(SOURCE_DISCOVERY, None, None, False, problem))
    for address, port_ in discovered:
        if found := check(SOURCE_DISCOVERY, address, port_):
            return found

    try:
        gateway_list = gateways()
    except Exception as exc:
        gateway_list = []
        attempts.append(Attempt(SOURCE_GATEWAY, None, None, False, f"could not read the gateway ({exc})"))
    else:
        if not gateway_list:
            attempts.append(Attempt(SOURCE_GATEWAY, None, None, False, "the PC has no default gateway (not connected to a network?)"))
    for gateway in gateway_list:
        if found := check(SOURCE_GATEWAY, gateway, port):
            return found
    return Resolution(None, None, None, tuple(attempts))


FAILURE_TITLE = "Neon glasses not found"
CHECKS = (
    "Check:",
    "  1. The phone and this PC are on the same network.",
    "  2. Guest, campus or hotel Wi-Fi often blocks devices from seeing each other - use the phone's hotspot instead.",
    "  3. Phone hotspot: join the hotspot from this PC, keep the Neon Companion app open with the glasses plugged in,",
    "     and type the address the Companion app shows into 'Neon address' in the start dialog (or leave it blank).",
    "  4. If this PC can't see the hotspot, turn on the hotspot's 2.4 GHz / 'Maximise compatibility' option.",
)


def failure_message(resolution: Resolution, network: dict | None = None) -> str:
    """The plain-language message shown (console + message box) when no address answered."""
    network = network or {}
    lines = [
        "Could not reach the Neon phone, so the experiment did not start.",
        "",
        "What was tried:",
        *(f"  {a.line()}" for a in resolution.attempts),
        "",
        "This PC:",
        f"  Wi-Fi network: {network.get('wifi_name') or 'unknown (or not on Wi-Fi)'}",
        f"  IPv4 address:  {network.get('ipv4') or 'unknown'}",
        f"  Gateway:       {', '.join(network.get('gateways') or []) or 'none'}",
        "",
        *CHECKS,
    ]
    return "\n".join(lines)
