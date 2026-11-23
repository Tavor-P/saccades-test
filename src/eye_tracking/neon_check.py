"""Standalone Neon connection check: `.\\scripts\\py.ps1 -m src.eye_tracking.neon_check` (or double-click neon_check.bat).

Runs the same address resolution as the experiment's start-up check (neon_address.resolve_neon_address), opens no window,
prints the result in plain words and writes neon_check_report.txt in the repo root (git-ignored) so it can be read later,
e.g. after testing on a phone hotspot that has no internet. The report holds network and device facts only - never
participant data. Every network/device call is injectable so tests never touch the real Neon."""

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

from include.eye_tracking.constants import NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT, NEON_PROBE_TIMEOUT_S
from src import timebase
from src.eye_tracking import neon_address
from src.eye_tracking.neon_address import parse_address, resolve_neon_address

REPORT_PATH = Path(__file__).resolve().parents[2] / "neon_check_report.txt"
GAZE_SAMPLE_S = 3.0
GAZE_START_TIMEOUT_S = 10.0  # the stream is set up on the first receive call; give up if no sample by then
ADAPTERS_TIMEOUT_S = 20.0  # Get-NetIPConfiguration took ~11 s on the lab PC
_ADAPTERS_PS = (
    "Get-NetIPConfiguration | Where-Object { $_.IPv4Address } | ForEach-Object { [pscustomobject]@{"
    " name = $_.InterfaceAlias; description = $_.InterfaceDescription;"
    " ipv4 = (@($_.IPv4Address.IPAddress) -join ', '); gateway = (@($_.IPv4DefaultGateway.NextHop) -join ', ');"
    " network = \"$($_.NetProfile.Name)\"; category = \"$($_.NetProfile.NetworkCategory)\" } } | ConvertTo-Json -Compress"
)


def parse_adapters(json_text: str) -> list[dict]:
    """PowerShell's ConvertTo-Json output (one object or a list) -> list of adapter dicts; [] if unreadable."""
    try:
        data = json.loads(json_text) if json_text.strip() else []
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [d for d in data if isinstance(d, dict)]


def network_adapters(run=neon_address._run) -> list[dict]:
    if sys.platform != "win32":
        return []
    return parse_adapters(run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _ADAPTERS_PS], ADAPTERS_TIMEOUT_S))


def fetch_status(address: str, port: int, timeout_s: float = NEON_PROBE_TIMEOUT_S) -> dict:
    """The Companion app's GET /api/status reply (raises on any failure)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{address}:{port}/api/status", timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def summarize_status(status: dict) -> list[str]:
    """Battery, glasses and sensor lines from a status reply. Reads only device facts (no wearer/recording names)."""
    lines = []
    items = status.get("result") if isinstance(status, dict) else None
    if not isinstance(items, list):
        return ["  (status reply not understood)"]
    sensors = []
    raw = []
    glasses = None
    for item in items:
        if not isinstance(item, dict):
            continue
        model, data = item.get("model"), item.get("data") or {}
        if model == "Phone":
            lines.append(f"  Phone battery: {data.get('battery_level', '?')} % ({data.get('battery_state', '?')})")
            lines.append(f"  Phone memory:  {data.get('memory_state', '?')}")
        if model in ("Hardware", "Sensor"):
            raw.append(f"  raw {model}: {json.dumps(data, sort_keys=True)}")
        if model == "Hardware":
            # Neon reports its module serial; Invisible its glasses serial; "-1"/blank = not plugged in
            serials = (str(data.get(key) or "").strip() for key in ("module_serial", "glasses_serial"))
            glasses = any(s and s not in ("-1", "unknown", "default") for s in serials)
        elif model == "Sensor":
            sensors.append(f"{data.get('sensor', '?')}({data.get('conn_type', '?')})={'on' if data.get('connected') else 'off'}")
    if glasses is not None:
        lines.append(f"  Glasses connected: {'yes' if glasses else 'NO - plug the glasses into the phone'} (guessed from the serials; unverified on hardware)")
    if sensors:
        lines.append(f"  Sensors: {', '.join(sensors)}")
    return (lines + raw) or ["  (status reply had no phone/glasses information)"]


def measure_gaze_rate(
    address: str, port: int, seconds: float = GAZE_SAMPLE_S, start_timeout_s: float = GAZE_START_TIMEOUT_S
) -> tuple[int, float, float | None]:
    """(samples, seconds counted, seconds until the first sample or None) from the Real-Time gaze stream. Counting starts at the
    first sample, so the stream's set-up time doesn't lower the rate. Lazy device import; raises NeonConnectionError on failure."""
    from pupil_labs.realtime_api.simple import Device

    from src.eye_tracking.neon_source import _build_device

    device = _build_device(Device, address, port)
    try:
        opened = timebase.now()
        while device.receive_gaze_datum(timeout_seconds=0.5) is None:
            if timebase.now() - opened >= start_timeout_s:
                return 0, 0.0, None
        start = timebase.now()
        first_sample_s = start - opened
        count = 0
        while (elapsed := timebase.now() - start) < seconds:
            if device.receive_gaze_datum(timeout_seconds=0.5) is not None:
                count += 1
        return count, elapsed, first_sample_s
    finally:
        device.close()


def build_report(
    address_text: str | None,
    *,
    resolve=resolve_neon_address,
    adapters=network_adapters,
    network=neon_address.network_summary,
    status=fetch_status,
    gaze_rate=measure_gaze_rate,
    progress=None,
    stamp=None,
) -> tuple[bool, str]:
    """(connected, report text). Nothing here raises: every failure becomes a line in the report."""
    host, port = parse_address(address_text)
    resolution = resolve(host, port, progress=progress)
    out = [
        "Neon connection check",
        f"Time: {stamp or datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Address typed: {(address_text or '').strip() or '(blank = find automatically)'}",
        "",
        "Network adapters (IPv4 / gateway / network / category):",
    ]
    if progress is not None:
        progress("  reading the network adapters (can take ~15 s) ...")
    try:
        adapter_list = adapters()
    except Exception as exc:  # report, never crash
        adapter_list, out = [], out + [f"  could not list adapters ({exc})"]
    for a in adapter_list:
        out.append(f"  {a.get('name') or '?'}: {a.get('ipv4') or '-'} / gw {a.get('gateway') or '-'} / "
                   f"{a.get('network') or '-'} / {a.get('category') or '-'}")
    if not adapter_list:
        try:
            net = network()
        except Exception:
            net = {}
        out.append(f"  (adapter list unavailable) Wi-Fi: {net.get('wifi_name') or '?'}, IPv4: {net.get('ipv4') or '?'}, "
                   f"gateway: {', '.join(net.get('gateways') or []) or 'none'}")
    out += ["", "What was tried:", *(f"  {a.line()}" for a in resolution.attempts)]
    discovery = [a for a in resolution.attempts if a.source == neon_address.SOURCE_DISCOVERY]
    if not discovery:
        found = "not needed (the typed address answered first)"
    elif any(a.address for a in discovery):
        found = "yes: " + ", ".join(f"{a.address}:{a.port}" for a in discovery if a.address)
    else:
        found = "no (" + discovery[0].detail + ")"
    out += ["", f"Auto-discovery found a Neon: {found}", ""]
    if not resolution.ok:
        out += ["RESULT: NOT CONNECTED - no address answered.", "", *neon_address.CHECKS]
        return False, "\n".join(out) + "\n"
    out.append(f"RESULT: CONNECTED to {resolution.label()}")
    out.append("Phone status:")
    try:
        out += summarize_status(status(resolution.address, resolution.port))
    except Exception as exc:
        out.append(f"  could not read the status ({type(exc).__name__}: {exc})")
    try:
        count, seconds, first_sample_s = gaze_rate(resolution.address, resolution.port)
        if first_sample_s is None:
            out.append(f"Gaze stream: NO GAZE within {GAZE_START_TIMEOUT_S:g} s - glasses plugged in and worn?")
        else:
            rate = count / seconds if seconds > 0 else 0.0
            verdict = "OK" if rate >= 100 else "LOW - glasses worn and plugged in?"
            out.append(f"Gaze stream: first sample after {first_sample_s:.1f} s; then {count} samples in {seconds:.1f} s = "
                       f"{rate:.0f} per second ({verdict}; Neon streams ~200)")
    except Exception as exc:
        out.append(f"Gaze stream: failed ({type(exc).__name__}: {exc})")
    if address_text is None or resolution.address != host:
        out.append(f"Tip: type {resolution.address}" + (f":{resolution.port}" if resolution.port != NEON_DEVICE_PORT else "")
                   + " into 'Neon address' in the start dialog.")
    return True, "\n".join(out) + "\n"


def main(argv=None, report_path: Path = REPORT_PATH) -> int:
    parser = argparse.ArgumentParser(description="Check that this PC can reach the Neon phone (no window, no participant data).")
    parser.add_argument("address", nargs="?", default=None,
                        help=f"phone IP or IP:port (default: the start dialog's last value, else {NEON_DEVICE_ADDRESS})")
    parser.add_argument("--report", type=Path, default=report_path, help="where to write the text report")
    args = parser.parse_args(argv)
    address_text = args.address if args.address is not None else _remembered_address()
    try:
        parse_address(address_text)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print("Checking the connection to the Neon phone (takes up to ~30 s)...", flush=True)
    ok, text = build_report(address_text, progress=lambda line: print(line, flush=True))
    print("", text, sep="\n")
    try:
        args.report.write_text(text, encoding="utf-8")
        print(f"Report saved: {args.report}")
    except OSError as exc:
        print(f"Could not save the report ({exc})", file=sys.stderr)
    return 0 if ok else 1


def _remembered_address() -> str:
    """The start dialog's remembered 'Neon address' (data/settings.json, read only), else the constant."""
    try:
        from src.experiment.settings import load_last_startup

        value = load_last_startup().get("neon_address")
    except Exception:
        value = None
    return value if isinstance(value, str) else (NEON_DEVICE_ADDRESS or "")


if __name__ == "__main__":
    sys.exit(main())
