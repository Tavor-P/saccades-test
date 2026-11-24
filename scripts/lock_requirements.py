"""Regenerate requirements-lock.txt: exact installed versions of requirements.txt and its dependencies.

    scripts/py.ps1 scripts/lock_requirements.py

Reads the environment only (importlib.metadata); installs nothing. Packages that come only from
PsychoPy are left out unless something in requirements.txt needs them.
"""

import re
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(_norm(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]))
    return names


def _dependencies(dist: metadata.Distribution) -> list[str]:
    deps = []
    for req in dist.requires or []:
        if "extra ==" in req:  # optional extras are not installed by default
            continue
        marker = req.split(";", 1)[1] if ";" in req else ""
        if "win32" in marker and sys.platform != "win32":
            continue
        deps.append(_norm(re.split(r"[<>=!~;\[ (]", req.strip(), maxsplit=1)[0]))
    return deps


def locked_versions(requirements_text: str) -> dict[str, str]:
    installed = {_norm(d.metadata["Name"]): d for d in metadata.distributions()}
    found: dict[str, str] = {}
    todo = _requirement_names(requirements_text)
    while todo:
        name = todo.pop()
        if name in found:
            continue
        dist = installed.get(name)
        if dist is None:
            print(f"warning: {name} is not installed, left out", file=sys.stderr)
            continue
        found[name] = dist.metadata["Name"] + "==" + dist.version
        todo.extend(_dependencies(dist))
    return dict(sorted(found.items()))


def main() -> None:
    pins = locked_versions((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    header = (
        f"# Exact versions of requirements.txt + its dependencies, as installed on {date.today().isoformat()}.\n"
        f"# Python {sys.version.split()[0]} (PsychoPy's). Regenerate: scripts/py.ps1 scripts/lock_requirements.py\n"
        "# Install: scripts/py.ps1 -m pip install -r requirements-lock.txt (requirements.txt gives looser versions)\n"
    )
    (ROOT / "requirements-lock.txt").write_text(header + "\n".join(pins.values()) + "\n", encoding="utf-8", newline="\r\n")
    print(f"locked {len(pins)} packages -> requirements-lock.txt")


if __name__ == "__main__":
    main()
