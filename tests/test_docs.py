"""Keeps CLAUDE.md honest: every file path it mentions in backticks must exist.
If this fails, either fix the doc (preferred) or restore the file."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "CLAUDE.md").read_text(encoding="utf8")

# Runtime output / user-local / generated paths the doc may mention without them being in git.
_EXEMPT_PREFIXES = ("data/", "models/", "results_", "C:", "%", "$", "include/**", "tests/.../")
_RUNTIME_FILES = {"_meta.json", "settings.json"}  # written under data/ at run time
_PATH_RE = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|md|ps1|sh|csv|txt|json|ini)|[A-Za-z0-9_.-]+/(?:[A-Za-z0-9_.-]+/)*)(?:::[\w]+)?`")


def _all_basenames() -> set[str]:
    return {p.name for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}


def _referenced_paths(text: str) -> list[str]:
    return sorted({m.group(1).replace("\\", "/") for m in _PATH_RE.finditer(text)})


def test_every_path_in_claude_md_exists():
    basenames = _all_basenames()
    missing = []
    for ref in _referenced_paths(DOC):
        if ref.startswith(_EXEMPT_PREFIXES) or ref.startswith("./") or ref in _RUNTIME_FILES:
            continue
        if "/" in ref:
            if not (ROOT / ref).exists():
                missing.append(ref)
        elif ref not in basenames:  # bare file name, e.g. `neon_source.py`
            missing.append(ref)
    assert not missing, f"CLAUDE.md mentions paths that don't exist: {missing}"


def test_repo_map_section_exists_and_is_nonempty():
    section = DOC.split("## Repo map", 1)[1].split("\n## ", 1)[0]
    assert len(_referenced_paths(section)) >= 20
