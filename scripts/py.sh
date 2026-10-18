#!/usr/bin/env bash
# Same as py.ps1, for Git Bash: runs PsychoPy's bundled Python from the project root.
#   scripts/py.sh -m pytest -q
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for py in "$PSYCHOPY_PYTHON" "$LOCALAPPDATA/Programs/PsychoPy/python.exe" "/c/Program Files/PsychoPy/python.exe" "/c/Program Files (x86)/PsychoPy/python.exe"; do
  if [ -n "$py" ] && [ -x "$py" ]; then cd "$root" && exec "$py" "$@"; fi
done
echo "PsychoPy's python.exe not found - install PsychoPy Standalone or set PSYCHOPY_PYTHON." >&2
exit 1
