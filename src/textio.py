"""Reading the experiment's own text files (no PsychoPy, no logging).

Files are written as UTF-8 (`encoding="utf-8"`, plus `newline=""` for csv). Runs recorded by older versions used the Windows
locale encoding (cp1252) for anything non-ASCII, so every reader goes through `read_text` / `read_csv` here, which fall back to cp1252.
"""

import csv
import io
from pathlib import Path


def read_text(path: Path | str) -> str:
    """The file's text as UTF-8 (a BOM is dropped); a file that is not valid UTF-8 was written with the Windows
    locale encoding by an older version and is read as cp1252 (the few bytes cp1252 leaves undefined become U+FFFD)."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def read_csv(path: Path | str) -> csv.DictReader:
    """A DictReader over the whole file (read through `read_text`, with csv's own newline handling). Raises OSError for a missing file."""
    return csv.DictReader(io.StringIO(read_text(path), newline=""))
