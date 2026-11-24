"""The dashboard's comparison, group-label and study-summary pages. Every number comes from `compare.compare` /
`study_summary.build_rows` (which use analyze_participant); this module only checks the selection, reads and writes the
labels, picks the output folder and reshapes the result for the page.

Selection: a side is a list of tokens: participant IDs (must be a participant folder with runs), `group:<label>` (must be
an existing label in data/participants.json) or `all`. Only these tokens ever go into a URL. The output of one selection
goes to a fixed folder, data/comparisons/dashboard/<hash of the resolved selection>/, so re-opening a bookmark overwrites
it instead of piling up folders; downloads re-resolve the query to find it, so no path or hash is ever in a URL.

Every data-folder call passes `data_dir=results._data_dir()` (tests point `data_access.DATA_DIR` at a tmp folder; the
experiment modules' own default would be the real data/). Analyses run under `results._LOCK` (pyplot is not thread-safe)."""

import hashlib
from pathlib import Path

from src.dashboard import results
from src.dashboard.results import FriendlyError
from src.experiment import compare as cmp
from src.experiment import participants as labels_module
from src.experiment import study_summary

N_BOOT = cmp.N_BOOT  # bootstrap resamples for the comparison CI (tests lower it)
DASHBOARD_DIR = "dashboard"  # data/comparisons/dashboard/<hash>/
FILES = {"csv": "comparison.csv", "json": "comparison.json", "report": "report.txt", "plot": "plot.png"}
DOWNLOAD_LABELS = results.DOWNLOAD_LABELS  # one table for all the pages (results.py)
MAX_GROUP_CHARS = 40
MAX_NOTE_CHARS = 200

HOW_TO_READ = {
    cmp.ONE_VS_ONE: ("Two people: the difference between their two suppression indices, with a 95 % interval from refitting each "
                     "person's flashes many times. If the interval includes 0, these data cannot tell the two apart. There is no "
                     "p-value: with one person per side there is no between-person spread to test against."),
    cmp.GROUP_VS_GROUP: ("Two groups: each person counts once, by their own index. The headline compares the group medians; the "
                         "p-value (Mann-Whitney) is the chance of a difference this large if both groups came from the same "
                         "population. A large p does NOT show the groups are the same, and with groups this small even a real "
                         "difference often gives a large p."),
    cmp.ONE_VS_GROUP: ("One person against a group: where that person's index falls among the group members (rank and percentile). "
                       "There is no p-value: a single person gives no estimate of how much people differ from one another, so "
                       "there is nothing to test against; the position is descriptive only."),
}
NOT_COMPARABLE_HELP = "Nothing could be compared; the table below says why each person was left out."


# -- selection -----------------------------------------------------------------------------------


def tokens(values: list[str]) -> list[str]:
    """Query values (repeated `a=P01&a=P02` and/or comma lists `a=P01,P02`) as stripped tokens, in order, no duplicates."""
    out: list[str] = []
    for value in values:
        for token in (t.strip() for t in str(value).split(",")):
            if token and token not in out:
                out.append(token)
    return out


def group_names() -> list[str]:
    """Existing group labels (as first typed; case-insensitive duplicates merged), sorted."""
    seen: dict[str, str] = {}
    for entry in labels_module.list_labels(results._data_dir()).values():
        group = entry["group"].strip()
        if group:
            seen.setdefault(group.casefold(), group)
    return sorted(seen.values(), key=str.casefold)


def resolve(side: str, side_tokens: list[str]) -> list[str]:
    """Participant IDs for one side's tokens. Raises FriendlyError for an empty side, an unknown ID or an unknown group."""
    if not side_tokens:
        raise FriendlyError(f"Set {side} is empty: tick at least one participant (or pick a group) on each side.", status=400)
    known = results.participant_ids()
    groups = {g.casefold(): g for g in group_names()}
    out: list[str] = []
    for token in side_tokens:
        if token.lower().startswith("group:"):
            name = token.split(":", 1)[1].strip()
            if name.casefold() not in groups:
                raise FriendlyError(f"Set {side}: there is no group '{name}'. Existing groups: {', '.join(groups.values()) or 'none yet'} "
                                    "(label participants on the sessions page).", status=404)
            ids = labels_module.participants_in_group(name, results._data_dir())
        elif token.lower() == "all":
            ids = known
        elif token in known:
            ids = [token]
        else:
            raise FriendlyError(f"Set {side}: there is no participant '{token}' with runs in the data folder.", status=404)
        out += [i for i in ids if i not in out]
    if not out:
        raise FriendlyError(f"Set {side} has nobody in it (the group has no members).", status=400)
    return out


def resolve_pair(a_tokens: list[str], b_tokens: list[str]) -> tuple[list[str], list[str]]:
    a, b = labels_module.everyone_else(a_tokens, resolve("A", a_tokens), b_tokens, resolve("B", b_tokens))  # "P01 vs all" = the rest
    if not a or not b:
        raise FriendlyError(f"Set {'A' if not a else 'B'} has nobody left: 'all' means everyone not on the other side.", status=400)
    both = [p for p in a if p in b]
    if both:
        raise FriendlyError(f"{', '.join(both)} {'is' if len(both) == 1 else 'are'} in both set A and set B. A person can only be on "
                            "one side: untick them on one side.", status=400)
    return a, b


def output_dir(a: list[str], b: list[str], include_test: bool) -> Path:
    key = "|".join([",".join(a), ",".join(b), "test" if include_test else "real"])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return results._data_dir() / cmp.COMPARISONS_DIR / DASHBOARD_DIR / digest


def output_file(a_tokens: list[str], b_tokens: list[str], include_test: bool, kind: str) -> Path | None:
    """The written file of this selection, or None (unknown kind, bad selection, not written yet)."""
    if kind not in FILES:
        return None
    try:
        a, b = resolve_pair(a_tokens, b_tokens)
    except FriendlyError:
        return None
    path = output_dir(a, b, include_test) / FILES[kind]
    return path if path.is_file() else None


# -- the comparison ------------------------------------------------------------------------------


def _ci_text(ci) -> str:
    return "" if not ci else f"{ci[0]:.2f} to {ci[1]:.2f}"


def _person_row(p, known: set) -> dict:
    return {
        "set": p.side, "participant": p.participant, "group": p.group or "", "has_page": p.participant in known,
        "index": "n/a" if p.index is None else f"{p.index:.2f}", "ci": _ci_text(p.ci), "method": p.method or "-",
        "n_saccade": p.n_saccade, "n_fixation": p.n_fixation, "runs_used": p.runs_used,
        "included": p.included, "reason": p.reason, "warnings": list(p.warnings),
    }


def run_comparison(a_tokens: list[str], b_tokens: list[str], include_test: bool) -> dict:
    """Runs compare() on the selection, writes its four files, returns what the page shows. Raises FriendlyError."""
    a, b = resolve_pair(a_tokens, b_tokens)
    data_dir = results._data_dir()
    out = output_dir(a, b, include_test)
    try:
        with results._LOCK:
            result = cmp.compare(a, b, data_dir=data_dir, include_test=include_test, n_boot=N_BOOT)
            cmp.write_comparison(result, out_dir=out, data_dir=data_dir)
    except ValueError as exc:  # compare's own checks (empty side, person on both sides); resolve_pair catches these first
        raise FriendlyError(f"These sets cannot be compared: {exc}.", status=400) from None
    except Exception as exc:  # noqa: BLE001 - shown as a sentence, traceback to the console
        raise results._unexpected("comparison", exc) from None
    included = [p for p in result.people if p.included]
    known = set(results.participant_ids())
    return {
        "headline": result.headline,
        "mode": result.mode,
        "mode_words": cmp.MODE_NAMES.get(result.mode or "", "not possible"),
        "requested_words": cmp.MODE_NAMES.get(result.requested_mode or "", ""),
        "how_to_read": HOW_TO_READ.get(result.mode or "", NOT_COMPARABLE_HELP),
        "name_a": result.name_a,
        "name_b": result.name_b,
        "people": [_person_row(p, known) for p in result.people],
        "everyone_excluded": not included,
        "warnings": list(result.warnings),
        "exploratory": result.exploratory,
        "details": list(result.details),
        "downloads": [k for k in FILES if (out / FILES[k]).is_file()],
        "report_text": result.report(),
        "generated_at": result.generated_at,
        "resolved_a": a,
        "resolved_b": b,
    }


def picker_rows() -> list[dict]:
    """One row per participant with runs for the two checkbox columns: ID, group, run counts."""
    labels = labels_module.list_labels(results._data_dir())
    return [{**row, "group": (labels.get(row["id"]) or {}).get("group", "")} for row in results.overview()]


# -- labels --------------------------------------------------------------------------------------


def overview_with_labels() -> list[dict]:
    """results.overview() plus each participant's group and note (for the sessions page)."""
    labels = labels_module.list_labels(results._data_dir())
    return [{**row, "group": (labels.get(row["id"]) or {}).get("group", ""), "note": (labels.get(row["id"]) or {}).get("note", "")}
            for row in results.overview()]


def save_label(pid: str, group, note) -> dict:
    """Stores one participant's group / note (both empty: label removed). Raises FriendlyError."""
    results.check_participant(pid)
    group = "" if group is None else str(group).strip()
    note = "" if note is None else str(note).strip()
    if len(group) > MAX_GROUP_CHARS or len(note) > MAX_NOTE_CHARS:
        raise FriendlyError(f"The group may have at most {MAX_GROUP_CHARS} characters and the note at most {MAX_NOTE_CHARS}.", status=400)
    if ":" in group or "," in group:
        raise FriendlyError("A group name cannot contain ':' or ',' (they separate names in a comparison).", status=400)
    data_dir = results._data_dir()
    if not group and not note:
        return labels_module.set_label(pid, None, None, data_dir=data_dir)
    return labels_module.set_label(pid, group, note, data_dir=data_dir)


# -- study summary -------------------------------------------------------------------------------


def study_page(include_test: bool) -> dict:
    """The study_summary rows (also writes data/study_summary.csv / .txt, as the CLI does). Raises FriendlyError."""
    data_dir = results._data_dir()
    try:
        with results._LOCK:
            out = study_summary.write_study_summary(data_dir, include_test)
    except Exception as exc:  # noqa: BLE001
        raise results._unexpected("study summary", exc) from None
    known = set(results.participant_ids())
    rows = [{**r, "has_page": r["participant"] in known} for r in out["rows"]]
    return {"rows": rows, "columns": study_summary.COLUMNS, "overview": out["overview"], "include_test": include_test}


def study_csv() -> Path | None:
    path = results._data_dir() / study_summary.CSV_FILE
    return path if path.is_file() else None
