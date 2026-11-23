"""Compare participants: 1 vs 1, 1 vs group, group vs group (`-m src.experiment.compare "P01" "group:control"`).

Input per person is the pooled analysis of `analyze_participant` (data/<participant>/analysis/summary.json). A person whose
saved analysis is missing, damaged, of another format or older than their runs is (re)analysed first, with the same
numbers `analyze_participant` would give; this module never changes a person's index.

The mode follows from how many usable people each side has (after exclusions):
- 1 vs 1: the two pooled indices and the difference A - B with a bootstrap 95 % CI. Each person's flashes are resampled
  within every contrast x condition cell (binomially, exactly as `analyze_session.suppression_index` bootstraps its own CI)
  and refitted with the same Weibull fit; the accuracy fallback is resampled per condition instead. Two people whose
  indices came from different methods (threshold ratio vs accuracy fallback) are reported as not comparable.
- group vs group (2+ people each): one index per person. Median and range per group, the difference in medians with a
  bootstrap 95 % CI (people resampled), a two-sided Mann-Whitney U p-value and the rank-biserial effect size.
- 1 vs group: where the single person falls among the group (rank and percentile), the group's median and range. No
  p-value: one person gives no estimate of how much people vary, so there is nothing to test against.
The index compared is always the pooled one (summary.json `index`; for return-jump runs both legs together, not by_leg).
Always: people whose runs used other contrasts, screen geometry or run kind (single vs return-jump/<trigger>), whose index could not be computed, or (groups) who
used a different method from the majority are excluded and listed with the reason. Everything with fewer than
EXPLORATORY_BELOW people per side is labelled exploratory (feasibility pilot).

`compare()` returns a `ComparisonResult` (a dataclass with `to_dict()`, `report()`); it writes no comparison files.
`write_comparison()` writes comparison.json / comparison.csv / report.txt / plot.png (default data/comparisons/<stamp>_<A>_vs_<B>/).
"""

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only renders PNG files
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import session_files
from src.experiment.analyze_participant import (
    _GEOMETRY_FIELDS,
    ANALYSIS_DIR,
    SCHEMA_VERSION,
    SINGLE_RUN_KIND,
    NoRunsError,
    analyze_participant,
    find_runs,
)
from src.experiment.analyze_session import BIN_EDGES_MS, CHANCE, DATA_DIR, _fit_pair, wilson
from src.experiment.flash_gates import FIXATION, SACCADE
from src.experiment.participants import everyone_else, list_labels, resolve_set

ONE_VS_ONE = "one_vs_one"
ONE_VS_GROUP = "one_vs_group"
GROUP_VS_GROUP = "group_vs_group"
MODE_NAMES = {ONE_VS_ONE: "1 vs 1", ONE_VS_GROUP: "1 vs group", GROUP_VS_GROUP: "group vs group"}

MIN_FLASHES_PER_CONDITION = 12  # fewer saccade or fixation flashes than this: the person's index is unstable (warned)
MIN_GROUP_SIZE = 3  # a group side with fewer people than this is warned about
EXPLORATORY_BELOW = 10  # fewer people than this on either side: every result is labelled exploratory
N_BOOT = 2000
COMPARISONS_DIR = "comparisons"  # data/comparisons/<stamp>_<A>_vs_<B>/ (no trials.csv inside, so never mistaken for runs)
_EXACT_PERMUTATION_LIMIT = 200_000  # enumerate every split of the people up to this many splits
_RUN_FILES = (session_files.TRIALS_FILE, session_files.SESSION_FILE, session_files.GAZE_TRACE_FILE)
CSV_COLUMNS = ("participant", "group", "set", "included", "reason", "index", "ci_low", "ci_high", "method",
               "n_saccade", "n_fixation", "n_flashes_total", "runs_used", "warnings")


# -- one person ----------------------------------------------------------------------------------


@dataclass
class Person:
    """One participant as the comparison sees them (numbers straight from their pooled summary.json)."""

    participant: str
    side: str = ""  # "A" | "B"
    group: str = ""  # label from data/participants.json ("" = unlabelled)
    index: float | None = None
    ci: list | None = None
    method: str | None = None  # "psychometric" | "accuracy" | None (cannot be computed)
    n_saccade: int = 0
    n_fixation: int = 0
    n_total: int = 0
    correct_saccade: int = 0
    correct_fixation: int = 0
    contrasts_pct: list | None = None
    geometry: dict | None = None
    run_kind: str = SINGLE_RUN_KIND  # "single" | "return_jump/<trigger>"; schema-1 summaries have none = single
    runs_used: int = 0
    runs_excluded: int = 0
    notes: list = field(default_factory=list)  # the analysis' own notes
    included: bool = True
    reason: str = ""  # why excluded
    warnings: list = field(default_factory=list)
    analysis_source: str = ""  # "saved" | "re-analysed (<why>)" | "analysed in memory (<why>)"
    summary_path: str | None = None
    by_contrast: list = field(default_factory=list)  # summary.json by_contrast rows
    flashes: list = field(default_factory=list, repr=False)  # (rel_ms or None, correct, condition) per pooled flash
    run_records: list = field(default_factory=list, repr=False)  # summary.json runs_used rows (gaze quality per run)

    @property
    def compat_key(self) -> tuple:
        geometry = self.geometry or {}
        return (tuple(self.contrasts_pct or ()), tuple(geometry.get(k) for k in _GEOMETRY_FIELDS), self.run_kind)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("flashes")
        out.pop("run_records")
        return _jsonable(out)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _read_flash_csv(path: Path) -> list | None:
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return [(None if r["flash_rel_onset_ms"] in ("", None) else float(r["flash_rel_onset_ms"]), r["outcome"] == "correct", r["condition"]) for r in rows]
    except (OSError, KeyError, ValueError):
        return None


def stale_reason(participant: str, data_dir: Path, include_test: bool, summary: dict | None, summary_path: Path) -> str | None:
    """Why the saved pooled analysis can't be used as is (None = up to date). Raises NoRunsError when there are no runs."""
    _, entries = find_runs(participant, data_dir)
    if summary is None:
        return "no saved analysis" if not summary_path.exists() else "saved analysis is damaged"
    if summary.get("schema_version") != SCHEMA_VERSION:
        return f"saved analysis has schema_version {summary.get('schema_version')}, expected {SCHEMA_VERSION}"
    if bool(summary.get("include_test")) != bool(include_test):
        return "saved analysis " + ("left out" if include_test else "included") + " test runs"
    candidates = entries if include_test else [e for e in entries if e.mode != "test"]
    saved = {r.get("run") for r in (summary.get("runs_used") or []) + (summary.get("runs_excluded") or [])}
    if {e.name for e in candidates} != saved:
        return "runs were added or removed since the saved analysis"
    saved_at = summary_path.stat().st_mtime
    for e in candidates:
        for name in _RUN_FILES:
            path = e.ref.folder / name
            if path.exists() and path.stat().st_mtime > saved_at:
                return f"run {e.name} changed since the saved analysis"
    flashes_csv = summary_path.parent / str(summary.get("flashes_csv") or "flashes.csv")
    if not flashes_csv.exists():
        return "saved analysis has no flashes.csv"
    return None


def load_person(participant: str, data_dir: Path = DATA_DIR, include_test: bool = False, refresh: bool = True) -> Person:
    """The participant's pooled result, (re)analysing first when the saved one is missing or out of date (refresh=False:
    use the saved one regardless). Never raises: a person without usable data comes back with included=False and a reason.

    A re-analysis is written to data/<participant>/analysis/ except when it would replace a saved default analysis with
    one that includes test runs (then it is done in memory only)."""
    data_dir = Path(data_dir)
    pid = session_files.sanitize_participant_id(participant)
    person = Person(pid)
    summary_path = data_dir / pid / ANALYSIS_DIR / "summary.json"
    summary = _read_json(summary_path)
    try:
        why = stale_reason(pid, data_dir, include_test, summary, summary_path) if (refresh or summary is None) else None
        if why is None and _read_flash_csv(summary_path.parent / str(summary.get("flashes_csv") or "flashes.csv")) is None:
            why = "saved flashes.csv is unreadable"
        if why is None:
            flashes = _read_flash_csv(summary_path.parent / str(summary.get("flashes_csv") or "flashes.csv"))
            person.analysis_source = "saved"
        else:
            write = (not include_test) or summary is None or bool(summary.get("include_test"))
            result = analyze_participant(pid, data_dir, include_test=include_test, write=write)
            summary = result.summary
            flashes = [(p.rel_ms, bool(p.flash.correct), g.condition) for p, g in zip(result.placed, result.gates)]
            person.analysis_source = ("re-analysed" if write else "analysed in memory") + f" ({why})"
    except NoRunsError as exc:
        person.included, person.reason = False, str(exc)
        return person
    except Exception as exc:  # one broken participant must not stop a comparison
        person.included, person.reason = False, f"could not be analysed ({type(exc).__name__}: {exc})"
        return person

    n, k = summary.get("n_flashes") or {}, summary.get("correct") or {}
    person.index, person.ci, person.method = summary.get("index"), summary.get("ci"), summary.get("method")
    person.n_saccade, person.n_fixation, person.n_total = int(n.get(SACCADE, 0)), int(n.get(FIXATION, 0)), int(n.get("total", 0))
    person.correct_saccade, person.correct_fixation = int(k.get(SACCADE, 0)), int(k.get(FIXATION, 0))
    person.contrasts_pct, person.geometry = summary.get("contrasts_pct"), summary.get("geometry")
    person.run_kind = summary.get("run_kind") or SINGLE_RUN_KIND  # schema 1 / no runs pooled: single
    person.runs_used, person.runs_excluded = len(summary.get("runs_used") or []), len(summary.get("runs_excluded") or [])
    person.notes = list(summary.get("notes") or [])
    person.by_contrast = list(summary.get("by_contrast") or [])
    person.summary_path = str(summary_path)
    person.flashes = flashes or []
    person.run_records = list(summary.get("runs_used") or [])
    if person.method is None:
        person.included = False
        note = person.notes[-1] if person.notes else "no flashes"
        person.reason = "suppression index cannot be computed: " + note.removeprefix("cannot compute: ")
    return person


# -- statistics (pure functions, no files) -------------------------------------------------------


def _cells(by_contrast: list, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted((r for r in by_contrast if r["condition"] == condition and r["n"] > 0), key=lambda r: r["contrast_pct"])
    return (np.asarray([r["contrast_pct"] / 100 for r in rows], dtype=float), np.asarray([r["n"] for r in rows], dtype=float),
            np.asarray([r["correct"] for r in rows], dtype=float))


def bootstrap_index(person: Person, method: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """n_boot bootstrap replicates of the person's index (NaN where the refit is unidentified).

    psychometric: every contrast x condition cell resampled binomially (= resampling that cell's flashes) and refitted
    with analyze_session's shared-slope Weibull fit; accuracy: each condition's total resampled, p_fixation - p_saccade."""
    out = np.full(n_boot, np.nan)
    if method == "psychometric":
        cs, ns, ks = _cells(person.by_contrast, SACCADE)
        cf, nf, kf = _cells(person.by_contrast, FIXATION)
        if len(cs) < 2 or len(cf) < 2:
            return out
        for i in range(n_boot):
            ts, tf, edge = _fit_pair(cs, ns, rng.binomial(ns.astype(int), ks / ns), cf, nf, rng.binomial(nf.astype(int), kf / nf))
            if not edge:
                out[i] = 1 - tf / ts
        return out
    if person.n_saccade and person.n_fixation:
        sac = rng.binomial(person.n_saccade, person.correct_saccade / person.n_saccade, n_boot) / person.n_saccade
        fix = rng.binomial(person.n_fixation, person.correct_fixation / person.n_fixation, n_boot) / person.n_fixation
        out[:] = fix - sac
    return out


def difference_ci(boot_a: np.ndarray, boot_b: np.ndarray) -> tuple[list | None, int, str | None]:
    """(95 % percentile CI of A - B or None, usable replicates, note). Same acceptance rule as suppression_index: at least
    75 % and 200 of the replicates must be identified for both people."""
    diff = boot_a - boot_b
    ok = diff[~np.isnan(diff)]
    if len(ok) >= 0.75 * len(diff) and len(ok) >= 200:
        return [float(v) for v in np.percentile(ok, [2.5, 97.5])], len(ok), None
    return None, len(ok), f"no CI for the difference: only {len(ok)} of {len(diff)} bootstrap refits were identified for both people (need >= 75 % and >= 200)"


def mann_whitney(a: list[float], b: list[float]) -> dict:
    """Two-sided Mann-Whitney U test of A vs B. U_A = number of (A, B) pairs where A is higher (ties count half).
    Exact (all splits enumerated, correct with ties) when there are at most _EXACT_PERMUTATION_LIMIT splits; scipy's
    exact test when there are no ties and more splits; otherwise scipy's normal approximation with tie correction."""
    a, b = [float(x) for x in a], [float(x) for x in b]
    na, nb = len(a), len(b)
    u_a = float(sum((x > y) + 0.5 * (x == y) for x in a for y in b))
    pooled = np.asarray(a + b)
    ties = len(set(pooled.tolist())) < len(pooled)
    n_splits = math.comb(na + nb, na)
    if n_splits <= _EXACT_PERMUTATION_LIMIT:
        # U_A = (sum of A's mid-ranks) - na (na + 1) / 2, so every split's U follows from its rank sum.
        ranks = np.asarray([(pooled < v).sum() + ((pooled == v).sum() + 1) / 2 for v in pooled])
        splits = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(na + nb), na)), dtype=np.int32).reshape(-1, na)
        u_all = ranks[splits].sum(axis=1) - na * (na + 1) / 2
        centre = na * nb / 2
        hits = int((np.abs(u_all - centre) >= abs(u_a - centre) - 1e-9).sum())
        return {"u_a": u_a, "p": hits / n_splits, "test": f"exact permutation Mann-Whitney U (all {n_splits} splits{', ties handled' if ties else ''})"}
    from scipy.stats import mannwhitneyu  # only for large groups; scipy ships with PsychoPy

    method = "asymptotic" if ties else "exact"
    res = mannwhitneyu(a, b, alternative="two-sided", method=method)
    return {"u_a": u_a, "p": float(res.pvalue), "test": f"scipy.stats.mannwhitneyu ({method}{', tie-corrected' if ties else ''})"}


def rank_biserial(u_a: float, na: int, nb: int) -> float:
    """r = 2 U_A / (n_A n_B) - 1: +1 = every A person is above every B person, 0 = complete overlap, -1 = every A below every B."""
    return 2 * u_a / (na * nb) - 1


def effect_size_words(r: float, name_a: str, name_b: str) -> str:
    share = 100 * (r + 1) / 2
    size = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small" if abs(r) >= 0.1 else "negligible"
    return (f"rank-biserial r = {r:+.2f} ({size}): picking one person from each group at random, the {name_a} person has the "
            f"higher index {share:.0f} % of the time (50 % = no difference, 100 % = every {name_a} person is higher).")


def median_difference_ci(a: list[float], b: list[float], n_boot: int, rng: np.random.Generator) -> list:
    """95 % percentile bootstrap CI of median(A) - median(B), people resampled with replacement within each group."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    diff = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    return [float(v) for v in np.percentile(diff, [2.5, 97.5])]


def position_in_group(value: float, group: list[float]) -> dict:
    """Where one person's index falls in a group: how many members are above / below / tied, and the percentile
    (share of the group below the person, ties counting half)."""
    above = sum(g > value for g in group)
    below = sum(g < value for g in group)
    tied = len(group) - above - below
    return {"members_above": above, "members_below": below, "members_tied": tied, "percentile": 100 * (below + 0.5 * tied) / len(group)}


# -- the comparison ------------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    mode: str | None  # ONE_VS_ONE | ONE_VS_GROUP | GROUP_VS_GROUP; None when nothing could be compared
    requested_mode: str  # from the requested set sizes (differs from mode when exclusions shrank a side)
    set_a: list  # requested IDs (sanitized)
    set_b: list
    name_a: str  # "A", or "A (TBI)" when everyone on the side shares a group label
    name_b: str
    comparable: bool
    method: str | None  # "psychometric" | "accuracy": the index scale everyone included shares
    headline: str
    details: list  # plain-language lines (meaning, statistics, caveats)
    warnings: list
    exploratory: bool
    stats: dict  # the mode's numbers (see compare())
    people: list  # Person, A then B, included and excluded
    include_test: bool
    generated_at: str

    def included(self, side: str) -> list:
        return [p for p in self.people if p.side == side and p.included]

    @property
    def excluded(self) -> list:
        return [p for p in self.people if not p.included]

    def to_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "people"}
        out["people"] = [p.to_dict() for p in self.people]
        return _jsonable(out)

    def report(self) -> str:
        return format_report(self)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def _side_name(side: str, people: list) -> str:
    groups = {p.group for p in people if p.side == side}
    return f"{side} ({groups.pop()})" if len(groups) == 1 and "" not in groups else side


def _ci_text(ci, signed: bool = False) -> str:
    if ci is None:
        return "95 % CI not available"
    lo, hi = (0.0 if round(v, 2) == 0 else v for v in ci)  # no "-0.00"
    fmt = "{:+.2f}" if signed else "{:.2f}"
    return f"95 % CI {fmt.format(lo)} to {fmt.format(hi)}" if (signed or lo < 0) else f"95 % CI {lo:.2f}-{hi:.2f}"


def _what_index(method: str | None) -> str:
    return "accuracy-based suppression (fallback)" if method == "accuracy" else "suppression"


def _exclude(person: Person, reason: str) -> None:
    person.included, person.reason = False, reason


def _fmt_key(key: tuple) -> str:
    contrasts, geometry, kind = key
    text = "contrasts " + ("/".join(f"{c:g}" for c in contrasts) + " %" if contrasts else "not recorded")
    if geometry and geometry[0] is not None and geometry[1] is not None:
        text += f", screen {geometry[0]:g} cm high at {geometry[1]:g} cm"
    return text + f", run kind {kind}"


def _apply_compatibility(people: list, one_vs_one: bool) -> tuple | None:
    """Excludes people whose contrasts/geometry differ from the reference set; returns the reference key, or None when
    a 1 vs 1 pair differs (the caller reports it as not comparable)."""
    usable = [p for p in people if p.included]
    if not usable:
        return None
    keys = Counter(p.compat_key for p in usable)
    if one_vs_one and len(keys) > 1:
        a, b = usable
        for p, other in ((a, b), (b, a)):
            _exclude(p, f"{_fmt_key(p.compat_key)} differ from {other.participant}'s {_fmt_key(other.compat_key)}")
        return None
    top = max(keys.values())
    tied = [k for k in keys if keys[k] == top]
    if len(tied) > 1:  # prefer the key of the larger side's members, then the first seen
        larger = "A" if sum(p.side == "A" for p in people) > sum(p.side == "B" for p in people) else "B"
        tied.sort(key=lambda k: -sum(p.side == larger and p.compat_key == k for p in usable))
    ref = tied[0]
    for p in usable:
        if p.compat_key != ref:
            _exclude(p, f"{_fmt_key(p.compat_key)} differ from the majority's {_fmt_key(ref)}; people tested with other settings are not mixed")
    return ref


def _majority_method(people: list) -> str | None:
    counts = Counter(p.method for p in people if p.included)
    if not counts:
        return None
    return max(counts, key=lambda m: (counts[m], m == "psychometric"))  # tie: the threshold ratio wins


def compare(set_a: list[str], set_b: list[str], data_dir: Path = DATA_DIR, include_test: bool = False,
            n_boot: int = N_BOOT, seed: int = 0, refresh: bool = True) -> ComparisonResult:
    """Compares set A with set B (participant IDs; see resolve_set() for "group:TBI"). The mode follows from the number
    of usable people per side. Raises ValueError for an empty side or a person in both sets; everything else (missing
    data, unusable people) ends up in the result.

    stats keys by mode:
      one_vs_one:     index_a, ci_a, index_b, ci_b, difference, difference_ci, ci_excludes_zero, n_boot, boot_usable
      group_vs_group: n_a, n_b, median_a, median_b, range_a, range_b, median_difference, median_difference_ci,
                      u_a, p_value, test, rank_biserial, prob_a_higher, n_boot
      one_vs_group:   single (side), single_id, index, ci, group_n, group_median, group_range, members_above,
                      members_below, members_tied, percentile, group_median_in_single_ci"""
    a_ids = list(dict.fromkeys(session_files.sanitize_participant_id(x) for x in set_a if str(x).strip()))
    b_ids = list(dict.fromkeys(session_files.sanitize_participant_id(x) for x in set_b if str(x).strip()))
    if not a_ids or not b_ids:
        raise ValueError("both sets need at least one participant")
    both = sorted(set(a_ids) & set(b_ids))
    if both:
        raise ValueError(f"in both sets: {', '.join(both)} (a person can only be on one side)")
    data_dir = Path(data_dir)
    labels = list_labels(data_dir)
    people = []
    for side, ids in (("A", a_ids), ("B", b_ids)):
        for pid in ids:
            person = load_person(pid, data_dir, include_test, refresh)
            person.side, person.group = side, (labels.get(pid) or {}).get("group", "")
            people.append(person)
    return compare_people(people, n_boot=n_boot, seed=seed, include_test=include_test)


def compare_people(people: list, n_boot: int = N_BOOT, seed: int = 0, include_test: bool = False) -> ComparisonResult:
    """The comparison itself, on already loaded Person objects (side "A"/"B" set; see compare()). Works on copies, so
    cached Person objects can be reused for several comparisons; the result's `people` hold the exclusions."""
    people = [replace(p, warnings=list(p.warnings)) for p in people]
    a_ids = [p.participant for p in people if p.side == "A"]
    b_ids = [p.participant for p in people if p.side == "B"]
    if not a_ids or not b_ids:
        raise ValueError("both sets need at least one participant")
    requested = _mode_for(len(a_ids), len(b_ids))
    warnings, details = [], []
    one_vs_one = requested == ONE_VS_ONE
    people = [p for p in people if p.side == "A"] + [p for p in people if p.side == "B"]
    _apply_compatibility(people, one_vs_one)
    method = None
    if one_vs_one:
        pa, pb = people
        if pa.included and pb.included:
            method = pa.method if pa.method == pb.method else None
    else:
        method = _majority_method(people)
        for p in people:
            if p.included and p.method != method:
                _exclude(p, f"index from the {p.method} method while most people's is from the {method} method; the two scales are not comparable")
    na, nb = len([p for p in people if p.side == "A" and p.included]), len([p for p in people if p.side == "B" and p.included])
    mode = _mode_for(na, nb) if na and nb else None
    if one_vs_one and method is None:
        mode = None  # different methods, or someone unusable
    if mode is not None and mode != requested:
        details.append(f"Asked for {MODE_NAMES[requested]}, but after exclusions only {na} (A) and {nb} (B) people are usable, so this is a {MODE_NAMES[mode]} comparison.")
    name_a, name_b = _side_name("A", [p for p in people if p.included] or people), _side_name("B", [p for p in people if p.included] or people)

    for p in people:
        if p.included and (p.n_saccade < MIN_FLASHES_PER_CONDITION or p.n_fixation < MIN_FLASHES_PER_CONDITION):
            p.warnings.append(f"only {p.n_saccade} saccade / {p.n_fixation} fixation flashes (fewer than {MIN_FLASHES_PER_CONDITION} per condition): index unstable")
            warnings.append(f"{p.participant}: {p.warnings[-1]}")
    if mode == GROUP_VS_GROUP:
        warnings += [f"group {name} has only {n} people (fewer than {MIN_GROUP_SIZE}): the p-value and CI are very rough"
                     for name, n in ((name_a, na), (name_b, nb)) if n < MIN_GROUP_SIZE]
    elif mode == ONE_VS_GROUP:
        n_group = max(na, nb)
        if n_group < MIN_GROUP_SIZE:
            warnings.append(f"the group has only {n_group} people (fewer than {MIN_GROUP_SIZE}): its median and range say little")
    exploratory = min(na, nb) < EXPLORATORY_BELOW

    result = ComparisonResult(mode, requested, a_ids, b_ids, name_a, name_b, False, method, "", details, warnings, exploratory, {},
                              people, include_test, datetime.now().isoformat(timespec="seconds"))
    rng = np.random.default_rng(seed)
    if mode is None:
        if one_vs_one and people[0].included and people[1].included:  # both usable, different methods
            result.headline = (f"{people[0].participant} and {people[1].participant} are not comparable: {people[0].participant}'s index comes from the "
                               f"{people[0].method} method and {people[1].participant}'s from the {people[1].method} method (different scales).")
            for p, other in ((people[0], people[1]), (people[1], people[0])):
                _exclude(p, f"{p.method} method, {other.participant} has the {other.method} method: different scales")
        elif one_vs_one and all("differ from" in p.reason for p in people):
            result.headline = f"{people[0].participant} and {people[1].participant} are not comparable: their runs used different settings ({people[0].reason})."
        else:
            empty = [n for n, k in ((name_a, na), (name_b, nb)) if not k]
            result.headline = f"Nothing to compare: no usable participant in set {' or '.join(empty)} (see the excluded list)."
    elif mode == ONE_VS_ONE:
        _one_vs_one(result, rng, n_boot)
    elif mode == GROUP_VS_GROUP:
        _group_vs_group(result, rng, n_boot)
    else:
        _one_vs_group(result)
    result.comparable = mode is not None and bool(result.stats)
    result.details += _caveats(result)
    return result


def _mode_for(na: int, nb: int) -> str:
    if na == 1 and nb == 1:
        return ONE_VS_ONE
    return ONE_VS_GROUP if min(na, nb) == 1 else GROUP_VS_GROUP


def _tail(result: ComparisonResult) -> str:
    return " - exploratory, small groups." if result.exploratory else "."


def _one_vs_one(result: ComparisonResult, rng, n_boot: int) -> None:
    (pa,), (pb,) = result.included("A"), result.included("B")
    boot_a, boot_b = bootstrap_index(pa, result.method, n_boot, rng), bootstrap_index(pb, result.method, n_boot, rng)
    ci, usable, note = difference_ci(boot_a, boot_b)
    diff = pa.index - pb.index
    excludes_zero = None if ci is None else (ci[0] > 0 or ci[1] < 0)
    result.stats = {"index_a": pa.index, "ci_a": pa.ci, "index_b": pb.index, "ci_b": pb.ci, "difference": diff, "difference_ci": ci,
                    "ci_excludes_zero": excludes_zero, "n_boot": n_boot, "boot_usable": usable}
    what = "accuracy-based suppression index (fallback, accuracy difference)" if result.method == "accuracy" else "suppression index"
    if excludes_zero is None:
        verdict = "no CI for the difference"
    elif excludes_zero:
        verdict = f"the CI excludes 0, so {pa.participant if diff > 0 else pb.participant} shows more suppression in these data"
    else:
        verdict = "the CI includes 0, so these data cannot tell the two apart"
    result.headline = (f"{pa.participant} {what} {pa.index:.2f} ({_ci_text(pa.ci)}) vs {pb.participant} {pb.index:.2f} ({_ci_text(pb.ci)}); "
                       f"difference A - B {diff:+.2f} ({_ci_text(ci, signed=True)}): {verdict} - exploratory, one person each.")
    if note:
        result.details.append(note)
    result.details.append(f"Difference CI: {n_boot} bootstrap refits per person ({usable} usable for both); each person's flashes were resampled "
                          + ("within every contrast x condition cell and refitted with the same Weibull fit as their own index."
                             if result.method == "psychometric" else "within each condition (saccade / fixation) and the accuracy difference recomputed."))


def _group_vs_group(result: ComparisonResult, rng, n_boot: int) -> None:
    a = [p.index for p in result.included("A")]
    b = [p.index for p in result.included("B")]
    mw = mann_whitney(a, b)
    r = rank_biserial(mw["u_a"], len(a), len(b))
    med_a, med_b = float(np.median(a)), float(np.median(b))
    ci = median_difference_ci(a, b, n_boot, rng)
    result.stats = {"n_a": len(a), "n_b": len(b), "median_a": med_a, "median_b": med_b, "range_a": [min(a), max(a)], "range_b": [min(b), max(b)],
                    "median_difference": med_a - med_b, "median_difference_ci": ci, "u_a": mw["u_a"], "p_value": mw["p"], "test": mw["test"],
                    "rank_biserial": r, "prob_a_higher": (r + 1) / 2, "n_boot": n_boot}
    what = _what_index(result.method)
    result.headline = (f"Group {result.name_a}, {len(a)} people, median {what} {med_a:.2f}; group {result.name_b}, {len(b)} people, {med_b:.2f}; "
                       f"difference {med_a - med_b:+.2f} ({_ci_text(ci, signed=True)}), Mann-Whitney p = {_p_text(mw['p'])}{_tail(result)}")
    result.details += [
        f"Ranges: {result.name_a} {min(a):.2f} to {max(a):.2f}; {result.name_b} {min(b):.2f} to {max(b):.2f}.",
        f"Effect size: {effect_size_words(r, result.name_a, result.name_b)}",
        f"Test: {mw['test']}, two-sided; U_A = {mw['u_a']:g} of {len(a) * len(b)} pairs. The p-value is the chance of a group difference "
        f"at least this large if both groups came from the same population; a small p (e.g. < 0.05) is evidence of a real difference, "
        f"a large p is NOT evidence that the groups are the same.",
        f"Difference CI: {n_boot} bootstrap resamples of the people in each group (percentile method).",
    ]


def _one_vs_group(result: ComparisonResult) -> None:
    single_side = "A" if len(result.included("A")) == 1 else "B"
    (single,) = result.included(single_side)
    group = result.included("B" if single_side == "A" else "A")
    group_name = result.name_b if single_side == "A" else result.name_a
    values = [p.index for p in group]
    pos = position_in_group(single.index, values)
    med = float(np.median(values))
    in_ci = None if single.ci is None else bool(single.ci[0] <= med <= single.ci[1])
    result.stats = {"single": single_side, "single_id": single.participant, "index": single.index, "ci": single.ci, "group_n": len(values),
                    "group_median": med, "group_range": [min(values), max(values)], **pos, "group_median_in_single_ci": in_ci}
    what = _what_index(result.method)
    n = len(values)
    tied = f", tied with {pos['members_tied']}" if pos["members_tied"] else ""
    if pos["members_above"] == n:
        where = f"is below all {n} members of group {group_name}"
    elif pos["members_below"] == n:
        where = f"is above all {n} members of group {group_name}"
    else:
        where = f"is below {pos['members_above']} of {n} members of group {group_name} and above {pos['members_below']}{tied}"
    result.headline = (f"{single.participant}'s {what} index {single.index:.2f} {where} ({pos['percentile']:.0f}th percentile); group median {med:.2f} "
                       f"(range {min(values):.2f} to {max(values):.2f}){_tail(result)}")
    result.details.append("No p-value: a single person gives no estimate of how much people differ from one another, so there is "
                          "nothing to test against; the position in the group is descriptive only.")
    if in_ci is not None:
        result.details.append(f"The group median {'lies inside' if in_ci else 'lies outside'} {single.participant}'s own 95 % CI "
                              f"({single.ci[0]:.2f} to {single.ci[1]:.2f}), i.e. {single.participant}'s measurement {'cannot' if in_ci else 'can'} be told apart from the group's typical value.")


def _p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def _caveats(result: ComparisonResult) -> list:
    out = []
    if result.method == "psychometric":
        out.append("Index scale: 1 - threshold_fixation / threshold_saccade. 0 = no suppression, 1 = the line is never seen during saccades, negative = seen better during saccades.")
    elif result.method == "accuracy":
        out.append("Index scale (fallback): proportion correct during fixation minus during saccades; 0 = no difference, about 0.5 = guessing during saccades. Not the threshold ratio.")
    if result.exploratory:
        out.append(f"EXPLORATORY: fewer than {EXPLORATORY_BELOW} people per side. This is a feasibility pilot; do not read it as a group result.")
    if result.mode in (GROUP_VS_GROUP, ONE_VS_GROUP):
        out.append("Each person counts once, by their index alone; how precise each person's index is (their own CI) is not weighted in.")
    return out


# -- report, files, plot -------------------------------------------------------------------------


def format_report(result: ComparisonResult) -> str:
    title = f"Comparison ({MODE_NAMES.get(result.mode, 'not possible')}): set {result.name_a} vs set {result.name_b}"
    lines = [title, "", result.headline, ""]
    if result.warnings:
        lines += ["Warnings:"] + [f"  - {w}" for w in result.warnings] + [""]
    if result.excluded:
        lines += [f"Left out ({len(result.excluded)}):"] + [f"  - {p.participant} (set {p.side}): {p.reason}" for p in result.excluded] + [""]
    lines.append("People:")
    lines.append(f"  {'set':<4} {'participant':<16} {'group':<12} {'index':>6}  {'95 % CI':<15} {'method':<12} {'saccade':>7} {'fixation':>8} {'runs':>4}  used")
    for p in result.people:
        ci = "" if p.ci is None else f"{p.ci[0]:.2f} to {p.ci[1]:.2f}"
        idx = "n/a" if p.index is None else f"{p.index:.2f}"
        lines.append(f"  {p.side:<4} {p.participant:<16} {p.group or '-':<12} {idx:>6}  {ci:<15} {p.method or '-':<12} {p.n_saccade:>7} {p.n_fixation:>8} {p.runs_used:>4}  {'yes' if p.included else 'no'}")
    lines += ["", "Details:"] + [f"  - {d}" for d in result.details]
    sources = Counter(p.analysis_source.split(" (")[0] for p in result.people if p.analysis_source)
    if sources:
        lines.append("  - Per-person analyses: " + ", ".join(f"{n} {s}" for s, n in sources.items()) + " (re-analysed = the saved one was missing or older than the runs).")
    lines += ["", "Files: report.txt (this), comparison.csv (one row per person), comparison.json (all numbers), plot.png."]
    return "\n".join(lines)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:40] or "set"


def _set_slug(result: ComparisonResult, side: str) -> str:
    ids = result.set_a if side == "A" else result.set_b
    name = result.name_a if side == "A" else result.name_b
    if len(ids) > 1 and name.startswith(f"{side} ("):
        return _slug(name[3:-1])  # the shared group label
    return _slug("+".join(ids))


def write_comparison(result: ComparisonResult, out_dir: Path | None = None, data_dir: Path = DATA_DIR) -> dict:
    """Writes comparison.json, comparison.csv, report.txt and plot.png; returns their paths."""
    if out_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(data_dir) / COMPARISONS_DIR / f"{stamp}_{_set_slug(result, 'A')}_vs_{_set_slug(result, 'B')}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "comparison.json", "csv": out / "comparison.csv", "report": out / "report.txt", "png": out / "plot.png"}
    paths["json"].write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for p in result.people:
            writer.writerow([p.participant, p.group, p.side, "yes" if p.included else "no", p.reason, _num(p.index),
                             _num(p.ci[0] if p.ci else None), _num(p.ci[1] if p.ci else None), p.method or "",
                             p.n_saccade, p.n_fixation, p.n_total, p.runs_used, "; ".join(p.warnings)])
    paths["report"].write_text(result.report() + "\n", encoding="utf-8")
    plot_comparison(result, paths["png"])
    return paths


def _num(v) -> str:
    return "" if v is None else f"{v:.4f}"


def _time_curve(flashes: list) -> tuple[list, list, list]:
    centres, acc, err = [], [], [[], []]
    for lo, hi in zip(BIN_EDGES_MS[:-1], BIN_EDGES_MS[1:]):
        members = [c for rel, c, _ in flashes if rel is not None and lo <= rel < hi]
        if members:
            k, n = sum(members), len(members)
            p = k / n
            w_lo, w_hi = wilson(k, n)
            centres.append((lo + hi) / 2)
            acc.append(100 * p)
            err[0].append(100 * (p - w_lo))
            err[1].append(100 * (w_hi - p))
    return centres, acc, err


def plot_comparison(result: ComparisonResult, out_path: Path) -> Path:
    colours = {"A": "#C44E52", "B": "#4C72B0"}
    if result.mode == ONE_VS_ONE:
        fig, (ax, ax_idx) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})
        for p in result.people:
            centres, acc, err = _time_curve(p.flashes)
            if centres:
                ax.errorbar(np.asarray(centres) + (-3 if p.side == "A" else 3), acc, yerr=err, fmt="o-", capsize=3, color=colours[p.side],
                            label=f"{p.side}: {p.participant} ({len(p.flashes)} flashes)")
        ax.axvline(0, color="#999999", linewidth=1)
        ax.axhline(100 * CHANCE, color="#999999", linestyle=":", linewidth=1)
        ax.set_xlim(BIN_EDGES_MS[0], BIN_EDGES_MS[-1])
        ax.set_ylim(0, 105)
        ax.set_xlabel("flash time relative to saccade onset (ms)")
        ax.set_ylabel("correct (%), with 95% CI")
        ax.set_title("Accuracy vs flash time, all contrasts")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        for x, p in enumerate(result.people):
            if p.index is not None:
                yerr = None if p.ci is None else [[p.index - p.ci[0]], [p.ci[1] - p.index]]
                ax_idx.errorbar([x], [p.index], yerr=yerr, fmt="o", capsize=5, color=colours[p.side], markersize=8)
        ax_idx.set_xticks(range(len(result.people)), [f"{p.side}: {p.participant}" for p in result.people])
        ax_idx.set_xlim(-0.6, len(result.people) - 0.4)
        _index_axis(ax_idx, result)
    else:
        fig, ax_idx = plt.subplots(figsize=(7, 5), layout="constrained")
        rng = np.random.default_rng(1)
        single = result.stats.get("single_id") if result.mode == ONE_VS_GROUP else None
        for x, side in enumerate(("A", "B")):
            members = result.included(side)
            if not members:
                continue
            values = [p.index for p in members]
            if single and len(members) == 1 and members[0].participant == single:
                p = members[0]
                yerr = None if p.ci is None else [[p.index - p.ci[0]], [p.ci[1] - p.index]]
                ax_idx.errorbar([x], [p.index], yerr=yerr, fmt="*", markersize=16, capsize=6, color="#DD8452", label=p.participant + (" (with own 95% CI)" if p.ci else " (no CI)"))
                continue
            xs = x + rng.uniform(-0.12, 0.12, len(values))
            ax_idx.scatter(xs, values, s=45, color=colours[side], alpha=0.8, zorder=3)
            ax_idx.hlines(np.median(values), x - 0.25, x + 0.25, color="black", linewidth=2, zorder=4)
            ax_idx.annotate(f"median {np.median(values):.2f}", (x + 0.27, np.median(values)), va="center", fontsize=8)
        ax_idx.set_xticks([0, 1], [f"{result.name_a} (n={len(result.included('A'))})", f"{result.name_b} (n={len(result.included('B'))})"])
        ax_idx.set_xlim(-0.6, 1.8)
        if single:
            ax_idx.legend(fontsize=8)
        _index_axis(ax_idx, result)
    fig.suptitle(_wrap(result.headline), fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _index_axis(ax, result: ComparisonResult) -> None:
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_ylabel("accuracy-based suppression index (fallback)" if result.method == "accuracy" else "suppression index (0 = none, 1 = never seen)")
    ax.set_title("One index per person" + (" (dots), medians (bars)" if result.mode == GROUP_VS_GROUP else ""))
    ax.grid(True, axis="y", alpha=0.3)


def _wrap(text: str, width: int = 120) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(lines + [line])


# -- CLI -----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare participants (1 vs 1, 1 vs group, group vs group). A set is comma-separated IDs, "
                                                 "group:<label> (see -m src.experiment.participants list) or all.")
    parser.add_argument("set_a", help='e.g. "P01" or "P01,P02" or "group:TBI"')
    parser.add_argument("set_b", help='e.g. "P03" or "group:control"')
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--include-test", action="store_true", help="also use test-mode runs")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default data/comparisons/<time>_<A>_vs_<B>/)")
    args = parser.parse_args(argv)
    a, b = resolve_set(args.set_a, args.data_dir), resolve_set(args.set_b, args.data_dir)
    a, b = everyone_else(args.set_a, a, args.set_b, b)  # "P01" vs "all" = P01 vs everyone else
    try:
        result = compare(a, b, args.data_dir, include_test=args.include_test)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    paths = write_comparison(result, args.out, args.data_dir)
    print(result.report())
    print(f"\nSaved in {paths['report'].parent}:")
    for p in paths.values():
        print(f"  {p}")
    return 0 if result.comparable else 2


if __name__ == "__main__":
    sys.exit(main())
