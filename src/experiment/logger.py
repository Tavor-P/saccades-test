"""ResultLogger: writes one session to data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/ - `trials.csv` (one row per trial),
`session.json` (participant, settings, hardware, results summary, gaze quality, glasses-video sync) and the paths of the
side files (gaze_trace.csv, analysis/, glasses/). TRIAL_COLUMNS is the single schema of trials.csv: it drives the writer
and `read_trial_rows`, which also reads sessions written in the old layout (loose results_<ts>.csv with other names)."""

import csv
import functools
import importlib
import json
import platform
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from include.experiment.constants import (
    LINE_OFFSET_DEG,
    LINE_SD_DEG,
    SCREEN_HEIGHT_CM,
    VIEWING_DISTANCE_CM,
)
from include.experiment.types import Side, Target, TrialResult
from src import timebase
from src.experiment import session_files

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# -- trials.csv schema ----------------------------------------------------------------------------


def _num(value: float | None, digits: int = 1) -> str:
    return "" if value is None else repr(round(float(value), digits))


def _yes_no(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "no")


def _enum_value(value) -> str:
    return "" if value is None else value.value


PHASE_WORDS = {"presaccade": "baseline"}  # old runs only: the removed fixation phase ("presaccade" in memory, "baseline" in trials.csv); kept so old files parse
OUTCOME_WORDS = {"incorrect": "wrong", "miss": "no_answer"}
_INVERSE = lambda table: {v: k for k, v in table.items()}  # noqa: E731


def _word_to_canonical(table: dict[str, str]) -> Callable[[str], str]:
    inverse = _INVERSE(table)
    return lambda text: inverse.get(text, text)


def _yes_no_to_canonical(text: str) -> str:
    return {"yes": "True", "no": "False", "true": "True", "false": "False"}.get(text.strip().lower(), "")


def _pct_to_fraction(text: str) -> str:
    return repr(round(float(text) / 100, 8)) if text else ""


def _is_correct(result: TrialResult) -> str:
    """yes/no for scored trials; blank for rt_test rows (they only time an eye movement) and for timeouts (no eye movement, nothing scored)."""
    return "" if result.phase == "rt_test" or result.outcome == "timeout" else _yes_no(result.outcome in ("correct", "correct_rejection"))


@dataclass(frozen=True)
class Column:
    name: str  # header in trials.csv
    field: str | None  # TrialResult field it holds (None: derived from other fields, ignored when reading)
    write: Callable[[TrialResult], str]
    to_canonical: Callable[[str], str] = lambda text: text  # file text -> the old CSV's text for that field (see read_trial_rows)

    @property
    def canonical_name(self) -> str | None:
        return None if self.field is None else _CANONICAL_NAMES.get(self.field, self.field)


# TrialResult field -> its column name in the old results_<ts>.csv (the stimulus was a grating then; read_trial_rows yields rows under these names)
_CANONICAL_NAMES = {
    "index": "trial_index",
    "flash_shown": "grating_shown",
    "line_side": "orientation",
    "answer_side": "response_orientation",
    "flash_unix_ms": "grating_shown_at_unix_ms",
}

# Left to right: who/what, stimulus contrast, the yes/no group (flash seen? answered? correct?), outcome, stimulus/answer pair,
# the millisecond columns (then the wall-clock flash time), quality.
TRIAL_COLUMNS: list[Column] = [
    Column("phase", "phase", lambda r: PHASE_WORDS.get(r.phase, r.phase), _word_to_canonical(PHASE_WORDS)),
    Column("trial", "index", lambda r: str(r.index)),
    Column("contrast_pct", "contrast", lambda r: _num(None if r.contrast is None else r.contrast * 100, 4), _pct_to_fraction),
    Column("flash_shown", "flash_shown", lambda r: _yes_no(r.flash_shown), _yes_no_to_canonical),
    Column("flash_during_saccade", "flash_during_saccade", lambda r: _yes_no(r.flash_during_saccade), _yes_no_to_canonical),
    Column("answered", "responded", lambda r: _yes_no(r.responded), _yes_no_to_canonical),
    Column("correct", None, _is_correct),
    Column("outcome", "outcome", lambda r: OUTCOME_WORDS.get(r.outcome, r.outcome), _word_to_canonical(OUTCOME_WORDS)),
    Column("line_side", "line_side", lambda r: _enum_value(r.line_side)),
    Column("answer_side", "answer_side", lambda r: _enum_value(r.answer_side)),
    Column("flash_leg", "flash_leg", lambda r: r.flash_leg or ""),
    Column("flash_ms_after_go_cue", "scheduled_delay_ms", lambda r: _num(r.scheduled_delay_ms)),
    Column("out_arrival_ms_after_go_cue", "out_arrival_ms_after_go_cue", lambda r: _num(r.out_arrival_ms_after_go_cue)),
    Column("jump_back_ms_after_go_cue", "jump_back_ms_after_go_cue", lambda r: _num(r.jump_back_ms_after_go_cue)),
    Column("reaction_ms", "reaction_latency_ms", lambda r: _num(r.reaction_latency_ms)),
    Column("return_reaction_ms", "return_reaction_ms", lambda r: _num(r.return_reaction_ms)),
    Column("answer_ms", "response_time_ms", lambda r: _num(r.response_time_ms)),
    Column("flash_unix_ms", "flash_unix_ms", lambda r: _num(r.flash_unix_ms)),
    Column("jump_back_forced", "jump_back_forced", lambda r: _yes_no(r.jump_back_forced), _yes_no_to_canonical),
    Column("left_target_before_jump", "left_target_before_jump", lambda r: _yes_no(r.left_target_before_jump), _yes_no_to_canonical),
    Column("tracking_ok", "tracking_ok", lambda r: _yes_no(r.tracking_ok), _yes_no_to_canonical),
]
FIELDNAMES = [c.name for c in TRIAL_COLUMNS]

# TrialResult fields that are NOT in trials.csv, with why. Per-phase medians of saccade_duration_ms and onset_detection_lag_ms go to session.json.
DROPPED_FROM_CSV = {
    "source": "the two symbols are the left and right dot; with return jump off trials alternate strictly (direction = trial parity within a phase), with it on every trial starts at the same symbol (out leg away from it, return leg back to it); analyze_session finds saccades in the gaze trace",
    "target": "same as source: the opposite symbol, i.e. follows from the trial parity (return jump off) or is the same every trial (return jump on)",
    "timing_offset_ms": "the actual delay is in flash_ms_after_go_cue and analyze_session recomputes timing (and the control flashes) from the gaze trace",
    "saccade_duration_ms": "live-detected duration from the live gaze stream; analyze_session measures saccades from the raw trace (median kept in session.json)",
    "onset_detection_lag_ms": "diagnostic of the live onset detector (median kept in session.json)",
    "flash_relative_to_onset_ms": "live estimate from the late gaze stream; analyze_session recomputes it post hoc from the trace",
    "avg_rt_used_ms": "reaction-time-average bookkeeping; flash_ms_after_go_cue already holds the resulting delay (final average is in session.json)",
    "gaze_latency_ms": "tracker diagnostic; latency statistics live in session.json gaze_quality",
}


def read_trial_rows(path: Path) -> Iterator[dict]:
    """Rows of a session's trials file as dicts keyed by the OLD results_<ts>.csv column names and text conventions
    ('True'/'False', contrast as a fraction, 'dot'/'cross', 'presaccade'), whichever layout wrote the file. Old files pass
    through unchanged; new ones are translated through TRIAL_COLUMNS (dropped columns are simply absent)."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        legacy = "trial_index" in (reader.fieldnames or [])
        for row in reader:
            if legacy:
                yield row
                continue
            canonical = {}
            for column in TRIAL_COLUMNS:
                if column.canonical_name is not None and column.name in row:
                    canonical[column.canonical_name] = column.to_canonical(row[column.name] or "")
            yield canonical


def _parse_float(value: str | None) -> float | None:
    return float(value) if value else None


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "yes")


def _parse_optional_bool(value: str | None) -> bool | None:
    return _parse_bool(value) if value else None


def _parse_enum(enum_class, value: str | None):
    """Unknown values (e.g. a row written by another experiment version) read as None rather than crashing the loader."""
    try:
        return enum_class(value) if value else None
    except ValueError:
        return None


# CSVs logged before the line stimulus hold the grating's 'vertical' / 'horizontal' in the orientation columns; they map to
# ABOVE / BELOW (an arbitrary but consistent relabel: answers are still scored against the same column).
_LEGACY_SIDES = {"vertical": Side.ABOVE, "horizontal": Side.BELOW}


def _parse_side(value: str | None) -> Side | None:
    return _LEGACY_SIDES.get(value) or _parse_enum(Side, value)


def trial_from_row(row: dict) -> TrialResult:
    """A TrialResult from a canonical row (see read_trial_rows); absent columns read as None."""
    offset = row.get("timing_offset_ms")
    return TrialResult(
        index=int(row["trial_index"]),
        phase=row["phase"],
        source=_parse_enum(Target, row.get("source")),
        target=_parse_enum(Target, row.get("target")),
        saccade_duration_ms=_parse_float(row.get("saccade_duration_ms")),
        flash_shown=_parse_bool(row.get("grating_shown")),
        contrast=_parse_float(row.get("contrast")),
        line_side=_parse_side(row.get("orientation")),
        responded=_parse_bool(row.get("responded")),
        answer_side=_parse_side(row.get("response_orientation")),
        response_time_ms=_parse_float(row.get("response_time_ms")),
        outcome=row["outcome"],
        reaction_latency_ms=_parse_float(row.get("reaction_latency_ms")),
        onset_detection_lag_ms=_parse_float(row.get("onset_detection_lag_ms")),
        flash_during_saccade=_parse_optional_bool(row.get("flash_during_saccade")),
        timing_offset_ms=int(float(offset)) if offset else None,
        flash_unix_ms=_parse_float(row.get("grating_shown_at_unix_ms")),
        tracking_ok=_parse_optional_bool(row.get("tracking_ok")),
        flash_relative_to_onset_ms=_parse_float(row.get("flash_relative_to_onset_ms")),
        avg_rt_used_ms=_parse_float(row.get("avg_rt_used_ms")),
        scheduled_delay_ms=_parse_float(row.get("scheduled_delay_ms")),
        gaze_latency_ms=_parse_float(row.get("gaze_latency_ms")),
        flash_leg=row.get("flash_leg") or None,
        jump_back_ms_after_go_cue=_parse_float(row.get("jump_back_ms_after_go_cue")),
        return_reaction_ms=_parse_float(row.get("return_reaction_ms")),
        jump_back_forced=_parse_optional_bool(row.get("jump_back_forced")),
        out_arrival_ms_after_go_cue=_parse_float(row.get("out_arrival_ms_after_go_cue")),
        left_target_before_jump=_parse_optional_bool(row.get("left_target_before_jump")),
    )


def read_trials(path: Path) -> list[TrialResult]:
    return [trial_from_row(row) for row in read_trial_rows(path)]


# -- session.json ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _git_state() -> dict:
    """Commit SHA + whether the tree had uncommitted changes, so a results file
    can be tied back to the exact code that produced it. Best effort."""
    root = Path(__file__).resolve().parents[2]
    try:
        run = lambda *args: subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"sha": None, "dirty": None}


# The tuning constants that matter for the science (the rest of the constants modules is in the git SHA).
_SETTINGS_CONSTANTS = {
    "include.experiment.constants": (
        "TIMING_OFFSETS_MS",
        "TIMING_JITTER_MS",
        "CONTROL_OFFSETS_MS",
        "FLASH_SPREAD_SD_MS",
        "FLASH_WINDOW_MIN_MS",
        "FLASH_WINDOW_MAX_MS",
        "WARNING_TONE_HZ",
        "WARNING_TONE_DURATION_S",
        "WARNING_TONE_VOLUME",
        "DEFAULT_REACTION_TIME_MS",
        "SACCADE_FIXED_TRIALS_TEST",
        "SACCADE_FIXED_TRIALS_REAL",
        "FOREPERIOD_MIN_MS",
        "FOREPERIOD_MAX_MS",
        "RESPONSE_WINDOW_MS",
        "LINE_DURATION_FRAMES",
        "DOT_POSITION",
        "CROSS_POSITION",
        "SHARP_GO_CUE",
        "TRACKING_LOSS_WINDOW_MS",
        "SACCADE_ONSET_DISPLACEMENT",
        "SACCADE_ONSET_STABILITY_MS",
        "RETURN_LEG_FLASH_FRACTION",
        "RETURN_JUMP_LANDING_TIMEOUT_MS",
        "RETURN_JUMP_TRIGGER",
        "RETURN_FLASH_SPREAD_SD_MS",
    ),
    "include.eye_tracking.constants": ("GAZE_DEAD_ZONE",),
}


def _settings_constants() -> dict:
    out = {}
    for module_name, names in _SETTINGS_CONSTANTS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name in names:
            if hasattr(module, name):
                value = getattr(module, name)
                out[name.lower()] = list(value) if isinstance(value, tuple) else value
    return out


def _psychopy_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("psychopy")
    except Exception:
        return None


def _neon_address() -> str | None:
    try:
        from include.eye_tracking.constants import NEON_DEVICE_ADDRESS, NEON_DEVICE_PORT

        return f"{NEON_DEVICE_ADDRESS}:{NEON_DEVICE_PORT}" if NEON_DEVICE_ADDRESS else "auto-discovered"
    except Exception:
        return None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _results_summary(rows: list[TrialResult]) -> dict:
    """Per phase: how many rows, how many flashes were shown and answered correctly, and the medians of the diagnostics
    that trials.csv does not carry."""
    phases: dict[str, dict] = {}
    for phase in sorted({r.phase for r in rows}):
        mine = [r for r in rows if r.phase == phase]
        flashes = [r for r in mine if r.flash_shown]
        correct = sum(r.outcome == "correct" for r in flashes)
        phases[PHASE_WORDS.get(phase, phase)] = {
            "rows": len(mine),
            "flashes_shown": len(flashes),
            "flashes_correct": correct,
            "accuracy_on_flashes": round(correct / len(flashes), 3) if flashes else None,
            "tracking_lost_near_flash": sum(r.tracking_ok is False for r in flashes),
            "median_reaction_ms": _median([r.reaction_latency_ms for r in mine if r.reaction_latency_ms is not None]),
            "median_saccade_duration_ms_live": _median([r.saccade_duration_ms for r in mine if r.saccade_duration_ms is not None]),
            "median_onset_detection_lag_ms_live": _median([r.onset_detection_lag_ms for r in mine if r.onset_detection_lag_ms is not None]),
        }
    return phases


class ResultLogger:
    """Writes one run of the experiment into its own folder (see module docstring).

    One trials.csv for the whole run; the `phase` column distinguishes saccade rows from rt_test rows (and, in old runs, baseline rows).
    """

    def __init__(
        self,
        participant_id: str,
        gaze_source_label: str | None = None,
        test_mode: bool | None = None,
    ) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        now = timebase.unix_now()
        self._timestamp = int(now)
        self._folder = session_files.create_session_folder(DATA_DIR, participant_id, now, test_mode)
        self._meta_path = self._folder / session_files.SESSION_FILE

        self._file = (self._folder / session_files.TRIALS_FILE).open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._rows: list[TrialResult] = []

        self._metadata = {
            "participant": {"id": participant_id, "gender": None, "age": None},
            "run": {
                "mode": None if test_mode is None else ("test" if test_mode else "real"),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "unix_timestamp": self._timestamp,
                "folder": f"{self._folder.parent.name}/{self._folder.name}",
                "git": _git_state(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "psychopy_version": _psychopy_version(),
            },
            "settings": {
                "gaze_source": gaze_source_label,
                "viewing_distance_cm": VIEWING_DISTANCE_CM,
                "screen_height_cm": SCREEN_HEIGHT_CM,
                "line_offset_deg": LINE_OFFSET_DEG,
                "line_sd_deg": LINE_SD_DEG,
                "saccade_contrasts_pct": None,
                "calibration": None,
                **_settings_constants(),
            },
            "hardware": {"display_refresh_hz": None, "neon_device": _neon_address()},
            "results": {"phases": {}, "saccade_phase_summary": {}},
            "gaze_quality": None,
            "head_motion": None,
            "ignored_space_presses": {},
            "glasses_video": {"recording": "not started"},
        }
        self._write_metadata()

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def display_refresh_hz(self) -> float | None:
        return self._metadata["hardware"]["display_refresh_hz"]

    @property
    def gaze_trace_path(self) -> Path:
        """Raw gaze recording for this session (see src/eye_tracking/gaze_trace.py)."""
        return self._folder / session_files.GAZE_TRACE_FILE

    @property
    def head_motion_path(self) -> Path:
        """Head speed/drift from the Neon IMU (see src/eye_tracking/head_monitor.py), written at the end of the saccade phase."""
        return self._folder / session_files.HEAD_MOTION_FILE

    @property
    def analysis_dir(self) -> Path:
        return self._folder / session_files.ANALYSIS_DIR

    @property
    def glasses_dir(self) -> Path:
        return self._folder / session_files.GLASSES_DIR

    def _write_metadata(self) -> None:
        self._metadata["results"]["phases"] = _results_summary(self._rows)
        self._meta_path.write_text(json.dumps(self._metadata, indent=2))

    def set_calibration(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._metadata["settings"]["calibration"] = {
            "left_ratio": left_ratio,
            "center_ratio": center_ratio,
            "right_ratio": right_ratio,
        }
        self._write_metadata()

    def set_phase_summary(self, phase: str, summary: dict) -> None:
        """Final numbers for the saccade phase: trial counts, final reaction-time average and its sample counts
        (merged into session.json "results")."""
        key = "saccade_phase_summary" if phase == "saccade" else f"{phase}_phase_summary"
        self._metadata["results"][key] = summary
        self._write_metadata()

    def set_geometry(self, geometry) -> None:
        """The screen geometry this run used (heights/distance, the line's size in degrees and screen-height units, placeholder or measured)."""
        self._metadata["settings"].update(geometry.as_settings())
        self._write_metadata()

    def set_saccade_contrasts(self, levels) -> None:
        """The fixed probe contrasts (0-1 fractions) used in the saccade phase of this session, stored as percent."""
        self._metadata["settings"]["saccade_contrasts_pct"] = [round(float(c) * 100, 4) for c in levels]
        self._write_metadata()

    def set_flash_spread_sd_ms(self, sd_ms: float) -> None:
        """The bell-curve width (SD, ms) of the flash times this run used (startup dialog; FLASH_SPREAD_SD_MS is the default)."""
        self._metadata["settings"]["flash_spread_sd_ms"] = float(sd_ms)
        self._write_metadata()

    def set_return_jump(self, enabled: bool) -> None:
        """Whether this run used return-jump trials (startup dialog "Return jump"; RETURN_JUMP_DEFAULT is the first-time default).
        Absent in older session.json files, which never jumped back."""
        self._metadata["settings"]["return_jump"] = bool(enabled)
        self._write_metadata()

    def set_display_refresh_hz(self, hz: float | None) -> None:
        self._metadata["hardware"]["display_refresh_hz"] = None if hz is None else round(float(hz), 2)
        self._write_metadata()

    def set_neon_device(self, label: str) -> None:
        """Where the Neon was actually reached, e.g. '192.168.43.1:8080 (PC's gateway (phone hotspot))' (start-up check in run_experiment)."""
        self._metadata["hardware"]["neon_device"] = label
        self._write_metadata()

    def set_neon_phone_ip(self, ip: str | None) -> None:
        """The IP the phone reports about itself (NeonGazeSource.phone_ip): clock sync and the gaze stream use it, so a value that
        differs from neon_device explains a failed clock offset. Absent in older session.json files."""
        self._metadata["hardware"]["neon_phone_ip"] = ip
        self._write_metadata()

    def set_gaze_quality(self, summary: dict) -> None:
        """Session-level gaze tracking quality (sample rate, gaps, latency,
        clock offset) from a GazeSource that reports it - see
        NeonGazeSource.quality_summary()."""
        self._metadata["gaze_quality"] = summary
        self._write_metadata()

    def set_head_motion(self, summary: dict) -> None:
        """Head-movement monitor summary (status, max speed/drift, flagged trials) from NeonGazeSource.head_motion_summary();
        status "unavailable (reason)" when the IMU could not be used."""
        self._metadata["head_motion"] = summary
        self._write_metadata()

    def set_ignored_space_presses(self, label: str, counts: dict) -> None:
        """SPACE presses that did nothing in one phase run, keyed "PHASE: reason" (see space_input.SpaceTally)."""
        self._metadata["ignored_space_presses"][label] = dict(counts)
        self._write_metadata()

    def update_glasses_video(self, **fields) -> None:
        """Phone recording status and the PC clock time of every sync flash (see glasses_video.py / README)."""
        self._metadata["glasses_video"].update(fields)
        self._write_metadata()

    def add_sync_flash(self, label: str, unix_ms: float) -> None:
        """One green sync-flash onset (PC wall clock, ms), appended to glasses_video.sync_flashes."""
        flashes = self._metadata["glasses_video"].setdefault("sync_flashes", [])
        flashes.append({"label": label, "pc_unix_ms": round(float(unix_ms), 1)})
        self._write_metadata()

    def log(self, result: TrialResult) -> None:
        self._writer.writerow({c.name: c.write(result) for c in TRIAL_COLUMNS})
        self._file.flush()
        self._rows.append(result)

    def close(self) -> None:
        self._file.close()
        self._write_metadata()
