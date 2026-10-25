"""SPACE-press bookkeeping shared by the sessions and the runner (no PsychoPy): a SPACE press must never silently disappear.

Rules (each is tested in tests/ai-tests/test_space_input.py):
- Calibration: one press is enough. If the gaze window is not full yet, the press is latched (`CalibrationLatch`) and the
  step completes the moment the window fills.
- Tutorial narration: a press made while the narrator is speaking is held (`NarrationKeyGate`) and delivered once speech ends.
- Trials: a press is accepted in the AWAITING_SPACE phases. During the foreperiod or the trial it is ignored (it must not
  restart the foreperiod or start the next trial), except a press after the participant has already answered, which
  starts the next trial the moment it is ready. Every ignored press is counted per phase with its reason (`SpaceTally`).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable


class SpaceTally:
    """Counts SPACE presses that did nothing, keyed by (phase, reason)."""

    def __init__(self, current_phase: Callable[[], str]) -> None:
        self._current_phase = current_phase
        self._counts: Counter[tuple[str, str]] = Counter()

    def record(self, reason: str, count: int = 1, phase: str | None = None) -> None:
        if count > 0:
            self._counts[(phase if phase is not None else self._current_phase(), reason)] += count

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def as_dict(self) -> dict[str, int]:
        return {f"{phase}: {reason}": n for (phase, reason), n in sorted(self._counts.items())}

    def summary(self, label: str) -> str:
        if not self._counts:
            return f"{label}: no ignored SPACE presses"
        parts = "; ".join(f"{key} x{n}" for key, n in self.as_dict().items())
        return f"{label}: {self.total} ignored SPACE press(es) - {parts}"


class CalibrationLatch:
    """One pending calibration press. `press()` returns the window's mean ratio when the window is already full; otherwise it
    latches (restarting the window so it holds only samples gathered after the press) and `poll()` returns the ratio once it fills."""

    def __init__(self) -> None:
        self.pending = False
        self._since: float | None = None

    def press(self, gaze, now: float) -> float | None:
        ratio = gaze.average_recent_ratio()
        if ratio is not None:
            self.clear()
            return ratio
        if not self.pending:
            self.pending = True
            self._since = now
            gaze.begin_calibration_sample()  # the participant is on the target now: collect a clean window from here
        return None

    def poll(self, gaze) -> float | None:
        if not self.pending:
            return None
        ratio = gaze.average_recent_ratio()
        if ratio is not None:
            self.clear()
        return ratio

    def clear(self) -> None:
        self.pending = False
        self._since = None

    def waiting_s(self, now: float) -> float | None:
        return None if not self.pending or self._since is None else now - self._since


class NarrationKeyGate:
    """Tutorial key routing: while the narrator speaks, nothing reaches the session, but one SPACE press is remembered and
    delivered as soon as speech ends (a response key made during speech is dropped - the question it answers was not heard
    yet). At most one SPACE per frame is delivered: a same-frame repeat is a duplicate, and chained presses could skip a stage."""

    def __init__(self) -> None:
        self.pending_space = False

    def route(self, keys: list[str], speaking: bool) -> tuple[bool, list[str], int]:
        """-> (deliver_space, response_keys_to_deliver, extra_space_presses_dropped)."""
        spaces = keys.count("space")
        if speaking:
            held_before = self.pending_space
            self.pending_space = self.pending_space or spaces > 0
            return False, [], max(0, spaces - (0 if held_before else 1))
        deliver = self.pending_space or spaces > 0
        extra = max(0, spaces + (1 if self.pending_space else 0) - 1)
        self.pending_space = False
        return deliver, [k for k in keys if k != "space"], extra


def deliver_space(session, presses: int) -> bool:
    """Forward this frame's SPACE presses to a session as ONE press: same-frame repeats are duplicates (and, with latched
    calibration presses, could chain two stages), so they are counted instead. Returns whether a press was delivered."""
    if presses <= 0:
        return False
    session.on_space()
    session.space_tally.record("duplicate press in the same frame", presses - 1)
    return True


def leaves_complete_screen(previous_frame_phase: str | None, space_pressed: bool) -> bool:
    """A SPACE press only dismisses the COMPLETE screen once the participant has had a frame of it: a press that arrives in the
    very frame the trials finish was meant for the last trial, not for a results screen they have not seen."""
    return space_pressed and previous_frame_phase == "COMPLETE"
