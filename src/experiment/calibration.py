"""The 3-point gaze calibration (left, center, right), repeated CALIBRATION_ROUNDS times; round 1 is discarded as the least reliable."""

from include.experiment.constants import CALIBRATION_ROUNDS
from src.experiment.space_input import CalibrationLatch, SpaceTally

STEP_LEFT, STEP_CENTER, STEP_RIGHT = 0, 1, 2


def average_calibration_rounds(
    round2: tuple[float, float, float], round3: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Combines the 2nd and 3rd rounds of a 3-round calibration into the final
    (left, center, right) ratios. Round 1 is deliberately not a parameter here
    - it's discarded by both ExperimentSession and TutorialSession before this
    is ever called, on the assumption that a participant's first calibration
    attempt is likely to be their least reliable one."""
    return (
        (round2[0] + round3[0]) / 2,
        (round2[1] + round3[1]) / 2,
        (round2[2] + round3[2]) / 2,
    )


class CalibrationRounds:
    """The calibration step/round state machine shared by ExperimentSession and TutorialSession. The owning session maps
    `step` onto its own phase; `press()`/`poll()` return the final (left, center, right) ratios once the last round is
    done, else None."""

    def __init__(self, gaze) -> None:
        self._gaze = gaze
        self.latch = CalibrationLatch()  # a press made before the gaze window filled
        self.step = STEP_LEFT
        self.round = 1
        self._rounds: list[tuple[float, float, float]] = []
        self._left: float | None = None
        self._center: float | None = None

    def start(self) -> None:
        self.round = 1
        self._rounds = []
        self.step = STEP_LEFT
        self._gaze.begin_calibration_sample()  # fresh window before the dot appears

    def press(self, now: float, tally: SpaceTally) -> tuple[float, float, float] | None:
        """One press is enough: if the gaze window is not full yet the press is latched and poll() completes it."""
        ratio = self.latch.press_recorded(self._gaze, now, tally)
        return None if ratio is None else self._advance(ratio)

    def poll(self) -> tuple[float, float, float] | None:
        ratio = self.latch.poll(self._gaze)
        return None if ratio is None else self._advance(ratio)

    def _advance(self, ratio: float) -> tuple[float, float, float] | None:
        self.latch.clear()
        if self.step == STEP_LEFT:
            self._left = ratio
            self.step = STEP_CENTER
        elif self.step == STEP_CENTER:
            self._center = ratio
            self.step = STEP_RIGHT
        else:
            if self._left is None or self._center is None:
                return None
            self._rounds.append((self._left, self._center, ratio))
            if self.round >= CALIBRATION_ROUNDS:
                final = average_calibration_rounds(self._rounds[-2], self._rounds[-1])
                self._gaze.calibrate(*final)
                return final
            self.round += 1
            self.step = STEP_LEFT
        self._gaze.begin_calibration_sample()  # fresh window before the next target appears
        return None
