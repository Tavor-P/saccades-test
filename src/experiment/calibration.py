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
