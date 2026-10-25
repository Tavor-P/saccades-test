"""Session-level scheduling decisions, PsychoPy-free: which phase runs first (counterbalanced by participant ID)."""

import zlib

PHASE_BASELINE = "baseline"
PHASE_SACCADE = "saccade"


def phase_order(participant_id: str, test_mode: bool) -> tuple[str, str]:
    """Test runs always go saccade-first (the phase under development). Real runs alternate by participant ID so that order effects (practice,
    fatigue) are balanced across participants: digit-only IDs alternate by parity (consecutive IDs get opposite orders), other IDs by a stable
    hash - never Python's salted hash(), which would differ between runs for the same ID."""
    if test_mode:
        return PHASE_SACCADE, PHASE_BASELINE
    pid = participant_id.strip()
    bucket = int(pid) % 2 if pid.isdigit() else zlib.crc32(pid.lower().encode("utf8")) % 2
    return (PHASE_BASELINE, PHASE_SACCADE) if bucket == 0 else (PHASE_SACCADE, PHASE_BASELINE)
