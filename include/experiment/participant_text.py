"""Everything the participant reads or hears (one string = the on-screen caption and the narrated line).
Plain strings only; rules enforced by tests/ai-tests/test_participant_text.py: each at most 15 words, one idea per screen,
dots named by position ("on the left"), never "the other dot" or "cross". The summary line of the end screen is added by the session."""

# -- tutorial intro (SPACE advances; the first line is the start screen) --
TUTORIAL_WELCOME = "Welcome. Keep your head still on the chin rest. Press SPACE."
TUTORIAL_INTRO = (
    "When the dot jumps to the other side, look at it right away.",
    "A thin line flashes briefly. Press UP if it was above, DOWN if below.",
    "The line can be very faint. If you are not sure, guess.",
    "You can rest between trials. Nothing starts until you press SPACE.",
)

# -- calibration (tutorial and, without a tutorial, the saccade phase) --
CALIBRATE_LEFT = "Look at the dot on the left, then press SPACE."
CALIBRATE_CENTER = "Look at the middle dot, then press SPACE."
CALIBRATE_RIGHT = "Look at the dot on the right, then press SPACE."

# -- quiz and practice (tutorial) --
QUIZ = "Press UP if the line is above the dot, DOWN if it is below."
PRACTICE_INTRO = "Practice lines are clear. Real ones are much fainter, so guessing is normal."
PRACTICE_DONE = "Practice done. The real test starts now. Press SPACE."

# -- saccade phase --
RT_TEST_INTRO = "Press SPACE. Look at the dot on the left. Follow it when it jumps."
ANSWER = "Press UP if the line was above, DOWN if below. If unsure, guess."
FIRST_TRIAL = "Nothing starts until you press SPACE. Follow the dot when it jumps."
FINISHED = "All done. Thank you!"

# -- glasses-video sync step --
SYNC_FLASH = "The screen will flash green a few times. This is normal."
