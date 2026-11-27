"""Everything the participant reads or hears (one string = the on-screen caption and the narrated line).
Plain strings only; rules enforced by tests/ai-tests/test_participant_text.py: each at most 15 words, one idea per screen,
dots named by position ("on the left"), never "the other dot" or "cross". The summary line of the end screen is added by the session."""

# -- tutorial intro (SPACE advances; the first line is the start screen) --
TUTORIAL_WELCOME = "Welcome. Keep your head still on the chin rest. Press SPACE."
TUTORIAL_INTRO = (
    "Look at the dot. When it jumps to the other side, look there right away.",
    "A thin line flashes. Press UP if it was above the dot, DOWN if below.",
    "The line can be very faint. If you are not sure, guess.",
    "You can rest between trials. Nothing starts until you press SPACE.",
)
# Run setting "Return jump" on: shown after the first intro line, and replacing the practice/first-trial lines below.
TUTORIAL_INTRO_RETURN_JUMP = "Then it jumps back to where it started. Look back right away."

# -- calibration (tutorial and, without a tutorial, the saccade phase) --
CALIBRATE_LEFT = "Look at the dot on the left, then press SPACE."
CALIBRATE_CENTER = "Look at the middle dot, then press SPACE."
CALIBRATE_RIGHT = "Look at the dot on the right, then press SPACE."
# Rounds 2 and 3 start again at the left; only that screen is spoken (the centre and right screens stay silent).
CALIBRATE_LEFT_AGAIN = "Once more: look at the dot on the left, then press SPACE."
CALIBRATE_LEFT_LAST = "Last time: look at the dot on the left, then press SPACE."

# -- quiz and practice (tutorial) --
QUIZ = "Press UP if the line is above the dot, DOWN if below. Not sure? Guess."
PRACTICE_INTRO = "Practice lines are clear. Real ones are much fainter, so guessing is normal."
PRACTICE_DONE = "Practice done. The real test starts now. Press SPACE."
PRACTICE_START_RETURN_JUMP = "Press SPACE. Look at the left dot. Follow it when it jumps and returns."

# -- saccade phase --
RT_TEST_INTRO = "Press SPACE. Look at the dot on the left. Follow it when it jumps."
ANSWER = "Press UP if the line was above, DOWN if below. If unsure, guess."
# SPACE pressed on a flashed trial that has no answer yet (practice and real trials): an answer is required.
ANSWER_FIRST = "First answer with UP or DOWN. Not sure? Guess."
FIRST_TRIAL = "Nothing starts until you press SPACE. Follow the dot when it jumps."
FIRST_TRIAL_RETURN_JUMP = "Nothing starts until you press SPACE. Follow the dot when it jumps and returns."
FINISHED = "All done. Thank you!"

# -- glasses-video sync step --
SYNC_FLASH = "The screen will flash green a few times. This is normal."
