"""Windows-only text-to-speech (SAPI via COM) on a background thread, used by the tutorial. See CLAUDE.md for the COM threading gotchas."""

import queue
import threading

# SAPI SpeechVoiceSpeakFlags: speak in the background (so a newer line can cut it off) / first drop whatever SAPI is saying.
_SPEAK_ASYNC = 1
_PURGE_BEFORE_SPEAK = 2
_POLL_MS = 20  # how often the thread, while a line plays, checks whether a newer one replaced it


class Narrator:
    """Speaks instruction text aloud on a background thread, using Windows'
    built-in SAPI text-to-speech (via pywin32, already a PsychoPy dependency -
    no new package needed). Participants have to hold fixation on specific
    on-screen targets throughout calibration and every trial, so instructions
    are spoken rather than relying on requiring a glance away to read them.
    A small caption at the bottom of the screen (see run_experiment.py's
    render_caption) shows the same text at the same time, for anyone who
    needs to double-check what was said without waiting to hear it repeated -
    it's peripheral/supplementary, not the primary channel.

    Runs on its own thread because SAPI's Speak() call blocks for as long as
    the speech takes (real seconds) - calling it directly from the PsychoPy
    render loop would stall rendering for that whole duration. COM objects
    are apartment-threaded, so the SAPI voice has to be created and used on
    the same thread that will call it - see _run().

    A new line replaces the old one: screens can be skipped faster than their
    lines are spoken, so speak() drops everything queued or playing. Each such
    call starts a new "generation"; the thread skips, or stops waiting for,
    anything from an older one.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[int, str, bool] | None] = queue.Queue()  # (generation, text, interrupt)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._pending = 0
        self._generation = 0
        self._thread.start()

    def _run(self) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            while True:
                item = self._queue.get()
                if item is None:
                    return
                try:
                    self._say(voice, *item)
                finally:
                    with self._lock:
                        self._pending -= 1
        finally:
            pythoncom.CoUninitialize()

    def _say(self, voice, generation: int, text: str, interrupt: bool) -> None:
        """Narrator thread only. Returns once `text` has been spoken or a newer generation replaced it (the newer line's
        purge flag is what cuts the audio)."""
        if generation != self._generation:
            return  # replaced while still queued: never started
        voice.Speak(text, _SPEAK_ASYNC | (_PURGE_BEFORE_SPEAK if interrupt else 0))
        while text and generation == self._generation and not voice.WaitUntilDone(_POLL_MS):
            pass

    def speak(self, text: str, interrupt: bool = True) -> None:
        """Speaks `text`, dropping any older speech still queued or playing. Callers should only call this when the text
        actually changes (see rendering.narrate_if_changed): every call restarts the line. `interrupt=False` instead
        says it after what is already queued (the head warning, which belongs to no screen); the next interrupting call
        drops it like any other line."""
        with self._lock:
            if interrupt:
                self._generation += 1
            self._pending += 1
            self._queue.put((self._generation, text, interrupt))

    def silence(self) -> None:
        """Drops all queued and playing speech."""
        self.speak("")

    @property
    def is_speaking(self) -> bool:
        """True while the latest line (or one queued after it) is still playing or waiting to play. The tutorial frame
        loop uses this to hold off on forwarding input until narration finishes."""
        with self._lock:
            return self._pending > 0

    def stop(self) -> None:
        """Signals the background thread to exit and waits (briefly) for it.
        Doesn't cut off speech already in progress - the sentinel is queued
        after whatever's already pending."""
        self._queue.put(None)
        self._thread.join(timeout=2)
