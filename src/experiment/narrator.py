import queue
import threading


class Narrator:
    """Speaks instruction text aloud on a background thread, using Windows'
    built-in SAPI text-to-speech (via pywin32, already a PsychoPy dependency -
    no new package needed). Participants have to hold fixation on specific
    on-screen targets throughout calibration and every trial; text at the
    bottom of the screen would mean looking away from those targets just to
    read it, which defeats the point of a saccade/fixation experiment. Speech
    replaces that text entirely instead of supplementing it.

    Runs on its own thread because SAPI's Speak() call blocks for as long as
    the speech takes (real seconds) - calling it directly from the PsychoPy
    render loop would stall rendering for that whole duration. COM objects
    are apartment-threaded, so the SAPI voice has to be created and used on
    the same thread that will call it - see _run().
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            while True:
                text = self._queue.get()
                if text is None:
                    return
                voice.Speak(text)
        finally:
            pythoncom.CoUninitialize()

    def speak(self, text: str) -> None:
        """Queues text to be spoken. Already-queued/in-progress speech isn't
        interrupted - calls just queue up and play in order, so callers
        should only call this when the text actually changes (see
        run_experiment.py), not every frame, or utterances would pile up
        far behind what's currently true on screen."""
        self._queue.put(text)

    def stop(self) -> None:
        """Signals the background thread to exit and waits (briefly) for it.
        Doesn't cut off speech already in progress - the sentinel is queued
        after whatever's already pending."""
        self._queue.put(None)
        self._thread.join(timeout=2)
