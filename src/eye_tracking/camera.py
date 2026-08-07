import threading
import time

import PySpin

from include.eye_tracking.constants import (
    CAMERA_EXPOSURE_TIME_US,
    CAMERA_FRAME_RATE,
    CAMERA_INDEX,
    CAMERA_PROBE_LIMIT,
    FRAME_HEIGHT,
    FRAME_OFFSET_X,
    FRAME_OFFSET_Y,
    FRAME_WIDTH,
)

_GRAB_TIMEOUT_MS = 1000  # GetNextImage() wait before we give the read loop a chance to check _running again


def list_available_cameras(max_index: int = CAMERA_PROBE_LIMIT) -> list[int]:
    """Returns positional indices (0..N-1) into the Spinnaker camera list for
    every FLIR camera currently connected - used to offer a camera picker
    when more than one is attached (e.g. a spare Blackfly S plugged in
    alongside the main one), rather than requiring CAMERA_INDEX to be
    hand-edited to switch between them."""
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        try:
            return list(range(min(cam_list.GetSize(), max_index)))
        finally:
            cam_list.Clear()
    finally:
        system.ReleaseInstance()


def _configure_camera(cam) -> None:
    """Initializes a Blackfly S for fixed-exposure, fixed-framerate Mono8
    capture - no auto-exposure/auto-gain hunting mid-experiment, and a known
    pixel format/frame rate the rest of the pipeline can rely on."""
    cam.Init()

    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)

    # Offsets must be zeroed before shrinking Width/Height - a nonzero offset
    # left over from a previous run can otherwise make offset+size exceed the
    # sensor bounds and reject the resize outright.
    cam.OffsetX.SetValue(0)
    cam.OffsetY.SetValue(0)

    width = min(FRAME_WIDTH, cam.WidthMax.GetValue())
    height = min(FRAME_HEIGHT, cam.HeightMax.GetValue())
    cam.Width.SetValue(width)
    cam.Height.SetValue(height)

    # Max offset shrinks as size grows (offset + size can't exceed the sensor),
    # so clamp against it now that Width/Height are set, same as width/height
    # above are clamped against the sensor's own max.
    cam.OffsetX.SetValue(min(FRAME_OFFSET_X, cam.OffsetX.GetMax()))
    cam.OffsetY.SetValue(min(FRAME_OFFSET_Y, cam.OffsetY.GetMax()))

    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(CAMERA_EXPOSURE_TIME_US)

    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(min(CAMERA_FRAME_RATE, cam.AcquisitionFrameRate.GetMax()))

    # Always hand the read loop the newest frame rather than letting unread
    # frames queue up and drag gaze samples behind real time.
    stream_nodemap = cam.GetTLStreamNodeMap()
    handling_mode = PySpin.CEnumerationPtr(stream_nodemap.GetNode("StreamBufferHandlingMode"))
    handling_mode.SetIntValue(handling_mode.GetEntryByName("NewestOnly").GetValue())


class Camera:
    """Continuously reads frames from a Teledyne FLIR Blackfly S (PySpin/
    Spinnaker SDK) on a background thread."""

    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self._index = index
        self._system = None
        self._cam_list = None
        self._cam = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_open(self) -> bool:
        return self._cam is not None and self._cam.IsStreaming()

    def start(self) -> None:
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        try:
            cam = self._cam_list.GetByIndex(self._index)
            _configure_camera(cam)
            cam.BeginAcquisition()
        except Exception:
            self._cam_list.Clear()
            self._cam_list = None
            self._system.ReleaseInstance()
            self._system = None
            raise
        self._cam = cam
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running and self._cam is not None:
            try:
                image = self._cam.GetNextImage(_GRAB_TIMEOUT_MS)
                try:
                    if image.IsIncomplete():
                        continue
                    # Raw Mono8 (grayscale), not converted to color here - this loop
                    # runs at the camera's full acquisition rate (~200fps) but only
                    # the single latest frame is ever consumed, so converting every
                    # captured frame wasted most of that work on frames that get
                    # overwritten before anyone reads them. .copy() is required:
                    # GetNDArray() views PySpin's own buffer, which image.Release()
                    # (below) returns to the SDK's pool.
                    frame = image.GetNDArray().copy()
                    with self._lock:
                        self._latest_frame = frame
                finally:
                    image.Release()
            except PySpin.SpinnakerException:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            # Give the read loop headroom to finish an in-flight GetNextImage()
            # call (up to _GRAB_TIMEOUT_MS) before we tear down the camera under it.
            self._thread.join(timeout=_GRAB_TIMEOUT_MS / 1000 + 0.5)
        if self._cam is not None:
            cam = self._cam
            self._cam = None
            if cam.IsStreaming():
                cam.EndAcquisition()
            cam.DeInit()
            del cam
        if self._cam_list is not None:
            self._cam_list.Clear()
            self._cam_list = None
        if self._system is not None:
            self._system.ReleaseInstance()
            self._system = None
