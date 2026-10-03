"""Sends the robot's camera frames from midterm_competition.py to config_gui.py.

When config_gui.py runs the script, it tells it where to send frames through
environment variables. Run on its own, the script doesn't get them and shows
its usual OpenCV window instead.
"""

import os
import secrets
import threading
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener

import cv2
import numpy as np

ADDRESS_ENV = "MIDTERM_VIDEO_ADDRESS"
KEY_ENV = "MIDTERM_VIDEO_KEY"


def sender_from_env():
    """Returns a FrameSender if config_gui.py started this script, else None."""
    address = os.environ.get(ADDRESS_ENV)
    key = os.environ.get(KEY_ENV)
    if not address or not key:
        return None
    host, port = address.rsplit(":", 1)
    return FrameSender((host, int(port)), bytes.fromhex(key))


class FrameSender:
    """Sends frames from a background thread, so the robot loop never waits.

    Only the newest frame is kept: if the window falls behind, older frames are
    dropped. If the window can't be reached, frames are quietly discarded.
    """

    def __init__(self, address, authkey):
        self._address = address
        self._authkey = authkey
        self._frame = None
        self._new_frame = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, frame):
        with self._new_frame:
            self._frame = frame
            self._new_frame.notify()

    def _run(self):
        try:
            with Client(self._address, authkey=self._authkey) as conn:
                while True:
                    with self._new_frame:
                        self._new_frame.wait_for(lambda: self._frame is not None)
                        frame, self._frame = self._frame, None
                    conn.send_bytes(frame)
        except (OSError, EOFError, AuthenticationError):
            # The window was closed: the robot carries on without video
            pass


class FrameReceiver:
    """Receives frames in a background thread and keeps the newest one.

    Frames are decoded and scaled here too, so the window only has to show them.
    """

    def __init__(self, size):
        self.size = size  # (width, height) that frames are scaled to fit
        # Only something given this key (the script we start) can connect
        self._authkey = secrets.token_bytes(32)
        self._listener = Listener(("127.0.0.1", 0), authkey=self._authkey)
        self._lock = threading.Lock()
        self._image = None
        threading.Thread(target=self._run, daemon=True).start()

    def env(self):
        """Environment variables that tell the script where to send frames."""
        host, port = self._listener.address
        return {
            ADDRESS_ENV: f"{host}:{port}",
            KEY_ENV: self._authkey.hex(),
        }

    def take_image(self):
        """Returns the newest frame as PNG data, or None if there's no new one."""
        with self._lock:
            image, self._image = self._image, None
        return image

    def _run(self):
        # One connection per run of the script
        while True:
            try:
                conn = self._listener.accept()
            except (AuthenticationError, OSError, EOFError):
                continue  # Something else tried to connect; ignore it
            with conn:
                try:
                    while True:
                        # recv_bytes, not recv: recv would unpickle what it gets
                        image = frame_to_png(conn.recv_bytes(), self.size)
                        if image is not None:
                            with self._lock:
                                self._image = image
                except (EOFError, OSError):
                    pass  # The script exited


def frame_to_png(frame, size):
    """Decodes a camera frame and scales it to fit size, keeping its shape.

    Returns PNG data (which Tk can show), or None if the frame can't be decoded.
    """
    try:
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        # e.g. an empty frame; raising would stop the video for good
        return None
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = min(size[0] / width, size[1] / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    # Light compression: the PNG only travels within this program
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return data.tobytes() if ok else None
