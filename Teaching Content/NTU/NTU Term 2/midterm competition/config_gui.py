"""Settings window for midterm_competition.py.

Run it the same way as the script: uv run config_gui.py

The constants live in midterm_competition.py itself: this window reads them from
there, and Save writes each changed value back into its own line, leaving
comments and the rest of the file alone. Editing the file directly still works;
use "Reload from file" to pick up those edits.
"""

import ast
import base64
import ipaddress
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import video_link

SCRIPT_PATH = Path(__file__).with_name("midterm_competition.py")
STOP_TIMEOUT_S = 5  # How long Stop waits for a clean exit before force-killing
POLL_MS = 33  # Also how often the video updates: about 30 frames a second
VIDEO_SIZE = (480, 360)  # Camera frames are scaled down to fit this


@dataclass(frozen=True)
class Field:
    name: str  # Constant name in midterm_competition.py
    label: str
    kind: str  # How the value is checked; see parse_value()
    hint: str = ""


# Same order and grouping as the constants block in midterm_competition.py
SECTIONS = [
    (
        "Robot",
        [
            Field("ROBOT_IP", "Robot IP", "ip"),
        ],
    ),
    (
        "Arm",
        [
            Field(
                "ARM_LOWERED_POSE",
                "Lowered pose",
                "pose",
                "joint angles (degrees) to grab / release",
            ),
            Field(
                "ARM_CARRY_POSE",
                "Carry pose",
                "pose",
                "joint angles (degrees) to carry",
            ),
            Field("ARM_MOVE_TIME_MS", "Arm move time", "int", "ms per arm move"),
            Field(
                "PICKUP_DELAY",
                "Pickup delay",
                "delay",
                "seconds to wait after each pickup step",
            ),
            Field(
                "PUTDOWN_DELAY",
                "Putdown delay",
                "delay",
                "seconds to wait after each putdown step",
            ),
        ],
    ),
    (
        "Line following",
        [
            Field("LINE_SPEED", "Speed", "number"),
            Field(
                "LINE_TURN_GAIN",
                "Turn gain",
                "number",
                "turn speed per px of line offset",
            ),
            Field(
                "LINE_MISSED_FRAMES_TO_STOP",
                "Missed frames to switch",
                "frame_count",
                "frames in a row with no line before looking for the face",
            ),
        ],
    ),
    (
        "Face approach",
        [
            Field("TARGET_FACE", "Target face", "text"),
            Field("FACE_FWD_SPEED", "Forward speed", "number"),
            Field("FACE_STRAFE_SPEED", "Strafe speed", "number"),
            Field(
                "FACE_CENTER_TOLERANCE_PX",
                "Center tolerance",
                "number",
                "px the face can be from center (each side)",
            ),
            Field(
                "FACE_STOP_HEIGHT_PX",
                "Stop height",
                "number",
                "face box height (px) that counts as close enough",
            ),
            Field(
                "FACE_MISSED_FRAMES_TO_STOP",
                "Missed frames to stop",
                "frame_count",
                "frames in a row without the face before stopping",
            ),
        ],
    ),
]
FIELDS = [field for _, fields in SECTIONS for field in fields]


class ConstantsError(Exception):
    """The script's constants can't be read or safely rewritten."""


# --- Checking values ---


def _whole_number(text):
    try:
        return int(text)
    except ValueError:
        raise ValueError("must be a whole number") from None


def _number(text):
    try:
        value = int(text)
    except ValueError:
        try:
            value = float(text)
        except ValueError:
            raise ValueError("must be a number") from None
    # inf / nan would be written as names Python doesn't know, breaking the script
    if not math.isfinite(value):
        raise ValueError("must be a number")
    return value


def parse_value(kind, texts):
    """Turns the text typed into a field into the value to save.

    texts has one string per input box (three for a pose). Raises ValueError
    with a short reason if the text isn't valid.
    """
    if kind == "pose":
        try:
            return tuple(int(text) for text in texts)
        except ValueError:
            raise ValueError("joint angles must be whole numbers") from None

    (text,) = texts
    if kind == "ip":
        try:
            ipaddress.ip_address(text)
        except ValueError:
            raise ValueError("must be an IP address like 192.168.1.204") from None
        return text
    if kind == "text":
        if not text.strip():
            raise ValueError("can't be empty")
        return text
    if kind == "int":
        return _whole_number(text)
    if kind == "number":
        return _number(text)
    if kind == "delay":
        value = _number(text)
        # time.sleep() crashes on a negative delay
        if value < 0:
            raise ValueError("can't be negative")
        return value
    if kind == "frame_count":
        value = _whole_number(text)
        # With 0 the robot would switch / stop even while it can still see
        if value < 1:
            raise ValueError("must be at least 1")
        return value
    raise ValueError(f"unknown field kind {kind!r}")


def value_to_texts(kind, value):
    """The text to show in a field's input box(es) for a value."""
    if kind == "pose":
        if not (isinstance(value, tuple) and len(value) == 3):
            raise ValueError("must be three joint angles, like (0, 45, 45)")
        return tuple(str(angle) for angle in value)
    return (str(value),)


# --- Reading and writing the script ---


def find_constants(source):
    """Finds each field's constant in the script source (bytes).

    Returns {name: (start, end, value)}, where start/end are byte offsets of
    the value's text, so it can be replaced without touching anything else.
    """
    wanted = {field.name for field in FIELDS}
    tree = ast.parse(source)
    # ast positions are (line, byte offset in that line); turn them into
    # offsets in the whole source
    line_starts = [0] + [i + 1 for i, byte in enumerate(source) if byte == ord("\n")]

    found = {}
    problems = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            continue
        name = node.targets[0].id
        if name in found:
            problems.append(f"{name} is set more than once")
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            problems.append(
                f"{name} isn't a plain value (for example it's calculated), "
                "so this window can't edit it"
            )
            value = None
        start = line_starts[node.value.lineno - 1] + node.value.col_offset
        end = line_starts[node.value.end_lineno - 1] + node.value.end_col_offset
        found[name] = (start, end, value)

    for field in FIELDS:
        if field.name not in found:
            problems.append(f"{field.name} is missing")
    if problems:
        raise ConstantsError("\n".join(problems))
    return found


def read_constants(source):
    """Returns {name: value} for every field, checked like the form checks input."""
    found = find_constants(source)
    values = {}
    problems = []
    for field in FIELDS:
        value = found[field.name][2]
        # Round-trip through the form's checks: anything the form would reject
        # (or show differently, like a number stored as text) is a problem
        reason = "has the wrong type"
        try:
            ok = parse_value(field.kind, value_to_texts(field.kind, value)) == value
        except ValueError as error:
            ok = False
            reason = str(error)
        if ok:
            values[field.name] = value
        else:
            problems.append(f"{field.name} {reason} (it's {value!r})")
    if problems:
        raise ConstantsError("\n".join(problems))
    return values


def format_literal(value):
    """Python source text for a value, in the style the script already uses."""
    if isinstance(value, str):
        # json.dumps gives a double-quoted string with Python-compatible escapes
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, tuple):
        return "(" + ", ".join(repr(item) for item in value) + ")"
    return repr(value)


def update_constants(source, values):
    """Returns the source with the given constants set to new values.

    Values that haven't changed are left exactly as written (so `1` doesn't
    become `1.0`). The result is read back and checked before it's returned,
    so a bad edit can't reach the file.
    """
    found = find_constants(source)
    edits = []
    for name, value in values.items():
        start, end, old = found[name]
        if value == old and type(value) is type(old):
            continue
        edits.append((start, end, format_literal(value).encode()))

    # Replace from the end of the file backwards so earlier offsets stay valid
    for start, end, literal in sorted(edits, reverse=True):
        source = source[:start] + literal + source[end:]

    try:
        written = read_constants(source)
    except (ConstantsError, SyntaxError) as error:
        raise ConstantsError(f"Saving would break the script:\n{error}") from None
    if any(written[name] != value for name, value in values.items()):
        raise ConstantsError("Saving would not store the values correctly")
    return source


# --- Running the script ---


class ScriptRunner:
    """Runs the robot script in its own process and collects its output.

    A separate process keeps the script's OpenCV camera window from fighting
    with this window over the main thread.
    """

    def __init__(self, script_path, stop_timeout=STOP_TIMEOUT_S):
        self.script_path = Path(script_path)
        self.stop_timeout = stop_timeout
        self.proc = None
        self._output = queue.Queue()
        self._stop_deadline = None
        self._killed = False

    @property
    def active(self):
        """True from start() until poll() has reported that the script exited."""
        return self.proc is not None

    @property
    def stopping(self):
        return self.proc is not None and self._stop_deadline is not None

    def start(self, extra_env=None):
        if self.active:
            raise RuntimeError("the script is already running")
        # Windows can only deliver Ctrl+Break to a child in its own process group
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            # The Python running this window is the project's, so it has the
            # script's packages; -u so output shows up in the log right away
            [sys.executable, "-u", str(self.script_path)],
            cwd=self.script_path.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            # Otherwise Windows uses its local code page, and printing a
            # non-English face name could crash the script
            env={**os.environ, "PYTHONIOENCODING": "utf-8", **(extra_env or {})},
            creationflags=flags,
        )
        self._stop_deadline = None
        self._killed = False
        self._reader = threading.Thread(
            target=self._read_output, args=(self.proc.stdout,), daemon=True
        )
        self._reader.start()

    def _read_output(self, stream):
        for line in stream:
            self._output.put(line)
        stream.close()

    def request_stop(self):
        """Asks the script to stop cleanly; it stops the wheels on its way out."""
        if self.proc is None or self.proc.poll() is not None or self.stopping:
            return
        self._stop_deadline = time.monotonic() + self.stop_timeout
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
        except OSError:
            # e.g. Windows without a console window to send Ctrl+Break through
            self._kill()

    def _kill(self):
        self.proc.kill()
        self._killed = True
        self._output.put(
            "--- The script didn't stop in time and was force-killed. "
            "The robot may still be moving! ---\n"
        )

    def poll(self):
        """Returns new log text. Call regularly while active."""
        if self.proc is None:
            return []
        if (
            self.stopping
            and not self._killed
            and self.proc.poll() is None
            and time.monotonic() > self._stop_deadline
        ):
            self._kill()

        texts = []
        while True:
            try:
                texts.append(self._output.get_nowait())
            except queue.Empty:
                break

        # Only finished once the reader has passed on everything the script printed
        if (
            self.proc.poll() is not None
            and not self._reader.is_alive()
            and self._output.empty()
        ):
            texts.append(f"--- Script exited (code {self.proc.returncode}) ---\n")
            self.proc = None
        return texts


# --- The window ---


class App:
    def __init__(self, root):
        self.root = root
        self.runner = ScriptRunner(SCRIPT_PATH)
        self.vars = {}  # Field name -> list of StringVars, one per input box
        self.loaded_source = None  # File contents as last loaded / saved
        self.loaded_texts = None  # Form text matching loaded_source
        self.closing = False
        self.frame_image = None  # Keeps the shown frame alive; Tk drops it otherwise
        try:
            self.video = video_link.FrameReceiver(VIDEO_SIZE)
        except OSError:
            # The script then shows the video in its own OpenCV window instead
            self.video = None

        self._build()
        self.reload(confirm=False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    def _build(self):
        self.root.title("UGOT Midterm Competition")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # Settings and buttons on the left, camera on the right, log underneath
        settings = ttk.Frame(main)
        settings.grid(row=0, column=0, sticky="nsew")
        settings.columnconfigure(0, weight=1)
        self._build_video(main)
        self.log = ScrolledText(main, height=10, state="disabled")
        self.log.grid(row=1, column=0, columnspan=2, sticky="nsew")

        row = 0
        for title, fields in SECTIONS:
            box = ttk.LabelFrame(settings, text=title, padding=(10, 5))
            box.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            row += 1
            for i, field in enumerate(fields):
                ttk.Label(box, text=field.label).grid(
                    row=i, column=0, sticky="w", padx=(0, 10), pady=2
                )
                self.vars[field.name] = self._add_inputs(box, i, field)
                ttk.Label(box, text=field.hint, foreground="gray").grid(
                    row=i, column=2, sticky="w", padx=(10, 0)
                )

        buttons = ttk.Frame(settings)
        buttons.grid(row=row, column=0, sticky="w", pady=(0, 8))
        self.reload_button = ttk.Button(
            buttons, text="Reload from file", command=self.reload
        )
        self.save_button = ttk.Button(buttons, text="Save", command=self.save)
        self.run_button = ttk.Button(
            buttons, text="Save & Run", command=self.save_and_run
        )
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop)
        for button in (
            self.reload_button,
            self.save_button,
            self.run_button,
            self.stop_button,
        ):
            button.pack(side="left", padx=(0, 6))

    def _build_video(self, parent):
        box = ttk.LabelFrame(parent, text="Camera", padding=5)
        box.grid(row=0, column=1, sticky="n", padx=(10, 0), pady=(0, 8))
        # Fixed size, so the layout doesn't jump when frames start or stop
        screen = tk.Frame(box, width=VIDEO_SIZE[0], height=VIDEO_SIZE[1], bg="black")
        screen.pack()
        screen.pack_propagate(False)
        self.video_label = tk.Label(screen, bg="black", fg="white", bd=0)
        self.video_label.pack(expand=True)
        if self.video is None:
            self.video_label["text"] = (
                "The video can't be shown here.\n"
                "It will open in its own window instead."
            )

    def _add_inputs(self, parent, row, field):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="w")
        if field.kind == "pose":
            variables = []
            for joint in range(3):
                ttk.Label(frame, text=f"J{joint + 1}").pack(
                    side="left", padx=(0 if joint == 0 else 8, 2)
                )
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=5).pack(side="left")
                variables.append(var)
            return variables
        var = tk.StringVar()
        width = 18 if field.kind in ("ip", "text") else 8
        ttk.Entry(frame, textvariable=var, width=width).pack(side="left")
        return [var]

    # --- Form contents ---

    def _form_texts(self):
        return {
            name: tuple(var.get() for var in variables)
            for name, variables in self.vars.items()
        }

    def _show_values(self, values):
        texts = {}
        for field in FIELDS:
            texts[field.name] = value_to_texts(field.kind, values[field.name])
            for var, text in zip(self.vars[field.name], texts[field.name]):
                var.set(text)
        self.loaded_texts = texts

    def _is_dirty(self):
        return self.loaded_texts is not None and self._form_texts() != self.loaded_texts

    def _read_form(self):
        """Returns ({name: value}, [problems])."""
        values = {}
        problems = []
        texts = self._form_texts()
        for field in FIELDS:
            try:
                values[field.name] = parse_value(field.kind, texts[field.name])
            except ValueError as error:
                problems.append(f"{field.label}: {error}")
        return values, problems

    # --- Buttons ---

    def reload(self, confirm=True):
        if (
            confirm
            and self._is_dirty()
            and not messagebox.askyesno(
                "Unsaved changes", "Discard your changes and reload from the file?"
            )
        ):
            return
        try:
            source = SCRIPT_PATH.read_bytes()
            values = read_constants(source)
        except (OSError, SyntaxError, ConstantsError) as error:
            self.loaded_source = None
            messagebox.showerror(
                "Can't read the settings",
                f"Couldn't read the constants from {SCRIPT_PATH.name}:\n\n{error}\n\n"
                "Fix the file, then press Reload from file.",
            )
        else:
            self.loaded_source = source
            self._show_values(values)
        self._update_buttons()

    def save(self):
        """Writes the form to the script. Returns True if it's saved."""
        values, problems = self._read_form()
        if problems:
            messagebox.showerror(
                "Can't save", "Please fix these first:\n\n" + "\n".join(problems)
            )
            return False
        try:
            current = SCRIPT_PATH.read_bytes()
            if current != self.loaded_source and not messagebox.askyesno(
                "File changed",
                f"{SCRIPT_PATH.name} was changed outside this window since it was "
                "loaded.\n\nOverwrite its settings with the ones in this window? "
                "(Choose No, then Reload from file, to see the file's settings.)",
                icon="warning",
            ):
                return False
            updated = update_constants(current, values)
            if updated != current:
                SCRIPT_PATH.write_bytes(updated)
        except (OSError, SyntaxError, ConstantsError) as error:
            messagebox.showerror("Can't save", str(error))
            return False
        self.loaded_source = updated
        self._show_values(values)
        return True

    def save_and_run(self):
        if self.runner.active or not self.save():
            return
        try:
            if self.video is None:
                self.runner.start()
            else:
                self.video.take_image()  # Drop the last frame of the previous run
                self.runner.start(self.video.env())
        except OSError as error:
            messagebox.showerror("Can't run", f"Couldn't start the script:\n\n{error}")
            return
        self._log(f"--- Running {SCRIPT_PATH.name} ---\n")
        self._update_buttons()

    def stop(self):
        self._log("--- Stopping... ---\n")
        self.runner.request_stop()
        self._update_buttons()

    def on_close(self):
        if self._is_dirty() and not messagebox.askyesno(
            "Unsaved changes", "Close without saving your changes?"
        ):
            return
        if self.runner.active:
            if not messagebox.askyesno(
                "Script running",
                "The robot script is still running. Stop it and close?",
            ):
                return
            # _poll() closes the window once the script has exited
            self.closing = True
            self.stop()
            return
        self.root.destroy()

    # --- Updating the window ---

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _update_buttons(self):
        loaded = self.loaded_source is not None
        active = self.runner.active
        stoppable = active and not self.runner.stopping
        self.save_button["state"] = "normal" if loaded else "disabled"
        self.run_button["state"] = "normal" if loaded and not active else "disabled"
        self.stop_button["state"] = "normal" if stoppable else "disabled"

    def _update_video(self):
        if self.video is None:
            return
        image = self.video.take_image()
        if not self.runner.active:
            # Black between runs, including late frames from a run that just ended
            if self.frame_image is not None:
                self.video_label["image"] = ""
                self.frame_image = None
        elif image is not None:
            # base64 text is the form of PNG data every Tk version accepts
            self.frame_image = tk.PhotoImage(
                data=base64.b64encode(image).decode("ascii"), format="png"
            )
            self.video_label["image"] = self.frame_image

    def _poll(self):
        for text in self.runner.poll():
            self._log(text)
        if self.closing and not self.runner.active:
            self.root.destroy()
            return
        self._update_video()
        self._update_buttons()
        self.root.after(POLL_MS, self._poll)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
