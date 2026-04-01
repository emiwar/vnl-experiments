#!/usr/bin/env python3
"""Print raw gamepad axis and button values in a loop.

Reads directly from /dev/input/js0 (Linux joystick API) — no GLFW needed.

Usage:
    .venv/bin/python vnl_experiments/tools/gamepad_test.py [--device /dev/input/js0]

Press Ctrl+C to exit.
"""

import argparse
import fcntl
import os
import struct
import time

# Linux joystick event: struct js_event { u32 time; s16 value; u8 type; u8 number; }
_JS_FMT          = "IhBB"
_JS_EVENT_SIZE   = struct.calcsize(_JS_FMT)
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS   = 0x02
_JS_EVENT_INIT   = 0x80


def read_joystick(path: str):
    """Yield (axes: list[float], buttons: list[int]) snapshots from a joystick device."""
    axes:    dict[int, float] = {}
    buttons: dict[int, int]   = {}

    fd = open(path, "rb")
    fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)

    while True:
        # Drain all pending events
        while True:
            try:
                data = fd.read(_JS_EVENT_SIZE)
            except BlockingIOError:
                break
            if not data or len(data) < _JS_EVENT_SIZE:
                break
            _, value, etype, number = struct.unpack(_JS_FMT, data)
            etype &= ~_JS_EVENT_INIT
            if etype == _JS_EVENT_AXIS:
                axes[number] = value / 32767.0
            elif etype == _JS_EVENT_BUTTON:
                buttons[number] = value

        yield axes, buttons


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/input/js0")
args = parser.parse_args()

print(f"Reading from {args.device}. Press Ctrl+C to stop.\n")

try:
    for axes, buttons in read_joystick(args.device):
        n_axes = max(axes.keys(), default=-1) + 1
        n_btns = max(buttons.keys(), default=-1) + 1
        ax_str = "  ".join(f"{i}:{axes.get(i, 0.0):+.3f}" for i in range(n_axes))
        bt_str = "  ".join(f"{i}:{buttons.get(i, 0)}" for i in range(n_btns))
        line = f"axes [{ax_str}]  |  btns [{bt_str}]"
        print(f"\r{line:<140}", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print()
