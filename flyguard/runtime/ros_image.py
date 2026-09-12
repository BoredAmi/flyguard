"""Decoding `sensor_msgs/Image` without cv_bridge.

Deliberately not an OpenCV dependency. The point of `flyguard.runtime` is
that it runs anywhere numpy, scipy and pandas do -- a companion computer, a
Raspberry Pi, a container with no graphics stack -- and pulling in cv_bridge
would mean pulling in OpenCV and a matching ROS2 build for the sake of about
fifteen lines of reshaping.

Nothing here imports ROS2 either, so the decoder is testable (and reusable)
without a ROS2 installation: it duck-types the message, needing only
`.encoding`, `.height`, `.width`, `.step` and `.data`.
"""

from __future__ import annotations

import numpy as np

#: Encodings this decoder handles. Anything else should be converted upstream
#: by `image_proc` rather than guessed at here.
SUPPORTED_ENCODINGS = ("rgb8", "bgr8", "mono8", "rgba8", "bgra8")

_CHANNELS = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}


def image_to_array(msg) -> np.ndarray:
    """`sensor_msgs/Image` -> HxW (mono) or HxWx3 (colour) uint8 array.

    Honours `step`, so a padded row stride does not silently shear the image
    -- a classic source of "the flow estimate is diagonal for no reason", and
    one that looks like a broken detector rather than a broken decode.

    Channel *order* is not corrected, because the detector converts to
    grayscale immediately and an unweighted channel mean is invariant to it.
    bgr8 therefore needs no swap.
    """
    if msg.encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; this decoder handles "
            f"{SUPPORTED_ENCODINGS}. Convert upstream with image_proc, or "
            f"republish as mono8."
        )
    channels = _CHANNELS[msg.encoding]
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    expected = msg.step * msg.height
    if buf.size < expected:
        raise ValueError(
            f"image buffer is {buf.size} bytes but height*step is {expected}; "
            f"the message is truncated or `step` is wrong"
        )
    if msg.step < msg.width * channels:
        raise ValueError(
            f"step {msg.step} is smaller than width*channels "
            f"({msg.width}*{channels}); the header does not describe the data"
        )
    rows = buf[:expected].reshape(msg.height, msg.step)
    img = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    return img[:, :, 0] if channels == 1 else img[:, :, :3]
