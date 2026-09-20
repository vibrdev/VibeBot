"""Screenshot downscaling.

Vision models charge by image area, and a browser screenshot is mostly empty
space. Measured on qwen3.5:4b, same eBay page, only the width changed:

    1440x900 -> 1280 prompt tokens, 20.9s
    1024x640 ->  660 prompt tokens, 17.2s
     768x480 ->  380 prompt tokens, 16.4s
     512x320 ->  180 prompt tokens, 16.5s

At the full 1440 the picture alone costs more than the entire llm.max_tokens
reply budget, which is how a reply gets cut off before it is valid JSON.

What this does *not* fix is memory: Ollama committed ~5.7 GB at every size
above, because that is the model, not the image. Measured, not assumed.
"""

from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

_warned = False


def downscale_png(png: bytes | None, max_width: int) -> bytes | None:
    """Shrink a PNG to `max_width` if it is wider. Returns the original on any
    problem — a slightly expensive screenshot beats no screenshot at all."""
    if not png or max_width <= 0:
        return png
    try:
        from PIL import Image  # noqa: PLC0415 - optional, and only on this path
    except ImportError:
        global _warned  # noqa: PLW0603 - once per process, not per step
        if not _warned:
            _warned = True
            log.warning(
                "Pillow is not installed, so screenshots go to the model at full size "
                "(roughly 3x the image tokens). Run: pip install Pillow"
            )
        return png

    try:
        with Image.open(io.BytesIO(png)) as image:
            if image.width <= max_width:
                return png
            height = max(1, round(image.height * max_width / image.width))
            resized = image.convert("RGB").resize((max_width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - never lose a step over a resize
        log.warning("screenshot downscale failed, sending it full size: %s", exc)
        return png
