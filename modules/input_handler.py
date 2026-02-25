"""
PatentScout — Input Handler Module

Responsible for receiving and validating user inputs from the Streamlit UI,
including text descriptions and optional sketch image uploads. Runs initial
preprocessing before data is passed to the query builder and Gemini API.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

from PIL import Image

from models.schemas import InventionInput

# Maximum file size allowed for uploaded images (10 MB)
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# Minimum description length required by the UI
MIN_DESCRIPTION_LENGTH = 50

# Maximum dimension (width or height) for images sent to the Gemini API
MAX_IMAGE_DIMENSION = 1024


# ---------------------------------------------------------------------------
# Module-level helper functions (used directly by app.py)
# ---------------------------------------------------------------------------


def validate_input(
    text: str,
    image_bytes: Optional[bytes] = None,
) -> Tuple[bool, str]:
    """
    Validate the user-supplied invention description and optional image.

    Args:
        text: Free-text invention description entered in the UI.
        image_bytes: Raw bytes of the uploaded image file, or ``None``.

    Returns:
        A ``(is_valid, error_message)`` tuple.  ``is_valid`` is ``True`` when
        the input passes all checks; ``error_message`` is an empty string on
        success or a human-readable explanation on failure.
    """
    stripped = text.strip() if text else ""

    if not stripped:
        return False, "Please enter a description of your invention."

    if len(stripped) < MIN_DESCRIPTION_LENGTH:
        remaining = MIN_DESCRIPTION_LENGTH - len(stripped)
        return (
            False,
            f"Description is too short. Please add at least {remaining} more "
            f"character{'s' if remaining != 1 else ''} "
            f"({len(stripped)}/{MIN_DESCRIPTION_LENGTH} minimum).",
        )

    if image_bytes is not None:
        if len(image_bytes) > MAX_IMAGE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            return (
                False,
                f"Image is too large ({size_mb:.1f} MB). "
                "Please upload an image smaller than 10 MB.",
            )

    return True, ""


def resize_image(file_bytes: bytes) -> bytes:
    """
    Resize an image so that neither dimension exceeds ``MAX_IMAGE_DIMENSION``
    pixels, preserving aspect ratio.  Returns the (possibly unchanged) image
    as PNG bytes.

    Args:
        file_bytes: Raw bytes of the source image file.

    Returns:
        PNG-encoded bytes of the resized image.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_image(file_bytes: bytes) -> str:
    """
    Base64-encode image bytes for inclusion in a Gemini API request.

    The image is first resized (if necessary) via :func:`resize_image` to cap
    the longest edge at 1024 px and avoid API timeouts.

    Args:
        file_bytes: Raw bytes of the uploaded image file.

    Returns:
        A base64-encoded string suitable for passing to the Gemini vision API.
    """
    resized = resize_image(file_bytes)
    return base64.b64encode(resized).decode("utf-8")


# ---------------------------------------------------------------------------
# Class-based interface (used by downstream pipeline modules)
# ---------------------------------------------------------------------------


class InputHandler:
    """Validates and pre-processes raw user input for downstream modules."""

    def __init__(self):
        pass

    def process(
        self,
        description: str,
        sketch_path: Optional[str] = None,
    ) -> InventionInput:
        """
        Validate inputs and return a populated InventionInput dataclass.

        Args:
            description: Free-text invention description from the user.
            sketch_path: Optional path to an uploaded sketch image.

        Returns:
            InventionInput dataclass ready for feature extraction.
        """
        raise NotImplementedError
