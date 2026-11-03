"""Image validation and preprocessing.

This module deliberately does not import TensorFlow, so uploads can be
validated and prepared even when no model is connected.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# MobileNetV2 is trained on 224 x 224 images, so inference must use the same size.
IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}  # The real file type, read from the file's contents.
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PIXELS = 50_000_000  # Protects memory against extremely large images.
MIN_DIMENSION = 32  # Anything smaller has too little detail to analyze.


class ImageValidationError(ValueError):
    """Raised with a user-friendly message when an upload cannot be used."""


def load_and_validate_image(data: bytes, filename: str) -> Image.Image:
    """Check an uploaded file and return it as an upright RGB image.

    Checking the extension alone is not enough: a renamed text file can still
    be called "leaf.png". The file is therefore fully decoded here, which also
    catches corrupted or truncated images.
    """
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            "Please upload a JPG, JPEG, or PNG image."
        )
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError("The image is larger than the 10 MB limit.")

    invalid_message = (
        "This file could not be read as an image. It may be corrupted "
        "or may not be a real JPG or PNG file."
    )
    try:
        image = Image.open(io.BytesIO(data))  # Reads only the header at this point.
        if image.format not in ALLOWED_FORMATS:
            raise ImageValidationError(
                f"This file is actually a {image.format} image. "
                "Please upload a JPG, JPEG, or PNG image."
            )
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise ImageValidationError("The image resolution is too large to process.")
        if min(width, height) < MIN_DIMENSION:
            raise ImageValidationError(
                f"The image is too small ({width} x {height} pixels). "
                f"Each side must be at least {MIN_DIMENSION} pixels."
            )
        image.load()  # Decodes every pixel, which exposes damaged files.
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as error:
        raise ImageValidationError(invalid_message) from error

    # Phones often store rotation as metadata instead of rotating the pixels.
    image = ImageOps.exif_transpose(image)
    return _to_rgb(image)


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert any image mode to RGB, placing transparency on a white background."""
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def resize_for_model(image: Image.Image) -> Image.Image:
    """Resize an RGB image to the model's input size."""
    return image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)


def preprocess_for_model(image: Image.Image) -> np.ndarray:
    """Convert an RGB image to the batch expected by the model.

    Returns a float32 array of shape (1, 224, 224, 3) with values in [-1, 1].
    This matches tf.keras.applications.mobilenet_v2.preprocess_input, and the
    training pipeline must apply that same scaling exactly once.
    """
    pixels = np.asarray(resize_for_model(image), dtype=np.float32)
    scaled = pixels / 127.5 - 1.0
    return np.expand_dims(scaled, axis=0)  # The model expects a batch dimension.
