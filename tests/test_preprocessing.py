import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    MAX_FILE_BYTES,
    ImageValidationError,
    load_and_validate_image,
    preprocess_for_model,
)


def encode(image, image_format):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def leaf(mode="RGB", size=(300, 200)):
    color = (40, 140, 60) if mode == "RGB" else (40, 140, 60, 0)
    return Image.new(mode, size, color)


@pytest.mark.parametrize(
    "filename, image_format",
    [("a.jpg", "JPEG"), ("a.jpeg", "JPEG"), ("a.png", "PNG"), ("A.PNG", "PNG")],
)
def test_accepts_supported_images(filename, image_format):
    image = load_and_validate_image(encode(leaf(), image_format), filename)
    assert image.mode == "RGB"
    assert image.size == (300, 200)


@pytest.mark.parametrize(
    "data, filename, expected",
    [
        (b"text", "notes.txt", "Unsupported"),
        (b"", "leaf.png", "empty"),
        (b"this is not an image", "leaf.png", "could not be read"),
        (b"x" * (MAX_FILE_BYTES + 1), "leaf.png", "10 MB"),
    ],
)
def test_rejects_invalid_files(data, filename, expected):
    with pytest.raises(ImageValidationError, match=expected):
        load_and_validate_image(data, filename)


def test_rejects_truncated_image():
    data = encode(Image.effect_noise((400, 400), 60).convert("RGB"), "JPEG")
    with pytest.raises(ImageValidationError):
        load_and_validate_image(data[: len(data) // 2], "leaf.jpg")


def test_rejects_other_format_with_allowed_extension():
    with pytest.raises(ImageValidationError, match="GIF"):
        load_and_validate_image(encode(leaf(), "GIF"), "leaf.png")


def test_rejects_tiny_image():
    with pytest.raises(ImageValidationError, match="too small"):
        load_and_validate_image(encode(leaf(size=(10, 10)), "PNG"), "leaf.png")


def test_transparency_is_placed_on_white():
    image = load_and_validate_image(encode(leaf("RGBA"), "PNG"), "leaf.png")
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_exif_rotation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # "Rotate 90 degrees" orientation tag.
    buffer = io.BytesIO()
    leaf(size=(300, 200)).save(buffer, format="JPEG", exif=exif)
    assert load_and_validate_image(buffer.getvalue(), "leaf.jpg").size == (200, 300)


def test_preprocessing_output_format():
    batch = preprocess_for_model(leaf())
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= -1 and batch.max() <= 1


def test_preprocessing_matches_keras_mobilenetv2():
    tf = pytest.importorskip("tensorflow")
    image = Image.effect_noise((224, 224), 80).convert("RGB")
    expected = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.asarray(image, dtype=np.float32)[np.newaxis]
    )
    np.testing.assert_allclose(preprocess_for_model(image), expected, atol=1e-5)
