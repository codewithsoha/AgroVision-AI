"""Tests the real pipeline with small models built only for testing.

Their predictions are meaningless for plants. These tests only prove that
saving, loading, prediction, and Grad-CAM work with the planned model structure.
"""

import json

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from src.gradcam import GradCamError, compute_heatmap, explain_prediction, render_heatmap
from src.model import ModelState, load_predictor
from src.preprocessing import preprocess_for_model

CLASS_NAMES = [f"Crop___Class_{i}" for i in range(5)]


def build_nested_model():
    """Same structure as the training guide: MobileNetV2 nested inside the model."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None, alpha=0.35
    )
    inputs = tf.keras.Input((224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    model_path = folder / "model.keras"
    classes_path = folder / "classes.json"
    build_nested_model().save(model_path)
    classes_path.write_text(json.dumps(CLASS_NAMES))
    return model_path, classes_path


@pytest.fixture
def image():
    return Image.effect_noise((224, 224), 80).convert("RGB")


def test_saved_model_loads_and_predicts(saved_model, image):
    result = load_predictor(*saved_model)
    assert result.state is ModelState.READY
    predictions = result.predictor.predict(image)
    assert len(predictions) == 3
    assert predictions[0].confidence >= predictions[1].confidence >= predictions[2].confidence


def test_class_count_mismatch_is_rejected(saved_model, tmp_path, image):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(CLASS_NAMES[:4]))
    result = load_predictor(saved_model[0], classes_path)
    assert result.state is ModelState.LOAD_FAILED
    assert "5 outputs" in result.message


def pattern_images(count, seed=0):
    """Dark 64 x 64 images holding a patch at a random position.

    Class 0 has a solid patch and class 1 a checkerboard patch, so only the
    patch can decide the class. Returns images, labels, and patch positions.
    """
    random = np.random.default_rng(seed)
    images = random.uniform(-1, -0.8, (count, 64, 64, 3)).astype("float32")
    labels = np.arange(count) % 2
    rows, columns = np.mgrid[0:20, 0:20]
    checkerboard = (((rows // 4 + columns // 4) % 2) * 2.0 - 1)[..., np.newaxis]
    positions = random.integers(0, 44, (count, 2))
    for image, label, (top, left) in zip(images, labels, positions):
        image[top:top + 20, left:left + 20] = 1.0 if label == 0 else checkerboard
    return images, labels, positions


def test_gradcam_points_at_the_patch_that_decides_the_class(tmp_path):
    """Train a small nested CNN, reload it as the app does, and check localization."""
    tf.keras.utils.set_random_seed(1)
    base_inputs = tf.keras.Input((64, 64, 3))
    x = base_inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    base = tf.keras.Model(base_inputs, x, name="small_base")

    inputs = tf.keras.Input((64, 64, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base(inputs))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2, activation="softmax")(x))
    model.compile(tf.keras.optimizers.Adam(2e-3), "sparse_categorical_crossentropy")
    images, labels, _ = pattern_images(300)
    model.fit(images, labels, epochs=15, batch_size=16, verbose=0)

    path = tmp_path / "trained.keras"
    model.save(path)
    model = tf.keras.models.load_model(path, compile=False)

    images, labels, positions = pattern_images(40, seed=2)
    correct = hits = 0
    for image, label, (top, left) in zip(images, labels, positions):
        batch = image[np.newaxis]
        if int(np.argmax(model(batch, training=False))) != label:
            continue
        correct += 1
        heatmap = compute_heatmap(model, batch, int(label))
        row, column = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        center_row, center_column = row * 8 + 4, column * 8 + 4  # 8 pixels per cell.
        hits += (top - 8 <= center_row <= top + 28) and (left - 8 <= center_column <= left + 28)
    assert correct >= 32, "The test model must learn before localization means anything."
    assert hits / correct >= 0.8


def test_nested_mobilenetv2_gives_heatmap_or_clear_error(saved_model, image):
    """Random weights may give no evidence, but must never fail in another way."""
    model = load_predictor(*saved_model).predictor.model
    batch = preprocess_for_model(image)
    try:
        result = explain_prediction(model, batch, image, class_index=0)
    except GradCamError as error:
        assert "no positive evidence" in str(error)
    else:
        assert result.overlay_image.size == image.size


def test_gradcam_on_flat_model(image):
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(5, activation="softmax")(x))
    try:
        heatmap = compute_heatmap(model, preprocess_for_model(image), 0)
    except GradCamError:
        pytest.skip("Random weights gave no positive evidence.")
    assert heatmap.shape == (222, 222)


def test_model_without_convolutions_raises_clear_error():
    inputs = tf.keras.Input((4,))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2)(inputs))
    with pytest.raises(GradCamError, match="No convolutional"):
        compute_heatmap(model, np.zeros((1, 4)), 0)


def test_render_highlights_hot_region(image):
    heatmap = np.zeros((7, 7))
    heatmap[0, 0] = 1
    result = render_heatmap(image, heatmap)
    corner = result.heatmap_image.getpixel((5, 5))
    far = result.heatmap_image.getpixel((200, 200))
    assert corner[0] > corner[2]  # Hot regions lean red.
    assert far[2] > far[0]  # Cold regions lean blue.
