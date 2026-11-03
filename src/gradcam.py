"""Grad-CAM: shows which image regions most influenced a prediction.

Steps:
1. Run the image through the model, keeping the last convolutional feature maps.
2. Compute the gradient of the predicted class score with respect to those maps.
3. Average the gradients for each map to get that map's importance weight.
4. Add up the maps using those weights and keep only the positive evidence.

With transfer learning, MobileNetV2 is usually one nested layer inside the
full model, so a single Keras graph cannot reach both its inner convolutional
layer and the classification head. The forward pass below is therefore done
in two explicit parts: the feature extractor, then the remaining layers.
This assumes the model is a simple chain of layers, as the training guide builds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

OVERLAY_ALPHA = 0.45


class GradCamError(RuntimeError):
    """Raised when a meaningful heatmap cannot be produced."""


@dataclass(frozen=True)
class GradCamResult:
    heatmap_image: Image.Image
    overlay_image: Image.Image


def _find_last_conv_layer(model):
    """Return (model containing the layer, layer) for the last 4D feature layer."""
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return _find_last_conv_layer(layer)
            except GradCamError:
                continue
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        output = getattr(layer, "output", None)
        if output is not None and len(output.shape) == 4:
            return model, layer
    raise GradCamError("No convolutional layer was found in this model.")


def compute_heatmap(model, batch: np.ndarray, class_index: int) -> np.ndarray:
    """Return a 2D Grad-CAM heatmap with values from 0 to 1."""
    import tensorflow as tf

    container, conv_layer = _find_last_conv_layer(model)
    feature_extractor = tf.keras.Model(
        container.inputs, [conv_layer.output, container.output]
    )

    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(batch)
        if container is model:  # Flat model: its output is already the prediction.
            features, x = feature_extractor(x, training=False)
        else:
            for layer in model.layers:
                if layer is container:
                    features, x = feature_extractor(x, training=False)
                elif not isinstance(layer, tf.keras.layers.InputLayer):
                    x = layer(x, training=False)
        class_score = x[:, class_index]

    gradients = tape.gradient(class_score, features)
    if gradients is None:
        raise GradCamError("Gradients could not be computed for this model.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))  # One weight per feature map.
    heatmap = tf.reduce_sum(features[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()  # Keep regions supporting the class.

    highest = float(heatmap.max())
    if not np.isfinite(highest) or highest <= 1e-8:
        # An all-zero map has no explanation to show, so none is drawn.
        raise GradCamError("The model produced no positive evidence for this class.")
    return heatmap / highest


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map values from 0-1 to a blue-to-red palette as uint8 RGB."""
    red = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def render_heatmap(image: Image.Image, heatmap: np.ndarray) -> GradCamResult:
    """Scale a heatmap to the image's size and blend it over the image."""
    image = image.convert("RGB")
    grayscale = Image.fromarray((heatmap * 255).astype(np.uint8))
    resized = grayscale.resize(image.size, Image.Resampling.BILINEAR)

    colored = _colorize(np.asarray(resized, dtype=np.float32) / 255)
    heatmap_image = Image.fromarray(colored)
    overlay_image = Image.blend(image, heatmap_image, OVERLAY_ALPHA)
    return GradCamResult(heatmap_image, overlay_image)


def explain_prediction(
    model, batch: np.ndarray, image: Image.Image, class_index: int
) -> GradCamResult:
    """Create the Grad-CAM heatmap and overlay for one predicted class."""
    return render_heatmap(image, compute_heatmap(model, batch, class_index))
