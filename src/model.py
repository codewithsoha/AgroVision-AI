"""Model loading and prediction.

No predictions exist unless a trained model file is present. When it is
missing, load_predictor() reports that state and the app stays in demo mode.

TensorFlow is imported only when a model file exists, so demo mode starts fast
and works even if TensorFlow is not installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import IMAGE_SIZE, preprocess_for_model
from src.recommendations import parse_class_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("AGROVISION_MODELS_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = MODELS_DIR / "agrovision_mobilenetv2.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

DEMO_MODE_MESSAGE = (
    "Demo mode: Image upload is working, but the trained disease detection "
    "model has not been connected yet."
)


class ModelState(Enum):
    READY = "ready"
    MODEL_MISSING = "model_missing"
    CLASSES_MISSING = "classes_missing"
    TENSORFLOW_MISSING = "tensorflow_missing"
    LOAD_FAILED = "load_failed"


@dataclass(frozen=True)
class Prediction:
    class_index: int
    class_name: str
    crop: str
    disease: str
    is_healthy: bool
    confidence: float  # Softmax score between 0 and 1.


class Predictor:
    """Pairs a trained Keras model with its class names."""

    def __init__(self, model, class_names: list[str]):
        self.model = model
        self.class_names = class_names

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        """Return the top_k predictions for an RGB image, best first."""
        batch = preprocess_for_model(image)
        scores = np.asarray(self.model(batch, training=False))[0]
        best_indices = np.argsort(scores)[::-1][:top_k]

        predictions = []
        for index in best_indices:
            class_name = self.class_names[index]
            crop, disease, is_healthy = parse_class_name(class_name)
            predictions.append(
                Prediction(
                    class_index=int(index),
                    class_name=class_name,
                    crop=crop,
                    disease=disease,
                    is_healthy=is_healthy,
                    confidence=float(scores[index]),
                )
            )
        return predictions


@dataclass(frozen=True)
class LoadResult:
    state: ModelState
    message: str
    predictor: Predictor | None = None

    @property
    def is_ready(self) -> bool:
        return self.state is ModelState.READY


def load_class_names(path: Path) -> list[str]:
    """Read class names from the JSON list saved during training.

    A name's position in the list is the model output index it belongs to.
    Keeping this in a file written by training avoids hardcoding an index
    mapping that could silently attach the wrong disease to a prediction.
    """
    with open(path, encoding="utf-8") as file:
        class_names = json.load(file)

    is_string_list = isinstance(class_names, list) and all(
        isinstance(name, str) and name for name in class_names
    )
    if not is_string_list or not class_names:
        raise ValueError("class_names.json must be a non-empty JSON list of strings.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names.json contains duplicate class names.")
    return class_names


def _check_model_shapes(model, class_count: int) -> None:
    """Reject a model that does not fit the app's image size or class list."""
    expected_input = (None, *IMAGE_SIZE, 3)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"The model expects input shape {model.input_shape}, "
            f"but the app provides {expected_input}."
        )
    output_count = model.output_shape[-1]
    if output_count != class_count:
        raise ValueError(
            f"The model has {output_count} outputs, but class_names.json "
            f"lists {class_count} classes."
        )


def load_predictor(
    model_path: Path = MODEL_PATH, classes_path: Path = CLASS_NAMES_PATH
) -> LoadResult:
    """Try to load the trained model. This never raises for an expected problem."""
    if not model_path.exists():
        return LoadResult(ModelState.MODEL_MISSING, DEMO_MODE_MESSAGE)
    if not classes_path.exists():
        return LoadResult(
            ModelState.CLASSES_MISSING,
            f"The model file was found, but {classes_path.name} is missing. "
            "It is needed to turn model outputs into class names, so the app "
            "is staying in demo mode.",
        )

    try:
        import tensorflow as tf
    except ImportError:
        return LoadResult(
            ModelState.TENSORFLOW_MISSING,
            "The model file was found, but TensorFlow is not installed. "
            "Run: pip install -r requirements.txt",
        )

    try:
        class_names = load_class_names(classes_path)
        # compile=False: the optimizer and loss are only needed for training.
        model = tf.keras.models.load_model(model_path, compile=False)
        _check_model_shapes(model, len(class_names))
    except Exception as error:  # Keras raises many types for damaged or incompatible files.
        return LoadResult(
            ModelState.LOAD_FAILED,
            f"The model could not be loaded, so the app is staying in demo mode. "
            f"Details: {error}",
        )

    return LoadResult(
        ModelState.READY,
        f"Trained model connected ({len(class_names)} classes).",
        Predictor(model, class_names),
    )
