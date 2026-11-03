import json

import numpy as np
import pytest
from PIL import Image

from src.model import (
    DEMO_MODE_MESSAGE,
    ModelState,
    Predictor,
    load_class_names,
    load_predictor,
)

CLASS_NAMES = ["Apple___healthy", "Tomato___Late_blight", "Potato___Early_blight"]


class FixedScoreModel:
    """A stand-in with known outputs, used only to test how results are ranked."""

    def __call__(self, batch, training=False):
        return np.array([[0.1, 0.7, 0.2]])


def test_missing_model_gives_demo_mode(tmp_path):
    result = load_predictor(tmp_path / "none.keras", tmp_path / "none.json")
    assert result.state is ModelState.MODEL_MISSING
    assert result.message == DEMO_MODE_MESSAGE
    assert result.predictor is None


def test_missing_class_file_is_reported(tmp_path):
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    result = load_predictor(model_path, tmp_path / "none.json")
    assert result.state is ModelState.CLASSES_MISSING
    assert result.predictor is None


def test_corrupted_model_does_not_crash(tmp_path):
    model_path = tmp_path / "model.keras"
    classes_path = tmp_path / "classes.json"
    model_path.write_bytes(b"not a model")
    classes_path.write_text(json.dumps(CLASS_NAMES))
    result = load_predictor(model_path, classes_path)
    assert result.state is ModelState.LOAD_FAILED
    assert result.predictor is None


@pytest.mark.parametrize("content", ['{"a": 1}', "[]", '["a", "a"]', "[1, 2]"])
def test_invalid_class_names_are_rejected(tmp_path, content):
    path = tmp_path / "classes.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_class_names(path)


def test_predictions_are_ranked_with_correct_labels():
    predictor = Predictor(FixedScoreModel(), CLASS_NAMES)
    predictions = predictor.predict(Image.new("RGB", (300, 300)))
    assert [p.class_index for p in predictions] == [1, 2, 0]
    assert predictions[0].crop == "Tomato"
    assert predictions[0].disease == "Late blight"
    assert predictions[0].confidence == pytest.approx(0.7)
    assert predictions[2].is_healthy
