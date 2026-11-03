"""Checks the Colab training notebook without training anything.

Cells tagged "pipeline" are real notebook code. They run here on a few tiny
generated images, and model.fit() is never called.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.gradcam import compute_heatmap
from src.model import CLASS_NAMES_PATH, MODEL_PATH, ModelState, load_predictor
from src.preprocessing import IMAGE_SIZE, load_and_validate_image, preprocess_for_model
from src.recommendations import ALL_CLASSES

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "AgroVision_Model_Training.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
CODE_CELLS = [cell for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]

# The label list in the TFDS plant_village builder (version 1.0.2).
TFDS_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy",
    "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


def source(cell):
    return "".join(cell["source"])


def python_source(cell):
    """Replace Colab shell lines, which are not Python, keeping the indentation."""
    lines = []
    for line in source(cell).splitlines():
        stripped = line.lstrip()
        is_shell = stripped.startswith(("!", "%"))
        lines.append(line[: len(line) - len(stripped)] + "pass" if is_shell else line)
    return "\n".join(lines)


def write_leaf(path, seed):
    random = np.random.default_rng(seed)
    pixels = Image.fromarray(random.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    pixels.resize((256, 256), Image.Resampling.BICUBIC).save(path, quality=90)


@pytest.fixture(scope="module")
def notebook():
    pytest.importorskip("tensorflow")
    namespace = {}
    for cell in CODE_CELLS:
        if "pipeline" in cell["metadata"].get("tags", []):
            exec(python_source(cell), namespace)
    return namespace


def test_every_code_cell_is_valid_python():
    for cell in CODE_CELLS:
        compile(python_source(cell), cell["id"], "exec")


def test_notebook_has_no_stored_results():
    assert all(not cell["outputs"] and cell["execution_count"] is None for cell in CODE_CELLS)


def test_notebook_targets_colab_t4():
    assert NOTEBOOK["metadata"]["accelerator"] == "GPU"
    assert NOTEBOOK["metadata"]["colab"]["gpuType"] == "T4"


def test_settings_match_the_app(notebook):
    assert tuple(notebook["IMAGE_SIZE"]) == IMAGE_SIZE
    assert notebook["MODEL_FILENAME"] == MODEL_PATH.name
    assert notebook["CLASS_NAMES_FILENAME"] == CLASS_NAMES_PATH.name


def test_class_check_lists_every_app_class():
    text = "".join(source(cell) for cell in CODE_CELLS)
    assert all(f'"{name}"' in text for name in ALL_CLASSES)


def test_training_preprocessing_matches_the_app(notebook, tmp_path):
    keras = notebook["keras"]
    for seed in range(5):
        path = tmp_path / f"{seed}.jpg"
        write_leaf(path, seed)
        training = keras.applications.mobilenet_v2.preprocess_input(
            notebook["load_image"](str(path))
        ).numpy()
        app = preprocess_for_model(load_and_validate_image(path.read_bytes(), path.name))[0]
        assert training.shape == app.shape
        assert np.abs(training - app).mean() * 127.5 < 1.0
        np.testing.assert_allclose(notebook["app_preprocess"](path), app, atol=1e-5)


def test_png_inputs_are_supported(notebook, tmp_path):
    path = tmp_path / "leaf.png"
    write_leaf(path, 0)
    assert notebook["load_image"](str(path)).shape == (224, 224, 3)


def test_datasets_keep_values_in_range_and_order(notebook, tmp_path):
    paths = []
    for seed in range(4):
        paths.append(str(tmp_path / f"{seed}.jpg"))
        write_leaf(paths[-1], seed)
    labels = [0, 1, 0, 1]

    images, batch_labels = next(iter(notebook["make_dataset"](paths, labels)))
    assert images.shape == (4, 224, 224, 3)
    assert batch_labels.numpy().tolist() == labels
    np.testing.assert_allclose(
        images[2].numpy(), notebook["app_preprocess"](paths[2]), atol=0.1
    )

    augmented, _ = next(iter(notebook["make_dataset"](paths, labels, training=True)))
    assert augmented.shape == (4, 224, 224, 3)
    assert float(np.min(augmented)) >= -1 and float(np.max(augmented)) <= 1


def test_splits_keep_duplicates_together_and_classes_balanced(notebook):
    records = []
    for label in ("a", "b"):
        for number in range(100):
            group = f"{label}{number // 2}" if number < 20 else f"{label}{number}"
            records.append((f"{label}/{number}", label, group))
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    assignments = notebook["assign_splits"](records, ratios, seed=42)

    assert len(assignments) == len(records)
    splits_by_group = {}
    for path, _, group in records:
        splits_by_group.setdefault(group, set()).add(assignments[path])
    assert all(len(splits) == 1 for splits in splits_by_group.values())

    for label in ("a", "b"):
        counts = [
            sum(1 for path, name, _ in records if name == label and assignments[path] == split)
            for split in ratios
        ]
        assert counts[0] == pytest.approx(70, abs=3)
        assert counts[1] == pytest.approx(15, abs=3)


def test_duplicate_images_get_the_same_fingerprint(notebook, tmp_path):
    first, second, other = (tmp_path / name for name in ("a.jpg", "b.png", "c.jpg"))
    write_leaf(first, 0)
    Image.open(first).save(second)
    write_leaf(other, 1)
    fingerprint = notebook["image_fingerprint"]
    assert fingerprint(first) == fingerprint(second)
    assert fingerprint(first) != fingerprint(other)
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    assert fingerprint(tmp_path / "bad.jpg") is None


def test_tfds_labels_translate_to_every_app_class(notebook):
    translated = {notebook["to_app_class_name"](name) for name in TFDS_LABELS}
    assert len(TFDS_LABELS) == 38
    assert translated == set(ALL_CLASSES)


def test_export_keeps_original_bytes_and_uses_app_names(notebook, tmp_path):
    image_path = tmp_path / "source.jpg"
    write_leaf(image_path, 0)
    data = image_path.read_bytes()
    output = tmp_path / "images"
    examples = [(data, TFDS_LABELS.index("Corn___Common_rust"), "a b.JPG")]

    assert notebook["export_examples"](examples, TFDS_LABELS, output) == 1
    assert (output / "Corn_(maize)___Common_rust_" / "a b.JPG").read_bytes() == data


@pytest.mark.parametrize(
    "class_count, image_count",
    [(15, 20_000), (15, 60_000), (37, 54_303), (38, 20_000)],
)
def test_partial_datasets_are_rejected(notebook, class_count, image_count):
    with pytest.raises(ValueError, match="Incomplete dataset"):
        notebook["check_dataset_is_complete"](list(ALL_CLASSES)[:class_count], image_count)


def test_complete_dataset_is_accepted(notebook):
    notebook["check_dataset_is_complete"](list(ALL_CLASSES), 54_303)


def test_notebook_uses_tfds_and_not_a_zip():
    text = "".join(source(cell) for cell in CODE_CELLS)
    assert 'TFDS_NAME = "plant_village"' in text
    assert "zipfile" not in text and "DATASET_ZIP" not in text


def test_class_weights_favor_rare_classes(notebook):
    weights = notebook["make_class_weights"]([0] * 90 + [1] * 10, 2)
    assert weights[1] > 1 > weights[0]


def test_notebook_model_works_with_the_app(notebook, tmp_path):
    """Builds the notebook's model (untrained), saves it, and loads it the app's way."""
    keras = notebook["keras"]
    model = notebook["build_model"](len(ALL_CLASSES), weights=None)
    assert not notebook["get_base"](model).trainable

    notebook["set_fine_tuning"](model, 30)
    base = notebook["get_base"](model)
    assert not base.layers[0].trainable
    unfrozen = [layer for layer in base.layers if layer.trainable]
    assert unfrozen and not any(
        isinstance(layer, keras.layers.BatchNormalization) for layer in unfrozen
    )
    notebook["compile_model"](model, 1e-5)
    assert notebook["training_callbacks"](tmp_path / "best.keras")

    model_path = tmp_path / MODEL_PATH.name
    classes_path = tmp_path / CLASS_NAMES_PATH.name
    model.save(model_path, include_optimizer=False)
    classes_path.write_text(json.dumps(sorted(ALL_CLASSES)))

    result = load_predictor(model_path, classes_path)
    assert result.state is ModelState.READY
    notebook["check_saved_model"](result.predictor.model, sorted(ALL_CLASSES))
    assert len(result.predictor.predict(Image.new("RGB", (300, 300)))) == 3

    batch = np.random.default_rng(0).uniform(-1, 1, (1, 224, 224, 3)).astype("float32")
    try:
        assert compute_heatmap(result.predictor.model, batch, 0).shape == (7, 7)
    except Exception as error:  # Untrained weights may give no evidence, but nothing else.
        assert "no positive evidence" in str(error)
