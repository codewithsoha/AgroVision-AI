# Training Guide (Google Colab)

[AgroVision_Model_Training.ipynb](AgroVision_Model_Training.ipynb) trains the
model on Google Colab. Its outputs load into the app without code changes.

Training has not been run yet, so the notebook contains no results.

## 1. Dataset

No manual download is needed. The notebook loads the complete dataset through
[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)
(`plant_village`) and stops unless it finds 38 classes, 14 crops, and at least
54,000 images. Earlier ZIP files are ignored and can be deleted from Drive.

## 2. Google Drive

The notebook creates these folders by itself:

```text
MyDrive/
└── AgroVision/
    ├── outputs/              Model, class names, and reports
    └── tensorflow_datasets/  Copy of the dataset for later runs (about 1 GB)
```

## 3. Run the notebook

1. Open [Google Colab](https://colab.research.google.com), choose **File > Upload
   notebook**, and select `AgroVision_Model_Training.ipynb`.
2. Select **Runtime > Change runtime type**, choose **T4 GPU**, and save.
3. Select **Runtime > Run all**, and allow Google Drive access when asked.
4. Keep the tab open. Free Colab sessions can disconnect or run out of GPU quota.

If a session ends, the best checkpoints remain in `outputs/`, but **Run all** starts
training again from the beginning.

## 4. Connect the model

Copy these two files from `MyDrive/AgroVision/outputs/` into the project:

```text
models/agrovision_mobilenetv2.keras
models/class_names.json
```

Restart the app. The sidebar changes from **Demo mode** to **Model connected**.

## Rules that must not change

- Keep the image size at 224 x 224.
- Scale images once, in the data pipeline, and not inside the model. The app
  applies the same scaling.
- Keep the model as a simple chain, with MobileNetV2 as one nested layer.
  Grad-CAM relies on this structure.
- Keep `class_names.json` in training order. A wrong order gives wrong labels
  with no error.
- Report only metrics measured by your own run.
