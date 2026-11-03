# AgroVision: AI-Based Crop Disease Detection

AgroVision is a Streamlit web application for identifying plant leaf diseases
from uploaded images, using MobileNetV2 transfer learning on the PlantVillage
dataset.

## Current status

**Phase 1 is implemented and the Phase 2 training notebook is ready, but the model
has not been trained yet.** The app therefore runs in demo mode: uploads are
validated and preprocessed, but no disease is predicted. No accuracy is reported
because no model has been evaluated.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Interface, upload, validation, preprocessing, demo mode, model interfaces | Implemented |
| 2 | Colab notebook and MobileNetV2 training | Notebook ready, training not yet run; see [the training guide](notebooks/README.md) |
| 3 | Verified predictions and Grad-CAM on the trained model, evaluation, deployment | Not started |

## Setup

Use Python 3.12. From the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project structure

```text
app.py                 Streamlit pages and application flow
src/preprocessing.py   Image validation, resizing, and scaling
src/model.py           Model loading and prediction
src/gradcam.py         Grad-CAM heatmaps and overlays
src/recommendations.py Editable treatment and prevention reference
src/ui.py              Reusable interface components
assets/style.css       Visual design
models/                Trained model and class names (not included yet)
notebooks/             Colab training notebook and guide
tests/                 Automated tests
```

## Connecting the trained model

Follow [the training guide](notebooks/README.md), then place these files in
`models/` and restart the app:

- `agrovision_mobilenetv2.keras`
- `class_names.json`

No interface changes are needed. Class names come from the file saved during
training, so the index-to-class mapping is never hardcoded.

## Deployment

For Streamlit Community Cloud, push this folder to its own GitHub repository and
select Python 3.12 under the app's advanced settings. Model files are ignored by
Git by default, so add the trained model deliberately:

```bash
git add -f models/agrovision_mobilenetv2.keras
```

## Disclaimer

Treatment information is general educational guidance, not professional
agricultural advice. Confirm any diagnosis and locally approved products with an
agricultural extension officer.
