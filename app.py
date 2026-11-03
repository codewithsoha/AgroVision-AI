"""AgroVision: Streamlit interface and application flow.

Run with: streamlit run app.py
"""

from __future__ import annotations

import hashlib

import streamlit as st
from PIL import Image

from src import ui
from src.gradcam import GradCamError, explain_prediction
from src.model import (
    CLASS_NAMES_PATH,
    DEMO_MODE_MESSAGE,
    MODEL_PATH,
    LoadResult,
    load_predictor,
)
from src.preprocessing import (
    ImageValidationError,
    load_and_validate_image,
    preprocess_for_model,
    resize_for_model,
)
from src.recommendations import (
    ALL_CLASSES,
    get_recommendation,
    parse_class_name,
)

PAGE_HOME = "Home"
PAGE_DETECT = "Detect Disease"
PAGE_LIBRARY = "Disease Library"
PAGE_ABOUT = "About Project"

# Below this score the top prediction is presented as uncertain.
LOW_CONFIDENCE_THRESHOLD = 0.5


@st.cache_resource(show_spinner="Loading the trained model...")
def load_cached_predictor(model_modified: float, classes_modified: float) -> LoadResult:
    """Load the model once. A changed file has a new timestamp and reloads it."""
    return load_predictor()


def get_load_result() -> LoadResult:
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        # Not cached, so adding the files is noticed on the next interaction.
        return load_predictor()
    return load_cached_predictor(
        MODEL_PATH.stat().st_mtime, CLASS_NAMES_PATH.stat().st_mtime
    )


def go_to_detect() -> None:
    st.session_state.page = PAGE_DETECT


def clear_upload() -> None:
    st.session_state.uploader_version += 1  # A new widget key empties the uploader.
    st.session_state.analysis = None


# ---------------------------------------------------------------- Home


def render_home(load_result: LoadResult) -> None:
    ui.hero(
        "AI-Powered Crop Disease Detection",
        "Plant diseases are easiest to manage when they are caught early. AgroVision "
        "uses deep learning to identify likely diseases from a photo of a leaf.",
        ["MobileNetV2", "TensorFlow", "PlantVillage", "Streamlit"],
    )
    st.write("")
    left, right = st.columns([3, 2], gap="medium", vertical_alignment="center")
    with left:
        ui.text_card(
            "About this project",
            "AgroVision is a web application built around a transfer-learning model. "
            "It is designed to show the most likely diseases, explain which leaf regions "
            "influenced the result, and offer treatment and prevention guidance.",
        )
    with right:
        st.button(
            "Upload a leaf image",
            type="primary",
            width="stretch",
            on_click=go_to_detect,
        )
        if not load_result.is_ready:
            st.caption("No trained model is connected yet, so the app runs in demo mode.")

    st.write("")
    st.subheader("Dataset at a glance")
    ui.stat_grid([
        ("~54,303", "PlantVillage leaf images"),
        ("38", "Disease and healthy classes"),
        ("14", "Crop species"),
        ("224 × 224", "Model input size"),
    ])
    st.write("")
    st.subheader("How it works")
    ui.step_grid([
        ("Upload", "Provide a clear JPG or PNG photo of a single leaf."),
        ("Analyze", "The image is validated and prepared for MobileNetV2."),
        ("Predict", "The model ranks the three most likely classes."),
        ("Explain and act", "Grad-CAM shows the evidence, followed by guidance."),
    ])


# ---------------------------------------------------------------- Detect


def analyze(image: Image.Image, digest: str, load_result: LoadResult) -> dict:
    """Run the pipeline. Predictions are only ever produced by a loaded model."""
    batch = preprocess_for_model(image)
    analysis = {
        "digest": digest,
        "tensor_shape": batch.shape,
        "value_range": (float(batch.min()), float(batch.max())),
        "predictions": None,
        "gradcam": None,
        "gradcam_error": None,
    }
    if load_result.is_ready:
        predictor = load_result.predictor
        predictions = predictor.predict(image)
        analysis["predictions"] = predictions
        try:
            analysis["gradcam"] = explain_prediction(
                predictor.model,
                batch,
                resize_for_model(image),
                predictions[0].class_index,
            )
        except GradCamError as error:
            analysis["gradcam_error"] = str(error)
        except Exception as error:  # A failed explanation must not lose the prediction.
            analysis["gradcam_error"] = f"Grad-CAM failed unexpectedly: {error}"
    return analysis


def render_detect(load_result: LoadResult) -> None:
    ui.page_heading(
        "Leaf Image Analysis",
        "Upload a clear photo of one leaf in JPG, JPEG, or PNG format, up to 10 MB.",
    )
    if not load_result.is_ready:
        ui.banner(
            load_result.message,
            "demo" if load_result.message == DEMO_MODE_MESSAGE else "error",
        )
        st.write("")

    upload_column, preview_column = st.columns(2, gap="large")
    with upload_column:
        upload = st.file_uploader(
            "Leaf image",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploader_version}",
        )
    if upload is None:
        st.session_state.analysis = None
        with preview_column:
            ui.image_placeholder("Your image preview will appear here.")
        return

    data = upload.getvalue()
    try:
        image = load_and_validate_image(data, upload.name)
    except ImageValidationError as error:
        st.session_state.analysis = None
        with preview_column:
            ui.banner(str(error), "error")
            st.button("Remove file", on_click=clear_upload)
        return

    digest = hashlib.sha256(data).hexdigest()
    analysis = st.session_state.analysis
    if analysis is not None and analysis["digest"] != digest:
        analysis = st.session_state.analysis = None  # Results belong to the old image.

    with preview_column:
        st.image(image, width="stretch")
        st.caption(f"{upload.name} · {image.width} × {image.height} px · {len(data) / 1024:.0f} KB")

    with upload_column:
        analyze_column, remove_column = st.columns(2)
        run_analysis = analyze_column.button("Analyze leaf", type="primary", width="stretch")
        remove_column.button("Remove image", width="stretch", on_click=clear_upload)

    if run_analysis:
        with st.spinner("Analyzing the leaf image..."):
            analysis = st.session_state.analysis = analyze(image, digest, load_result)

    if analysis is None:
        return
    st.divider()
    if analysis["predictions"] is None:
        render_demo_results(analysis, image)
    else:
        render_model_results(analysis, image)


def render_demo_results(analysis: dict, image: Image.Image) -> None:
    st.subheader("Analysis status")
    low, high = analysis["value_range"]
    steps_column, details_column = st.columns(2, gap="medium")
    with steps_column:
        ui.progress_card(
            "Pipeline progress",
            [
                ("Image uploaded", True),
                ("Image validated", True),
                ("Image preprocessed for MobileNetV2", True),
                ("Disease prediction", False),
                ("Grad-CAM explanation", False),
                ("Treatment guidance", False),
            ],
        )
    with details_column:
        ui.text_card(
            "Prepared model input",
            f"Your image was converted into a tensor of shape {analysis['tensor_shape']} "
            f"with values from {low:.2f} to {high:.2f}, which is the format MobileNetV2 expects.",
            "No disease has been identified. Predictions will become available after "
            "the model is trained and integrated.",
        )

    st.write("")
    st.subheader("Treatment recommendations")
    ui.banner(
        "No recommendation is shown because no disease has been detected. "
        "Guidance will be matched to the prediction once the model is connected.",
    )

    st.write("")
    st.subheader("Grad-CAM explainability")
    ui.banner(
        "Grad-CAM needs a trained model's gradients, so no heatmap is drawn in demo mode.",
    )
    st.write("")
    original, heatmap, overlay = st.columns(3)
    with original:
        st.image(resize_for_model(image), caption="Original leaf", width="stretch")
    with heatmap:
        ui.image_placeholder("Grad-CAM heatmap: available after model integration")
        st.caption("Heatmap")
    with overlay:
        ui.image_placeholder("Heatmap overlay: available after model integration")
        st.caption("Overlay")


def render_model_results(analysis: dict, image: Image.Image) -> None:
    predictions = analysis["predictions"]
    top = predictions[0]

    st.subheader("Detection results")
    if top.confidence < LOW_CONFIDENCE_THRESHOLD:
        ui.banner(
            f"Low confidence ({top.confidence:.1%}). This prediction is uncertain. "
            "Try a clearer photo of a single leaf, or seek expert advice.",
            "demo",
        )
        st.write("")

    result_column, chart_column = st.columns(2, gap="medium")
    with result_column:
        ui.prediction_card(top)
    with chart_column:
        ui.confidence_chart(predictions)
    st.write("")
    ui.text_card(
        "What this result means",
        f"Among the classes it was trained on, the model scored {top.crop}: {top.disease} "
        f"highest at {top.confidence:.1%}. Confidence is the model's own score, not a "
        "guarantee of accuracy. The model cannot recognize crops or diseases outside "
        "its training classes.",
    )

    st.write("")
    st.subheader("Treatment recommendations")
    recommendation = get_recommendation(top.class_name)
    if recommendation is None:
        ui.banner(f"No guidance is available yet for {top.class_name}.")
    else:
        ui.recommendation_card("Suggested guidance", recommendation)

    st.write("")
    st.subheader("Grad-CAM explainability")
    gradcam = analysis["gradcam"]
    if gradcam is None:
        ui.banner(f"No heatmap is available: {analysis['gradcam_error']}", "error")
        return
    original, heatmap, overlay = st.columns(3)
    original.image(resize_for_model(image), caption="Original leaf", width="stretch")
    heatmap.image(gradcam.heatmap_image, caption="Grad-CAM heatmap", width="stretch")
    overlay.image(gradcam.overlay_image, caption="Overlay", width="stretch")
    st.caption(
        "Warmer colors mark the regions that most increased the score of the predicted class."
    )


# ---------------------------------------------------------------- Library


def render_library() -> None:
    ui.page_heading(
        "Disease Library",
        "Browse the guidance prepared for each class in the dataset.",
    )
    ui.banner("Reference information only. Browsing this library is not a diagnosis.")
    st.write("")

    by_crop: dict[str, dict[str, str]] = {}
    for class_name in ALL_CLASSES:
        crop, condition, _ = parse_class_name(class_name)
        by_crop.setdefault(crop, {})[condition] = class_name

    crop_column, condition_column = st.columns(2)
    crop = crop_column.selectbox("Crop", sorted(by_crop))
    condition = condition_column.selectbox("Condition", sorted(by_crop[crop]))
    ui.recommendation_card(f"{crop}: {condition}", get_recommendation(by_crop[crop][condition]))


# ---------------------------------------------------------------- About


def render_about() -> None:
    ui.page_heading("About Project", "The problem, the approach, and the current status.")

    ui.text_card(
        "Problem statement",
        "Crop diseases reduce yield and income, and expert diagnosis is not always available "
        "when it is needed. Late identification can lead to avoidable losses or unnecessary "
        "chemical use. AgroVision explores whether a lightweight deep learning model can "
        "give fast, explainable first-level guidance from a leaf photo.",
    )
    st.write("")
    left, right = st.columns(2, gap="medium")
    with left:
        ui.list_card("Objectives", [
            "Classify leaf images across 38 PlantVillage classes.",
            "Show the top three predictions with confidence scores.",
            "Explain predictions visually with Grad-CAM.",
            "Offer disease-specific treatment and prevention guidance.",
        ])
    with right:
        ui.list_card("Dataset: PlantVillage", [
            "Approximately 54,305 leaf images.",
            "38 classes, including healthy leaves, across 14 crop species.",
            "Images are resized to 224 × 224 pixels.",
            "Limitation: many images have controlled backgrounds, so field photos may be harder.",
        ])
    st.write("")
    left, right = st.columns(2, gap="medium")
    with left:
        ui.list_card("Algorithm overview", [
            "MobileNetV2 pretrained on ImageNet extracts visual features.",
            "Its base is frozen while a new 38-class head is trained.",
            "Selected top layers can then be fine-tuned at a low learning rate.",
            "Grad-CAM uses gradients to highlight influential regions.",
        ])
    with right:
        ui.list_card("Technology stack", [
            "Python, TensorFlow/Keras, and MobileNetV2.",
            "Google Colab for model training.",
            "Streamlit for the web application.",
            "Streamlit Community Cloud as the deployment target.",
        ])
    st.write("")
    ui.list_card("Potential applications and impact", [
        "Mobile-assisted crop diagnosis",
        "Precision agriculture",
        "Targeted treatment that reduces unnecessary spraying",
        "Early-warning systems",
        "Support for agricultural extension services",
        "Crop-loss reduction",
    ])
    st.caption("These are potential future applications, not features that are already implemented.")
    st.write("")
    ui.list_card("Project status", [
        "Phase 1, working foundation: complete in this version.",
        "Phase 2, model training in Google Colab: notebook prepared, training not yet run.",
        "Phase 3, verified predictions, Grad-CAM, evaluation, and deployment: planned.",
        "Model accuracy is not reported because no model has been trained or evaluated yet.",
    ])


# ---------------------------------------------------------------- App


def main() -> None:
    st.set_page_config(
        page_title="AgroVision | Crop Disease Detection",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="auto",
    )
    ui.apply_styles()

    st.session_state.setdefault("page", PAGE_HOME)
    st.session_state.setdefault("uploader_version", 0)
    st.session_state.setdefault("analysis", None)

    load_result = get_load_result()

    with st.sidebar:
        ui.sidebar_brand()
        page = st.radio(
            "Navigation",
            [PAGE_HOME, PAGE_DETECT, PAGE_LIBRARY, PAGE_ABOUT],
            key="page",
            label_visibility="collapsed",
        )
        st.write("")
        ui.sidebar_model_status(load_result.is_ready, load_result.message)

    if page == PAGE_HOME:
        render_home(load_result)
    elif page == PAGE_DETECT:
        render_detect(load_result)
    elif page == PAGE_LIBRARY:
        render_library()
    else:
        render_about()


if __name__ == "__main__":
    main()
