"""Runs the real Streamlit script headlessly in demo mode."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.model import DEMO_MODE_MESSAGE, MODEL_PATH

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app():
    assert not MODEL_PATH.exists(), "These tests cover the no-model state."
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    return at


def html_text(at):
    return " ".join(element.proto.body for element in at.get("html"))


def test_home_shows_heading_and_demo_state(app):
    assert "AI-Powered Crop Disease Detection" in html_text(app)


@pytest.mark.parametrize("page", ["Detect Disease", "Disease Library", "About Project"])
def test_every_page_runs_without_errors(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception


def test_upload_button_opens_detect_page(app):
    app.button[0].click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Detect Disease"
    assert DEMO_MODE_MESSAGE in html_text(app)


def test_no_accuracy_claim_anywhere(app):
    for page in ["Home", "Detect Disease", "Disease Library", "About Project"]:
        app.sidebar.radio[0].set_value(page).run()
        text = html_text(app)
        for claim in ["97%", "98%", "99%"]:
            assert claim not in text
