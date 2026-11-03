"""Reusable HTML components for the Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.model import Prediction
from src.recommendations import DISCLAIMER, DiseaseInfo

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def apply_styles() -> None:
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


def _bullets(items: tuple[str, ...] | list[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def page_heading(title: str, subtitle: str) -> None:
    st.html(
        f'<h1 class="av-page-title">{escape(title)}</h1>'
        f'<p class="av-page-subtitle">{escape(subtitle)}</p>'
    )


def hero(heading: str, subtitle: str, badges: list[str]) -> None:
    badge_html = "".join(f'<span class="av-badge">{escape(b)}</span>' for b in badges)
    st.html(
        f"""<section class="av-hero">
        <p class="av-eyebrow">AgroVision</p>
        <h1>{escape(heading)}</h1>
        <p class="av-subtitle">{escape(subtitle)}</p>
        <div class="av-badges">{badge_html}</div>
        </section>"""
    )


def banner(message: str, kind: str = "info") -> None:
    """Show a message; kind is 'demo', 'info', or 'error'."""
    st.html(f'<div class="av-banner {kind}">{escape(message)}</div>')


def card(title: str, body_html: str) -> None:
    """Show a card. body_html must already be safe HTML."""
    st.html(f'<div class="av-card"><h3>{escape(title)}</h3>{body_html}</div>')


def text_card(title: str, *paragraphs: str) -> None:
    card(title, "".join(f"<p>{escape(text)}</p>" for text in paragraphs))


def list_card(title: str, items: list[str]) -> None:
    card(title, _bullets(items))


def stat_grid(stats: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<div class="av-card"><div class="av-stat-value">{escape(value)}</div>'
        f'<div class="av-stat-label">{escape(label)}</div></div>'
        for value, label in stats
    )
    st.html(f'<div class="av-grid">{items}</div>')


def step_grid(steps: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<div class="av-card"><div class="av-step-number">{number}</div>'
        f"<h3>{escape(title)}</h3><p>{escape(text)}</p></div>"
        for number, (title, text) in enumerate(steps, start=1)
    )
    st.html(f'<div class="av-grid">{items}</div>')


def progress_card(title: str, steps: list[tuple[str, bool]]) -> None:
    """Show pipeline steps, each marked as complete or not yet available."""
    rows = "".join(
        f"<li><span>{escape(label)}</span>"
        + ('<span class="av-tag done">Complete</span>' if done
           else '<span class="av-tag pending">Awaiting model</span>')
        + "</li>"
        for label, done in steps
    )
    card(title, f'<ul class="av-progress">{rows}</ul>')


def sidebar_brand() -> None:
    st.html(
        '<div class="av-brand"><div class="av-brand-mark">A</div>'
        '<div class="av-brand-name">AgroVision</div></div>'
        '<p class="av-brand-tag">Crop disease detection</p>'
    )


def sidebar_model_status(is_ready: bool, message: str) -> None:
    title = "Model connected" if is_ready else "Demo mode"
    dot_class = "av-dot ready" if is_ready else "av-dot"
    detail = message if is_ready else "No trained model is connected, so no predictions are made."
    st.html(
        f'<div class="av-model-status"><strong><span class="{dot_class}"></span>'
        f"{title}</strong>{escape(detail)}</div>"
    )


def prediction_card(prediction: Prediction) -> None:
    st.html(
        f"""<div class="av-card">
        <div class="av-result-crop">{escape(prediction.crop)}</div>
        <div class="av-result-disease">{escape(prediction.disease)}</div>
        <p>Model confidence: <strong>{prediction.confidence:.1%}</strong></p>
        </div>"""
    )


def confidence_chart(predictions: list[Prediction]) -> None:
    rows = ""
    for prediction in predictions:
        label = f"{prediction.crop}: {prediction.disease}"
        rows += (
            f'<div class="av-bar-row"><div class="av-bar-label">'
            f"<span>{escape(label)}</span>"
            f"<strong>{prediction.confidence:.1%}</strong></div>"
            f'<div class="av-bar-track"><div class="av-bar-fill" '
            f'style="width:{prediction.confidence * 100:.2f}%"></div></div></div>'
        )
    card("Top 3 predictions", rows)


def recommendation_card(title: str, info: DiseaseInfo) -> None:
    columns = "".join(
        f'<div class="av-grid-item"><h4>{heading}</h4>{_bullets(items)}</div>'
        for heading, items in (
            ("Organic and cultural", info.organic),
            ("Chemical options", info.chemical),
            ("Prevention", info.prevention),
        )
    )
    card(
        title,
        f"<p><strong>{escape(info.name)}</strong> &middot; {escape(info.cause)}</p>"
        f"<p>{escape(info.summary)}</p>"
        f'<div class="av-grid">{columns}</div>'
        f'<p class="av-muted" style="margin-top:12px">{escape(DISCLAIMER)}</p>',
    )


def image_placeholder(text: str) -> None:
    st.html(f'<div class="av-placeholder">{escape(text)}</div>')
