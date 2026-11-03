import pytest

from src.recommendations import (
    ALL_CLASSES,
    DISEASES,
    HEALTHY_GUIDANCE,
    get_recommendation,
    parse_class_name,
)


def test_covers_all_38_classes_and_14_crops():
    assert len(ALL_CLASSES) == len(set(ALL_CLASSES)) == 38
    assert len({parse_class_name(name)[0] for name in ALL_CLASSES}) == 14


@pytest.mark.parametrize("info", DISEASES.values(), ids=lambda info: info.name)
def test_entries_are_complete(info):
    assert info.cause and info.summary
    assert info.organic and info.chemical and info.prevention


def test_parse_disease_and_healthy():
    assert parse_class_name("Tomato___Late_blight") == ("Tomato", "Late blight", False)
    assert parse_class_name("Corn_(maize)___healthy") == ("Corn (maize)", "Healthy", True)


def test_healthy_and_unknown_lookup():
    assert get_recommendation("Apple___healthy") is HEALTHY_GUIDANCE
    assert get_recommendation("Mango___Unknown") is None
