import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from app.models import OcrResult


def _make_ocr(text: str) -> OcrResult:
    return OcrResult(text=text, regions=[])


@pytest.fixture(autouse=True)
def mock_gliner_module():
    """Inject a fake gliner module so tests run without installing the package."""
    gliner_mod = types.ModuleType("gliner")
    gliner_mod.GLiNER = MagicMock()
    with patch.dict(sys.modules, {"gliner": gliner_mod}):
        # Remove cached engine module so each test gets a fresh import with the mock
        sys.modules.pop("app.engines.gliner_engine", None)
        yield gliner_mod.GLiNER


async def test_extracts_single_author(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Brandon Sanderson", "label": "author", "score": 0.95}
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Brandon Sanderson Mistborn"))

    assert result.potential_authors == ["Brandon Sanderson"]


async def test_returns_empty_for_empty_text(mock_gliner_module):
    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("   "))

    assert result.potential_authors == []
    mock_gliner_module.from_pretrained.return_value.predict_entities.assert_not_called()


async def test_normalizes_all_caps_text(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = []

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    await engine.analyze(_make_ocr("BRANDON SANDERSON MISTBORN"))

    call_args = mock_gliner_module.from_pretrained.return_value.predict_entities.call_args
    text_passed = call_args[0][0]
    assert text_passed == "Brandon Sanderson Mistborn"


async def test_leaves_mixed_case_text_unchanged(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = []

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    await engine.analyze(_make_ocr("Brandon Sanderson MISTBORN"))

    call_args = mock_gliner_module.from_pretrained.return_value.predict_entities.call_args
    text_passed = call_args[0][0]
    assert text_passed == "Brandon Sanderson MISTBORN"


async def test_extracts_multiple_authors(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Fonda Lee", "label": "author", "score": 0.90},
        {"text": "Brandon Sanderson", "label": "author", "score": 0.85},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Fonda Lee Brandon Sanderson"))

    assert result.potential_authors == ["Fonda Lee", "Brandon Sanderson"]


async def test_deduplicates_case_insensitive(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Brandon Sanderson", "label": "author", "score": 0.95},
        {"text": "BRANDON SANDERSON", "label": "author", "score": 0.80},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Brandon Sanderson"))

    assert len(result.potential_authors) == 1
    assert result.potential_authors[0] == "Brandon Sanderson"


async def test_ignores_non_author_entities(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Mistborn", "label": "title", "score": 0.92},
        {"text": "Brandon Sanderson", "label": "author", "score": 0.95},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Brandon Sanderson Mistborn"))

    assert result.potential_authors == ["Brandon Sanderson"]
    assert "Mistborn" not in result.potential_authors


async def test_extracts_single_title(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Mistborn", "label": "book title", "score": 0.92},
        {"text": "Brandon Sanderson", "label": "author", "score": 0.95},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Brandon Sanderson Mistborn"))

    assert result.potential_titles == ["Mistborn"]
    assert "Brandon Sanderson" not in result.potential_titles


async def test_deduplicates_titles_case_insensitive(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Mistborn", "label": "book title", "score": 0.92},
        {"text": "MISTBORN", "label": "book title", "score": 0.80},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Mistborn"))

    assert len(result.potential_titles) == 1
    assert result.potential_titles[0] == "Mistborn"


async def test_titles_not_contaminated_by_author_entities(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Brandon Sanderson", "label": "author", "score": 0.95},
        {"text": "Mistborn", "label": "book title", "score": 0.92},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Brandon Sanderson Mistborn"))

    assert "Brandon Sanderson" not in result.potential_titles
    assert "Mistborn" not in result.potential_authors


async def test_authors_not_contaminated_by_title_entities(mock_gliner_module):
    mock_gliner_module.from_pretrained.return_value.predict_entities.return_value = [
        {"text": "Mistborn", "label": "book title", "score": 0.92},
        {"text": "Brandon Sanderson", "label": "author", "score": 0.95},
    ]

    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("Brandon Sanderson Mistborn"))

    assert "Mistborn" not in result.potential_authors
    assert result.potential_authors == ["Brandon Sanderson"]


async def test_empty_text_returns_empty_titles(mock_gliner_module):
    from app.engines.gliner_engine import GlinerNlpEngine
    engine = GlinerNlpEngine()
    result = await engine.analyze(_make_ocr("   "))

    assert result.potential_titles == []
    mock_gliner_module.from_pretrained.return_value.predict_entities.assert_not_called()
