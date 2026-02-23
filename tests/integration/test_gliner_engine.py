from pathlib import Path

from app.models import OcrResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ocr(image_stem: str) -> OcrResult:
    return OcrResult.model_validate_json(
        (FIXTURES_DIR / f"{image_stem}.json").read_text()
    )


def _authors(result) -> list[str]:
    return [a.lower() for a in result.potential_authors]


import pytest


@pytest.fixture(scope="session")
def gliner_engine():
    from app.engines.gliner_engine import GlinerNlpEngine
    return GlinerNlpEngine()


async def test_gliner_mistborn(gliner_engine):
    result = await gliner_engine.analyze(_ocr("mistborn"))
    authors = _authors(result)
    assert any("sanderson" in a for a in authors)
    assert not any("mistborn" in a for a in authors)


async def test_gliner_jade_city(gliner_engine):
    result = await gliner_engine.analyze(_ocr("jade-city"))
    authors = _authors(result)
    assert any("fonda" in a or "lee" in a for a in authors)
    assert not any("jade" in a for a in authors)
    assert not any("city" in a for a in authors)


async def test_gliner_snow_crash(gliner_engine):
    result = await gliner_engine.analyze(_ocr("snow-crash"))
    authors = _authors(result)
    assert any("stephenson" in a for a in authors)
    assert not any("snow" in a for a in authors)
    assert not any("crash" in a for a in authors)


async def test_gliner_to_kill_a_mockingbird(gliner_engine):
    result = await gliner_engine.analyze(_ocr("to-kill-a-mockingbird"))
    authors = _authors(result)
    assert any("harper" in a or "lee" in a for a in authors)
    assert not any("mockingbird" in a for a in authors)
    assert not any("kill" in a for a in authors)


async def test_gliner_a_restless_truth(gliner_engine):
    result = await gliner_engine.analyze(_ocr("a-restless-truth"))
    authors = _authors(result)
    assert any("marske" in a for a in authors)
    assert not any("restless" in a for a in authors)
    assert not any("truth" in a for a in authors)


async def test_gliner_gardens_of_the_moon(gliner_engine):
    result = await gliner_engine.analyze(_ocr("gardens-of-the-moon"))
    authors = _authors(result)
    assert any("erikson" in a for a in authors)
    assert not any("gardens" in a for a in authors)
    assert not any("moon" in a for a in authors)


async def test_gliner_under_the_whispering_door(gliner_engine):
    result = await gliner_engine.analyze(_ocr("under-the-whispering-door"))
    authors = _authors(result)
    assert any("klune" in a for a in authors)
    assert not any("whispering" in a for a in authors)
    assert not any("door" in a for a in authors)
