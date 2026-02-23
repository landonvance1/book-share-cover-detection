from pathlib import Path

from app.models import OcrResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ocr(image_stem: str) -> OcrResult:
    return OcrResult.model_validate_json(
        (FIXTURES_DIR / f"{image_stem}.json").read_text()
    )


def _authors(result) -> list[str]:
    return [a.lower() for a in result.potential_authors]


def _titles(result) -> list[str]:
    return [t.lower() for t in result.potential_titles]


import pytest


@pytest.fixture(scope="session")
def gliner_engine():
    from app.engines.gliner_engine import GlinerNlpEngine
    return GlinerNlpEngine()


async def test_gliner_mistborn(gliner_engine):
    result = await gliner_engine.analyze(_ocr("mistborn"))
    authors = _authors(result)
    assert any("sanderson" in a for a in authors), authors
    assert not any("mistborn" in a for a in authors), authors


async def test_gliner_jade_city(gliner_engine):
    result = await gliner_engine.analyze(_ocr("jade-city"))
    authors = _authors(result)
    assert any("fonda" in a or "lee" in a for a in authors), authors
    assert not any("jade" in a for a in authors), authors
    assert not any("city" in a for a in authors), authors


async def test_gliner_snow_crash(gliner_engine):
    result = await gliner_engine.analyze(_ocr("snow-crash"))
    authors = _authors(result)
    assert any("stephenson" in a for a in authors), authors
    assert not any("snow" in a for a in authors), authors
    assert not any("crash" in a for a in authors), authors


async def test_gliner_to_kill_a_mockingbird(gliner_engine):
    result = await gliner_engine.analyze(_ocr("to-kill-a-mockingbird"))
    authors = _authors(result)
    assert any("harper" in a or "lee" in a for a in authors), authors
    assert not any("mockingbird" in a for a in authors), authors
    assert not any("kill" in a for a in authors), authors


async def test_gliner_a_restless_truth(gliner_engine):
    result = await gliner_engine.analyze(_ocr("a-restless-truth"))
    authors = _authors(result)
    assert any("marske" in a for a in authors), authors
    assert not any("restless" in a for a in authors), authors
    assert not any("truth" in a for a in authors), authors


async def test_gliner_gardens_of_the_moon(gliner_engine):
    result = await gliner_engine.analyze(_ocr("gardens-of-the-moon"))
    authors = _authors(result)
    assert any("erikson" in a for a in authors), authors
    assert not any("gardens" in a for a in authors), authors
    assert not any("moon" in a for a in authors), authors


async def test_gliner_under_the_whispering_door(gliner_engine):
    result = await gliner_engine.analyze(_ocr("under-the-whispering-door"))
    authors = _authors(result)
    assert any("klune" in a for a in authors), authors
    assert not any("whispering" in a for a in authors), authors
    assert not any("door" in a for a in authors), authors


async def test_gliner_mistborn_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("mistborn"))
    titles = _titles(result)
    print("Urgent message")
    assert any("mistborn" in t for t in titles), titles
    assert not any("brandon" in t for t in titles), titles
    assert not any("sanderson" in t for t in titles), titles
    assert not any("bestselling" in t for t in titles), titles
    assert not any("new" in t for t in titles), titles
    assert not any("york" in t for t in titles), titles


async def test_gliner_jade_city_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("jade-city"))
    titles = _titles(result)
    assert any("jade city" in t for t in titles), titles
    assert not any("fonda" in t for t in titles), titles
    assert not any("lee" in t for t in titles), titles
    assert not any("family" in t for t in titles), titles
    assert not any("duty" in t for t in titles), titles
    assert not any("magic" in t for t in titles), titles
    assert not any("power" in t for t in titles), titles
    assert not any("honor" in t for t in titles), titles
    assert not any("everything" in t for t in titles), titles
    assert not any("world" in t for t in titles), titles
    assert not any("fantasy" in t for t in titles), titles
    assert not any("award" in t for t in titles), titles


async def test_gliner_snow_crash_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("snow-crash"))
    titles = _titles(result)
    assert any("snow crash" in t for t in titles), titles
    assert not any("neal" in t for t in titles), titles
    assert not any("stephenson" in t for t in titles), titles
    assert not any("novel" in t for t in titles), titles


async def test_gliner_to_kill_a_mockingbird_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("to-kill-a-mockingbird"))
    titles = _titles(result)
    assert any("mockingbird" in t for t in titles), titles
    assert not any("harper" in t for t in titles), titles
    assert not any("lee" in t for t in titles), titles
    assert not any("timeless" in t for t in titles), titles
    assert not any("dignity" in t for t in titles), titles


async def test_gliner_a_restless_truth_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("a-restless-truth"))
    titles = _titles(result)
    assert any("a restless truth" in t for t in titles), titles
    assert not any("freya" in t for t in titles), titles
    assert not any("marske" in t for t in titles), titles
    assert not any("new" in t for t in titles), titles
    assert not any("york" in t for t in titles), titles


async def test_gliner_gardens_of_the_moon_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("gardens-of-the-moon"))
    titles = _titles(result)
    assert any("gardens of the moon" in t for t in titles), titles
    assert not any("steven" in t for t in titles), titles
    assert not any("erikson" in t for t in titles), titles
    assert not any("fallen" in t for t in titles), titles


async def test_gliner_under_the_whispering_door_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("under-the-whispering-door"))
    titles = _titles(result)
    assert any("under the whispering door" in t for t in titles), titles
    assert not any("tj" in t for t in titles), titles
    assert not any("klune" in t for t in titles), titles
    assert not any("death" in t for t in titles), titles
    assert not any("beginning" in t for t in titles), titles

