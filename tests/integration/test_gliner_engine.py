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


async def test_gliner_mistborn_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("mistborn"))
    authors = _authors(result)
    assert authors[0] == "brandon sanderson", authors
    assert not any("mistborn" in a for a in authors), authors


async def test_gliner_jade_city_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("jade-city"))
    authors = _authors(result)
    assert authors[0] == "fonda lee", authors
    assert not any("jade" in a for a in authors), authors
    assert not any("city" in a for a in authors), authors


async def test_gliner_snow_crash_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("snow-crash"))
    authors = _authors(result)
    assert authors[0] == "neal stephenson", authors
    assert not any("snow" in a for a in authors), authors
    assert not any("crash" in a for a in authors), authors


async def test_gliner_to_kill_a_mockingbird_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("to-kill-a-mockingbird"))
    authors = _authors(result)
    assert authors[0] == "harper lee", authors
    assert not any("mockingbird" in a for a in authors), authors
    assert not any("kill" in a for a in authors), authors


async def test_gliner_a_restless_truth_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("a-restless-truth"))
    authors = _authors(result)
    assert authors[0] == "freya marske", authors
    assert not any("restless" in a for a in authors), authors
    assert not any("truth" in a for a in authors), authors


async def test_gliner_gardens_of_the_moon_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("gardens-of-the-moon"))
    authors = _authors(result)
    assert authors[0] == "steven erikson", authors
    assert not any("gardens" in a for a in authors), authors
    assert not any("moon" in a for a in authors), authors


async def test_gliner_under_the_whispering_door_author(gliner_engine):
    result = await gliner_engine.analyze(_ocr("under-the-whispering-door"))
    authors = _authors(result)
    assert authors[0] == "tj klune", authors
    assert not any("whispering" in a for a in authors), authors
    assert not any("door" in a for a in authors), authors


async def test_gliner_mistborn_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("mistborn"))
    titles = _titles(result)
    print("Urgent message")
    assert titles[0] == "mistborn", titles


async def test_gliner_jade_city_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("jade-city"))
    titles = _titles(result)
    assert titles[0] == "jade city", titles


async def test_gliner_snow_crash_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("snow-crash"))
    titles = _titles(result)
    assert titles[0] == "snow crash", titles


async def test_gliner_to_kill_a_mockingbird_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("to-kill-a-mockingbird"))
    titles = _titles(result)
    assert titles[0] == "to kill a mockingbird", titles


async def test_gliner_a_restless_truth_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("a-restless-truth"))
    titles = _titles(result)
    assert titles[0] == "a restless truth", titles


async def test_gliner_gardens_of_the_moon_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("gardens-of-the-moon"))
    titles = _titles(result)
    assert titles[0] == "gardens of the moon", titles


async def test_gliner_under_the_whispering_door_title(gliner_engine):
    result = await gliner_engine.analyze(_ocr("under-the-whispering-door"))
    titles = _titles(result)
    assert titles[0] == "under the whispering door", titles

