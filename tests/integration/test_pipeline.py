import pytest
from pathlib import Path

from app.engines.florence2_engine import Florence2OcrEngine

IMAGES_DIR = Path(__file__).parent / "images"


@pytest.fixture(scope="session")
def florence2_engine():
    return Florence2OcrEngine()


def _load(filename: str) -> bytes:
    return (IMAGES_DIR / filename).read_bytes()


async def test_florence2_a_restless_truth(florence2_engine):
    result = await florence2_engine.extract_text(_load("a-restless-truth.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "freya" in text
    assert "marske" in text
    assert "truth" in text


async def test_florence2_gardens_of_the_moon(florence2_engine):
    result = await florence2_engine.extract_text(_load("gardens-of-the-moon.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "steven" in text
    assert "erikson" in text
    assert "gardens" in text
    assert "moon" in text


async def test_florence2_jade_city(florence2_engine):
    result = await florence2_engine.extract_text(_load("jade-city.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "fonda" in text
    assert "lee" in text
    assert "jade" in text
    assert "city" in text


async def test_florence2_mistborn(florence2_engine):
    result = await florence2_engine.extract_text(_load("mistborn.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "brandon" in text
    assert "sanderson" in text
    assert "mistborn" in text


async def test_florence2_snow_crash(florence2_engine):
    result = await florence2_engine.extract_text(_load("snow-crash.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "neal" in text
    assert "stephenson" in text
    assert "snow" in text
    assert "crash" in text


async def test_florence2_to_kill_a_mockingbird(florence2_engine):
    result = await florence2_engine.extract_text(_load("to-kill-a-mockingbird.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "harper" in text
    assert "lee" in text
    assert "kill" in text
    assert "mockingbird" in text


async def test_florence2_under_the_whispering_door(florence2_engine):
    result = await florence2_engine.extract_text(_load("under-the-whispering-door.jpg"))
    assert result.text != ""
    text = result.text.lower()
    assert "klune" in text
    assert "tj" in text
    assert "under" in text
    assert "whispering" in text
    assert "door" in text
