import pytest

from app.interfaces.book_search import BookSearchClient
from app.interfaces.nlp import NlpEngine
from app.interfaces.ocr import OcrEngine
from app.models import BookMatch, NlpAnalysis, OcrBoundingBox, OcrResult


class MockOcrEngine(OcrEngine):
    def __init__(self, result: OcrResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result


class MockNlpEngine(NlpEngine):
    def __init__(self, result: NlpAnalysis | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def analyze(self, ocr_result: OcrResult) -> NlpAnalysis:
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result


class MockBookSearchClient(BookSearchClient):
    def __init__(self, results: list[BookMatch] | None = None, error: Exception | None = None):
        self._results = results or []
        self._error = error

    async def search(self, query: str, limit: int = 11) -> list[BookMatch]:
        if self._error:
            raise self._error
        return self._results


@pytest.fixture
def sample_ocr_regions() -> list[OcrBoundingBox]:
    return [
        OcrBoundingBox(
            text="The Great Gatsby",
            confidence=0.95,
            coordinates=[[0, 0], [200, 0], [200, 50], [0, 50]],
        ),
        OcrBoundingBox(
            text="F Scott Fitzgerald",
            confidence=0.88,
            coordinates=[[0, 60], [200, 60], [200, 100], [0, 100]],
        ),
    ]


@pytest.fixture
def sample_ocr_result(sample_ocr_regions) -> OcrResult:
    return OcrResult(
        text="The Great Gatsby F Scott Fitzgerald",
        regions=sample_ocr_regions,
    )


@pytest.fixture
def sample_nlp_analysis() -> NlpAnalysis:
    return NlpAnalysis(
        detected_title="The Great Gatsby",
        title_confidence=0.92,
        detected_author="F Scott Fitzgerald",
        author_confidence=0.87,
    )


@pytest.fixture
def sample_book_matches() -> list[BookMatch]:
    return [
        BookMatch(
            title="The Great Gatsby",
            author="F. Scott Fitzgerald",
            isbn="9780743273565",
            thumbnail_url="https://covers.openlibrary.org/b/id/12345-M.jpg",
        ),
        BookMatch(
            title="Gatsby: A Novel",
            author="F. Scott Fitzgerald",
            isbn="9780000000001",
            thumbnail_url=None,
        ),
    ]
