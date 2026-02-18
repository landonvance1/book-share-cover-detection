import pytest

from app.interfaces.book_search import BookSearchClient
from app.interfaces.nlp import NlpEngine
from app.interfaces.ocr import OcrEngine


class TestOcrEngineABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            OcrEngine()

    def test_subclass_must_implement_extract_text(self):
        class IncompleteOcr(OcrEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteOcr()


class TestNlpEngineABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            NlpEngine()

    def test_subclass_must_implement_analyze(self):
        class IncompleteNlp(NlpEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteNlp()


class TestBookSearchClientABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BookSearchClient()

    def test_subclass_must_implement_search(self):
        class IncompleteSearch(BookSearchClient):
            pass

        with pytest.raises(TypeError):
            IncompleteSearch()
