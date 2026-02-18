import pytest

from app.models import BookMatch, NlpAnalysis, OcrResult
from app.services.analyzer import CoverAnalyzer
from tests.conftest import MockBookSearchClient, MockNlpEngine, MockOcrEngine


class TestCoverAnalyzer:
    @pytest.mark.asyncio
    async def test_successful_pipeline(
        self, sample_ocr_result, sample_nlp_analysis, sample_book_matches
    ):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(result=sample_nlp_analysis)
        search = MockBookSearchClient(results=sample_book_matches)
        analyzer = CoverAnalyzer(ocr, nlp, search)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysis.is_success is True
        assert result.analysis.extracted_text == sample_ocr_result.text
        assert result.nlp_analysis is not None
        assert result.nlp_analysis.detected_title == "The Great Gatsby"

    @pytest.mark.asyncio
    async def test_ocr_failure(self):
        ocr = MockOcrEngine(error=RuntimeError("OCR crashed"))
        nlp = MockNlpEngine(result=NlpAnalysis())
        search = MockBookSearchClient()
        analyzer = CoverAnalyzer(ocr, nlp, search)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysis.is_success is False
        assert "OCR failed" in result.analysis.error_message

    @pytest.mark.asyncio
    async def test_nlp_failure(self, sample_ocr_result):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(error=RuntimeError("NLP crashed"))
        search = MockBookSearchClient()
        analyzer = CoverAnalyzer(ocr, nlp, search)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysis.is_success is False
        assert "NLP analysis failed" in result.analysis.error_message

    @pytest.mark.asyncio
    async def test_search_failure(self, sample_ocr_result, sample_nlp_analysis):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(result=sample_nlp_analysis)
        search = MockBookSearchClient(error=RuntimeError("Search crashed"))
        analyzer = CoverAnalyzer(ocr, nlp, search)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysis.is_success is False
        assert "Book search failed" in result.analysis.error_message

    @pytest.mark.asyncio
    async def test_no_matches(self, sample_ocr_result, sample_nlp_analysis):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(result=sample_nlp_analysis)
        search = MockBookSearchClient(results=[])
        analyzer = CoverAnalyzer(ocr, nlp, search)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysis.is_success is True
        assert result.matched_books == []

    @pytest.mark.asyncio
    async def test_data_flows_through_pipeline(self):
        ocr_result = OcrResult(text="Specific Test Text", regions=[])
        nlp_result = NlpAnalysis(
            detected_title="Specific Test",
            title_confidence=0.95,
            detected_author="Text Author",
            author_confidence=0.85,
        )
        books = [
            BookMatch(title="Specific Test", author="Text Author"),
        ]

        ocr = MockOcrEngine(result=ocr_result)
        nlp = MockNlpEngine(result=nlp_result)
        search = MockBookSearchClient(results=books)
        analyzer = CoverAnalyzer(ocr, nlp, search)

        result = await analyzer.analyze(b"image")

        assert result.analysis.extracted_text == "Specific Test Text"
        assert result.nlp_analysis.detected_title == "Specific Test"
        assert result.nlp_analysis.detected_author == "Text Author"
