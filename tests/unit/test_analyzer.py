import pytest

from app.models import BookMatch, NlpAnalysis, OcrResult
from app.services.analyzer import CoverAnalyzer
from tests.conftest import MockNlpEngine, MockOcrEngine


class TestCoverAnalyzer:
    @pytest.mark.asyncio
    async def test_successful_pipeline(
        self, sample_ocr_result, sample_nlp_analysis
    ):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(result=sample_nlp_analysis)
        analyzer = CoverAnalyzer(ocr, nlp)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysisStatus.is_success is True
        assert result.ocr_result.text == sample_ocr_result.text
        assert result.nlp_analysis is not None
        assert result.nlp_analysis.potential_authors == ["F Scott Fitzgerald"]

    @pytest.mark.asyncio
    async def test_ocr_failure(self):
        ocr = MockOcrEngine(error=RuntimeError("OCR crashed"))
        nlp = MockNlpEngine(result=NlpAnalysis())
        analyzer = CoverAnalyzer(ocr, nlp)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysisStatus.is_success is False
        assert "OCR failed" in result.analysisStatus.error_message

    @pytest.mark.asyncio
    async def test_nlp_failure(self, sample_ocr_result):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(error=RuntimeError("NLP crashed"))
        analyzer = CoverAnalyzer(ocr, nlp)

        result = await analyzer.analyze(b"fake image bytes")

        assert result.analysisStatus.is_success is False
        assert "NLP analysis failed" in result.analysisStatus.error_message

    @pytest.mark.asyncio
    async def test_data_flows_through_pipeline(self):
        ocr_result = OcrResult(text="Specific Test Text", regions=[])
        nlp_result = NlpAnalysis(potential_authors=["Text Author"])
        books = [
            BookMatch(title="Specific Test", author="Text Author"),
        ]

        ocr = MockOcrEngine(result=ocr_result)
        nlp = MockNlpEngine(result=nlp_result)
        analyzer = CoverAnalyzer(ocr, nlp)

        result = await analyzer.analyze(b"image")

        assert result.ocr_result.text == "Specific Test Text"
        assert result.nlp_analysis.potential_authors == ["Text Author"]
