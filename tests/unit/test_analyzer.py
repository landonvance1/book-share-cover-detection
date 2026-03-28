import logging
from unittest.mock import patch

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
    async def test_successful_pipeline_records_histograms(self, sample_ocr_result, sample_nlp_analysis):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(result=sample_nlp_analysis)
        analyzer = CoverAnalyzer(ocr, nlp)

        with patch("app.services.analyzer._OCR_DURATION") as mock_ocr, \
             patch("app.services.analyzer._NLP_DURATION") as mock_nlp, \
             patch("app.services.analyzer._TOTAL_DURATION") as mock_total:
            await analyzer.analyze(b"fake image bytes")

        mock_ocr.observe.assert_called_once()
        mock_nlp.observe.assert_called_once()
        mock_total.observe.assert_called_once()
        assert mock_ocr.observe.call_args[0][0] >= 0
        assert mock_nlp.observe.call_args[0][0] >= 0
        assert mock_total.observe.call_args[0][0] >= 0

    @pytest.mark.asyncio
    async def test_successful_pipeline_logs_durations(self, sample_ocr_result, sample_nlp_analysis, caplog):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(result=sample_nlp_analysis)
        analyzer = CoverAnalyzer(ocr, nlp)

        with caplog.at_level(logging.INFO, logger="app.services.analyzer"):
            await analyzer.analyze(b"fake image bytes")

        assert "OCR completed" in caplog.messages
        assert "NLP completed" in caplog.messages
        assert "Analysis completed" in caplog.messages
        ocr_record = next(r for r in caplog.records if r.getMessage() == "OCR completed")
        assert hasattr(ocr_record, "duration_ms")

    @pytest.mark.asyncio
    async def test_ocr_failure_logs_error(self, caplog):
        ocr = MockOcrEngine(error=RuntimeError("OCR crashed"))
        nlp = MockNlpEngine(result=NlpAnalysis())
        analyzer = CoverAnalyzer(ocr, nlp)

        with caplog.at_level(logging.ERROR, logger="app.services.analyzer"):
            await analyzer.analyze(b"fake image bytes")

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert error_records[0].getMessage() == "OCR failed"
        assert hasattr(error_records[0], "error")

    @pytest.mark.asyncio
    async def test_nlp_failure_logs_error(self, sample_ocr_result, caplog):
        ocr = MockOcrEngine(result=sample_ocr_result)
        nlp = MockNlpEngine(error=RuntimeError("NLP crashed"))
        analyzer = CoverAnalyzer(ocr, nlp)

        with caplog.at_level(logging.ERROR, logger="app.services.analyzer"):
            await analyzer.analyze(b"fake image bytes")

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert error_records[0].getMessage() == "NLP analysis failed"
        assert hasattr(error_records[0], "error")

    @pytest.mark.asyncio
    async def test_ocr_failure_does_not_record_histograms(self):
        ocr = MockOcrEngine(error=RuntimeError("OCR crashed"))
        nlp = MockNlpEngine(result=NlpAnalysis())
        analyzer = CoverAnalyzer(ocr, nlp)

        with patch("app.services.analyzer._OCR_DURATION") as mock_ocr, \
             patch("app.services.analyzer._NLP_DURATION") as mock_nlp, \
             patch("app.services.analyzer._TOTAL_DURATION") as mock_total:
            await analyzer.analyze(b"fake image bytes")

        mock_ocr.observe.assert_not_called()
        mock_nlp.observe.assert_not_called()
        mock_total.observe.assert_not_called()

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
