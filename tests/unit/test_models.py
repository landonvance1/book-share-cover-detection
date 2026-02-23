from app.models import (
    AnalysisStatus,
    BookMatch,
    CoverAnalysisResponse,
    HealthResponse,
    NlpAnalysis,
    OcrBoundingBox,
    OcrResult,
)


class TestCamelCaseSerialization:
    def test_nlp_analysis_camel_case(self):
        result = NlpAnalysis(potential_authors=["Author One", "Author Two"])
        data = result.model_dump(by_alias=True)
        assert "potentialAuthors" in data
        assert data["potentialAuthors"] == ["Author One", "Author Two"]

    def test_book_match_camel_case(self):
        book = BookMatch(
            title="Test", author="Author", isbn="123", thumbnail_url="http://example.com"
        )
        data = book.model_dump(by_alias=True)
        assert "thumbnailUrl" in data

    def test_analysis_status_camel_case(self):
        status = AnalysisStatus(is_success=True, error_message=None, extracted_text="text")
        data = status.model_dump(by_alias=True)
        assert "isSuccess" in data
        assert "errorMessage" in data

    def test_cover_analysis_response_camel_case(self):
        response = CoverAnalysisResponse(
            analysisStatus=AnalysisStatus(is_success=True),
            matched_books=[],
            nlp_analysis=NlpAnalysis(),
        )
        data = response.model_dump(by_alias=True)
        assert "nlpAnalysis" in data
        assert "ocrResult" in data

    def test_ocr_bounding_box_camel_case(self):
        box = OcrBoundingBox(text="test", confidence=0.9, coordinates=[[0, 0]])
        data = box.model_dump(by_alias=True)
        assert "text" in data
        assert "confidence" in data

    def test_health_response_serialization(self):
        health = HealthResponse(status="healthy", version="0.1.0")
        data = health.model_dump(by_alias=True)
        assert data == {"status": "healthy", "version": "0.1.0"}


class TestOptionalFields:
    def test_nlp_analysis_defaults(self):
        result = NlpAnalysis()
        assert result.potential_authors == []

    def test_book_match_optional_fields(self):
        book = BookMatch(title="Test", author="Author")
        assert book.isbn is None
        assert book.thumbnail_url is None

    def test_analysis_status_optional_fields(self):
        status = AnalysisStatus(is_success=False)
        assert status.error_message is None

    def test_cover_analysis_response_defaults(self):
        response = CoverAnalysisResponse(
            analysisStatus=AnalysisStatus(is_success=True),
        )
        assert response.nlp_analysis is None
        assert response.ocr_result is None


class TestOcrResult:
    def test_ocr_result_with_regions(self, sample_ocr_regions):
        result = OcrResult(text="full text", regions=sample_ocr_regions)
        assert result.text == "full text"
        assert len(result.regions) == 2

    def test_ocr_result_empty_regions(self):
        result = OcrResult(text="", regions=[])
        assert result.text == ""
        assert result.regions == []
