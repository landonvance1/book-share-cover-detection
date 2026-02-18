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
        result = NlpAnalysis(
            detected_title="Test",
            title_confidence=0.9,
            detected_author="Author",
            author_confidence=0.8,
        )
        data = result.model_dump(by_alias=True)
        assert "detectedTitle" in data
        assert "titleConfidence" in data
        assert "detectedAuthor" in data
        assert "authorConfidence" in data

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
        assert "extractedText" in data

    def test_cover_analysis_response_camel_case(self):
        response = CoverAnalysisResponse(
            analysis=AnalysisStatus(is_success=True),
            matched_books=[],
            nlp_analysis=NlpAnalysis(),
        )
        data = response.model_dump(by_alias=True)
        assert "matchedBooks" in data
        assert "exactMatch" in data
        assert "nlpAnalysis" in data

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
        assert result.detected_title is None
        assert result.title_confidence == 0.0
        assert result.detected_author is None
        assert result.author_confidence == 0.0

    def test_book_match_optional_fields(self):
        book = BookMatch(title="Test", author="Author")
        assert book.isbn is None
        assert book.thumbnail_url is None

    def test_analysis_status_optional_fields(self):
        status = AnalysisStatus(is_success=False)
        assert status.error_message is None
        assert status.extracted_text is None

    def test_cover_analysis_response_defaults(self):
        response = CoverAnalysisResponse(
            analysis=AnalysisStatus(is_success=True),
        )
        assert response.matched_books == []
        assert response.exact_match is None
        assert response.nlp_analysis is None


class TestOcrResult:
    def test_ocr_result_with_regions(self, sample_ocr_regions):
        result = OcrResult(text="full text", regions=sample_ocr_regions)
        assert result.text == "full text"
        assert len(result.regions) == 2

    def test_ocr_result_empty_regions(self):
        result = OcrResult(text="", regions=[])
        assert result.text == ""
        assert result.regions == []
