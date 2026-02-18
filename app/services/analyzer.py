from app.interfaces.book_search import BookSearchClient
from app.interfaces.nlp import NlpEngine
from app.interfaces.ocr import OcrEngine
from app.models import AnalysisStatus, CoverAnalysisResponse


class CoverAnalyzer:
    def __init__(
        self,
        ocr_engine: OcrEngine,
        nlp_engine: NlpEngine,
        book_search: BookSearchClient,
    ) -> None:
        self._ocr = ocr_engine
        self._nlp = nlp_engine
        self._search = book_search

    async def analyze(self, image_bytes: bytes) -> CoverAnalysisResponse:
        try:
            ocr_result = await self._ocr.extract_text(image_bytes)
        except Exception as e:
            return CoverAnalysisResponse(
                analysis=AnalysisStatus(
                    is_success=False,
                    error_message=f"OCR failed: {e}",
                ),
            )

        try:
            nlp_analysis = await self._nlp.analyze(ocr_result)
        except Exception as e:
            return CoverAnalysisResponse(
                analysis=AnalysisStatus(
                    is_success=False,
                    error_message=f"NLP analysis failed: {e}",
                ),
            )

        query_parts = []
        if nlp_analysis.detected_title:
            query_parts.append(nlp_analysis.detected_title)
        if nlp_analysis.detected_author:
            query_parts.append(nlp_analysis.detected_author)
        query = " ".join(query_parts) if query_parts else ocr_result.text

        try:
            raw_results = await self._search.search(query)
        except Exception as e:
            return CoverAnalysisResponse(
                analysis=AnalysisStatus(
                    is_success=False,
                    error_message=f"Book search failed: {e}",
                ),
            )

        ocr_words = self._search.build_word_set(ocr_result.text)
        scored = self._search.score_results(raw_results, ocr_words)
        matched_books = [book for book, _score in scored]

        return CoverAnalysisResponse(
            analysis=AnalysisStatus(
                is_success=True,
                extracted_text=ocr_result.text,
            ),
            matched_books=matched_books,
            nlp_analysis=nlp_analysis,
        )
