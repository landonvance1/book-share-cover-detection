from app.interfaces.nlp import NlpEngine
from app.interfaces.ocr import OcrEngine
from app.models import AnalysisStatus, CoverAnalysisResponse


class CoverAnalyzer:
    def __init__(
        self,
        ocr_engine: OcrEngine,
        nlp_engine: NlpEngine,
    ) -> None:
        self._ocr = ocr_engine
        self._nlp = nlp_engine

    async def analyze(self, image_bytes: bytes) -> CoverAnalysisResponse:
        try:
            ocr_result = await self._ocr.extract_text(image_bytes)
        except Exception as e:
            return CoverAnalysisResponse(
                analysisStatus=AnalysisStatus(
                    is_success=False,
                    error_message=f"OCR failed: {e}",
                ),
            )

        try:
            nlp_analysis = await self._nlp.analyze(ocr_result)
        except Exception as e:
            return CoverAnalysisResponse(
                analysisStatus=AnalysisStatus(
                    is_success=False,
                    error_message=f"NLP analysis failed: {e}",
                ),
            )

        except Exception as e:
            return CoverAnalysisResponse(
                analysisStatus=AnalysisStatus(
                    is_success=False,
                    error_message=f"Book search failed: {e}",
                ),
            )

        return CoverAnalysisResponse(
            analysisStatus=AnalysisStatus(
                is_success=True
            ),
            ocr_result=ocr_result,
            nlp_analysis=nlp_analysis,
        )
