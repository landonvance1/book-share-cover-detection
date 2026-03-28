import logging
import time

from prometheus_client import Histogram

from app.interfaces.nlp import NlpEngine
from app.interfaces.ocr import OcrEngine
from app.models import AnalysisStatus, CoverAnalysisResponse

logger = logging.getLogger(__name__)

_OCR_DURATION = Histogram(
    "cover_detection_ocr_duration_seconds",
    "Time spent in the OCR stage",
)
_NLP_DURATION = Histogram(
    "cover_detection_nlp_duration_seconds",
    "Time spent in the NLP stage",
)
_TOTAL_DURATION = Histogram(
    "cover_detection_total_duration_seconds",
    "Total analysis time (OCR + NLP)",
)


class CoverAnalyzer:
    def __init__(
        self,
        ocr_engine: OcrEngine,
        nlp_engine: NlpEngine,
    ) -> None:
        self._ocr = ocr_engine
        self._nlp = nlp_engine

    async def analyze(self, image_bytes: bytes) -> CoverAnalysisResponse:
        t_start = time.perf_counter()

        try:
            t_ocr_start = time.perf_counter()
            ocr_result = await self._ocr.extract_text(image_bytes)
            ocr_duration = time.perf_counter() - t_ocr_start
            _OCR_DURATION.observe(ocr_duration)
            logger.info("OCR completed", extra={"duration_ms": round(ocr_duration * 1000, 1)})
        except Exception as e:
            logger.error("OCR failed", extra={"error": str(e)})
            return CoverAnalysisResponse(
                analysisStatus=AnalysisStatus(
                    is_success=False,
                    error_message=f"OCR failed: {e}",
                ),
            )

        try:
            t_nlp_start = time.perf_counter()
            nlp_analysis = await self._nlp.analyze(ocr_result)
            nlp_duration = time.perf_counter() - t_nlp_start
            _NLP_DURATION.observe(nlp_duration)
            logger.info("NLP completed", extra={"duration_ms": round(nlp_duration * 1000, 1)})
        except Exception as e:
            logger.error("NLP analysis failed", extra={"error": str(e)})
            return CoverAnalysisResponse(
                analysisStatus=AnalysisStatus(
                    is_success=False,
                    error_message=f"NLP analysis failed: {e}",
                ),
            )

        total_duration = time.perf_counter() - t_start
        _TOTAL_DURATION.observe(total_duration)
        logger.info("Analysis completed", extra={"duration_ms": round(total_duration * 1000, 1)})

        return CoverAnalysisResponse(
            analysisStatus=AnalysisStatus(
                is_success=True
            ),
            ocr_result=ocr_result,
            nlp_analysis=nlp_analysis,
        )
