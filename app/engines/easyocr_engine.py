import asyncio
import io

import easyocr
import numpy as np
from PIL import Image

from app.interfaces.ocr import OcrEngine
from app.models import OcrBoundingBox, OcrResult


class EasyOcrEngine(OcrEngine):
    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self._reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        image_array = _bytes_to_array(image_bytes)
        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(
            None, lambda: self._reader.readtext(image_array)
        )
        return _build_ocr_result(raw_results)


def _bytes_to_array(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def _build_ocr_result(
    raw_results: list[tuple[list[list[float]], str, float]],
) -> OcrResult:
    regions = [
        OcrBoundingBox(
            text=text,
            confidence=float(confidence),
            coordinates=bounding_box,
        )
        for bounding_box, text, confidence in raw_results
    ]
    return OcrResult(text=" ".join(r.text for r in regions), regions=regions)
