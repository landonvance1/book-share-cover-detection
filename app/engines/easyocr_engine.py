from app.interfaces.ocr import OcrEngine
from app.models import OcrResult


class EasyOcrEngine(OcrEngine):
    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        raise NotImplementedError("EasyOCR engine not yet implemented")
