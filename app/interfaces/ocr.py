from abc import ABC, abstractmethod

from app.models import OcrResult


class OcrEngine(ABC):
    @abstractmethod
    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        ...
