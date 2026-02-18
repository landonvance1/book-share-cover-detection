from abc import ABC, abstractmethod

from app.models import NlpAnalysis, OcrResult


class NlpEngine(ABC):
    @abstractmethod
    async def analyze(self, ocr_result: OcrResult) -> NlpAnalysis:
        ...
