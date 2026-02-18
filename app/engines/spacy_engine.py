from app.interfaces.nlp import NlpEngine
from app.models import NlpAnalysis, OcrResult


class SpacyNlpEngine(NlpEngine):
    async def analyze(self, ocr_result: OcrResult) -> NlpAnalysis:
        raise NotImplementedError("SpaCy NLP engine not yet implemented")
