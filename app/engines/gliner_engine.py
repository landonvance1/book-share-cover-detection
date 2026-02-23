from __future__ import annotations

import asyncio

from app.interfaces.nlp import NlpEngine
from app.models import NlpAnalysis, OcrResult


class GlinerNlpEngine(NlpEngine):
    DEFAULT_MODEL = "urchade/gliner_small-v2.1"
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        from gliner import GLiNER  # lazy import — gliner is heavy and optional at import time
        self._model = GLiNER.from_pretrained(model_name)
        self._threshold = threshold

    async def analyze(self, ocr_result: OcrResult) -> NlpAnalysis:
        text = ocr_result.text
        if not text.strip():
            return NlpAnalysis(potential_authors=[])

        normalized = text.title() if text == text.upper() else text

        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            None, lambda: self._model.predict_entities(normalized, ["author"], threshold=self._threshold)
        )

        authors, seen = [], set()
        for entity in entities:
            if entity["label"] != "author":
                continue
            name = entity["text"].strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                authors.append(name)

        return NlpAnalysis(potential_authors=authors)
