from __future__ import annotations

import asyncio

from app.interfaces.nlp import NlpEngine
from app.models import NlpAnalysis, OcrResult


_HEIGHT_THRESHOLD = 0.2  # keep regions >= 20% of the tallest region's height


def _filter_text_by_height(ocr_result: OcrResult) -> str:
    """Return only text from regions whose height is >= _HEIGHT_THRESHOLD * max height.

    Falls back to the full OCR text string when no region coordinates are available.
    """
    if not ocr_result.regions:
        return ocr_result.text

    heights = [
        max(c[1] for c in r.coordinates) - min(c[1] for c in r.coordinates)
        for r in ocr_result.regions
    ]
    cutoff = max(heights) * _HEIGHT_THRESHOLD
    return " ".join(
        r.text for r, h in zip(ocr_result.regions, heights) if h >= cutoff
    )


class GlinerNlpEngine(NlpEngine):
    DEFAULT_MODEL = "urchade/gliner_large-v2.1"
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        from gliner import GLiNER  # lazy import — gliner is heavy and optional at import time
        self._model = GLiNER.from_pretrained(model_name)
        self._threshold = threshold

    async def analyze(self, ocr_result: OcrResult) -> NlpAnalysis:
        text = _filter_text_by_height(ocr_result)
        if not text.strip():
            return NlpAnalysis(potential_authors=[], potential_titles=[])

        normalized = text.title() if text == text.upper() else text

        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            None, lambda: self._model.predict_entities(normalized, ["author", "book title"], threshold=self._threshold)
        )

        authors, seen_authors = [], set()
        titles, seen_titles = [], set()
        for entity in entities:
            name = entity["text"].strip()
            if entity["label"] == "author":
                if name.lower() not in seen_authors:
                    seen_authors.add(name.lower())
                    authors.append(name)
            elif entity["label"] == "book title":
                if name.lower() not in seen_titles:
                    seen_titles.add(name.lower())
                    titles.append(name)

        return NlpAnalysis(potential_authors=authors, potential_titles=titles)
