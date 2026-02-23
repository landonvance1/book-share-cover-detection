from __future__ import annotations

import asyncio

from app.interfaces.nlp import NlpEngine
from app.models import NlpAnalysis, OcrResult


def _region_height(region) -> float:
    ys = [c[1] for c in region.coordinates]
    return max(ys) - min(ys)


def _regions_with_heights(
    ocr_result: OcrResult,
) -> list[tuple[str, float]]:
    """Return (text, height) for every OCR region."""
    return [(r.text, _region_height(r)) for r in ocr_result.regions]


def _build_text_with_spans(
    regions: list[tuple[str, float]],
) -> tuple[str, list[tuple[int, int, float]]]:
    """
    Returns (concatenated_text, [(char_start, char_end, height), ...]).
    Spans let us map a GLiNER entity's character positions back to region heights.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, float]] = []
    pos = 0
    for text, height in regions:
        start = pos
        end = pos + len(text)
        parts.append(text)
        spans.append((start, end, height))
        pos = end + 1  # +1 for the space separator
    return " ".join(parts), spans


def _entity_height(
    entity: dict,
    spans: list[tuple[int, int, float]],
) -> float:
    """Max height of OCR regions that overlap the entity's character span."""
    e_start, e_end = entity.get("start", 0), entity.get("end", 0)
    matching = [h for (s, e, h) in spans if s < e_end and e > e_start]
    return max(matching) if matching else 0.0


class GlinerNlpEngine(NlpEngine):
    DEFAULT_MODEL = "urchade/gliner_large-v2.1"
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        from gliner import GLiNER  # lazy import — gliner is heavy and optional at import time
        self._model = GLiNER.from_pretrained(model_name)
        self._threshold = threshold

    async def analyze(self, ocr_result: OcrResult) -> NlpAnalysis:
        regions = _regions_with_heights(ocr_result)
        if regions:
            raw_text, spans = _build_text_with_spans(regions)
        else:
            raw_text = ocr_result.text
            spans = []

        if not raw_text.strip():
            return NlpAnalysis(potential_authors=[], potential_titles=[])

        normalized = raw_text.title() if raw_text == raw_text.upper() else raw_text

        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            None,
            lambda: self._model.predict_entities(
                normalized, ["author", "book title"], threshold=self._threshold
            ),
        )

        authors: list[tuple[str, float]] = []
        titles: list[tuple[str, float]] = []
        seen_authors: set[str] = set()
        seen_titles: set[str] = set()

        for entity in entities:
            name = entity["text"].strip()
            height = _entity_height(entity, spans)
            if entity["label"] == "author" and name.lower() not in seen_authors:
                seen_authors.add(name.lower())
                authors.append((name, height))
            elif entity["label"] == "book title" and name.lower() not in seen_titles:
                seen_titles.add(name.lower())
                titles.append((name, height))

        authors.sort(key=lambda x: x[1], reverse=True)
        titles.sort(key=lambda x: x[1], reverse=True)

        return NlpAnalysis(
            potential_authors=[a for a, _ in authors],
            potential_titles=[t for t, _ in titles],
        )
