# 002 — NLP Engine Selection

**Date:** 2026-02-23
**Status:** Accepted

---

## Context

Florence-2 OCR returns all visible text as a flat string (e.g., `"BRANDON SANDERSON MISTBORN"`). To query OpenLibrary effectively, the service needs to identify which tokens are the author's name vs. the book title vs. noise.

The parent project (book-share-api) used height-based heuristics — retaining only text whose bounding box height was ≥20% of the tallest text on the cover. That approach is coupled to Azure Vision's bounding box output and discards useful information. The goal here was a proper NLP-based approach with a confidence signal.

The full evaluation is in `docs/research/nlp-author-extraction.md`. A summary of options considered is below.

---

## Options Considered

### SpaCy NER (`en_core_web_sm`)

**Result: 2/7**

SpaCy's NER is trained on OntoNotes (news articles with full sentence context). Book cover OCR text has no surrounding context, so the model has no signal to classify entities. All-caps OCR output removes the capitalisation cues the model relies on.

Larger SpaCy models (`en_core_web_lg`, `en_core_web_trf`) were considered and ruled out — SpaCy's own documentation notes that English model size differences are marginal, and the root cause is structural (no context), not capacity.

### Hugging Face Transformer NER (`dslim/distilbert-NER`)

50–100ms on CPU, trained on CoNLL-2003 (Reuters news). BERT subword tokenisation gives slightly better handling of unfamiliar names than SpaCy, but the model is still context-dependent and would still struggle with all-caps isolated text.

Not tested against the integration images — ruled out on the same structural grounds as SpaCy.

### Name Database Lookup

Sub-millisecond lookup using a corpus of known first/last names (e.g., US Census surnames + SSA first names). No model, no context required, deterministic.

**Main risk:** fictional character names in titles (e.g., "Harry Potter", "Jade City") match the same name patterns as real authors. Mitigation would require combining with a second signal.

Not spike-tested. Remains a viable alternative if sub-millisecond latency becomes a hard requirement.

### GLiNER (`urchade/gliner_small-v2.1`)

Zero-shot NER that scores all possible text spans against a custom label description (`"author"`) in a single forward pass. No context required — the label itself provides the signal.

Community benchmarks reported ~15s CPU inference, which initially led to this option being ruled out. On investigation, that figure is for paragraph-length text (~150–300 tokens). Book cover OCR output is 10–30 tokens. Actual inference time on our test covers is **~1–2 seconds on CPU** — well within acceptable range.

**Result: 6/7 clean** (correct primary author extracted), **7/7 inclusive** (correct author present in results at threshold=0.4).

The one difficult case (jade-city) has mixed-case OCR output that prevents all-caps normalisation, causing `FONDA LEE` to be scored below the default threshold. At `threshold=0.4`, GLiNER returns `['FONDA LEE', 'ANN LECKIE', 'SCOTT LYNCH']` — the real author is present, alongside two blurb authors from the cover.

---

## Decision

**Use GLiNER (`urchade/gliner_small-v2.1`) as the NLP engine.**

Key parameters:
- **Confidence threshold: 0.4** (default GLiNER is 0.5; lowered to favour recall over precision — missing the real author is a worse outcome than returning an extra blurb author)
- **All-caps normalisation:** if `text == text.upper()`, apply `.title()` before inference to restore capitalisation signal
- **Label:** `"author"` (single zero-shot label)

The model (~330 MB) downloads automatically from HuggingFace on first use.

---

## Consequences

### Positive
- 7/7 integration test coverage with real OCR input
- No context required — works on isolated OCR text
- Custom `"author"` label outperforms fixed PERSON NER for this task
- CPU inference viable for short OCR text (~1–2s)
- No training data or fine-tuning required

### Negative / Risks

**Blurb author false positives**
Covers with visible reviewer attributions (e.g., `"-ANN LECKIE"`) will produce extra entries in `potential_authors`. This is acceptable — the consumer (OpenLibrary search) uses the list as candidates, and the primary author typically appears first after height-based ranking (not yet implemented, see below).

**No height-based ranking**
`potential_authors` is currently returned in GLiNER extraction order, not ranked by visual prominence. On most covers this is fine (single author returned). On covers like jade-city, the primary author happens to score highest and appears first, but this is not guaranteed. Height-based ranking using `OcrResult.regions` bounding boxes would make the ordering reliable. Not implemented in this iteration.

**All-caps normalisation is partial**
The `.title()` normalisation only triggers when the entire string is all-caps. Mixed-case OCR output with embedded all-caps author names (jade-city pattern) bypasses normalisation. A per-token approach would improve recall.

**GLiNER + transformers 5.x incompatibility**
GLiNER 0.2.x accesses `TokenizersBackend.additional_special_tokens`, which was removed in `transformers` 5.0. `requirements.txt` pins `transformers<5.0.0` to avoid this. Tracked upstream in [GLiNER issue #327](https://github.com/urchade/GLiNER/issues/327).
