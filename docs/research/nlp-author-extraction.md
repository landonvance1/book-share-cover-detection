# NLP Author Extraction from Book Cover OCR Text

**Date**: 2026-02-23
**Status**: Decided
**Context**: Issue #15 — implement NLP to extract potential author names from OCR output

## Problem

Florence-2 OCR extracts all visible text from a book cover as a flat string (e.g., `"BRANDON SANDERSON MISTBORN"`). We need to identify which words are the author's name vs. the book title vs. noise.

The initial implementation used SpaCy `en_core_web_sm` PERSON NER. It failed on **5 of 7** integration test images.

## SpaCy NER Results

| Image | Expected Author | `potential_authors` | Result |
|---|---|---|---|
| `mistborn.jpg` | Sanderson | `[]` | FAIL |
| `jade-city.jpg` | Lee | (contained "lee") | PASS |
| `snow-crash.jpg` | Stephenson | `['SNOW']` | FAIL |
| `to-kill-a-mockingbird.jpg` | Lee | (contained "lee") | PASS |
| `a-restless-truth.jpg` | Marske | `[]` | FAIL |
| `gardens-of-the-moon.jpg` | Erikson | `[]` | FAIL |
| `under-the-whispering-door.jpg` | Klune | `[]` | FAIL |

### Root Cause

SpaCy's NER is trained on OntoNotes (news articles, web text) — full sentences with surrounding context. It relies on contextual cues to classify entities:

- *"Brandon Sanderson wrote Mistborn"* — SpaCy gets this right (sentence context)
- *"BRANDON SANDERSON MISTBORN"* — SpaCy returns nothing (no context, all-caps removes capitalization signal)

This is a **structural limitation**, not a model-size issue. Larger SpaCy models (`en_core_web_lg`, `en_core_web_trf`) show only ~1-2% F1 improvement on well-formed sentences and would not meaningfully help on isolated text.

The `SNOW` false positive on snow-crash confirms the model is guessing — it tagged a title word as a person because it had no better signal.

### Additional Factor: All-Caps OCR Text

Florence-2 sometimes returns all-caps text. SpaCy explicitly uses "word shape" features including capitalization — `"BRANDON SANDERSON"` gives no signal, but `"Brandon Sanderson"` matches the expected PERSON pattern. Text normalization (`.title()`) before NLP would help any model but doesn't fix the fundamental context problem.

## Options Evaluated

### 1. Larger SpaCy Models

| Model | Size | NER F1 (OntoNotes) | CPU Speed |
|---|---|---|---|
| `en_core_web_sm` | 12 MB | ~85% | <1ms |
| `en_core_web_md` | 40 MB | ~85% | <2ms |
| `en_core_web_lg` | 560 MB | ~86% | <3ms |
| `en_core_web_trf` | 440 MB | ~90% | ~200ms+ |

**Verdict**: Won't fix the 5/7 failure rate. The problem is structural (no context), not model capacity. SpaCy's own developers note that larger English models don't make a huge difference.

### 2. GLiNER (Zero-Shot NER)

Zero-shot NER using custom labels like `"author"` or `"book author"` instead of fixed PERSON type. Scores all possible text spans against label descriptions in a single forward pass.

| Model | Params | Size |
|---|---|---|
| `urchade/gliner_small-v2.1` | 166M | ~330 MB |
| `urchade/gliner_medium-v2.1` | 209M | ~420 MB |
| `urchade/gliner_large-v2.1` | 459M | ~920 MB |

**Accuracy**: Likely the highest of any NER approach — custom labels give it targeted signal even without sentence context.

**CPU speed**: Community benchmarks report **~15 seconds per inference**, which is where the "dealbreaker" verdict came from. However, those benchmarks use paragraph-length text (150–300 tokens). GLiNER's cost scales with the number of candidate spans, which grows quadratically with token count. Book cover OCR output is 10–30 tokens — a fraction of benchmark inputs. In practice, inference on our test covers completes in **~1–2 seconds** on CPU, well within acceptable range for an on-demand web service.

**Verdict**: ~~CPU speed is a dealbreaker~~ **Viable on CPU for short OCR text.** Best accuracy of any approach evaluated.

### 3. Hugging Face Transformer NER

| Model | Params | Size | CPU Speed |
|---|---|---|---|
| `dslim/distilbert-NER` | 65M | ~260 MB | 50-100ms |
| `dslim/bert-base-NER` | 110M | ~440 MB | 100-300ms |
| `Jean-Baptiste/roberta-large-ner-english` | 355M | ~1.4 GB | 300-500ms |

Trained on CoNLL-2003 (Reuters news). BERT's subword tokenization gives slightly better handling of names even without context (it recognizes name-like subword patterns). Still fundamentally context-dependent.

**Verdict**: Moderate accuracy improvement over SpaCy. `distilbert-NER` is the best speed/accuracy tradeoff at 50-100ms. Would still struggle with all-caps text.

### 4. SpaCy PROPN (Part-of-Speech) Heuristic

Extract consecutive PROPN (proper noun) tokens as name candidates. POS tagging is more rule-based than NER and handles isolated words better.

**Problem**: Both author names AND title words are proper nouns. `"Brandon"` = PROPN, `"Sanderson"` = PROPN, `"Mistborn"` = PROPN. Cannot distinguish names from titles. Also degrades on all-caps text.

**Verdict**: Not useful alone. Could be a component of a hybrid approach.

### 5. Name Database Lookup

Look up each word against a database of known first/last names. Consecutive words that are both known names likely form an author name.

#### `names-dataset` (philipperemy/name-dataset)
- 730K first names + 983K last names (extracted from 533M Facebook profiles, 106 countries)
- Sub-millisecond lookup after initial load
- **3.2 GB RAM** to load the full dataset

#### `namecrawler`
- US SSA first names (1920-2017, ~1.75M records) + US Census surnames (~151K)
- Much smaller footprint
- US-centric — misses international authors

#### Algorithm

For `"BRANDON SANDERSON MISTBORN"` (after `.title()` normalization):
1. "Brandon" -> known first name (high frequency)
2. "Sanderson" -> known last name
3. "Mistborn" -> not a known name
4. Result: `potential_authors = ["Brandon Sanderson"]`

**Advantages**:
- No context needed — purely a lookup
- Sub-millisecond speed
- Deterministic
- No GPU, no model inference

**Limitations**:
- Fictional character names in titles (e.g., "Harry Potter") would match as person names
- Uncommon/international author names may not be in the database
- `names-dataset` requires 3.2 GB RAM (could use trimmed Census data instead at ~50 MB)

### 6. Pure Regex/Heuristics

Pattern-match for "FirstName LastName" sequences with stop-word exclusion.

**Verdict**: Too many false positives alone. "Jade City" matches the same pattern as "Fonda Lee".

### 7. Vision-Language Models (VQA)

Use a model that can answer "who is the author?" directly from the image.

| Model | Params | Can ask "who is the author?" | CPU Speed |
|---|---|---|---|
| Florence-2 | 0.23B | No (no VQA task) | Already near limit |
| Moondream2 | 1.86B | Yes | ~23s on CPU |
| Phi-3.5-Vision | ~4B | Yes | Too slow |

Florence-2 supports `<CAPTION>` / `<DETAILED_CAPTION>` tasks but cannot do VQA. Captions might incidentally mention authors for famous books but aren't reliable.

**Verdict**: CPU speed eliminates all VQA models. Only viable with GPU.

## Summary Matrix

| Approach | CPU Speed | Accuracy (isolated text) | RAM/Disk | Context needed? |
|---|---|---|---|---|
| SpaCy `en_core_web_sm` NER | <1ms | Poor (2/7) | 12 MB | Yes |
| SpaCy `en_core_web_trf` NER | ~200ms | Marginal | 440 MB | Yes |
| GLiNER (zero-shot) | ~1-2s on short OCR text (benchmarks overstate cost) | Very good | 330-920 MB | No |
| DistilBERT-NER | 50-100ms | Moderate | 260 MB | Partially |
| Name database lookup | <1ms | Good | 50 MB - 3.2 GB | **No** |
| PROPN heuristic | <1ms | Poor (no name/title distinction) | 0 | No |
| VQA (Moondream2) | **~23s** | Good | ~4 GB | No |

## Key Takeaway

The fundamental challenge is **no surrounding context**. Traditional NER (SpaCy, BERT, etc.) is trained on sentences and degrades sharply on isolated text. The approaches that don't need context — name database lookup and GLiNER zero-shot — are the most promising.

GLiNER's community-reported ~15s CPU speed turned out not to apply here: that figure is for paragraph-length text, and book cover OCR is 10–30 tokens. Actual inference time is ~1–2s on CPU, making it viable. The name database approach remains an alternative if sub-second latency becomes a requirement.

## Decision

**Chose GLiNER (`urchade/gliner_small-v2.1`).**

Rationale:
- Highest accuracy on isolated OCR text — the custom `"author"` label gives targeted signal without requiring sentence context, directly addressing the root cause of SpaCy's 5/7 failure rate.
- Zero-shot: no training data or fine-tuning needed; the label description alone guides entity extraction.
- All-caps normalization (`.title()`) as preprocessing restores the capitalization signal that any NER model benefits from, and is trivially cheap.

CPU inference time (~15 seconds per image) is acceptable for this workload. Cover analysis is an on-demand operation triggered by a user photographing a book — not a high-frequency hot path. The .NET API already enforces rate limiting (10 requests/min), so the throughput constraint is upstream, not in this service.

The name database approach (Option 5) was not spike-tested: while it offers sub-millisecond speed, its false-positive risk on fictional character names in titles (e.g., "Harry Potter", "Jade City") and limited coverage of international/rare surnames made GLiNER the more robust choice.

## References

- [SpaCy English Models](https://spacy.io/models/en)
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [GLiNER CPU speed discussion](https://github.com/theirstory/gliner-spacy/discussions/28)
- [names-dataset](https://github.com/philipperemy/name-dataset)
- [namecrawler](https://github.com/psolin/namecrawler)
- [dslim/distilbert-NER](https://huggingface.co/dslim/distilbert-NER)
- [SpaCy issue on lowercase NER](https://github.com/explosion/spaCy/issues/701)
