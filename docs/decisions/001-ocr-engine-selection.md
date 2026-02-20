# 001 — OCR Engine Selection

**Date:** 2026-02-20
**Status:** Accepted

---

## Context

This service needs to extract text from book cover images. The parent project
(book-share-api) used Azure Computer Vision's cloud Read API — a key motivation
for building this microservice was eliminating that dependency. We needed a local
OCR engine that could reliably read the range of typographic styles found on
real book covers: standard typeset, decorative serifs, embossed/low-contrast
text, and fragmented cover art.

Three engines were evaluated against a fixed set of 7 integration test images
chosen to represent difficult real-world cases:

| Image | Challenge |
|---|---|
| `a-restless-truth.jpg` | Decorative serif font |
| `gardens-of-the-moon.jpg` | Standard typeset (control) |
| `jade-city.jpg` | Embossed, jade-on-jade (low contrast) |
| `mistborn.jpg` | Stylised display font |
| `snow-crash.jpg` | Title rendered as glitchy cover art |
| `to-kill-a-mockingbird.jpg` | Standard typeset (control) |
| `under-the-whispering-door.jpg` | Decorative, multi-word title |

---

## Options Considered

### EasyOCR

A two-stage pipeline: CRAFT text detector → CRNN recogniser.

**Result: 4/7**

Failures:
- `mistborn.jpg` — CRAFT reads "BRANDOU"/"HNDERSON" regardless of preprocessing
- `snow-crash.jpg` — title is cover art, not typeset; unreadable by any detector
- `under-the-whispering-door.jpg` — CRAFT segments "WHISPERING" as two boxes;
  the joined output string never contains the contiguous word

### DocTR

A two-stage pipeline: FAST text detector → parseq recogniser.

**Result: 4/7** (different 4 from EasyOCR)

Failures:
- `a-restless-truth.jpg` — parseq misreads "TRUTH" as "RUTI"/"RUTH"
- `jade-city.jpg` — FAST cannot detect jade-on-jade embossed text at any contrast setting
- `snow-crash.jpg` — same art-vs-text issue as EasyOCR

The two engines are complementary: between them they cover 6/7 images. A
dual-engine strategy (run both, merge word sets) was briefly considered but
would double inference latency and still not crack snow-crash.

### Preprocessing to improve EasyOCR / DocTR

CLAHE, bilateral filter, NLM denoising, unsharp masking, and upscaling were
swept across all 7 images and both engines. Full results in
`experiments/PREPROCESSING_FINDINGS.md`.

**Result: no improvement in pass count for either engine.**

All failures were caused by detector or recogniser architecture limitations,
not image quality. The preprocessing that helped individual words on one image
reliably regressed other words on the same or different images.

Key findings:
- CLAHE actively harms DocTR (fragments word boundaries)
- NLM h=5 fixes "BRANDOU"→"BRANDON" on mistborn but simultaneously degrades
  "MISTBORN"→"MSTBORN" — no net gain
- Upscaling and sharpening helped nothing

### Florence-2 (Microsoft)

A vision-language model (0.23B parameters) that generates text end-to-end from
the full image in a single forward pass, rather than running separate detection
and recognition stages. Uses the `<OCR_WITH_REGION>` task prompt to return text
labels with quad bounding boxes.

**Result: 7/7**, including all three cases that defeated both two-stage engines:
- `jade-city.jpg` — the VLM "sees" the embossed text that CRAFT and FAST cannot detect
- `mistborn.jpg` — reads the stylised font correctly
- `under-the-whispering-door.jpg` — no word segmentation; generates full words directly
- `snow-crash.jpg` — reads "SNOW" and "CRASH" from the cover art (unexpected)

---

## Decision

**Use Florence-2 (`microsoft/Florence-2-base`) as the sole OCR engine.**

EasyOCR and DocTR were removed. Florence-2 is wired as the production engine
in `app/main.py` and is the only engine with integration tests.

Greedy decoding (`num_beams=1`) is the default — beam search (`num_beams=3`)
adds ~3× generation time with no measurable accuracy improvement for OCR of
printed text.

---

## Consequences

### Positive
- 7/7 integration test pass rate vs 4/7 for the previous engines
- Single engine, no dual-engine complexity
- No cloud dependency, no API keys

### Negative / Risks

**`trust_remote_code=True`**
Florence-2 is not a built-in transformers architecture. Loading it executes
Python downloaded from the HuggingFace Hub (`microsoft/Florence-2-base`).
This is mitigated by the snapshot being pinned by hash after first download
(subsequent runs use the local cache), but it remains a supply-chain concern
for production deployment.

The resolution is to export Florence-2 to ONNX, which produces a
self-contained graph with no custom Python at inference time. Tracked in
[issue #12](https://github.com/landonvance1/book-share-cover-detection/issues/12).

**`flash_attn` monkey-patch**
Florence-2's model file unconditionally imports `flash_attn` (CUDA-only). A
module-level patch to `transformers.dynamic_module_utils.get_imports` strips
this from the import check so the engine loads on CPU. This patch is in
`app/engines/florence2_engine.py` and goes away once the ONNX path lands.

**transformers version pin**
`transformers>=4.44.0` breaks `Florence2LanguageConfig` (changed
`PretrainedConfig.__getattribute__` behaviour). Pinned to `<4.44.0`. This
constraint is also resolved by the ONNX export — the ONNX path requires only
`onnxruntime`, not `transformers` at inference time.

**Inference speed**
~9s/image on CPU with greedy decoding. Acceptable for an experimental baseline.
The ONNX path is expected to yield a 3–5× improvement. GPU deployment would
give ~20× improvement over CPU.
