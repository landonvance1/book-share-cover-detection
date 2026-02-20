# Preprocessing Sweep — Findings

## Summary

Tested CLAHE, bilateral filter, NLM denoising, unsharp masking, and upscaling on
all 7 integration test images against both EasyOCR and DocTR.

**Baseline scores:**
- EasyOCR baseline: 4/7 pass (fails mistborn, snow-crash, under-the-whispering-door)
- DocTR baseline:   4/7 pass (fails a-restless-truth, jade-city, snow-crash)

**Best preprocessing scores:**
- EasyOCR best: still 4/7 (no config improved overall pass count)
- DocTR best: still 4/7 (no config improved overall pass count)

---

## Per-Image Analysis

### 🟢 Consistent passes (both engines, all configs)
- `gardens-of-the-moon.jpg`
- `to-kill-a-mockingbird.jpg`

### 🟡 Engine-dependent (one engine passes baseline, the other doesn't)

| Image | EasyOCR | DocTR |
|---|---|---|
| `a-restless-truth.jpg` | ✓ (all configs) | ✗ ("TRUTH" → "RUTI/RUTH") |
| `jade-city.jpg` | ✓ (all configs) | ✗ (JADE CITY never detected) |
| `mistborn.jpg` | ✗ (BRANDON/SANDERSON misread) | ✓ (all configs) |
| `under-the-whispering-door.jpg` | ✗ ("WHISPERING" split) | ✓ (all configs) |

### 🔴 Both engines fail on all configs
- `snow-crash.jpg`: "snow" and "crash" are never found by either engine.

---

## Root Cause Analysis

### snow-crash.jpg (both fail)
The Snow Crash cover renders the title as fragmented/glitchy digital art — it is
cover artwork, not typeset text. Neither CRAFT (EasyOCR) nor FAST (DocTR) text
detectors can find it. No preprocessing can fix a detection model that sees art,
not glyphs. **This is a test expectation problem**: the author name (NEAL
STEPHENSON) is consistently readable by both engines; the title keywords ("snow",
"crash") should be removed from the test assertions.

### jade-city.jpg (DocTR only)
"JADE CITY" is embossed jade-on-jade (similar hue and luminance), making it
invisible to DocTR's FAST text detection model. EasyOCR's CRAFT detector finds
it. No CLAHE (up to clip=8.0), sharpening, or upscaling (up to 2x) helped —
the text/background luminance contrast is simply too low for FAST to detect.
This is a detector architecture difference, not an image quality issue.

### a-restless-truth.jpg (DocTR only)
"TRUTH" is consistently misread as "RUTI" (baseline) or "RUTH" (with denoising).
The decorative serif on this word confuses DocTR's parseq recognition model.
Mild denoising (NLM h=5, bilateral d=9 σ=75) gets it to "RUTH" — closer, but
not a fix. CLAHE fragments the word further ("RUT", "HERTELE").

### mistborn.jpg (EasyOCR only)
"BRANDON" → "BRANDOU" (baseline) and "SANDERSON" → "HNDERSON" consistently.
NLM h=5 fixes BRANDON but then MISTBORN drops to "MSTBORN" and SANDERSON
remains wrong. There is no NLM level that simultaneously fixes all three words.
CLAHE makes recognition worse. DocTR reads this cover perfectly.

### under-the-whispering-door.jpg (EasyOCR only)
EasyOCR's CRAFT detector segments "WHISPERING" as two bounding boxes ("WHispe" +
"Ringl"), and the joined `result.text` string never contains the contiguous
substring "whispering". DocTR's parseq model reads the whole word correctly.
No preprocessing changes the segmentation behavior.

---

## What Preprocessing Does and Doesn't Help

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **DocTR**: Consistently harmful. Even low clip (1.0) breaks word boundaries
  in under-the-whispering-door ("WH HISPERING"). Higher clips fragment more text.
- **EasyOCR**: No improvement on any failing image. Marginally different errors.
- **Verdict**: Do not use CLAHE in production. It degrades DocTR and does not
  help EasyOCR.

### Bilateral Filter
- **DocTR**: Neutral. Doesn't hurt passing images. Gets a-restless-truth "RUTH"
  (vs "RUTI" baseline) but doesn't fix it. jade-city unaffected.
- **EasyOCR**: Neutral. Gets "ANDERSON" on mistborn (vs "HNDERSON") — closer
  but still missing the leading "S".
- **Verdict**: Very mild bilateral (d=5, σ=50) is safe to apply without risk of
  regression on either engine, but doesn't improve pass rates.

### NLM (Non-Local Means)
- **DocTR**: Neutral at low values (h=3-5). Gets "RUTH" on a-restless-truth.
  Higher values (h=10+) start introducing noise in passing tests.
- **EasyOCR**: NLM h=5 is the best single finding — fixes "BRANDOU" → "BRANDON"
  on mistborn — but simultaneously degrades "MISTBORN" → "MSTBORN". No net
  improvement in pass count.
- **Verdict**: NLM h=3 is safe for DocTR (no regression). EasyOCR NLM has a
  tricky tradeoff at mistborn.

### Sharpening (Unsharp Mask)
- Neither engine shows any improvement from sharpening (sigma 1.0–2.0, amount
  1.0–2.0). EasyOCR produces identical output to baseline on all failing images.
- **Verdict**: Sharpening does not help. Don't apply it.

### Upscaling (LANCZOS4, 1.5x–2.0x)
- No improvement on any failing image for either engine. All failures remain.
- DocTR jade-city: still completely absent even at 2x.
- EasyOCR under-the-whispering-door: same word segmentation errors at all scales.
- **Verdict**: Upscaling doesn't help for these covers. Avoid — it adds latency.

---

## Recommendations

### For production (DocTR as primary engine)
**Apply no preprocessing.** The DocTR baseline is already optimal. CLAHE actively
harms results; other preprocessing is at best neutral.

If a very mild denoising pass is desired for noisy real-world phone photos
(not reflected in our test set), **NLM h=3** is safe for DocTR — it produces
no regressions on any of the 7 test images.

### For EasyOCR
**No preprocessing.** Nothing reliably improves accuracy without introducing
regressions elsewhere.

### For snow-crash test
Update the test assertion to only check for the author name. The title is not
OCR-readable text on this cover edition.

### Bigger wins available elsewhere
The two engines are complementary: between them they cover 6/7 images. A
**dual-engine strategy** (run both, merge word sets) would outperform any
preprocessing-only improvement:

| Image | EasyOCR | DocTR | Merged |
|---|---|---|---|
| a-restless-truth | ✓ | ✗ | ✓ |
| gardens-of-the-moon | ✓ | ✓ | ✓ |
| jade-city | ✓ | ✗ | ✓ |
| mistborn | ✗ | ✓ | ✓ |
| snow-crash | ✗ | ✗ | ✗ (title art) |
| to-kill-a-mockingbird | ✓ | ✓ | ✓ |
| under-the-whispering-door | ✗ | ✓ | ✓ |
