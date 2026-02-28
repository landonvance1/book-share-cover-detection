# book-share-cover-detection

A Python microservice that analyzes book cover images to identify the title and author

## Overview

When a user photographs a book cover in the BookSharing mobile app, this service:

1. Runs **Florence-2** locally to extract text from the image
2. Uses **GLiNER** (zero-shot NER) to identify which text is the author name
4. Returns likely author and title information

This replaces Azure Vision OCR + height-based filtering heuristics with a fully local, cost-free pipeline. Results are sorted by bounding box height so the most visually prominent match appears first.

## API

### `POST /analyze`

Accepts a multipart image upload (JPEG, PNG, or WebP, max 4 MB) and returns structured book matches.

**Response:**

```json
{
  "analysis": {
    "isSuccess": true,
    "errorMessage": null,
  },
  "nlpAnalysis": {
    "potentialAuthors": ["F Scott Fitzgerald"],
    "potentialTitles": ["The Great Gatsby"]
  },
  "ocrResult": {
    "text": "BRANDON SANDERSON MISTBORN",
    "regions": [
      {
        "text": "BRANDON",
        "confidence": 1.0,
        "coordinates": [
          [
            442.55999755859375,
            546.1499633789062
          ],
          [
            1490.8800048828125,
            523.0499877929688
          ],
          [
            1492.7999267578125,
            948.75
          ],
          [
            446.3999938964844,
            965.25
          ]
        ]
      },
      {
        "text": "SANDERSON",
        "confidence": 1.0,
        "coordinates": [
          [
            246.72000122070312,
            975.1499633789062
          ],
          [
            1730.8800048828125,
            958.6499633789062
          ],
          [
            1732.7999267578125,
            1440.449951171875
          ],
          [
            248.63999938964844,
            1456.949951171875
          ]
        ]
      },
      {
        "text": "MISTBORN",
        "confidence": 1.0,
        "coordinates": [
          [
            331.1999816894531,
            2529.449951171875
          ],
          [
            1684.7999267578125,
            2506.349853515625
          ],
          [
            1686.719970703125,
            3024.449951171875
          ],
          [
            333.1199951171875,
            3057.449951171875
          ]
        ]
      }
  ]}
}
```

### `GET /health`

Returns service health status for container orchestration.

## Architecture

```
book-share-api (.NET 8)
    │  POST /books/analyze/cover
    │
    ▼
book-share-cover-detection (this service)
    │
    ├── Florence-2    — local vision-language model OCR (no cloud dependency)
    ├── GLiNER        — zero-shot NER for author name extraction
```

### OCR Engine

Florence-2 (`microsoft/Florence-2-base`, 0.23B parameters) is a vision-language model that generates text end-to-end from the full image in a single pass, rather than running separate text detection and recognition stages. It passes 7/7 integration tests across a range of difficult cover types including embossed text, decorative fonts, and stylised display type.

Two implementations are available:

- **Florence-2 PyTorch** (`Florence2OcrEngine`) — default, downloads the model from HuggingFace on first use (~1 GB). Requires `trust_remote_code=True` for the model forward pass.
- **Florence-2 ONNX** (`Florence2OnnxEngine`) — uses pre-exported ONNX models from `onnx-community/Florence-2-base-ft` (q4 quantized by default). Runs on ONNX Runtime without `trust_remote_code` for model computation (still needed for the tokenizer/post-processor). Expected 3-5x speedup on CPU.

#### ONNX Engine Setup

Download the pre-exported ONNX model:

```bash
pip install huggingface-hub
huggingface-cli download onnx-community/Florence-2-base-ft --local-dir florence2-onnx
```

The `florence2-onnx/` directory is excluded from git via `.gitignore`. The download includes all quantization variants; the engine defaults to `q4`.

Set ONNX_NUM_THREADS in .env file (unless defauly of 4 is sufficient)

EasyOCR and DocTR were evaluated and discarded — each achieved 4/7 on a different subset of images and no preprocessing strategy improved either engine's pass rate. See `docs/decisions/001-ocr-engine-selection.md` for the full evaluation and `experiments/PREPROCESSING_FINDINGS.md` for preprocessing sweep results.

### NLP Engine

GLiNER (`urchade/gliner_small-v2.1`, 166M parameters) is a zero-shot NER model that uses custom labels `"author"` and `"book title"` to score all possible text spans in a single forward pass. Unlike SpaCy or BERT-based NER, it does not require sentence-level context — it works directly on isolated OCR text like `"BRANDON SANDERSON MISTBORN"`.

Results are sorted by bounding box height (derived from Florence-2 region coordinates), so the most visually prominent author and title appear at index 0. All OCR regions are passed to the model — no text is discarded before inference.

The model (~330 MB) downloads automatically from HuggingFace on first use and is cached locally. CPU inference takes approximately 15 seconds per image, which is acceptable for this on-demand workload.

All-caps OCR text (a common Florence-2 output pattern) is normalized via `.title()` before inference to restore the capitalization signal that GLiNER uses for name recognition.

### Abstractions

The OCR and NLP engines are both behind interfaces, making it straightforward to swap in alternatives:

- **OCR**: Florence-2 PyTorch (default), Florence-2 ONNX (see ONNX Engine Setup above)
- **NLP**: GLiNER (default), Hugging Face transformers, custom models

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) CUDA-capable GPU for faster inference

### Installation

```bash
pip install -r requirements.txt
```

Florence-2 (~1 GB) and GLiNER (~330 MB) both download automatically from HuggingFace on first use and are cached locally.

### Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

#### Quick Start

```bash
docker build -t book-share-cover-detection .
docker run -p 8000:8000 book-share-cover-detection
```

Or with Docker Compose:

```bash
docker compose up --build
```

#### Build Options

By default, the image uses the **ONNX engine** (Florence-2 ONNX with q4 quantization). To build with the PyTorch engine instead:

```bash
docker build --build-arg OCR_ENGINE=pytorch -t book-share-cover-detection .
```

**Engine comparison:**
- **ONNX (default)**: Pre-quantized, ~3-5x faster on CPU, smaller memory footprint (~2 GB image)
- **PyTorch**: Slower on CPU but supports GPU acceleration, larger image (~3 GB)

#### Environment Variables

Copy `.env.example` to `.env` and adjust values:

```bash
cp .env.example .env
```

Key settings:
- `OCR_ENGINE`: Must match the build ARG (`onnx` or `pytorch`)
- `ONNX_MODEL_PATH`: Path to the ONNX model directory (default: `/opt/hf_cache/florence2-onnx`)
- `ONNX_PROCESSOR_NAME`: HuggingFace model name for the ONNX processor (default: `microsoft/Florence-2-base-ft`)
- `ONNX_NUM_THREADS`: ONNX Runtime thread count (default: 4)

#### Offline Deployment

The Docker image bakes in all required models (`florence2`, `gliner`, and tokenizer dependencies) during build. Set `HF_HUB_OFFLINE=1` to prevent any runtime network calls—useful for airgapped or unreliable network environments. The Dockerfile does this by default.

## Test App

A minimal browser-based UI for manually testing the `/analyze` endpoint with a live camera. Useful for testing with real book covers on a phone without needing the full mobile app.

### Setup

```bash
# 1. Generate a self-signed TLS cert (one-time — cert files are gitignored)
openssl req -x509 -newkey rsa:2048 \
  -keyout test_app/key.pem -out test_app/cert.pem \
  -days 365 -nodes -subj '/CN=localhost'

# 2. Enable the test app
echo "ENABLE_TEST_APP=true" >> .env

# 3. Start uvicorn with TLS
uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile test_app/key.pem --ssl-certfile test_app/cert.pem
```

Then visit `https://<your-machine-ip>:8000/test/` on a phone connected to the same WiFi network. Accept the browser security warning once.

> **Why HTTPS?** Browsers block camera access (`getUserMedia`) on non-localhost origins without a valid TLS connection.

### Chrome on Android

Chrome may block `getUserMedia` for self-signed certs even after accepting the security warning. Fix it by adding your origin as a trusted secure origin:

1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. Add `https://<your-machine-ip>:8000`
3. Enable the flag and relaunch

Safari on iOS works without this step.

## Development

### Running Tests

```bash
# Unit tests (fast, no model download)
pytest tests/unit/

# Integration tests — PyTorch engine (downloads Florence-2 on first run, ~1 GB)
pytest tests/integration/ -v -k "florence2 and not onnx"

# Integration tests — ONNX engine (requires model download, see ONNX Engine Setup)
pytest tests/integration/ -v -k "onnx"

# All integration tests
pytest tests/integration/ -v
```

### Project Structure

```
app/
├── main.py              # FastAPI app and routes
├── interfaces/
│   ├── ocr.py           # OCR abstract base class
│   └── nlp.py           # NLP abstract base class
├── engines/
│   ├── florence2_engine.py       # Florence-2 PyTorch implementation
│   ├── florence2_onnx_engine.py  # Florence-2 ONNX implementation
│   ├── gliner_engine.py     # GLiNER zero-shot NER implementation
│   └── spacy_engine.py      # SpaCy implementation (unused stub)
├── services/
│   ├── analyzer.py      # Orchestrates OCR → NLP → search
docs/
└── decisions/           # Architecture Decision Records
    └── 001-ocr-engine-selection.md
experiments/
└── PREPROCESSING_FINDINGS.md  # EasyOCR/DocTR preprocessing sweep results
tests/
├── unit/
└── integration/
    └── images/          # Sample book cover images for testing
```

## Related Projects

- [book-share-api](https://github.com/landonvance1/book-share-api) — .NET 8 backend that calls this service
