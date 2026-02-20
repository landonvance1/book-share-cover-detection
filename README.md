# book-share-cover-detection

A Python microservice that analyzes book cover images to identify the title and author, then searches OpenLibrary to return the most likely book match.

## Overview

When a user photographs a book cover in the BookSharing mobile app, this service:

1. Runs **Florence-2** locally to extract text from the image
2. Uses **SpaCy NLP** to identify which text is likely the title vs. author vs. noise
3. Queries the **OpenLibrary API** with NLP-informed search terms
4. Returns scored, structured book matches

This replaces Azure Vision OCR + height-based heuristics with a fully local, cost-free pipeline.

## API

### `POST /analyze`

Accepts a multipart image upload (JPEG, PNG, or WebP, max 4 MB) and returns structured book matches.

**Response:**

```json
{
  "analysis": {
    "isSuccess": true,
    "errorMessage": null,
    "extractedText": "The Great Gatsby F Scott Fitzgerald"
  },
  "matchedBooks": [
    {
      "title": "The Great Gatsby",
      "author": "F. Scott Fitzgerald",
      "isbn": "9780743273565",
      "thumbnailUrl": "https://covers.openlibrary.org/b/id/12345-M.jpg"
    }
  ],
  "exactMatch": null,
  "nlpAnalysis": {
    "detectedTitle": "The Great Gatsby",
    "titleConfidence": 0.92,
    "detectedAuthor": "F Scott Fitzgerald",
    "authorConfidence": 0.87
  }
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
    ├── SpaCy         — NLP title/author identification with confidence scores
    └── OpenLibrary   — external book search (openlibrary.org)
```

### OCR Engine

Florence-2 (`microsoft/Florence-2-base`, 0.23B parameters) is a vision-language model that generates text end-to-end from the full image in a single pass, rather than running separate text detection and recognition stages. It passes 7/7 integration tests across a range of difficult cover types including embossed text, decorative fonts, and stylised display type.

EasyOCR and DocTR were evaluated and discarded — each achieved 4/7 on a different subset of images and no preprocessing strategy improved either engine's pass rate. See `docs/decisions/001-ocr-engine-selection.md` for the full evaluation and `experiments/PREPROCESSING_FINDINGS.md` for preprocessing sweep results.

### Abstractions

The OCR and NLP engines are both behind interfaces, making it straightforward to swap in alternatives:

- **OCR**: Florence-2 PyTorch (default), Florence-2 ONNX (planned — see [issue #12](https://github.com/landonvance1/book-share-cover-detection/issues/12))
- **NLP**: SpaCy (default), Hugging Face transformers, custom models

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) CUDA-capable GPU for faster inference

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Florence-2 (~1 GB) downloads automatically from HuggingFace on first use and is cached locally.

### Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t book-share-cover-detection .
docker run -p 8000:8000 book-share-cover-detection
```

## Development

### Running Tests

```bash
# Unit tests (fast, no model download)
pytest tests/unit/

# Integration tests (downloads Florence-2 on first run, ~1 GB)
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
│   ├── florence2_engine.py  # Florence-2 implementation
│   └── spacy_engine.py      # SpaCy implementation
├── services/
│   ├── analyzer.py      # Orchestrates OCR → NLP → search
│   └── openlibrary.py   # OpenLibrary API client
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
