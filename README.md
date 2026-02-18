# book-share-cover-detection

A Python microservice that analyzes book cover images to identify the title and author, then searches OpenLibrary to return the most likely book match.

## Overview

When a user photographs a book cover in the BookSharing mobile app, this service:

1. Runs **EasyOCR** locally to extract text from the image
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
    ├── EasyOCR       — local text extraction (no cloud dependency)
    ├── SpaCy         — NLP title/author identification with confidence scores
    └── OpenLibrary   — external book search (openlibrary.org)
```

### Abstractions

The OCR and NLP engines are both behind interfaces, making it straightforward to swap in alternatives:

- **OCR**: EasyOCR (default), Tesseract, PaddleOCR, cloud APIs
- **NLP**: SpaCy (default), Hugging Face transformers, custom models

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) CUDA-capable GPU for faster OCR

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

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
pytest
```

Tests cover:
- OCR output parsing (mocked engine)
- NLP analysis logic (mocked engine)
- OpenLibrary search and scoring
- Integration tests with real book cover images

### Project Structure

```
app/
├── main.py              # FastAPI app and routes
├── interfaces/
│   ├── ocr.py           # OCR abstract base class
│   └── nlp.py           # NLP abstract base class
├── engines/
│   ├── easyocr.py       # EasyOCR implementation
│   └── spacy.py         # SpaCy implementation
├── services/
│   ├── analyzer.py      # Orchestrates OCR → NLP → search
│   └── openlibrary.py   # OpenLibrary API client
tests/
├── unit/
└── integration/
    └── images/          # Sample book cover images for testing
```

## Related Projects

- [book-share-api](https://github.com/landonvance1/book-share-api) — .NET 8 backend that calls this service
