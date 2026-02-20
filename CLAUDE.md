# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**book-share-cover-detection** is a Python microservice that analyzes book cover images to identify the book's title and author, then searches OpenLibrary to return the most likely match. It replaces the Azure Computer Vision dependency currently embedded in the [book-share-api](https://github.com/landonvance1/book-share-api) (.NET backend).

### Why This Exists

The parent project (BookSharingWebAPI) currently handles cover analysis inline using:
1. **Azure Vision OCR** — sends image to Azure's cloud Read API, polls for results, gets text + bounding boxes
2. **Height-based filtering** — uses bounding box geometry to keep only visually prominent text (≥20% of max text size), with a "sharpening" retry strategy that drops smaller text tiers when OpenLibrary returns no matches
3. **OpenLibrary search** — searches `openlibrary.org/search.json?q={text}` and scores results by word overlap

This microservice replaces that with:
1. **Florence-2** (Microsoft, local vision-language model) for text extraction
2. **SpaCy NLP** for intelligent title/author identification (replaces the height heuristic)
3. **OpenLibrary search** with NLP-informed queries (same external API)

### Design Goals

- **Eliminate Azure dependency** — no cloud OCR costs, no API keys needed for core functionality
- **Better accuracy** — NLP-based title/author detection should outperform height-based heuristics
- **Separation of concerns** — isolate the compute-heavy OCR/NLP pipeline from the lightweight .NET API
- **Independent scaling** — the OCR/NLP workload can scale separately from the API
- **Abstraction** — OCR and NLP implementations behind interfaces so alternatives (Tesseract, Hugging Face, etc.) can be swapped in

## Architecture

### Service Boundaries

```
Mobile App (React Native)
    │
    ▼
book-share-api (.NET 8, Minimal APIs)
    │  POST /books/analyze/cover
    │  Receives image from mobile app
    │  Forwards to microservice
    │  Merges results with local DB books
    │  Returns response to mobile app
    │
    ▼
book-share-cover-detection (this repo, Python)
    │  Receives image
    │  OCR → NLP → OpenLibrary search
    │  Returns structured analysis result
    │
    ├── Florence-2 (local vision-language model OCR)
    ├── SpaCy (local NLP engine)
    └── OpenLibrary API (external book search)
```

### Interface Abstractions

Two core abstractions must be defined as interfaces/protocols:

**OCR Interface** — Takes an image, returns extracted text with positional/confidence metadata.
- Default implementation: Florence-2 (`microsoft/Florence-2-base`)
- Future alternatives: Florence-2 ONNX export (see issue #12), Tesseract, PaddleOCR, cloud APIs

**NLP Interface** — Takes raw OCR output, returns structured analysis with confidence scores identifying which text is likely the title vs. author vs. noise.
- Default implementation: SpaCy
- Future alternatives: Hugging Face transformers, custom models

### Expected Response Contract

The .NET API currently expects this shape from cover analysis (adapt to match):

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

Key additions over the current .NET response:
- `nlpAnalysis` — structured NLP output with confidence scores (new capability)
- `matchedBooks` uses `BookLookupResult`-style objects (title, author, isbn, thumbnailUrl) rather than full DB entities
- The .NET API will handle merging with its local database — this service only returns OpenLibrary results

### OpenLibrary API Details

The search endpoint used by the current system:
- **URL**: `https://openlibrary.org/search.json?q={query}&limit=11`
- **User-Agent header required**: `Community Bookshare App (landonpvance@gmail.com)`
- Cover thumbnails: `https://covers.openlibrary.org/b/id/{coverId}-M.jpg`
- ISBN lookup: `https://openlibrary.org/isbn/{isbn}.json`
- Rate limits: Be respectful, no official limit but avoid hammering

### Current Scoring Algorithm (for reference)

The .NET API scores OpenLibrary results like this — the new service should aim to improve on it:
1. Build a set of OCR words (lowercased, punctuation stripped, length > 2)
2. For each OpenLibrary result, count how many of its title+author words appear in the OCR set
3. Score = matchCount / bookWordCount
4. Filter: keep results with score ≥ 0.5
5. Sort by score descending

The NLP analysis should enable smarter searching (e.g., search title and author separately rather than dumping all OCR text into one query).

## Parent Project Context

### BookSharingWebAPI (book-share-api)

The .NET 8 backend for a community-based book lending platform. Key facts:

- **Stack**: ASP.NET Core 8.0 Minimal APIs, PostgreSQL, EF Core, JWT auth, SignalR
- **Purpose**: Users share physical books within communities — discover, request, borrow, return
- **Current cover analysis endpoint**: `POST /books/analyze/cover` (multipart image upload, JWT required, rate limited to 10/min)
- **Repo**: https://github.com/landonvance1/book-share-api

The .NET API will be modified to call this microservice instead of Azure Vision directly. The API will:
1. Receive the image from the mobile app
2. Forward it to this microservice
3. Take the microservice response and merge `matchedBooks` with its local database (matching by title+author, assigning local IDs to known books, negative IDs to external-only books)
4. Return the final response to the mobile app

### Mobile App (BookSharingApp)

React Native + Expo mobile app that consumes the API. The cover analysis flow from the user's perspective:
1. User takes a photo of a book cover (or selects from gallery)
2. App uploads image to `POST /books/analyze/cover`
3. App displays matched books for the user to select
4. User picks the correct match to add to their library

### Image Constraints

The mobile app and API enforce these constraints before the image reaches this service:
- **Max file size**: 4 MB
- **Accepted formats**: JPEG, PNG, WebP
- These constraints can be relaxed in this service since we're not bound by Azure Vision limits, but the upstream will send images within these bounds

## Development Guidelines

### Git Standards
- **Commits and PRs**: Use conventional commit syntax: https://www.conventionalcommits.org/en/v1.0.0/

### Technology Choices

- **Language**: Python 3.11+
- **OCR**: Florence-2 (`microsoft/Florence-2-base`, 0.23B params, GPU-optional)
- **NLP**: SpaCy (lightweight, fast, good NER out of the box)
- **HTTP framework**: FastAPI recommended (async, automatic OpenAPI docs, similar developer experience to the .NET Minimal APIs pattern used in the parent project)
- **OpenLibrary client**: httpx or aiohttp for async HTTP

### Key Design Principles

1. **Interface-first design** — Define OCR and NLP as abstract base classes/protocols before implementing. This is a core requirement, not optional.
2. **Confidence scoring** — The NLP layer should output confidence scores for title and author detection. This is a key improvement over the current height-based heuristic.
3. **Stateless** — This service has no database. It receives an image, processes it, and returns results.
4. **Container-ready** — Should run in Docker alongside the .NET API and PostgreSQL. The .NET project uses Docker Compose already.
5. **Health check endpoint** — Include a `/health` endpoint for container orchestration.

### Testing Strategy

- Unit tests for OCR output parsing and NLP analysis logic (mock the OCR/NLP engines)
- Unit tests for OpenLibrary search and scoring
- Integration tests with real images (keep a small set of test book cover images in the repo)
- pytest as the test framework

### Error Handling

Return structured error responses, not exceptions. The .NET API maps errors like:
- OCR failure → 400
- External service timeout → 504
- Invalid image → 400

This service should return clear error objects so the .NET API can map them appropriately.
