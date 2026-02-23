# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**book-share-cover-detection** is a Python microservice that analyzes book cover images to identify the book's title and author. It replaces the Azure Computer Vision dependency currently embedded in the [book-share-api](https://github.com/landonvance1/book-share-api) (.NET backend).

### Why This Exists

The parent project (BookSharingWebAPI) currently handles cover analysis inline using:
1. **Azure Vision OCR** — sends image to Azure's cloud Read API, polls for results, gets text + bounding boxes
2. **Height-based filtering** — uses bounding box geometry to keep only visually prominent text (≥20% of max text size), with a "sharpening" retry strategy that drops smaller text tiers when OpenLibrary returns no matches

This microservice replaces that with:
1. **Florence-2** (Microsoft, local vision-language model) for text extraction
2. **GLiNER** (zero-shot NER) to classify text as author or title; results are sorted by bounding box height so the most visually prominent match appears first

### Design Goals

- **Eliminate Azure dependency** — no cloud OCR costs, no API keys needed for core functionality
- **Better accuracy** — NLP-based title/author detection combined with height-sorted results should outperform height-based filtering heuristics
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
    │  OCR → NLP
    │  Returns structured analysis result
    │
    ├── Florence-2 (local vision-language model OCR)
    ├── SpaCy (local NLP engine)
```

### Interface Abstractions

Two core abstractions must be defined as interfaces/protocols:

**OCR Interface** — Takes an image, returns extracted text with positional/confidence metadata.
- Default implementation: Florence-2 (`microsoft/Florence-2-base`)
- Future alternatives: Florence-2 ONNX export (see issue #12), Tesseract, PaddleOCR, cloud APIs

**NLP Interface** — Takes raw OCR output, returns structured analysis identifying which text is likely the title vs. author. Results are ordered by bounding box height (most visually prominent first).
- Default implementation: GLiNER (`urchade/gliner_small-v2.1`)
- Future alternatives: Hugging Face transformers, custom models

### Expected Response Contract

The .NET API currently expects this shape from cover analysis (adapt to match):

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
- **NLP**: GLiNER
- **HTTP framework**: FastAPI recommended (async, automatic OpenAPI docs, similar developer experience to the .NET Minimal APIs pattern used in the parent project)

### Key Design Principles

1. **Interface-first design** — Define OCR and NLP as abstract base classes/protocols before implementing. This is a core requirement, not optional.
2. **Height-sorted results** — The NLP layer sorts `potentialAuthors` and `potentialTitles` by the bounding box height of their constituent OCR regions, so the most visually prominent match is always at index 0. This replaces the old height-based filtering heuristic.
3. **Stateless** — This service has no database. It receives an image, processes it, and returns results.
4. **Container-ready** — Should run in Docker alongside the .NET API and PostgreSQL. The .NET project uses Docker Compose already.
5. **Health check endpoint** — Include a `/health` endpoint for container orchestration.

### Testing Strategy

- Unit tests for OCR output parsing and NLP analysis logic (mock the OCR/NLP engines)
- Integration tests with real images (keep a small set of test book cover images in the repo)
- pytest as the test framework

### Error Handling

Return structured error responses, not exceptions. The .NET API maps errors like:
- OCR failure → 400
- External service timeout → 504
- Invalid image → 400

This service should return clear error objects so the .NET API can map them appropriately.
