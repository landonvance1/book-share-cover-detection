from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.engines.easyocr_engine import EasyOcrEngine
from app.engines.spacy_engine import SpacyNlpEngine
from app.models import CoverAnalysisResponse, HealthResponse
from app.services.analyzer import CoverAnalyzer
from app.services.openlibrary import OpenLibraryClient

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 4 * 1024 * 1024  # 4 MB

analyzer: CoverAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    ocr_engine = EasyOcrEngine()
    nlp_engine = SpacyNlpEngine()
    book_search = OpenLibraryClient()
    analyzer = CoverAnalyzer(ocr_engine, nlp_engine, book_search)
    yield
    analyzer = None


app = FastAPI(title="Book Cover Detection", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/analyze", response_model=CoverAnalysisResponse)
async def analyze_cover(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Accepted: JPEG, PNG, WebP",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(image_bytes)} bytes. Max: {MAX_FILE_SIZE} bytes",
        )

    assert analyzer is not None
    return await analyzer.analyze(image_bytes)
