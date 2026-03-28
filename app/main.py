import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.engines.gliner_engine import GlinerNlpEngine
from app.logging_config import setup_logging
from app.models import CoverAnalysisResponse, HealthResponse
from app.services.analyzer import CoverAnalyzer

setup_logging()

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 4 * 1024 * 1024  # 4 MB

analyzer: CoverAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-apply after uvicorn's own logging.config.dictConfig call, which runs
    # during server startup and reinstalls plain-text handlers on uvicorn loggers.
    setup_logging()
    global analyzer
    logger.info("Starting cover detection service", extra={"ocr_engine": settings.ocr_engine})
    if settings.ocr_engine == "onnx":
        from app.engines.florence2_onnx_engine import Florence2OnnxEngine
        ocr_engine = Florence2OnnxEngine(
            model_path=settings.onnx_model_path,
            processor_name=settings.onnx_processor_name,
        )
    else:
        from app.engines.florence2_engine import Florence2OcrEngine
        ocr_engine = Florence2OcrEngine(
            model_name=settings.pytorch_model_name,
            revision=settings.pytorch_florence2_revision,
        )
    nlp_engine = GlinerNlpEngine(revision=settings.gliner_model_revision)
    analyzer = CoverAnalyzer(ocr_engine, nlp_engine)
    logger.info("Models loaded, service ready", extra={"ocr_engine": settings.ocr_engine})
    yield
    analyzer = None


app = FastAPI(title="Book Cover Detection", version="0.1.0", lifespan=lifespan)
Instrumentator(excluded_handlers=["/health"]).instrument(app).expose(app, endpoint="/metrics")


@app.get("/health", response_model=HealthResponse)
async def health():
    status = "healthy" if analyzer is not None else "starting"
    return HealthResponse(status=status, version="0.1.0")


@app.post("/analyze", response_model=CoverAnalysisResponse)
async def analyze_cover(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            "Invalid content type rejected",
            extra={"content_type": file.content_type},
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Accepted: JPEG, PNG, WebP",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        logger.warning(
            "Oversized file rejected",
            extra={"file_size_bytes": len(image_bytes), "max_bytes": MAX_FILE_SIZE},
        )
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(image_bytes)} bytes. Max: {MAX_FILE_SIZE} bytes",
        )

    assert analyzer is not None
    return await analyzer.analyze(image_bytes)


if settings.enable_test_app:
    app.mount("/test", StaticFiles(directory="test_app", html=True), name="test_app")
