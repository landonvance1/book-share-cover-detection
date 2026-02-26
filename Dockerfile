FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Store HuggingFace models in a predictable location (not user home)
ENV HF_HOME=/opt/hf_cache

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user and model cache dir before model downloads
# so downloaded model files are owned by appuser from the start
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /opt/hf_cache \
    && chown -R appuser:appuser /opt/hf_cache /app

USER appuser

# Copy constants.py early so model download commands can import from it.
# This ensures model revisions are defined in a single place (app/constants.py).
COPY --chown=appuser:appuser app/constants.py ./app/constants.py

# Select which Florence-2 model to download based on OCR_ENGINE build ARG.
# Default is "onnx" (onnx-community/Florence-2-base-ft, flat layout at /opt/hf_cache/florence2-onnx).
# Use --build-arg OCR_ENGINE=pytorch for the PyTorch model (microsoft/Florence-2-base).
# The local_dir option writes a flat file layout so the path is predictable at runtime.
# Model revisions are imported from app/constants.py for single-source-of-truth.
ARG OCR_ENGINE=onnx
RUN if [ "$OCR_ENGINE" = "onnx" ]; then \
      python -c "\
import sys; sys.path.insert(0, '/app'); \
from app.constants import FLORENCE2_ONNX_MODEL, FLORENCE2_ONNX_REVISION; \
from huggingface_hub import snapshot_download; \
snapshot_download(FLORENCE2_ONNX_MODEL, local_dir='/opt/hf_cache/florence2-onnx', revision=FLORENCE2_ONNX_REVISION) \
"; \
    else \
      python -c "\
import sys; sys.path.insert(0, '/app'); \
from app.constants import FLORENCE2_PYTORCH_MODEL, FLORENCE2_PYTORCH_REVISION; \
from huggingface_hub import snapshot_download; \
snapshot_download(FLORENCE2_PYTORCH_MODEL, local_dir='/opt/hf_cache/florence2-pytorch', revision=FLORENCE2_PYTORCH_REVISION) \
"; \
    fi

RUN python -c "\
import sys; sys.path.insert(0, '/app'); \
from app.constants import FLORENCE2_PROCESSOR_MODEL; \
from transformers import AutoProcessor; \
AutoProcessor.from_pretrained(FLORENCE2_PROCESSOR_MODEL, trust_remote_code=True) \
"

# Pre-download GLiNER (~330 MB) during build.
# Model name and revision are imported from app/constants.py for single-source-of-truth.
RUN python -c "\
import sys; sys.path.insert(0, '/app'); \
from app.constants import GLINER_MODEL, GLINER_REVISION; \
from huggingface_hub import snapshot_download; \
snapshot_download(GLINER_MODEL, revision=GLINER_REVISION) \
"

# GLiNER loads its backbone tokenizer from config.model_name at runtime.
# gliner_large-v2.1 uses microsoft/deberta-v3-large as its backbone.
# Pre-download it so the lookup works in offline mode.
RUN python -c "\
import sys; sys.path.insert(0, '/app'); \
from app.constants import GLINER_BACKBONE_MODEL; \
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained(GLINER_BACKBONE_MODEL) \
"

# Local path for the pytorch Florence-2 model (only used when OCR_ENGINE=pytorch).
# snapshot_download with local_dir uses a flat layout that from_pretrained can read directly.
ENV PYTORCH_MODEL_NAME=/opt/hf_cache/florence2-pytorch

# Prevent runtime network calls — all models are baked in above.
ENV HF_HUB_OFFLINE=1

# Copy application code (after model downloads for better layer caching)
COPY --chown=appuser:appuser app/ ./app/

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
