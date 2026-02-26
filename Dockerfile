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

# Select which Florence-2 model to download based on OCR_ENGINE build ARG.
# Default is "onnx" (onnx-community/Florence-2-base-ft, flat layout at /opt/hf_cache/florence2-onnx).
# Use --build-arg OCR_ENGINE=pytorch for the PyTorch model (microsoft/Florence-2-base).
# The local_dir option writes a flat file layout so the path is predictable at runtime.
ARG OCR_ENGINE=onnx
RUN if [ "$OCR_ENGINE" = "onnx" ]; then \
      python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('onnx-community/Florence-2-base-ft', local_dir='/opt/hf_cache/florence2-onnx')"; \
      python -c "\
from transformers import AutoProcessor; \
AutoProcessor.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True)"; \
    else \
      python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/Florence-2-base', local_dir='/opt/hf_cache/florence2-pytorch', revision='5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac')"; \
    fi

# Pre-download GLiNER (~330 MB) during build.
# To update: check https://huggingface.co/urchade/gliner_large-v2.1/commits/main
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('urchade/gliner_large-v2.1', revision='abd49a1f1ebc12af1be84d06f6848221cf96dcad')"

# GLiNER loads its backbone tokenizer from config.model_name at runtime.
# gliner_large-v2.1 uses microsoft/deberta-v3-large as its backbone.
# Pre-download it so the lookup works in offline mode.
RUN python -c "\
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')"

ENV GLINER_MODEL_REVISION=abd49a1f1ebc12af1be84d06f6848221cf96dcad
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
