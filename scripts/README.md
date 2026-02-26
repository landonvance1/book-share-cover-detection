# Scripts

Utility scripts for development and testing.

## test_api.sh

Tests the running API against all test images in the integration test suite and displays NLP results (detected authors and titles).

### Prerequisites

- API running locally on `http://localhost:8000`
- `curl` and `jq` installed
- Test images present in `tests/integration/images/`

### Usage

```bash
bash scripts/test_api.sh
```

### Output

For each test image, shows:
- Image filename
- Request time (milliseconds)
- Detected authors and titles
- Any errors during analysis

Example:
```
📚 a-restless-truth
  ⏱  2341ms
  Authors:
    - Freya Marske
  Titles:
    - A Restless Truth

📚 gardens-of-the-moon
  ⏱  1987ms
  Authors:
    - Steven Erikson
  Titles:
    - Gardens of the Moon
```

## sync_onnx_model.py

Syncs the local ONNX Florence-2 model download with the pinned revision in `app/constants.py`.

This ensures local development stays in sync with Docker builds. The ONNX model is the only pre-downloaded model in local dev (loaded from a fixed directory path). PyTorch and GLiNER are downloaded on-demand at runtime via HuggingFace's caching, so they don't need explicit syncing.

### Usage

```bash
# Download to current directory
python scripts/sync_onnx_model.py

# Download to a specific directory
python scripts/sync_onnx_model.py --cache-dir /path/to/models
```

The script will:
1. Remove any existing `florence2-onnx/` directory
2. Download the ONNX model at the pinned revision from `app/constants.FLORENCE2_ONNX_REVISION`
3. Download the processor model needed for tokenization

### When to run

- **First setup**: After cloning the repo
- **Model updates**: When `FLORENCE2_ONNX_REVISION` in `app/constants.py` is updated
- **Troubleshooting**: If you get ONNX model loading errors in local dev
