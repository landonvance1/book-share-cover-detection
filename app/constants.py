"""Model configuration constants for book-share-cover-detection.

This module centralizes all model names, revisions, and other constants
to ensure consistent pinning across the codebase and Dockerfile.
"""

# Florence-2 PyTorch model (used when OCR_ENGINE=pytorch)
FLORENCE2_PYTORCH_MODEL = "microsoft/Florence-2-base"
FLORENCE2_PYTORCH_REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"

# Florence-2 ONNX model (used when OCR_ENGINE=onnx)
FLORENCE2_ONNX_MODEL = "onnx-community/Florence-2-base-ft"
FLORENCE2_ONNX_REVISION = "e88a44eaf3791a35eae0c5a47b3dbcd36e67eb6f"

# Florence-2 processor (used for both PyTorch and ONNX engines)
# ONNX engine uses this for tokenization since the flat local_dir layout
# doesn't include the custom tokenizer code that requires trust_remote_code
FLORENCE2_PROCESSOR_MODEL = "microsoft/Florence-2-base-ft"

# GLiNER NLP model for author/title extraction
GLINER_MODEL = "urchade/gliner_large-v2.1"
GLINER_REVISION = "abd49a1f1ebc12af1be84d06f6848221cf96dcad"

# GLiNER backbone tokenizer (used as dependency by gliner_large-v2.1)
GLINER_BACKBONE_MODEL = "microsoft/deberta-v3-large"
