from pydantic_settings import BaseSettings, SettingsConfigDict

from app import constants


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Which OCR engine to use at runtime.
    # Must match the OCR_ENGINE build ARG used when building the Docker image.
    # Options: "onnx", "pytorch"
    ocr_engine: str = "onnx"

    # Path to the pre-downloaded ONNX model directory (used when ocr_engine="onnx").
    onnx_model_path: str = "/opt/hf_cache/florence2-onnx"

    # HuggingFace model name used to load the processor/tokenizer for the ONNX engine.
    # Must be resolvable via the standard HF cache — the flat local_dir layout used for
    # the ONNX weights doesn't include the custom tokenizer Python files that
    # trust_remote_code needs. Defaults to the source model of the ONNX export.
    onnx_processor_name: str = constants.FLORENCE2_PROCESSOR_MODEL

    # HuggingFace model name or local path for the PyTorch engine (used when ocr_engine="pytorch").
    pytorch_model_name: str = constants.FLORENCE2_PYTORCH_MODEL

    # Pinned revision of the PyTorch Florence-2 model.
    # Used when loading the model at runtime in local dev.
    # Set PYTORCH_FLORENCE2_REVISION in the environment to override.
    pytorch_florence2_revision: str = constants.FLORENCE2_PYTORCH_REVISION

    # Pinned revision of the GLiNER model to load from the HF cache.
    # Must match the revision used in the Dockerfile snapshot_download step.
    # Set GLINER_MODEL_REVISION in the environment to override.
    gliner_model_revision: str = constants.GLINER_REVISION

    # ONNX Runtime thread count per session.
    # Set ONNX_NUM_THREADS in the environment or .env to override.
    # Rule of thumb: match the number of physical cores available to the
    # container. Setting it too high (past physical cores) causes contention
    # and significant slowdowns — see benchmark results in issue #12.
    onnx_num_threads: int = 4

    # When true, logs per-stage timing for each ONNX inference call.
    # Set ONNX_LOG_TIMING=true in the environment or .env to enable.
    # Default is false to avoid overhead in production.
    onnx_log_timing: bool = False


settings = Settings()
