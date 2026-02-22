from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
