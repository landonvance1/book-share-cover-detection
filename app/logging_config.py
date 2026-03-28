import datetime
import logging

from pythonjsonlogger import jsonlogger


class _JsonFormatter(jsonlogger.JsonFormatter):
    """JsonFormatter that emits ISO 8601 timestamps with milliseconds and UTC 'Z' suffix.

    The default datefmt-based formatting omits sub-second precision and timezone info.
    Alloy uses the timestamp field for event time rather than ingestion time, so the
    format needs to be unambiguous and parseable by standard ISO 8601 parsers.
    Example output: "2026-03-28T12:00:00.123Z"
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z"


def setup_logging() -> None:
    formatter = _JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger_name",
        },
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    # Uvicorn installs its own plain-text StreamHandler on startup (via
    # logging.config.dictConfig). Clear those handlers and enable propagation
    # so all uvicorn output — including access logs — routes through the root
    # JSON handler instead.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True
