"""
Structured logging.

Emits JSON lines in prod (machine-parseable, ships cleanly to any log
aggregator) and readable console output in dev. Every log call can carry
a `job_id` / `agent` field so a single review run's logs are traceable
end to end via `grep job_id=...`.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("job_id", "agent", "sub_question", "duration_ms", "paper_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(
            f"{k}={getattr(record, k)}"
            for k in ("job_id", "agent", "sub_question", "duration_ms", "paper_id")
            if hasattr(record, k)
        )
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<8} {record.name}: {record.getMessage()}"
        return f"{base} | {extras}" if extras else base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger(name), extra={})


def with_context(logger: logging.LoggerAdapter, **context: Any) -> logging.LoggerAdapter:
    """Return a logger adapter that always attaches the given context fields."""
    merged = {**logger.extra, **context}
    return logging.LoggerAdapter(logger.logger, extra=merged)
