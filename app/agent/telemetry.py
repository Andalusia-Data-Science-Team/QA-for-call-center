"""Dedicated JSONL logging helpers for graph telemetry."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.config import settings


_node_executions: ContextVar[list[str] | None] = ContextVar(
    "node_executions", default=None
)

def _resolve_log_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def build_jsonl_logger(logger_name: str, configured_path: str) -> tuple[logging.Logger, Path]:
    """Return a non-propagating rotating logger that writes one JSON object per line."""
    log_path = _resolve_log_path(configured_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_logger = logging.getLogger(logger_name)
    jsonl_logger.setLevel(logging.INFO)
    jsonl_logger.propagate = False
    resolved_path = str(log_path.resolve())
    if not any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", None) == resolved_path
        for handler in jsonl_logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=settings.TOKEN_USAGE_LOG_MAX_BYTES,
            backupCount=settings.TOKEN_USAGE_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        jsonl_logger.addHandler(handler)
    return jsonl_logger, log_path


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def write_jsonl(jsonl_logger: logging.Logger, record: dict[str, Any]) -> None:
    """Serialize a telemetry record while preserving structured Pydantic values."""
    for handler in jsonl_logger.handlers:
        if not isinstance(handler, RotatingFileHandler):
            continue
        if Path(handler.baseFilename).exists():
            continue
        handler.acquire()
        try:
            if handler.stream is not None:
                handler.stream.close()
                handler.stream = None
        finally:
            handler.release()
    jsonl_logger.info(json.dumps(record, ensure_ascii=False, default=_json_default))


def begin_node_execution_tracking() -> Token:
    """Start an execution list isolated to the current chat analysis."""
    return _node_executions.set([])


def current_node_executions() -> list[str] | None:
    """Return the active execution list, if tracking has been started."""
    return _node_executions.get()


def record_node_execution(node_name: str) -> None:
    """Record one graph-node invocation without changing its state output."""
    executions = _node_executions.get()
    if executions is not None:
        executions.append(node_name)


def end_node_execution_tracking(token: Token) -> list[str]:
    """Return the completed execution list and restore the prior context."""
    executions = list(_node_executions.get() or [])
    _node_executions.reset(token)
    return executions
