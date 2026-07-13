"""
Observabilidade estruturada — structured JSON logging.

Design para plugabilidade cloud:
- Localmente: FileHandler + StreamHandler com JSONFormatter
- Para adicionar cloud, basta adicionar um handler em setup_logging():

  # Datadog
  from datadog_logger import DatadogHandler
  logger.addHandler(DatadogHandler(api_key=..., service="risk-gestacional"))

  # AWS CloudWatch
  import watchtower
  logger.addHandler(watchtower.CloudWatchLogHandler(log_group="/risk-gestacional/app"))

  # GCP Cloud Logging
  import google.cloud.logging
  client = google.cloud.logging.Client()
  client.setup_logging()

  # OpenTelemetry (vendor-neutral)
  from opentelemetry.sdk._logs import LoggerProvider
  # ... configurar exporter para o backend desejado

Todos consomem JSON estruturado — nenhuma mudança no código de negócio é necessária.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

LOGGER_NAME = "risk_gestacional"


class _JSONFormatter(logging.Formatter):
    """Formata cada log record como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Campos estruturados injetados via log_event()
        for field in ("event", "session_id", "data"):
            value = getattr(record, field, None)
            if value is not None:
                entry[field] = value

        if record.exc_info:
            entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configura o logger principal da aplicação.

    Handlers locais:
    - FileHandler  → logs/app.log   (JSON, rotação manual ou via logrotate)
    - StreamHandler → stdout         (JSON, útil para docker logs / tail)

    Para adicionar um handler cloud, basta instanciá-lo aqui e chamar
    logger.addHandler(cloud_handler) — o formato JSON já é compatível.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = _JSONFormatter()

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Retorna o logger configurado (chama setup_logging() se ainda não foi feito)."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        setup_logging()
    return logger


def log_event(
    event: str,
    session_id: str | None = None,
    level: int = logging.INFO,
    **data: Any,
) -> None:
    """
    Loga um evento estruturado.

    Parâmetros
    ----------
    event      : identificador do evento (ex: "session.started", "llm.call.completed")
    session_id : ID da sessão do usuário (para correlacionar eventos de uma mesma sessão)
    level      : nível de log (logging.INFO, logging.WARNING, etc.)
    **data     : campos adicionais do evento (ex: risk_level="high risk", elapsed_ms=320)

    Exemplo
    -------
    log_event("model.prediction", session_id=sid, risk_level="high risk", elapsed_ms=45)

    Produz:
    {
      "timestamp": "2026-07-13T14:23:01.123Z",
      "level": "INFO",
      "logger": "risk_gestacional",
      "message": "model.prediction",
      "event": "model.prediction",
      "session_id": "a3f9...",
      "data": {"risk_level": "high risk", "elapsed_ms": 45}
    }
    """
    logger = get_logger()
    extra = {
        "event": event,
        "session_id": session_id,
        "data": data if data else None,
    }
    logger.log(level, event, extra=extra)
