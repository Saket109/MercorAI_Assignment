# serve/config.py
"""
Application configuration loaded from environment / .env file.

Uses a plain dataclass + classmethod factory instead of a heavy settings
framework — keeps the dependency footprint small and explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Walk up to find .env at project root
_root = Path(__file__).resolve().parent.parent
_dotenv = _root / ".env"
if _dotenv.exists():
    load_dotenv(_dotenv)


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


@dataclass(frozen=True)
class AppConfig:
    """Immutable snapshot of all runtime knobs."""

    # vLLM backend
    backend_url: str
    model_name: str

    # local gateway
    gateway_host: str
    gateway_port: int

    # httpx pool
    pool_size: int
    keepalive: int
    request_timeout: float
    max_retries: int

    @classmethod
    def load(cls) -> AppConfig:
        """Build from environment variables (or defaults)."""
        return cls(
            backend_url=_env("VLLM_BASE", "https://i46h0gnb0vm4nt-8888.proxy.runpod.net"),
            model_name=_env("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1"),
            gateway_host=_env("HOST", "0.0.0.0"),
            gateway_port=_env_int("PORT", 9000),
            pool_size=_env_int("POOL_SIZE", 80),
            keepalive=_env_int("KEEPALIVE", 20),
            request_timeout=_env_float("REQUEST_TIMEOUT", 120.0),
            max_retries=_env_int("MAX_RETRIES", 3),
        )

    @property
    def completions_endpoint(self) -> str:
        return f"{self.backend_url.rstrip('/')}/v1/chat/completions"

    @property
    def models_endpoint(self) -> str:
        return f"{self.backend_url.rstrip('/')}/v1/models"


# Singleton — import directly where needed.
cfg = AppConfig.load()
