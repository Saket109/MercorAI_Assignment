# eval_runner/cache.py
"""
Persistent memo-table for prompt→response pairs.

Every (prompt, params) combination is hashed into a SHA-256 key and
stored in a JSON file.  This gives two benefits:
  1. Determinism — identical prompts always return the cached response.
  2. Efficiency — repeated eval runs don't re-query the model.

The store is intentionally simple (single JSON file) because the
evaluation datasets are small (~1-5 k entries).  For production-scale
caching, swap in Redis / SQLite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskMemo:
    """SHA-256-keyed JSON store for caching LLM responses."""

    def __init__(self, filepath: str | Path) -> None:
        self._path = Path(filepath)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._table: dict[str, Any] = {}
        if self._path.exists():
            with open(self._path, encoding="utf-8") as fh:
                self._table = json.load(fh)

    # ── public API ───────────────────────────────────────────────────

    def lookup(self, *key_parts: Any) -> Any | None:
        """Return cached value or ``None`` on miss."""
        return self._table.get(self._make_key(key_parts))

    def store(self, value: Any, *key_parts: Any) -> None:
        """Insert a value and persist to disk."""
        self._table[self._make_key(key_parts)] = value
        self._persist()

    @property
    def entries(self) -> int:
        return len(self._table)

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _make_key(parts: tuple) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _persist(self) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._table, fh)
