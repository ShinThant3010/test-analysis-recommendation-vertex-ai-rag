"""
Simple helper to append token usage entries to ``token_log.json``.
Each entry keeps the usage label, token counts, and runtime for a single activity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import TOKEN_LOG_PATH

LOG_FILE = Path(TOKEN_LOG_PATH)


def log_token_usage(
    usage: str,
    input_tokens: int | None,
    output_tokens: int | None,
    runtime_seconds: float | None,
) -> None:
    """
    Append a token usage entry to ``token_log.json``.
    Values default to zero when None is provided.
    """
    entry = {
        "usage": usage,
        "input token": input_tokens if input_tokens is not None else 0,
        "output token": output_tokens if output_tokens is not None else 0,
        "runtime": round(runtime_seconds or 0.0, 4),
    }
    _write_entry(entry)


def extract_token_counts(response: Any) -> tuple[int | None, int | None]:
    """
    Attempt to extract input/output token counts from a Gemini response object.
    Works with dict-like or attribute-style metadata payloads.
    """
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is None and isinstance(response, dict):
        usage_meta = response.get("usage_metadata")
    if usage_meta is None:
        return None, None

    input_tokens = _get_value(
        usage_meta,
        ["input_tokens", "prompt_token_count", "prompt_tokens"],
    )
    output_tokens = _get_value(
        usage_meta,
        ["output_tokens", "candidates_token_count", "completion_token_count"],
    )
    return input_tokens, output_tokens


def _read_existing_entries() -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_entry(entry: dict[str, Any]) -> None:
    entries = _read_existing_entries()
    entries.append(entry)
    LOG_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _get_value(usage_meta: Any, possible_keys: list[str]) -> int | None:
    for key in possible_keys:
        if isinstance(usage_meta, dict) and key in usage_meta:
            return usage_meta[key]
        if hasattr(usage_meta, key):
            return getattr(usage_meta, key)
    return None
