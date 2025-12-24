"""
Lightweight run-scoped logging helpers for token usage.
Agents append token stats here; pipeline reads once per run and writes to run_log.json.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

_token_entries: List[Dict[str, Any]] = []


def reset_token_log() -> None:
    """Clear token log for a new pipeline run."""
    _token_entries.clear()


def log_token_usage(
    usage: str,
    input_tokens: int | None,
    output_tokens: int | None,
    runtime_seconds: float | None,
) -> None:
    entry = {
        "usage": usage,
        "input_token": input_tokens if input_tokens is not None else 0,
        "output_token": output_tokens if output_tokens is not None else 0,
        "runtime": round(runtime_seconds or 0.0, 4),
    }
    _token_entries.append(entry)


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


def get_token_entries() -> List[Dict[str, Any]]:
    """Return a shallow copy of the token log entries for the current run."""
    return list(_token_entries)


def _get_value(usage_meta: Any, possible_keys: list[str]) -> int | None:
    for key in possible_keys:
        if isinstance(usage_meta, dict) and key in usage_meta:
            return usage_meta[key]
        if hasattr(usage_meta, key):
            return getattr(usage_meta, key)
    return None
