"""
Sandbox Protocol — intercept dangerous tools in non-production environments.

Any tool decorated with @sandboxed_tool will:
- In production (ENCLAVE_ENV=production): execute normally
- In all other environments: log the action to sandbox_logs/ instead

This provides a safety net for tools that have real-world side effects
(sending emails, charging credit cards, booking meetings, etc.).

Usage:
    from core.safety.sandbox import sandboxed_tool

    @sandboxed_tool("send_email")
    async def send_email(to: str, subject: str, body: str) -> dict:
        # This only runs in production.
        # In dev/staging, the call + args are logged to sandbox_logs/
        return await real_email_provider.send(to, subject, body)

Environment:
    ENCLAVE_ENV: "production" | "staging" | "development" | "test"
    Default: "development" (safe by default)
"""

from __future__ import annotations

import functools
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Directory for sandbox log files
_SANDBOX_LOG_DIR = Path(__file__).parent.parent.parent / "sandbox_logs"


def _get_env() -> str:
    """Get the current environment. Defaults to 'development' (safe)."""
    return os.environ.get("ENCLAVE_ENV", "development").lower().strip()


def _is_production() -> bool:
    """Check if we're running in production."""
    return _get_env() == "production"


def _ensure_log_dir() -> Path:
    """Ensure the sandbox log directory exists and return it."""
    _SANDBOX_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _SANDBOX_LOG_DIR


def _log_sandboxed_call(
    tool_name: str,
    args: tuple,
    kwargs: dict[str, Any],
    log_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Log a sandboxed tool call to a JSON file.

    Returns the log entry dict for testing/inspection.
    """
    log_dir = log_dir or _ensure_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    entry = {
        "timestamp": timestamp.isoformat(),
        "environment": _get_env(),
        "tool_name": tool_name,
        "args": _serialize_args(args),
        "kwargs": _serialize_args(kwargs),
        "action": "SANDBOXED — not executed",
    }

    # Write to a per-tool log file (append mode)
    log_file = log_dir / f"{tool_name}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    logger.warning(
        f"SANDBOX: Tool '{tool_name}' intercepted in '{_get_env()}' env. "
        f"Call logged to {log_file.name}"
    )

    return entry


def _serialize_args(obj: Any) -> Any:
    """Best-effort serialization for log entries."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize_args(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_args(item) for item in obj]
    # Fallback: use str() for complex objects
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def sandboxed_tool(
    tool_name: str,
    *,
    log_dir: Optional[Path] = None,
) -> Callable:
    """
    Decorator that intercepts tool execution in non-production environments.

    Args:
        tool_name: Human-readable name for logging (e.g., "send_email").
        log_dir: Override log directory (useful for testing).

    Returns:
        Decorator that wraps both sync and async functions.

    Example:
        @sandboxed_tool("send_email")
        async def send_email(to, subject, body):
            ...  # Only executes in production
    """

    def decorator(func: Callable) -> Callable:
        # Mark the function so we can inspect it
        func._sandboxed = True
        func._sandbox_tool_name = tool_name

        if _is_async(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
                if _is_production():
                    return await func(*args, **kwargs)

                entry = _log_sandboxed_call(
                    tool_name, args, kwargs, log_dir=log_dir
                )
                return {
                    "sandboxed": True,
                    "tool_name": tool_name,
                    "environment": _get_env(),
                    "message": f"Tool '{tool_name}' was sandboxed — not executed",
                    "logged_at": entry["timestamp"],
                }

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
                if _is_production():
                    return func(*args, **kwargs)

                entry = _log_sandboxed_call(
                    tool_name, args, kwargs, log_dir=log_dir
                )
                return {
                    "sandboxed": True,
                    "tool_name": tool_name,
                    "environment": _get_env(),
                    "message": f"Tool '{tool_name}' was sandboxed — not executed",
                    "logged_at": entry["timestamp"],
                }

            return sync_wrapper

    return decorator


def _is_async(func: Callable) -> bool:
    """Check if a function is a coroutine function."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


def is_sandboxed(func: Callable) -> bool:
    """Check if a function has been decorated with @sandboxed_tool."""
    return getattr(func, "_sandboxed", False)


def get_sandbox_tool_name(func: Callable) -> Optional[str]:
    """Get the sandbox tool name from a decorated function."""
    return getattr(func, "_sandbox_tool_name", None)
