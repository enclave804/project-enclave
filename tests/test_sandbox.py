"""
Unit tests for the Sandbox Protocol (@sandboxed_tool decorator).

Tests tool interception in non-production, production passthrough,
log file writing, and introspection helpers.
"""

import asyncio
import json
import os

import pytest

from core.safety.sandbox import (
    sandboxed_tool,
    is_sandboxed,
    get_sandbox_tool_name,
    _log_sandboxed_call,
)


# ─── Sync Tool Tests ─────────────────────────────────────────────


class TestSandboxedSyncTool:
    """Tests for synchronous sandboxed tools."""

    def test_intercepted_in_development(self, tmp_path, monkeypatch):
        """In dev env, tool should NOT execute — just log."""
        monkeypatch.setenv("ENCLAVE_ENV", "development")
        call_count = 0

        @sandboxed_tool("test_sync", log_dir=tmp_path)
        def dangerous_action(x: int, y: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"executed": True}

        result = dangerous_action(42, y="hello")

        assert call_count == 0  # NOT executed
        assert result["sandboxed"] is True
        assert result["tool_name"] == "test_sync"
        assert result["environment"] == "development"

    def test_executes_in_production(self, monkeypatch):
        """In production, tool should execute normally."""
        monkeypatch.setenv("ENCLAVE_ENV", "production")

        @sandboxed_tool("test_sync")
        def real_action(x: int) -> dict:
            return {"executed": True, "value": x}

        result = real_action(99)

        assert result["executed"] is True
        assert result["value"] == 99

    def test_intercepted_in_staging(self, tmp_path, monkeypatch):
        """Staging should also be intercepted (not production)."""
        monkeypatch.setenv("ENCLAVE_ENV", "staging")

        @sandboxed_tool("test_sync", log_dir=tmp_path)
        def send_email(to: str) -> dict:
            return {"sent": True}

        result = send_email(to="user@example.com")
        assert result["sandboxed"] is True

    def test_intercepted_when_env_unset(self, tmp_path, monkeypatch):
        """Default env (development) should be intercepted."""
        monkeypatch.delenv("ENCLAVE_ENV", raising=False)

        @sandboxed_tool("test_sync", log_dir=tmp_path)
        def dangerous() -> dict:
            return {"bad": True}

        result = dangerous()
        assert result["sandboxed"] is True
        assert result["environment"] == "development"

    def test_log_file_created(self, tmp_path, monkeypatch):
        """Should write a JSONL log file with the tool call details."""
        monkeypatch.setenv("ENCLAVE_ENV", "development")

        @sandboxed_tool("log_test_tool", log_dir=tmp_path)
        def action(a: int, b: str) -> dict:
            return {}

        action(1, b="two")

        log_file = tmp_path / "log_test_tool.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["tool_name"] == "log_test_tool"
        assert entry["action"] == "SANDBOXED — not executed"
        assert entry["kwargs"]["b"] == "two"

    def test_multiple_calls_append(self, tmp_path, monkeypatch):
        """Multiple sandboxed calls should append to the same log file."""
        monkeypatch.setenv("ENCLAVE_ENV", "test")

        @sandboxed_tool("multi_tool", log_dir=tmp_path)
        def action(x: int) -> dict:
            return {}

        action(1)
        action(2)
        action(3)

        log_file = tmp_path / "multi_tool.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3


# ─── Async Tool Tests ────────────────────────────────────────────


class TestSandboxedAsyncTool:
    """Tests for asynchronous sandboxed tools."""

    def test_async_intercepted_in_development(self, tmp_path, monkeypatch):
        """Async tools should also be intercepted in non-production."""
        monkeypatch.setenv("ENCLAVE_ENV", "development")
        call_count = 0

        @sandboxed_tool("async_test", log_dir=tmp_path)
        async def send_email(to: str, body: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"sent": True}

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(send_email("a@b.com", body="hi"))
        finally:
            loop.close()

        assert call_count == 0
        assert result["sandboxed"] is True
        assert result["tool_name"] == "async_test"

    def test_async_executes_in_production(self, monkeypatch):
        """Async tools should execute normally in production."""
        monkeypatch.setenv("ENCLAVE_ENV", "production")

        @sandboxed_tool("async_test")
        async def book_meeting(time: str) -> dict:
            return {"booked": True, "time": time}

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(book_meeting("2025-03-01T10:00"))
        finally:
            loop.close()

        assert result["booked"] is True


# ─── Introspection Tests ─────────────────────────────────────────


class TestSandboxIntrospection:
    """Tests for is_sandboxed() and get_sandbox_tool_name()."""

    def test_is_sandboxed_true(self):
        @sandboxed_tool("check_me")
        def my_func():
            pass

        assert is_sandboxed(my_func) is True

    def test_is_sandboxed_false(self):
        def plain_func():
            pass

        assert is_sandboxed(plain_func) is False

    def test_get_tool_name(self):
        @sandboxed_tool("stripe_charge")
        def charge(amount: int):
            pass

        assert get_sandbox_tool_name(charge) == "stripe_charge"

    def test_get_tool_name_none_for_plain(self):
        def plain():
            pass

        assert get_sandbox_tool_name(plain) is None

    def test_preserves_function_name(self):
        @sandboxed_tool("test")
        def my_original_function():
            """Original docstring."""
            pass

        assert my_original_function.__name__ == "my_original_function"
        assert "Original docstring" in (my_original_function.__doc__ or "")


# ─── Log Entry Tests ─────────────────────────────────────────────


class TestLogSandboxedCall:
    """Tests for the _log_sandboxed_call helper."""

    def test_creates_log_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENCLAVE_ENV", "test")
        entry = _log_sandboxed_call(
            "my_tool", (1, "two"), {"key": "value"}, log_dir=tmp_path
        )
        assert entry["tool_name"] == "my_tool"
        assert entry["args"] == [1, "two"]
        assert entry["kwargs"]["key"] == "value"
        assert "timestamp" in entry

    def test_handles_complex_args(self, tmp_path, monkeypatch):
        """Should serialize complex objects without crashing."""
        monkeypatch.setenv("ENCLAVE_ENV", "test")

        class Unserializable:
            pass

        entry = _log_sandboxed_call(
            "complex_tool",
            (Unserializable(), [1, 2, 3]),
            {"nested": {"deep": True}},
            log_dir=tmp_path,
        )
        assert isinstance(entry["args"][0], str)  # Fell back to str()
        assert entry["kwargs"]["nested"]["deep"] is True
