"""
Tests for Voice & SMS MCP Tools — MockTwilioClient and MCP tool functions.

Covers:
- MockTwilioClient: SMS, calls, logs, phone number purchase
- send_sms: sandbox safety, delegation, error handling
- make_call: synthesis + call flow, sandbox safety
- get_call_logs: summary computation, read-only
- get_sms_logs: message history
- buy_phone_number: sandbox safety, area code
- transcribe_audio: delegation to Transcriber
- MCP server registration (29 tools)
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.mcp.tools.voice_tools import (
    MockTwilioClient,
    buy_phone_number,
    get_call_logs,
    get_sms_logs,
    make_call,
    send_sms,
    transcribe_audio,
)


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════
# 1. MockTwilioClient
# ═══════════════════════════════════════════════════════════════════════


class TestMockTwilioClient:
    """Tests for the mock Twilio client."""

    def test_send_sms_returns_sid(self):
        client = MockTwilioClient()
        result = _run(client.send_sms(to="+15551234567", body="Hello"))
        assert result["sid"].startswith("SM_mock_")
        assert result["to"] == "+15551234567"
        assert result["body"] == "Hello"
        assert result["status"] == "queued"

    def test_send_sms_increments_counter(self):
        client = MockTwilioClient()
        r1 = _run(client.send_sms(to="+1", body="a"))
        r2 = _run(client.send_sms(to="+2", body="b"))
        assert r1["sid"] != r2["sid"]

    def test_send_sms_default_from_number(self):
        client = MockTwilioClient()
        result = _run(client.send_sms(to="+1", body="test"))
        assert result["from"] == "+15551234567"

    def test_send_sms_custom_from_number(self):
        client = MockTwilioClient()
        result = _run(client.send_sms(to="+1", body="test", from_number="+19995551234"))
        assert result["from"] == "+19995551234"

    def test_make_call_returns_sid(self):
        client = MockTwilioClient()
        result = _run(client.make_call(to="+15559876543"))
        assert result["sid"].startswith("CA_mock_")
        assert result["to"] == "+15559876543"
        assert result["status"] == "queued"

    def test_make_call_with_audio_url(self):
        client = MockTwilioClient()
        result = _run(client.make_call(to="+1", audio_url="https://example.com/audio.mp3"))
        assert result["audio_url"] == "https://example.com/audio.mp3"

    def test_get_call_logs_includes_history(self):
        client = MockTwilioClient()
        logs = _run(client.get_call_logs())
        assert len(logs) >= 2  # At least the 2 mock history entries

    def test_get_call_logs_includes_session_calls(self):
        client = MockTwilioClient()
        _run(client.make_call(to="+1"))
        logs = _run(client.get_call_logs())
        assert len(logs) >= 3  # 2 history + 1 session call

    def test_get_call_logs_respects_limit(self):
        client = MockTwilioClient()
        logs = _run(client.get_call_logs(limit=1))
        assert len(logs) == 1

    def test_get_sms_logs_includes_history(self):
        client = MockTwilioClient()
        logs = _run(client.get_sms_logs())
        assert len(logs) >= 1  # At least the 1 mock history entry

    def test_get_sms_logs_includes_session_sms(self):
        client = MockTwilioClient()
        _run(client.send_sms(to="+1", body="test"))
        logs = _run(client.get_sms_logs())
        assert len(logs) >= 2  # 1 history + 1 session

    def test_buy_phone_number_returns_details(self):
        client = MockTwilioClient()
        result = _run(client.buy_phone_number(area_code="212"))
        assert result["sid"].startswith("PN_mock_")
        assert "212" in result["phone_number"]
        assert result["capabilities"]["voice"] is True
        assert result["capabilities"]["sms"] is True

    def test_buy_phone_number_default_area_code(self):
        client = MockTwilioClient()
        result = _run(client.buy_phone_number())
        assert "415" in result["phone_number"]


# ═══════════════════════════════════════════════════════════════════════
# 2. send_sms MCP Tool
# ═══════════════════════════════════════════════════════════════════════


class TestSendSmsTool:
    """Tests for the send_sms MCP tool."""

    def test_is_sandboxed(self):
        from core.safety.sandbox import is_sandboxed, get_sandbox_tool_name
        assert is_sandboxed(send_sms)
        assert get_sandbox_tool_name(send_sms) == "send_sms"

    def test_sends_via_mock_client(self):
        client = MockTwilioClient()
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(send_sms(
                to_number="+15559999999",
                message="Test message",
                _client=client,
            ))
        result = json.loads(result_json)
        assert result["to"] == "+15559999999"
        assert result["body"] == "Test message"
        assert result["status"] == "queued"

    def test_truncates_long_messages(self):
        client = MockTwilioClient()
        long_msg = "x" * 2000
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(send_sms(
                to_number="+1",
                message=long_msg,
                _client=client,
            ))
        result = json.loads(result_json)
        assert len(result["body"]) <= 1600

    def test_uses_env_var_from_number(self):
        client = MockTwilioClient()
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production", "TWILIO_PHONE_NUMBER": "+18001234567"}):
            result_json = _run(send_sms(
                to_number="+1",
                message="hi",
                _client=client,
            ))
        result = json.loads(result_json)
        assert result["from"] == "+18001234567"

    def test_handles_client_error(self):
        client = AsyncMock()
        client.send_sms.side_effect = Exception("Network error")
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(send_sms(
                to_number="+1",
                message="fail",
                _client=client,
            ))
        result = json.loads(result_json)
        assert "error" in result

    def test_intercepted_in_dev(self):
        with patch.dict(os.environ, {"ENCLAVE_ENV": "development"}):
            result = _run(send_sms(
                to_number="+1",
                message="test",
            ))
            assert result["sandboxed"] is True
            assert result["tool_name"] == "send_sms"


# ═══════════════════════════════════════════════════════════════════════
# 3. make_call MCP Tool
# ═══════════════════════════════════════════════════════════════════════


class TestMakeCallTool:
    """Tests for the make_call MCP tool."""

    def test_is_sandboxed(self):
        from core.safety.sandbox import is_sandboxed, get_sandbox_tool_name
        assert is_sandboxed(make_call)
        assert get_sandbox_tool_name(make_call) == "make_call"

    def test_makes_call_via_mock_client(self):
        mock_client = MockTwilioClient()
        mock_synth = AsyncMock()
        mock_synth.synthesize.return_value = {
            "audio_path": "/tmp/test.mp3",
            "cached": False,
            "source": "mock",
        }

        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(make_call(
                to_number="+15551112222",
                script_text="Hello, this is a test call",
                _client=mock_client,
                _synthesizer=mock_synth,
            ))
        result = json.loads(result_json)
        assert result["to"] == "+15551112222"
        assert result["status"] == "queued"
        assert result["voice_preset"] == "professional_female"

    def test_includes_synthesis_metadata(self):
        mock_client = MockTwilioClient()
        mock_synth = AsyncMock()
        mock_synth.synthesize.return_value = {
            "audio_path": "/tmp/cached.mp3",
            "cached": True,
            "source": "cache",
        }

        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(make_call(
                to_number="+1",
                script_text="Test",
                _client=mock_client,
                _synthesizer=mock_synth,
            ))
        result = json.loads(result_json)
        assert result["audio_cached"] is True
        assert result["audio_source"] == "cache"

    def test_truncates_script_in_result(self):
        mock_client = MockTwilioClient()
        mock_synth = AsyncMock()
        mock_synth.synthesize.return_value = {"audio_path": "", "cached": False, "source": "mock"}

        long_script = "x" * 500
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(make_call(
                to_number="+1",
                script_text=long_script,
                _client=mock_client,
                _synthesizer=mock_synth,
            ))
        result = json.loads(result_json)
        assert len(result["script_text"]) <= 200

    def test_handles_synthesis_failure(self):
        mock_client = MockTwilioClient()
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            # No synthesizer provided, and import will fail in test env
            result_json = _run(make_call(
                to_number="+1",
                script_text="Test",
                _client=mock_client,
            ))
        result = json.loads(result_json)
        # Should still make the call even if synthesis fails
        assert result["status"] == "queued"

    def test_intercepted_in_dev(self):
        with patch.dict(os.environ, {"ENCLAVE_ENV": "development"}):
            result = _run(make_call(
                to_number="+1",
                script_text="test",
            ))
            assert result["sandboxed"] is True
            assert result["tool_name"] == "make_call"


# ═══════════════════════════════════════════════════════════════════════
# 4. get_call_logs MCP Tool
# ═══════════════════════════════════════════════════════════════════════


class TestGetCallLogsTool:
    """Tests for the get_call_logs MCP tool (read-only, not sandboxed)."""

    def test_returns_json_with_calls_and_summary(self):
        client = MockTwilioClient()
        result_json = _run(get_call_logs(_client=client))
        result = json.loads(result_json)
        assert "calls" in result
        assert "summary" in result
        assert len(result["calls"]) >= 2

    def test_summary_computes_correct_counts(self):
        client = MockTwilioClient()
        result_json = _run(get_call_logs(_client=client))
        result = json.loads(result_json)
        summary = result["summary"]
        assert summary["total_calls"] == len(result["calls"])
        assert summary["inbound_count"] >= 1
        assert summary["outbound_count"] >= 1

    def test_summary_computes_total_duration(self):
        client = MockTwilioClient()
        result_json = _run(get_call_logs(_client=client))
        result = json.loads(result_json)
        summary = result["summary"]
        assert summary["total_duration_seconds"] > 0
        assert summary["avg_duration_seconds"] > 0

    def test_respects_limit(self):
        client = MockTwilioClient()
        result_json = _run(get_call_logs(limit=1, _client=client))
        result = json.loads(result_json)
        assert result["summary"]["total_calls"] == 1

    def test_handles_error(self):
        client = AsyncMock()
        client.get_call_logs.side_effect = Exception("DB error")
        result_json = _run(get_call_logs(_client=client))
        result = json.loads(result_json)
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════
# 5. get_sms_logs MCP Tool
# ═══════════════════════════════════════════════════════════════════════


class TestGetSmsLogsTool:
    """Tests for the get_sms_logs MCP tool."""

    def test_returns_json_with_messages(self):
        client = MockTwilioClient()
        result_json = _run(get_sms_logs(_client=client))
        result = json.loads(result_json)
        assert "messages" in result
        assert "total_count" in result
        assert len(result["messages"]) >= 1

    def test_handles_error(self):
        client = AsyncMock()
        client.get_sms_logs.side_effect = Exception("Timeout")
        result_json = _run(get_sms_logs(_client=client))
        result = json.loads(result_json)
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════
# 6. buy_phone_number MCP Tool
# ═══════════════════════════════════════════════════════════════════════


class TestBuyPhoneNumberTool:
    """Tests for the buy_phone_number MCP tool."""

    def test_is_sandboxed(self):
        from core.safety.sandbox import is_sandboxed, get_sandbox_tool_name
        assert is_sandboxed(buy_phone_number)
        assert get_sandbox_tool_name(buy_phone_number) == "buy_phone_number"

    def test_buys_via_mock_client(self):
        client = MockTwilioClient()
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(buy_phone_number(area_code="212", _client=client))
        result = json.loads(result_json)
        assert result["sid"].startswith("PN_mock_")
        assert "212" in result["phone_number"]
        assert result["capabilities"]["sms"] is True

    def test_default_area_code(self):
        client = MockTwilioClient()
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(buy_phone_number(_client=client))
        result = json.loads(result_json)
        assert "415" in result["phone_number"]

    def test_handles_error(self):
        client = AsyncMock()
        client.buy_phone_number.side_effect = Exception("No numbers available")
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result_json = _run(buy_phone_number(_client=client))
        result = json.loads(result_json)
        assert "error" in result

    def test_intercepted_in_dev(self):
        with patch.dict(os.environ, {"ENCLAVE_ENV": "development"}):
            result = _run(buy_phone_number())
            assert result["sandboxed"] is True
            assert result["tool_name"] == "buy_phone_number"


# ═══════════════════════════════════════════════════════════════════════
# 7. transcribe_audio MCP Tool
# ═══════════════════════════════════════════════════════════════════════


class TestTranscribeAudioTool:
    """Tests for the transcribe_audio MCP tool."""

    def test_transcribes_via_injected_transcriber(self):
        mock_t = AsyncMock()
        mock_t.transcribe.return_value = {
            "text": "Hello from the caller",
            "language": "en",
            "source": "whisper_api",
        }

        result_json = _run(transcribe_audio(
            audio_path="/path/to/recording.mp3",
            _transcriber=mock_t,
        ))
        result = json.loads(result_json)
        assert result["text"] == "Hello from the caller"
        assert result["source"] == "whisper_api"

    def test_passes_language_param(self):
        mock_t = AsyncMock()
        mock_t.transcribe.return_value = {"text": "Hola", "language": "es"}

        _run(transcribe_audio(
            audio_path="/path/file.mp3",
            language="es",
            _transcriber=mock_t,
        ))
        mock_t.transcribe.assert_called_once_with("/path/file.mp3", language="es")

    def test_handles_transcription_error(self):
        mock_t = AsyncMock()
        mock_t.transcribe.side_effect = Exception("File not found")

        result_json = _run(transcribe_audio(
            audio_path="/missing.mp3",
            _transcriber=mock_t,
        ))
        result = json.loads(result_json)
        assert "error" in result
        assert result["audio_path"] == "/missing.mp3"


# ═══════════════════════════════════════════════════════════════════════
# 8. MCP Server Voice Tool Registration
# ═══════════════════════════════════════════════════════════════════════


class TestMCPServerVoiceRegistration:
    """Tests that voice tools are registered on the MCP server."""

    def test_server_has_29_tools(self):
        """MCP server should have 29 registered tools (23 previous + 6 voice)."""
        try:
            from core.mcp.server import create_mcp_server

            server = create_mcp_server(name="Test Voice")
            tool_names = list(server._tool_manager._tools.keys())
            assert len(tool_names) == 29
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_voice_tools_importable(self):
        """All 6 voice tools should be importable."""
        from core.mcp.tools.voice_tools import (
            send_sms,
            make_call,
            get_call_logs,
            get_sms_logs,
            buy_phone_number,
            transcribe_audio,
        )
        assert callable(send_sms)
        assert callable(make_call)
        assert callable(get_call_logs)
        assert callable(get_sms_logs)
        assert callable(buy_phone_number)
        assert callable(transcribe_audio)

    def test_voice_tools_on_server(self):
        """Voice tools should appear in the MCP server tool list."""
        try:
            from core.mcp.server import create_mcp_server

            server = create_mcp_server(name="Test Voice")
            tool_names = list(server._tool_manager._tools.keys())
            assert "send_sms" in tool_names
            assert "make_call" in tool_names
            assert "get_call_logs" in tool_names
            assert "get_sms_logs" in tool_names
            assert "buy_phone_number" in tool_names
            assert "transcribe_audio" in tool_names
        except ImportError:
            pytest.skip("fastmcp not installed")
