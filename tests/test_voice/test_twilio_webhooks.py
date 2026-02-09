"""
Tests for Twilio Webhook Processing — signature verification + event processing.

Covers:
- Twilio HMAC-SHA1 signature verification
- WebhookProcessor: inbound call, SMS received, recording received
- TwiML response generation
- EventBus dispatch integration for Twilio events
- FastAPI router Twilio endpoints
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.integrations.webhook_server import (
    WebhookProcessor,
    create_webhook_router,
    verify_twilio_signature,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Twilio Signature Verification
# ═══════════════════════════════════════════════════════════════════════


class TestVerifyTwilioSignature:
    """Tests for Twilio webhook HMAC-SHA1 signature verification."""

    def _make_signature(
        self,
        url: str,
        params: dict[str, str],
        auth_token: str,
    ) -> str:
        """Build a valid Twilio signature for testing."""
        data = url
        for key in sorted(params.keys()):
            data += key + params[key]
        return base64.b64encode(
            hmac.new(
                auth_token.encode("utf-8"),
                data.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")

    def test_accepts_valid_signature(self):
        auth_token = "my_twilio_token"
        url = "https://example.com/webhooks/twilio/voice"
        params = {"CallSid": "CA123", "From": "+15551234567"}
        sig = self._make_signature(url, params, auth_token)

        assert verify_twilio_signature(url, params, sig, auth_token) is True

    def test_rejects_invalid_signature(self):
        auth_token = "my_twilio_token"
        url = "https://example.com/webhooks/twilio/voice"
        params = {"CallSid": "CA123"}
        assert verify_twilio_signature(url, params, "invalid_sig", auth_token) is False

    def test_rejects_tampered_params(self):
        auth_token = "my_twilio_token"
        url = "https://example.com/webhooks/twilio/voice"
        original_params = {"CallSid": "CA123", "From": "+15551234567"}
        sig = self._make_signature(url, original_params, auth_token)

        tampered_params = {"CallSid": "CA999", "From": "+15551234567"}
        assert verify_twilio_signature(url, tampered_params, sig, auth_token) is False

    def test_rejects_tampered_url(self):
        auth_token = "my_twilio_token"
        url = "https://example.com/webhooks/twilio/voice"
        params = {"CallSid": "CA123"}
        sig = self._make_signature(url, params, auth_token)

        assert verify_twilio_signature(
            "https://evil.com/webhooks/twilio/voice", params, sig, auth_token
        ) is False

    def test_accepts_without_auth_token_dev_mode(self):
        """In dev mode (no auth token), accept all webhooks."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TWILIO_AUTH_TOKEN", None)
            assert verify_twilio_signature(
                "https://example.com", {}, "any_sig", auth_token=None
            ) is True

    def test_uses_env_var_token(self):
        auth_token = "env_token_456"
        url = "https://example.com/webhooks/twilio/sms"
        params = {"MessageSid": "SM123", "Body": "Hello"}
        sig = self._make_signature(url, params, auth_token)

        with patch.dict(os.environ, {"TWILIO_AUTH_TOKEN": auth_token}):
            assert verify_twilio_signature(url, params, sig) is True

    def test_empty_params(self):
        auth_token = "token123"
        url = "https://example.com/hook"
        params: dict[str, str] = {}
        sig = self._make_signature(url, params, auth_token)
        assert verify_twilio_signature(url, params, sig, auth_token) is True

    def test_params_sorted_correctly(self):
        """Parameters must be sorted alphabetically for signature computation."""
        auth_token = "token"
        url = "https://example.com"
        # Two sets with same params in different insertion order
        params1 = {"Zebra": "1", "Alpha": "2"}
        params2 = {"Alpha": "2", "Zebra": "1"}
        sig = self._make_signature(url, params1, auth_token)
        # Both should produce the same valid signature
        assert verify_twilio_signature(url, params2, sig, auth_token) is True


# ═══════════════════════════════════════════════════════════════════════
# 2. WebhookProcessor — Inbound Call
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorInboundCall:
    """Tests for processing Twilio inbound voice call webhooks."""

    def test_processes_inbound_call(self):
        processor = WebhookProcessor()
        payload = {
            "CallSid": "CA_test123",
            "From": "+15551234567",
            "To": "+15559876543",
            "CallStatus": "ringing",
        }
        result = processor.process_twilio_inbound_call(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "inbound_call"
        assert result["call_sid"] == "CA_test123"

    def test_returns_twiml(self):
        processor = WebhookProcessor()
        payload = {"CallSid": "CA_test", "From": "+1", "To": "+2"}
        result = processor.process_twilio_inbound_call(payload)
        assert "twiml" in result
        assert "<Response>" in result["twiml"]
        assert "<Say" in result["twiml"]
        assert "<Record" in result["twiml"]
        assert "maxLength" in result["twiml"]

    def test_twiml_has_recording_action(self):
        processor = WebhookProcessor()
        payload = {"CallSid": "CA_test", "From": "+1"}
        result = processor.process_twilio_inbound_call(payload)
        assert "/webhooks/twilio/recording" in result["twiml"]

    def test_custom_greeting(self):
        with patch.dict(os.environ, {"VOICE_GREETING": "Welcome to Acme Corp!"}):
            processor = WebhookProcessor()
            result = processor.process_twilio_inbound_call({"CallSid": "CA1"})
            assert "Welcome to Acme Corp!" in result["twiml"]

    def test_records_event_in_history(self):
        processor = WebhookProcessor()
        processor.process_twilio_inbound_call({"CallSid": "CA_hist"})
        assert len(processor.received_events) == 1
        assert processor.received_events[0]["event_type"] == "inbound_call"

    def test_dispatches_to_event_bus(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = ["task_voice_1"]
        processor = WebhookProcessor(event_bus=mock_bus)

        payload = {"CallSid": "CA_bus", "From": "+15551234567"}
        result = processor.process_twilio_inbound_call(payload)

        mock_bus.dispatch.assert_called_once()
        call_kwargs = mock_bus.dispatch.call_args[1]
        assert call_kwargs["event_type"] == "inbound_call"
        assert call_kwargs["source_agent_id"] == "twilio_webhook"
        assert result["tasks_dispatched"] == 1

    def test_works_without_event_bus(self):
        processor = WebhookProcessor(event_bus=None)
        result = processor.process_twilio_inbound_call({"CallSid": "CA_no_bus"})
        assert result["tasks_dispatched"] == 0

    def test_extracts_caller_info(self):
        processor = WebhookProcessor()
        payload = {"CallSid": "CA_info", "From": "+15551234567", "Caller": "+15551234567"}
        processor.process_twilio_inbound_call(payload)
        event = processor.received_events[-1]
        assert event["data"]["caller"] == "+15551234567"
        assert event["data"]["direction"] == "inbound"


# ═══════════════════════════════════════════════════════════════════════
# 3. WebhookProcessor — SMS Received
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorSmsReceived:
    """Tests for processing Twilio inbound SMS webhooks."""

    def test_processes_sms(self):
        processor = WebhookProcessor()
        payload = {
            "MessageSid": "SM_test123",
            "From": "+15551234567",
            "To": "+15559876543",
            "Body": "I need a quote for your services",
            "NumMedia": "0",
        }
        result = processor.process_twilio_sms_received(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "sms_received"
        assert result["message_sid"] == "SM_test123"
        assert "quote" in result["body_preview"]

    def test_returns_empty_twiml(self):
        processor = WebhookProcessor()
        result = processor.process_twilio_sms_received({"MessageSid": "SM1", "Body": "Hi"})
        assert "twiml" in result
        assert "<Response></Response>" in result["twiml"]

    def test_records_event_in_history(self):
        processor = WebhookProcessor()
        processor.process_twilio_sms_received({"MessageSid": "SM_hist", "Body": "Test"})
        assert len(processor.received_events) == 1
        assert processor.received_events[0]["event_type"] == "sms_received"

    def test_event_data_includes_body(self):
        processor = WebhookProcessor()
        processor.process_twilio_sms_received({
            "MessageSid": "SM_body",
            "Body": "Hello, I need help with my order",
        })
        event = processor.received_events[-1]
        assert event["data"]["body"] == "Hello, I need help with my order"

    def test_dispatches_to_event_bus(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = ["task_sms_1"]
        processor = WebhookProcessor(event_bus=mock_bus)

        result = processor.process_twilio_sms_received({
            "MessageSid": "SM_bus",
            "Body": "Help",
        })

        mock_bus.dispatch.assert_called_once()
        assert mock_bus.dispatch.call_args[1]["event_type"] == "sms_received"
        assert result["tasks_dispatched"] == 1

    def test_handles_media_count(self):
        processor = WebhookProcessor()
        processor.process_twilio_sms_received({
            "MessageSid": "SM_media",
            "Body": "Check this out",
            "NumMedia": "2",
        })
        event = processor.received_events[-1]
        assert event["data"]["num_media"] == 2

    def test_fallback_sms_sid(self):
        """Uses SmsSid if MessageSid is missing."""
        processor = WebhookProcessor()
        result = processor.process_twilio_sms_received({
            "SmsSid": "SM_fallback",
            "Body": "Test",
        })
        assert result["message_sid"] == "SM_fallback"

    def test_body_preview_truncated(self):
        processor = WebhookProcessor()
        long_body = "x" * 200
        result = processor.process_twilio_sms_received({
            "MessageSid": "SM_long",
            "Body": long_body,
        })
        assert len(result["body_preview"]) <= 100


# ═══════════════════════════════════════════════════════════════════════
# 4. WebhookProcessor — Recording
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorRecording:
    """Tests for processing Twilio recording completion webhooks."""

    def test_processes_recording(self):
        processor = WebhookProcessor()
        payload = {
            "CallSid": "CA_rec123",
            "RecordingSid": "RE_rec123",
            "RecordingUrl": "https://api.twilio.com/recordings/RE_rec123",
            "RecordingDuration": "45",
            "From": "+15551234567",
        }
        result = processor.process_twilio_recording(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "voice_message_received"
        assert result["recording_sid"] == "RE_rec123"
        assert result["recording_duration"] == 45

    def test_records_event_in_history(self):
        processor = WebhookProcessor()
        processor.process_twilio_recording({
            "CallSid": "CA_hist",
            "RecordingSid": "RE_hist",
            "RecordingUrl": "https://example.com/rec.mp3",
            "RecordingDuration": "30",
        })
        assert len(processor.received_events) == 1
        assert processor.received_events[0]["event_type"] == "voice_message_received"

    def test_event_data_includes_recording_url(self):
        processor = WebhookProcessor()
        processor.process_twilio_recording({
            "CallSid": "CA_url",
            "RecordingSid": "RE_url",
            "RecordingUrl": "https://api.twilio.com/recordings/RE_url",
            "RecordingDuration": "60",
        })
        event = processor.received_events[-1]
        assert event["data"]["recording_url"] == "https://api.twilio.com/recordings/RE_url"
        assert event["data"]["recording_duration"] == 60

    def test_dispatches_to_event_bus(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = ["task_rec_1"]
        processor = WebhookProcessor(event_bus=mock_bus)

        result = processor.process_twilio_recording({
            "CallSid": "CA_bus",
            "RecordingSid": "RE_bus",
            "RecordingUrl": "https://example.com/rec.mp3",
            "RecordingDuration": "10",
        })

        mock_bus.dispatch.assert_called_once()
        assert mock_bus.dispatch.call_args[1]["event_type"] == "voice_message_received"
        assert result["tasks_dispatched"] == 1

    def test_works_without_event_bus(self):
        processor = WebhookProcessor(event_bus=None)
        result = processor.process_twilio_recording({
            "CallSid": "CA_no",
            "RecordingSid": "RE_no",
            "RecordingDuration": "5",
        })
        assert result["tasks_dispatched"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 5. Combined Twilio Event History
# ═══════════════════════════════════════════════════════════════════════


class TestTwilioEventHistory:
    """Tests for Twilio event tracking alongside commerce events."""

    def test_accumulates_twilio_events(self):
        processor = WebhookProcessor()
        processor.process_twilio_inbound_call({"CallSid": "CA1"})
        processor.process_twilio_sms_received({"MessageSid": "SM1", "Body": "Hi"})
        processor.process_twilio_recording({
            "CallSid": "CA1",
            "RecordingSid": "RE1",
            "RecordingDuration": "10",
        })
        assert len(processor.received_events) == 3

    def test_mixed_commerce_and_twilio_events(self):
        processor = WebhookProcessor()
        processor.process_shopify_order_created({"id": 1, "total_price": "100"})
        processor.process_twilio_inbound_call({"CallSid": "CA1"})
        processor.process_twilio_sms_received({"MessageSid": "SM1", "Body": "Hi"})

        types = [e["event_type"] for e in processor.received_events]
        assert types == ["order_created", "inbound_call", "sms_received"]


# ═══════════════════════════════════════════════════════════════════════
# 6. FastAPI Router — Twilio Endpoints
# ═══════════════════════════════════════════════════════════════════════


_has_fastapi = True
try:
    import fastapi  # noqa: F401
except ImportError:
    _has_fastapi = False


@pytest.mark.skipif(not _has_fastapi, reason="FastAPI not installed")
class TestTwilioRouterEndpoints:
    """Tests that Twilio endpoints are registered on the router."""

    def test_router_has_twilio_voice(self):
        router = create_webhook_router()
        route_paths = [r.path for r in router.routes]
        assert "/twilio/voice" in route_paths

    def test_router_has_twilio_sms(self):
        router = create_webhook_router()
        route_paths = [r.path for r in router.routes]
        assert "/twilio/sms" in route_paths

    def test_router_has_twilio_recording(self):
        router = create_webhook_router()
        route_paths = [r.path for r in router.routes]
        assert "/twilio/recording" in route_paths

    def test_router_has_7_total_endpoints(self):
        """4 commerce + 3 twilio = 7 endpoints."""
        router = create_webhook_router()
        route_paths = [r.path for r in router.routes]
        assert len(route_paths) == 7
