"""
Tests for the Webhook Server — Shopify + Stripe webhook processing.

Covers:
- HMAC-SHA256 signature verification (Shopify)
- Stripe signature verification
- WebhookProcessor event processing (4 event types)
- Event bus dispatch integration
- FastAPI router creation
- Edge cases (missing secrets, malformed payloads)
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
    verify_shopify_hmac,
    verify_stripe_signature,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Shopify HMAC Verification
# ═══════════════════════════════════════════════════════════════════════


class TestVerifyShopifyHmac:
    """Tests for Shopify HMAC-SHA256 signature verification."""

    def test_accepts_valid_signature(self):
        secret = "my_shopify_secret"
        body = b'{"id": 12345, "total_price": "99.00"}'
        expected = base64.b64encode(
            hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        ).decode("utf-8")

        assert verify_shopify_hmac(body, expected, secret) is True

    def test_rejects_invalid_signature(self):
        secret = "my_shopify_secret"
        body = b'{"id": 12345}'
        assert verify_shopify_hmac(body, "invalid_hmac", secret) is False

    def test_rejects_tampered_body(self):
        secret = "my_shopify_secret"
        original_body = b'{"id": 12345}'
        hmac_header = base64.b64encode(
            hmac.new(secret.encode("utf-8"), original_body, hashlib.sha256).digest()
        ).decode("utf-8")

        tampered_body = b'{"id": 99999}'
        assert verify_shopify_hmac(tampered_body, hmac_header, secret) is False

    def test_accepts_without_secret_dev_mode(self):
        """In dev mode (no secret set), accept all webhooks."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure SHOPIFY_WEBHOOK_SECRET is not set
            os.environ.pop("SHOPIFY_WEBHOOK_SECRET", None)
            assert verify_shopify_hmac(b"anything", "anything", secret=None) is True

    def test_uses_env_var_secret(self):
        secret = "env_secret_123"
        body = b'{"order": true}'
        expected = base64.b64encode(
            hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        ).decode("utf-8")

        with patch.dict(os.environ, {"SHOPIFY_WEBHOOK_SECRET": secret}):
            assert verify_shopify_hmac(body, expected) is True


# ═══════════════════════════════════════════════════════════════════════
# 2. Stripe Signature Verification
# ═══════════════════════════════════════════════════════════════════════


class TestVerifyStripeSignature:
    """Tests for Stripe webhook signature verification."""

    def _make_signature(self, body: bytes, secret: str, timestamp: str = "1234567890") -> str:
        signed_payload = f"{timestamp}.{body.decode('utf-8')}"
        sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"t={timestamp},v1={sig}"

    def test_accepts_valid_signature(self):
        secret = "whsec_test123"
        body = b'{"type": "payment_intent.succeeded"}'
        sig_header = self._make_signature(body, secret)
        assert verify_stripe_signature(body, sig_header, secret) is True

    def test_rejects_invalid_signature(self):
        secret = "whsec_test123"
        body = b'{"type": "payment_intent.succeeded"}'
        assert verify_stripe_signature(body, "t=123,v1=invalid", secret) is False

    def test_rejects_tampered_body(self):
        secret = "whsec_test123"
        body = b'{"type": "payment_intent.succeeded"}'
        sig_header = self._make_signature(body, secret)

        tampered = b'{"type": "payment_intent.failed"}'
        assert verify_stripe_signature(tampered, sig_header, secret) is False

    def test_accepts_without_secret_dev_mode(self):
        """In dev mode (no secret), accept all webhooks."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STRIPE_WEBHOOK_SECRET", None)
            assert verify_stripe_signature(b"anything", "anything", secret=None) is True

    def test_handles_malformed_header(self):
        """Malformed signature header should return False, not crash."""
        secret = "whsec_test"
        assert verify_stripe_signature(b"body", "not_a_valid_header", secret) is False

    def test_uses_env_var_secret(self):
        secret = "whsec_env_secret"
        body = b'{"data": true}'
        sig_header = self._make_signature(body, secret)

        with patch.dict(os.environ, {"STRIPE_WEBHOOK_SECRET": secret}):
            assert verify_stripe_signature(body, sig_header) is True


# ═══════════════════════════════════════════════════════════════════════
# 3. WebhookProcessor — Order Created
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorOrderCreated:
    """Tests for processing Shopify order/create webhooks."""

    def test_extracts_order_data(self):
        processor = WebhookProcessor()
        payload = {
            "id": 12345,
            "email": "whale@acme.com",
            "customer": {"first_name": "Jane", "last_name": "Doe"},
            "total_price": "999.99",
            "currency": "USD",
            "financial_status": "paid",
            "fulfillment_status": None,
            "line_items": [{"title": "Widget", "quantity": 1}],
        }

        result = processor.process_shopify_order_created(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "order_created"
        assert result["order_id"] == "12345"

    def test_detects_vip_order(self):
        processor = WebhookProcessor()
        payload = {
            "id": 99,
            "email": "vip@corp.com",
            "total_price": "5000.00",
        }

        processor.process_shopify_order_created(payload)
        event = processor.received_events[-1]
        assert event["data"]["is_vip"] is True

    def test_non_vip_order(self):
        processor = WebhookProcessor()
        payload = {
            "id": 100,
            "email": "regular@corp.com",
            "total_price": "49.99",
        }

        processor.process_shopify_order_created(payload)
        event = processor.received_events[-1]
        assert event["data"]["is_vip"] is False

    def test_dispatches_to_event_bus(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = ["task_1"]
        processor = WebhookProcessor(event_bus=mock_bus)

        payload = {"id": 1, "total_price": "100.00"}
        result = processor.process_shopify_order_created(payload)

        mock_bus.dispatch.assert_called_once()
        call_kwargs = mock_bus.dispatch.call_args
        assert call_kwargs[1]["event_type"] == "order_created"
        assert result["tasks_dispatched"] == 1

    def test_works_without_event_bus(self):
        processor = WebhookProcessor(event_bus=None)
        payload = {"id": 1, "total_price": "50.00"}

        result = processor.process_shopify_order_created(payload)
        assert result["tasks_dispatched"] == 0

    def test_handles_missing_customer_info(self):
        processor = WebhookProcessor()
        payload = {"id": 1, "total_price": "100.00"}

        result = processor.process_shopify_order_created(payload)
        assert result["status"] == "received"

    def test_records_event_in_history(self):
        processor = WebhookProcessor()
        assert len(processor.received_events) == 0

        processor.process_shopify_order_created({"id": 1, "total_price": "10"})
        assert len(processor.received_events) == 1
        assert processor.received_events[0]["event_type"] == "order_created"


# ═══════════════════════════════════════════════════════════════════════
# 4. WebhookProcessor — Order Paid
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorOrderPaid:
    """Tests for processing Shopify order/paid webhooks."""

    def test_processes_paid_order(self):
        processor = WebhookProcessor()
        payload = {
            "id": 555,
            "total_price": "250.00",
            "currency": "EUR",
            "gateway": "stripe",
        }

        result = processor.process_shopify_order_paid(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "order_paid"
        assert result["order_id"] == "555"

    def test_dispatches_order_paid_event(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = []
        processor = WebhookProcessor(event_bus=mock_bus)

        processor.process_shopify_order_paid({"id": 1, "total_price": "50"})

        mock_bus.dispatch.assert_called_once()
        assert mock_bus.dispatch.call_args[1]["event_type"] == "order_paid"

    def test_extracts_payment_gateway(self):
        processor = WebhookProcessor()
        payload = {"id": 1, "total_price": "100", "gateway": "paypal"}

        processor.process_shopify_order_paid(payload)
        event = processor.received_events[-1]
        assert event["data"]["payment_gateway"] == "paypal"


# ═══════════════════════════════════════════════════════════════════════
# 5. WebhookProcessor — Stripe Payment Success
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorStripeSuccess:
    """Tests for processing Stripe payment_intent.succeeded webhooks."""

    def test_processes_payment_success(self):
        processor = WebhookProcessor()
        payload = {
            "data": {
                "object": {
                    "id": "pi_abc123",
                    "amount": 9999,
                    "currency": "usd",
                    "receipt_email": "whale@acme.com",
                }
            }
        }

        result = processor.process_stripe_payment_success(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "payment_received"
        assert result["payment_intent_id"] == "pi_abc123"

    def test_records_payment_event(self):
        processor = WebhookProcessor()
        payload = {
            "data": {"object": {"id": "pi_xyz", "amount": 5000, "currency": "eur"}}
        }

        processor.process_stripe_payment_success(payload)
        event = processor.received_events[-1]
        assert event["event_type"] == "payment_received"
        assert event["data"]["amount_cents"] == 5000
        assert event["data"]["status"] == "succeeded"

    def test_dispatches_to_event_bus(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = ["t1", "t2"]
        processor = WebhookProcessor(event_bus=mock_bus)

        payload = {"data": {"object": {"id": "pi_bus", "amount": 100}}}
        result = processor.process_stripe_payment_success(payload)

        mock_bus.dispatch.assert_called_once()
        assert mock_bus.dispatch.call_args[1]["event_type"] == "payment_received"
        assert result["tasks_dispatched"] == 2

    def test_flat_payload_fallback(self):
        """When payload is flat (no data.object wrapper), use it directly."""
        processor = WebhookProcessor()
        payload = {"id": "pi_flat", "amount": 1500, "currency": "gbp"}

        result = processor.process_stripe_payment_success(payload)
        assert result["payment_intent_id"] == "pi_flat"


# ═══════════════════════════════════════════════════════════════════════
# 6. WebhookProcessor — Stripe Payment Failed
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorStripeFailed:
    """Tests for processing Stripe payment_intent.payment_failed webhooks."""

    def test_processes_payment_failure(self):
        processor = WebhookProcessor()
        payload = {
            "data": {
                "object": {
                    "id": "pi_fail123",
                    "amount": 3000,
                    "currency": "usd",
                    "last_payment_error": {
                        "message": "Card declined"
                    },
                }
            }
        }

        result = processor.process_stripe_payment_failed(payload)
        assert result["status"] == "received"
        assert result["event_type"] == "payment_failed"
        assert result["payment_intent_id"] == "pi_fail123"

    def test_extracts_error_message(self):
        processor = WebhookProcessor()
        payload = {
            "data": {
                "object": {
                    "id": "pi_err",
                    "amount": 500,
                    "last_payment_error": {"message": "Insufficient funds"},
                }
            }
        }

        processor.process_stripe_payment_failed(payload)
        event = processor.received_events[-1]
        assert event["data"]["error_message"] == "Insufficient funds"
        assert event["data"]["status"] == "failed"

    def test_default_error_message(self):
        processor = WebhookProcessor()
        payload = {"data": {"object": {"id": "pi_noerr", "amount": 100}}}

        processor.process_stripe_payment_failed(payload)
        event = processor.received_events[-1]
        assert event["data"]["error_message"] == "Payment failed"

    def test_dispatches_failure_event(self):
        mock_bus = MagicMock()
        mock_bus.dispatch.return_value = []
        processor = WebhookProcessor(event_bus=mock_bus)

        payload = {"data": {"object": {"id": "pi_x", "amount": 100}}}
        processor.process_stripe_payment_failed(payload)

        mock_bus.dispatch.assert_called_once()
        assert mock_bus.dispatch.call_args[1]["event_type"] == "payment_failed"


# ═══════════════════════════════════════════════════════════════════════
# 7. WebhookProcessor — Event History
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookProcessorHistory:
    """Tests for event recording and tracking."""

    def test_accumulates_events(self):
        processor = WebhookProcessor()
        processor.process_shopify_order_created({"id": 1, "total_price": "10"})
        processor.process_shopify_order_paid({"id": 1, "total_price": "10"})
        processor.process_stripe_payment_success(
            {"data": {"object": {"id": "pi_1", "amount": 1000}}}
        )
        assert len(processor.received_events) == 3

    def test_events_are_immutable_copy(self):
        processor = WebhookProcessor()
        processor.process_shopify_order_created({"id": 1, "total_price": "10"})

        events = processor.received_events
        events.clear()  # Clearing the copy should not affect internal state
        assert len(processor.received_events) == 1

    def test_event_types_tracked_correctly(self):
        processor = WebhookProcessor()
        processor.process_shopify_order_created({"id": 1, "total_price": "10"})
        processor.process_stripe_payment_failed(
            {"data": {"object": {"id": "pi_f", "amount": 100}}}
        )

        types = [e["event_type"] for e in processor.received_events]
        assert types == ["order_created", "payment_failed"]


# ═══════════════════════════════════════════════════════════════════════
# 8. FastAPI Router Factory
# ═══════════════════════════════════════════════════════════════════════


_has_fastapi = True
try:
    import fastapi  # noqa: F401
except ImportError:
    _has_fastapi = False


@pytest.mark.skipif(not _has_fastapi, reason="FastAPI not installed")
class TestCreateWebhookRouter:
    """Tests for the FastAPI router factory."""

    def test_creates_router(self):
        router = create_webhook_router()
        assert router is not None

    def test_router_has_routes(self):
        router = create_webhook_router()
        # FastAPI APIRouter stores routes
        route_paths = [r.path for r in router.routes]
        assert "/shopify/orders/create" in route_paths
        assert "/shopify/orders/paid" in route_paths
        assert "/stripe/payment/success" in route_paths
        assert "/stripe/payment/failed" in route_paths

    def test_router_has_prefix(self):
        router = create_webhook_router()
        assert router.prefix == "/webhooks"

    def test_router_accepts_event_bus(self):
        mock_bus = MagicMock()
        router = create_webhook_router(event_bus=mock_bus)
        assert router is not None


class TestCreateWebhookRouterWithoutFastapi:
    """Tests for graceful degradation without FastAPI."""

    def test_returns_none_without_fastapi(self):
        """When FastAPI is not available, router creation returns None."""
        if _has_fastapi:
            pytest.skip("FastAPI is installed, can't test absence")
        router = create_webhook_router()
        assert router is None
