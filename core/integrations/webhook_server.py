"""
Webhook receiver for external commerce events.

Listens for incoming webhooks from Shopify and Stripe,
validates HMAC signatures, and dispatches events to the EventBus.

This is a FastAPI router — mount it on your main app:

    from core.integrations.webhook_server import create_webhook_router
    app.include_router(create_webhook_router(event_bus))

Endpoints:
    POST /webhooks/shopify/orders/create   -> order_created event
    POST /webhooks/shopify/orders/paid     -> order_paid event
    POST /webhooks/stripe/payment/success  -> payment_received event
    POST /webhooks/stripe/payment/failed   -> payment_failed event

Security:
    - Shopify HMAC-SHA256 signature verification (SHOPIFY_WEBHOOK_SECRET)
    - Stripe signature verification (STRIPE_WEBHOOK_SECRET)
    - Falls back to accepting all webhooks if secrets are not set (dev mode)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature Verification
# ---------------------------------------------------------------------------


def verify_shopify_hmac(
    body: bytes,
    hmac_header: str,
    secret: Optional[str] = None,
) -> bool:
    """
    Verify a Shopify webhook HMAC-SHA256 signature.

    Args:
        body: Raw request body bytes.
        hmac_header: Value of X-Shopify-Hmac-SHA256 header.
        secret: Shopify webhook secret. Falls back to SHOPIFY_WEBHOOK_SECRET env var.

    Returns:
        True if signature is valid, False otherwise.
    """
    secret = secret or os.environ.get("SHOPIFY_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("shopify_webhook_secret_not_set: accepting webhook without verification")
        return True

    import base64
    computed = base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")

    return hmac.compare_digest(computed, hmac_header)


def verify_stripe_signature(
    body: bytes,
    sig_header: str,
    secret: Optional[str] = None,
) -> bool:
    """
    Verify a Stripe webhook signature.

    Args:
        body: Raw request body bytes.
        sig_header: Value of Stripe-Signature header.
        secret: Stripe webhook secret. Falls back to STRIPE_WEBHOOK_SECRET env var.

    Returns:
        True if signature is valid, False otherwise.
    """
    secret = secret or os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("stripe_webhook_secret_not_set: accepting webhook without verification")
        return True

    # Stripe signature format: t=timestamp,v1=signature
    try:
        parts = dict(pair.split("=", 1) for pair in sig_header.split(","))
        timestamp = parts.get("t", "")
        expected_sig = parts.get("v1", "")

        signed_payload = f"{timestamp}.{body.decode('utf-8')}"
        computed = hmac.new(
            secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, expected_sig)
    except Exception as e:
        logger.error("stripe_signature_verification_failed", extra={"error": str(e)})
        return False


# ---------------------------------------------------------------------------
# Webhook Event Processing
# ---------------------------------------------------------------------------


class WebhookProcessor:
    """
    Processes incoming webhooks and dispatches events to the EventBus.

    This is decoupled from any web framework — it accepts raw payloads
    and returns structured results. The FastAPI router wraps this.
    """

    def __init__(self, event_bus: Any = None):
        self.event_bus = event_bus
        self._received_events: list[dict[str, Any]] = []

    def process_shopify_order_created(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process a Shopify order/create webhook.

        Extracts order data and dispatches an 'order_created' event.
        """
        order_id = payload.get("id") or payload.get("order_id", "unknown")
        customer_email = (
            payload.get("email")
            or payload.get("customer", {}).get("email", "")
        )
        customer_name = (
            payload.get("customer", {}).get("first_name", "")
            + " "
            + payload.get("customer", {}).get("last_name", "")
        ).strip() or payload.get("customer_name", "")

        total_price = float(payload.get("total_price", 0))

        event_data = {
            "order_id": str(order_id),
            "customer_email": customer_email,
            "customer_name": customer_name,
            "total_price": total_price,
            "currency": payload.get("currency", "USD"),
            "financial_status": payload.get("financial_status", ""),
            "fulfillment_status": payload.get("fulfillment_status", ""),
            "line_items": payload.get("line_items", []),
            "is_vip": total_price >= 500.0,
            "source": "shopify_webhook",
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

        self._received_events.append({
            "event_type": "order_created",
            "data": event_data,
        })

        # Dispatch to EventBus
        tasks_created = []
        if self.event_bus:
            tasks_created = self.event_bus.dispatch(
                event_type="order_created",
                payload=event_data,
                source_agent_id="shopify_webhook",
            )

        logger.info(
            "webhook_order_created",
            extra={
                "order_id": str(order_id),
                "total_price": total_price,
                "is_vip": event_data["is_vip"],
                "tasks_dispatched": len(tasks_created),
            },
        )

        return {
            "status": "received",
            "event_type": "order_created",
            "order_id": str(order_id),
            "tasks_dispatched": len(tasks_created),
        }

    def process_shopify_order_paid(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a Shopify order/paid webhook."""
        order_id = payload.get("id") or payload.get("order_id", "unknown")
        total_price = float(payload.get("total_price", 0))

        event_data = {
            "order_id": str(order_id),
            "total_price": total_price,
            "currency": payload.get("currency", "USD"),
            "payment_gateway": payload.get("gateway", ""),
            "source": "shopify_webhook",
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

        self._received_events.append({
            "event_type": "order_paid",
            "data": event_data,
        })

        tasks_created = []
        if self.event_bus:
            tasks_created = self.event_bus.dispatch(
                event_type="order_paid",
                payload=event_data,
                source_agent_id="shopify_webhook",
            )

        logger.info(
            "webhook_order_paid",
            extra={"order_id": str(order_id), "total_price": total_price},
        )

        return {
            "status": "received",
            "event_type": "order_paid",
            "order_id": str(order_id),
            "tasks_dispatched": len(tasks_created),
        }

    def process_stripe_payment_success(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a Stripe payment_intent.succeeded webhook."""
        data = payload.get("data", {}).get("object", payload)
        payment_intent_id = data.get("id", "unknown")
        amount = data.get("amount", 0)
        customer_email = data.get("receipt_email") or data.get("customer_email", "")

        event_data = {
            "payment_intent_id": payment_intent_id,
            "amount_cents": amount,
            "currency": data.get("currency", "usd"),
            "customer_email": customer_email,
            "status": "succeeded",
            "source": "stripe_webhook",
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

        self._received_events.append({
            "event_type": "payment_received",
            "data": event_data,
        })

        tasks_created = []
        if self.event_bus:
            tasks_created = self.event_bus.dispatch(
                event_type="payment_received",
                payload=event_data,
                source_agent_id="stripe_webhook",
            )

        logger.info(
            "webhook_payment_received",
            extra={
                "payment_intent_id": payment_intent_id[:16],
                "amount_cents": amount,
            },
        )

        return {
            "status": "received",
            "event_type": "payment_received",
            "payment_intent_id": payment_intent_id,
            "tasks_dispatched": len(tasks_created),
        }

    def process_stripe_payment_failed(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a Stripe payment_intent.payment_failed webhook."""
        data = payload.get("data", {}).get("object", payload)
        payment_intent_id = data.get("id", "unknown")
        error_message = (
            data.get("last_payment_error", {}).get("message", "")
            or data.get("error_message", "Payment failed")
        )

        event_data = {
            "payment_intent_id": payment_intent_id,
            "amount_cents": data.get("amount", 0),
            "currency": data.get("currency", "usd"),
            "status": "failed",
            "error_message": error_message,
            "source": "stripe_webhook",
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

        self._received_events.append({
            "event_type": "payment_failed",
            "data": event_data,
        })

        tasks_created = []
        if self.event_bus:
            tasks_created = self.event_bus.dispatch(
                event_type="payment_failed",
                payload=event_data,
                source_agent_id="stripe_webhook",
            )

        logger.info(
            "webhook_payment_failed",
            extra={
                "payment_intent_id": payment_intent_id[:16],
                "error": error_message[:100],
            },
        )

        return {
            "status": "received",
            "event_type": "payment_failed",
            "payment_intent_id": payment_intent_id,
            "tasks_dispatched": len(tasks_created),
        }

    @property
    def received_events(self) -> list[dict[str, Any]]:
        """Return all events received (useful for testing)."""
        return list(self._received_events)


# ---------------------------------------------------------------------------
# FastAPI Router Factory
# ---------------------------------------------------------------------------


def create_webhook_router(event_bus: Any = None) -> Any:
    """
    Create a FastAPI router for webhook endpoints.

    Usage:
        from fastapi import FastAPI
        from core.integrations.webhook_server import create_webhook_router

        app = FastAPI()
        app.include_router(create_webhook_router(event_bus))

    Args:
        event_bus: EventBus instance for dispatching events.

    Returns:
        FastAPI APIRouter with webhook endpoints.
    """
    try:
        from fastapi import APIRouter, Request, Response
    except ImportError:
        logger.warning("fastapi_not_installed: webhook router unavailable")
        return None

    router = APIRouter(prefix="/webhooks", tags=["webhooks"])
    processor = WebhookProcessor(event_bus=event_bus)

    @router.post("/shopify/orders/create")
    async def shopify_order_created(request: Request):
        body = await request.body()
        hmac_header = request.headers.get("X-Shopify-Hmac-SHA256", "")

        if not verify_shopify_hmac(body, hmac_header):
            return Response(content="Invalid signature", status_code=401)

        payload = json.loads(body)
        result = processor.process_shopify_order_created(payload)
        return result

    @router.post("/shopify/orders/paid")
    async def shopify_order_paid(request: Request):
        body = await request.body()
        hmac_header = request.headers.get("X-Shopify-Hmac-SHA256", "")

        if not verify_shopify_hmac(body, hmac_header):
            return Response(content="Invalid signature", status_code=401)

        payload = json.loads(body)
        result = processor.process_shopify_order_paid(payload)
        return result

    @router.post("/stripe/payment/success")
    async def stripe_payment_success(request: Request):
        body = await request.body()
        sig_header = request.headers.get("Stripe-Signature", "")

        if not verify_stripe_signature(body, sig_header):
            return Response(content="Invalid signature", status_code=401)

        payload = json.loads(body)
        result = processor.process_stripe_payment_success(payload)
        return result

    @router.post("/stripe/payment/failed")
    async def stripe_payment_failed(request: Request):
        body = await request.body()
        sig_header = request.headers.get("Stripe-Signature", "")

        if not verify_stripe_signature(body, sig_header):
            return Response(content="Invalid signature", status_code=401)

        payload = json.loads(body)
        result = processor.process_stripe_payment_failed(payload)
        return result

    return router
