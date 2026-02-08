"""
n8n integration client for Project Enclave.

Triggers n8n workflows via webhook and handles incoming webhook data.
n8n handles the "nervous system" — connecting to Gmail, LinkedIn, banks,
and other external services.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class N8NClient:
    """
    Client for triggering n8n workflows via webhook.

    n8n must be self-hosted (v1.121.0+) with webhook endpoints
    configured for each workflow.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        self.base_url = (
            base_url
            or os.environ.get("N8N_BASE_URL", "http://localhost:5678")
        ).rstrip("/")
        self.webhook_secret = (
            webhook_secret or os.environ.get("N8N_WEBHOOK_SECRET", "")
        )

    async def trigger_webhook(
        self,
        webhook_path: str,
        payload: dict[str, Any],
        method: str = "POST",
    ) -> dict[str, Any]:
        """
        Trigger an n8n webhook workflow.

        Args:
            webhook_path: The webhook path (e.g., '/webhook/send-email').
            webhook_path should NOT include the base URL.
            payload: JSON payload to send.
            method: HTTP method (default POST).

        Returns:
            Response from n8n workflow.
        """
        url = f"{self.base_url}/webhook/{webhook_path.lstrip('/')}"

        headers = {"Content-Type": "application/json"}
        if self.webhook_secret:
            headers["X-Webhook-Secret"] = self.webhook_secret

        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "POST":
                response = await client.post(url, json=payload, headers=headers)
            elif method.upper() == "GET":
                response = await client.get(url, params=payload, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()

            try:
                return response.json()
            except Exception:
                return {"status": "ok", "text": response.text}

    # ------------------------------------------------------------------
    # Pre-built workflow triggers
    # ------------------------------------------------------------------

    async def trigger_send_email(
        self,
        to_email: str,
        to_name: str,
        subject: str,
        body_html: str,
        body_text: str,
        reply_to: str,
        unsubscribe_url: str,
        tracking_id: str,
        physical_address: str,
    ) -> dict[str, Any]:
        """
        Trigger the email sending workflow in n8n.

        The n8n workflow handles:
        1. Sending via SendGrid/Mailgun
        2. Adding tracking pixels
        3. Injecting unsubscribe headers
        4. Appending physical address
        5. Returning the message ID
        """
        return await self.trigger_webhook(
            "send-email",
            {
                "to_email": to_email,
                "to_name": to_name,
                "subject": subject,
                "body_html": body_html,
                "body_text": body_text,
                "reply_to": reply_to,
                "unsubscribe_url": unsubscribe_url,
                "tracking_id": tracking_id,
                "physical_address": physical_address,
            },
        )

    async def trigger_daily_pipeline(
        self, vertical_id: str
    ) -> dict[str, Any]:
        """
        Trigger the daily pipeline run via n8n.

        This kicks off the lead processing batch for a vertical.
        """
        return await self.trigger_webhook(
            "daily-pipeline",
            {"vertical_id": vertical_id},
        )

    async def trigger_nightly_classification(
        self, vertical_id: str
    ) -> dict[str, Any]:
        """
        Trigger the nightly reply classification job.

        Classifies sentiment and intent for all unprocessed replies.
        """
        return await self.trigger_webhook(
            "nightly-classification",
            {"vertical_id": vertical_id},
        )

    async def trigger_weekly_pattern_extraction(
        self, vertical_id: str
    ) -> dict[str, Any]:
        """
        Trigger the weekly pattern extraction job.

        Analyzes outreach results and extracts winning patterns for RAG.
        """
        return await self.trigger_webhook(
            "weekly-patterns",
            {"vertical_id": vertical_id},
        )

    async def health_check(self) -> bool:
        """Check if n8n is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/healthz")
                return response.status_code == 200
        except Exception:
            return False
