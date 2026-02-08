"""
Email MCP tools for the Sovereign Venture Engine.

Wraps EmailEngine.send_email() as an MCP tool with the critical
@sandboxed_tool decorator applied. In non-production environments,
the email is intercepted and logged to sandbox_logs/ instead of
being actually sent.

This is the safety net for the most dangerous tool in the system.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.safety.sandbox import sandboxed_tool

logger = logging.getLogger(__name__)


@sandboxed_tool("send_email")
async def send_email(
    to_email: str,
    to_name: str,
    subject: str,
    body_html: str,
    body_text: str = "",
    from_name: str = "Enclave Guard",
    unsubscribe_url: Optional[str] = None,
    tracking_id: Optional[str] = None,
    *,
    _email_engine: Any = None,
) -> dict[str, Any]:
    """
    Send an email via the configured provider (SendGrid/Mailgun).

    SAFETY: This tool is wrapped with @sandboxed_tool("send_email").
    In development/staging/test environments, the call is intercepted
    and logged to sandbox_logs/send_email.jsonl instead of sending.

    Args:
        to_email: Recipient email address.
        to_name: Recipient name.
        subject: Email subject line.
        body_html: HTML email body.
        body_text: Plain text fallback (auto-generated from HTML if empty).
        from_name: Sender display name.
        unsubscribe_url: One-click unsubscribe URL (required for CAN-SPAM).
        tracking_id: Internal tracking correlation ID.
        _email_engine: Injected EmailEngine instance (for testing/DI).

    Returns:
        Dict with message_id, status, and provider info.
    """
    if _email_engine is None:
        from core.outreach.email_engine import EmailEngine

        _email_engine = EmailEngine()

    # Auto-generate plain text if not provided
    if not body_text:
        # Simple HTML-to-text: strip tags
        import re
        body_text = re.sub(r"<[^>]+>", "", body_html).strip()

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "send_email",
            "to_email": to_email,
            "subject": subject[:50],
        },
    )

    result = await _email_engine.send_email(
        to_email=to_email,
        to_name=to_name,
        subject=subject,
        body_html=body_html,
        body_text=body_text,
        from_name=from_name,
        unsubscribe_url=unsubscribe_url,
        tracking_id=tracking_id,
    )

    logger.info(
        "email_send_result",
        extra={
            "tool_name": "send_email",
            "status": result.get("status"),
            "message_id": result.get("message_id", ""),
        },
    )

    return result
