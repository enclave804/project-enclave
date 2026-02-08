"""
Email sending engine for Project Enclave.

Handles email dispatch via SendGrid or Mailgun, including
open/click tracking setup and delivery monitoring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class EmailEngine:
    """
    Email sending engine with provider abstraction.

    Supports SendGrid and Mailgun. Handles:
    - Email sending with tracking
    - Unsubscribe header injection (RFC 8058)
    - Rate limiting awareness
    - Delivery status tracking
    """

    def __init__(
        self,
        provider: str = "sendgrid",
        api_key_env: Optional[str] = None,
        sending_domain: Optional[str] = None,
        reply_to: Optional[str] = None,
    ):
        self.provider = provider

        if provider == "sendgrid":
            self.api_key = os.environ.get(
                api_key_env or "SENDGRID_API_KEY", ""
            )
            self.base_url = "https://api.sendgrid.com/v3"
        elif provider == "mailgun":
            self.api_key = os.environ.get(
                api_key_env or "MAILGUN_API_KEY", ""
            )
            self.base_url = f"https://api.mailgun.net/v3/{sending_domain}"
        else:
            raise ValueError(f"Unsupported email provider: {provider}")

        self.sending_domain = sending_domain or ""
        self.reply_to = reply_to or ""

    async def send_email(
        self,
        to_email: str,
        to_name: str,
        subject: str,
        body_html: str,
        body_text: str,
        from_email: Optional[str] = None,
        from_name: str = "Enclave Guard",
        unsubscribe_url: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        tracking_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send an email via the configured provider.

        Args:
            to_email: Recipient email address.
            to_name: Recipient name.
            subject: Email subject line.
            body_html: HTML body content.
            body_text: Plain text body content.
            from_email: Sender email (defaults to noreply@sending_domain).
            from_name: Sender display name.
            unsubscribe_url: One-click unsubscribe URL (required for CAN-SPAM).
            custom_headers: Additional headers to include.
            tracking_id: Internal tracking ID for correlation.

        Returns:
            Dict with 'message_id', 'status', and provider-specific data.
        """
        from_email = from_email or f"noreply@{self.sending_domain}"

        if self.provider == "sendgrid":
            return await self._send_sendgrid(
                to_email=to_email,
                to_name=to_name,
                subject=subject,
                body_html=body_html,
                body_text=body_text,
                from_email=from_email,
                from_name=from_name,
                unsubscribe_url=unsubscribe_url,
                custom_headers=custom_headers,
                tracking_id=tracking_id,
            )
        elif self.provider == "mailgun":
            return await self._send_mailgun(
                to_email=to_email,
                to_name=to_name,
                subject=subject,
                body_html=body_html,
                body_text=body_text,
                from_email=from_email,
                from_name=from_name,
                unsubscribe_url=unsubscribe_url,
                custom_headers=custom_headers,
                tracking_id=tracking_id,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _send_sendgrid(self, **kwargs: Any) -> dict[str, Any]:
        """Send via SendGrid v3 API."""
        headers_list = []

        # Add RFC 8058 one-click unsubscribe
        if kwargs.get("unsubscribe_url"):
            headers_list.append({
                "List-Unsubscribe": f"<{kwargs['unsubscribe_url']}>",
                "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
            })

        payload = {
            "personalizations": [
                {
                    "to": [{"email": kwargs["to_email"], "name": kwargs["to_name"]}],
                    "subject": kwargs["subject"],
                }
            ],
            "from": {
                "email": kwargs["from_email"],
                "name": kwargs["from_name"],
            },
            "reply_to": {"email": self.reply_to} if self.reply_to else None,
            "content": [
                {"type": "text/plain", "value": kwargs["body_text"]},
                {"type": "text/html", "value": kwargs["body_html"]},
            ],
            "tracking_settings": {
                "click_tracking": {"enable": True},
                "open_tracking": {"enable": True},
            },
        }

        # Add custom headers
        if kwargs.get("custom_headers") or headers_list:
            all_headers = {}
            for h in headers_list:
                all_headers.update(h)
            if kwargs.get("custom_headers"):
                all_headers.update(kwargs["custom_headers"])
            payload["headers"] = all_headers

        # Add tracking ID as custom arg
        if kwargs.get("tracking_id"):
            payload["personalizations"][0]["custom_args"] = {
                "tracking_id": kwargs["tracking_id"]
            }

        # Clean None values
        payload = {k: v for k, v in payload.items() if v is not None}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/mail/send",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if response.status_code in (200, 201, 202):
                message_id = response.headers.get("X-Message-Id", "")
                return {
                    "message_id": message_id,
                    "status": "sent",
                    "provider": "sendgrid",
                    "status_code": response.status_code,
                }
            else:
                logger.error(
                    f"SendGrid error: {response.status_code} - {response.text}"
                )
                return {
                    "message_id": "",
                    "status": "failed",
                    "provider": "sendgrid",
                    "status_code": response.status_code,
                    "error": response.text,
                }

    async def _send_mailgun(self, **kwargs: Any) -> dict[str, Any]:
        """Send via Mailgun API."""
        data = {
            "from": f"{kwargs['from_name']} <{kwargs['from_email']}>",
            "to": f"{kwargs['to_name']} <{kwargs['to_email']}>",
            "subject": kwargs["subject"],
            "text": kwargs["body_text"],
            "html": kwargs["body_html"],
            "o:tracking": "yes",
            "o:tracking-clicks": "yes",
            "o:tracking-opens": "yes",
        }

        if self.reply_to:
            data["h:Reply-To"] = self.reply_to

        if kwargs.get("unsubscribe_url"):
            data["h:List-Unsubscribe"] = f"<{kwargs['unsubscribe_url']}>"
            data["h:List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

        if kwargs.get("tracking_id"):
            data["v:tracking_id"] = kwargs["tracking_id"]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                auth=("api", self.api_key),
                data=data,
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "message_id": result.get("id", ""),
                    "status": "sent",
                    "provider": "mailgun",
                    "message": result.get("message", ""),
                }
            else:
                logger.error(
                    f"Mailgun error: {response.status_code} - {response.text}"
                )
                return {
                    "message_id": "",
                    "status": "failed",
                    "provider": "mailgun",
                    "status_code": response.status_code,
                    "error": response.text,
                }

    @staticmethod
    def text_to_html(text: str) -> str:
        """Convert plain text email to minimal HTML."""
        import html
        escaped = html.escape(text)
        paragraphs = escaped.split("\n\n")
        html_parts = [f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs]
        return f"""<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; color: #333;">
{''.join(html_parts)}
</body>
</html>"""
