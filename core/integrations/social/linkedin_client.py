"""
LinkedIn API client for the Sovereign Venture Engine.

Uses the LinkedIn Marketing API for posting text + image shares.
Automatically falls back to mock mode when API credentials are missing.

Environment Variables:
    LINKEDIN_ACCESS_TOKEN   — OAuth 2.0 access token
    LINKEDIN_ORG_ID         — Organization/Company page ID (for org posts)
    LINKEDIN_PERSON_URN     — Person URN for personal posts (urn:li:person:XXX)

Mock Mode:
    When LINKEDIN_ACCESS_TOKEN is not set, all operations are logged to
    storage/social_mock.json instead of calling the real API.

Usage:
    client = LinkedInClient.from_env()
    result = await client.post_share(
        text="Cybersecurity trends for 2025 🔒",
        image_url="https://example.com/image.png",
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Default mock storage path
MOCK_STORAGE_DIR = Path(__file__).parent.parent.parent.parent / "storage"
MOCK_STORAGE_FILE = MOCK_STORAGE_DIR / "social_mock.json"

LINKEDIN_API_BASE = "https://api.linkedin.com/v2"


class LinkedInClient:
    """
    LinkedIn API client with automatic mock mode fallback.

    When real credentials are available, uses LinkedIn REST API.
    When credentials are missing, logs operations to a JSON file.
    """

    def __init__(
        self,
        access_token: str = "",
        org_id: str = "",
        person_urn: str = "",
        *,
        mock_mode: Optional[bool] = None,
        mock_storage_path: Optional[Path] = None,
    ):
        self.access_token = access_token
        self.org_id = org_id
        self.person_urn = person_urn

        # Auto-detect mock mode
        if mock_mode is None:
            self.mock_mode = not bool(access_token)
        else:
            self.mock_mode = mock_mode

        self.mock_storage_path = mock_storage_path or MOCK_STORAGE_FILE

        if self.mock_mode:
            logger.info(
                "linkedin_mock_mode",
                extra={"reason": "LINKEDIN_ACCESS_TOKEN not set"},
            )
        else:
            logger.info("linkedin_client_initialized")

    @classmethod
    def from_env(cls, **kwargs: Any) -> LinkedInClient:
        """Create a LinkedInClient from environment variables."""
        return cls(
            access_token=os.getenv("LINKEDIN_ACCESS_TOKEN", ""),
            org_id=os.getenv("LINKEDIN_ORG_ID", ""),
            person_urn=os.getenv("LINKEDIN_PERSON_URN", ""),
            **kwargs,
        )

    # ─── Mock Helpers ──────────────────────────────────────────────────

    def _log_mock_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Log a mock action to the JSON storage file."""
        entry = {
            "platform": "linkedin",
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "mock": True,
        }

        self.mock_storage_path.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        if self.mock_storage_path.exists():
            try:
                raw = self.mock_storage_path.read_text()
                entries = json.loads(raw) if raw.strip() else []
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append(entry)
        entries = entries[-500:]
        self.mock_storage_path.write_text(json.dumps(entries, indent=2))

        logger.info(
            f"linkedin_mock_{action}",
            extra={"data_preview": str(data)[:200]},
        )
        return entry

    def _get_headers(self) -> dict[str, str]:
        """Build API request headers."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": "202401",
        }

    def _get_author_urn(self) -> str:
        """Get the author URN (org or person)."""
        if self.org_id:
            return f"urn:li:organization:{self.org_id}"
        return self.person_urn or "urn:li:person:unknown"

    # ─── Public API ────────────────────────────────────────────────────

    async def post_share(
        self,
        text: str,
        *,
        image_url: Optional[str] = None,
        link_url: Optional[str] = None,
        link_title: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Post a text share on LinkedIn (with optional image or link).

        Args:
            text: Post text (max ~3000 chars for LinkedIn).
            image_url: URL of an image to include (optional).
            link_url: URL to share as a link card (optional).
            link_title: Title for the link card (optional).

        Returns:
            Dict with post_id, text, created_at, and mock flag.
        """
        if len(text) > 3000:
            text = text[:2997] + "..."

        if self.mock_mode:
            return self._log_mock_action("post_share", {
                "text": text,
                "image_url": image_url,
                "link_url": link_url,
            })

        author_urn = self._get_author_urn()

        # Build UGC Post body (LinkedIn v2 API)
        share_content: dict[str, Any] = {
            "shareCommentary": {"text": text},
            "shareMediaCategory": "NONE",
        }

        # Add link attachment if provided
        if link_url:
            share_content["shareMediaCategory"] = "ARTICLE"
            share_content["media"] = [{
                "status": "READY",
                "originalUrl": link_url,
                "title": {"text": link_title or ""},
            }]
        elif image_url:
            share_content["shareMediaCategory"] = "IMAGE"
            share_content["media"] = [{
                "status": "READY",
                "originalUrl": image_url,
            }]

        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": share_content,
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{LINKEDIN_API_BASE}/ugcPosts",
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

            post_id = data.get("id", "")
            logger.info(
                "linkedin_share_posted",
                extra={
                    "post_id": post_id,
                    "text_length": len(text),
                },
            )

            return {
                "platform": "linkedin",
                "action": "post_share",
                "post_id": post_id,
                "text": text,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mock": False,
            }

        except Exception as e:
            logger.error(
                "linkedin_post_failed",
                extra={"error": str(e)[:200]},
            )
            return {"error": str(e)[:200], "mock": False}

    async def get_company_updates(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get recent company page updates.

        Returns list of post dicts with id, text, likes, comments.
        """
        if self.mock_mode:
            self._log_mock_action("get_company_updates", {"limit": limit})
            return []

        if not self.org_id:
            logger.warning("linkedin_no_org_id: Cannot fetch company updates")
            return []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{LINKEDIN_API_BASE}/ugcPosts",
                    params={
                        "q": "authors",
                        "authors": f"List(urn:li:organization:{self.org_id})",
                        "count": min(limit, 50),
                    },
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

            posts = []
            for element in data.get("elements", []):
                content = element.get("specificContent", {}).get(
                    "com.linkedin.ugc.ShareContent", {}
                )
                commentary = content.get("shareCommentary", {}).get("text", "")
                posts.append({
                    "id": element.get("id", ""),
                    "text": commentary,
                    "created_at": element.get("created", {}).get("time", ""),
                })

            return posts

        except Exception as e:
            logger.error(
                "linkedin_updates_failed",
                extra={"error": str(e)[:200]},
            )
            return []

    async def get_engagement_metrics(
        self,
        post_id: str,
    ) -> dict[str, Any]:
        """
        Get engagement metrics for a specific post.

        Returns dict with likes, comments, shares, impressions.
        """
        if self.mock_mode:
            self._log_mock_action("get_engagement", {"post_id": post_id})
            return {
                "likes": 0,
                "comments": 0,
                "shares": 0,
                "impressions": 0,
                "mock": True,
            }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{LINKEDIN_API_BASE}/socialActions/{post_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

            return {
                "likes": data.get("likesSummary", {}).get("totalLikes", 0),
                "comments": data.get("commentsSummary", {}).get("totalFirstLevelComments", 0),
                "shares": 0,  # Not directly available in v2
                "impressions": 0,  # Requires analytics API
                "mock": False,
            }

        except Exception as e:
            logger.debug(f"Engagement metrics failed: {e}")
            return {"likes": 0, "comments": 0, "shares": 0, "impressions": 0}
