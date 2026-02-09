"""
Twitter/X API client for the Sovereign Venture Engine.

Uses Twitter API v2 (via tweepy) for posting, replying, and reading mentions.
Automatically falls back to mock mode when API credentials are missing,
logging all actions to a JSON file for testing without burning API limits.

Environment Variables:
    TWITTER_API_KEY         — Consumer API key
    TWITTER_API_SECRET      — Consumer API secret
    TWITTER_ACCESS_TOKEN    — User access token
    TWITTER_ACCESS_SECRET   — User access token secret
    TWITTER_BEARER_TOKEN    — Bearer token for read-only endpoints

Mock Mode:
    When TWITTER_API_KEY is not set, all operations are logged to
    storage/social_mock.json instead of calling the real API.

Usage:
    client = TwitterClient.from_env()
    result = await client.post_tweet("Hello from Enclave Guard! 🔒")
    mentions = await client.get_mentions(limit=10)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default mock storage path
MOCK_STORAGE_DIR = Path(__file__).parent.parent.parent.parent / "storage"
MOCK_STORAGE_FILE = MOCK_STORAGE_DIR / "social_mock.json"


class TwitterClient:
    """
    Twitter/X API client with automatic mock mode fallback.

    When real credentials are available, uses tweepy for API v2.
    When credentials are missing, logs operations to a JSON file.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
        access_secret: str = "",
        bearer_token: str = "",
        *,
        mock_mode: Optional[bool] = None,
        mock_storage_path: Optional[Path] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.bearer_token = bearer_token

        # Auto-detect mock mode if not explicitly set
        if mock_mode is None:
            self.mock_mode = not bool(api_key)
        else:
            self.mock_mode = mock_mode

        self.mock_storage_path = mock_storage_path or MOCK_STORAGE_FILE
        self._client: Any = None  # tweepy.Client (lazy)
        self._api: Any = None     # tweepy.API for media uploads (lazy)

        if self.mock_mode:
            logger.info(
                "twitter_mock_mode",
                extra={"reason": "TWITTER_API_KEY not set"},
            )
        else:
            logger.info("twitter_client_initialized")

    @classmethod
    def from_env(cls, **kwargs: Any) -> TwitterClient:
        """Create a TwitterClient from environment variables."""
        return cls(
            api_key=os.getenv("TWITTER_API_KEY", ""),
            api_secret=os.getenv("TWITTER_API_SECRET", ""),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.getenv("TWITTER_ACCESS_SECRET", ""),
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
            **kwargs,
        )

    def _get_client(self) -> Any:
        """Lazy-initialize the tweepy v2 Client."""
        if self._client is None:
            try:
                import tweepy
                self._client = tweepy.Client(
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_secret,
                    bearer_token=self.bearer_token,
                )
            except ImportError:
                logger.warning("tweepy not installed — falling back to mock mode")
                self.mock_mode = True
                return None
        return self._client

    # ─── Mock Helpers ──────────────────────────────────────────────────

    def _log_mock_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Log a mock action to the JSON storage file."""
        entry = {
            "platform": "twitter",
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "mock": True,
        }

        # Ensure directory exists
        self.mock_storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing entries
        entries: list[dict] = []
        if self.mock_storage_path.exists():
            try:
                raw = self.mock_storage_path.read_text()
                entries = json.loads(raw) if raw.strip() else []
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append(entry)

        # Write back (keep last 500 entries)
        entries = entries[-500:]
        self.mock_storage_path.write_text(json.dumps(entries, indent=2))

        logger.info(
            f"twitter_mock_{action}",
            extra={"data_preview": str(data)[:200]},
        )
        return entry

    # ─── Public API ────────────────────────────────────────────────────

    async def post_tweet(
        self,
        text: str,
        *,
        image_path: Optional[str] = None,
        in_reply_to: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Post a tweet.

        Args:
            text: Tweet text (max 280 characters).
            image_path: Optional path to an image to attach.
            in_reply_to: Optional tweet ID to reply to.

        Returns:
            Dict with tweet_id, text, created_at, and mock flag.
        """
        if len(text) > 280:
            text = text[:277] + "..."

        if self.mock_mode:
            return self._log_mock_action("post_tweet", {
                "text": text,
                "image_path": image_path,
                "in_reply_to": in_reply_to,
            })

        client = self._get_client()
        if client is None:
            return self._log_mock_action("post_tweet", {"text": text})

        try:
            # Handle media upload if image provided
            media_ids = None
            if image_path:
                media_ids = await self._upload_media(image_path)

            kwargs: dict[str, Any] = {"text": text}
            if in_reply_to:
                kwargs["in_reply_to_tweet_id"] = in_reply_to
            if media_ids:
                kwargs["media_ids"] = media_ids

            response = client.create_tweet(**kwargs)
            tweet_data = response.data if response.data else {}

            logger.info(
                "twitter_tweet_posted",
                extra={
                    "tweet_id": tweet_data.get("id", ""),
                    "text_length": len(text),
                },
            )

            return {
                "platform": "twitter",
                "action": "post_tweet",
                "tweet_id": tweet_data.get("id", ""),
                "text": text,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mock": False,
            }

        except Exception as e:
            logger.error(
                "twitter_post_failed",
                extra={"error": str(e)[:200]},
            )
            return {"error": str(e)[:200], "mock": False}

    async def reply(
        self,
        tweet_id: str,
        text: str,
    ) -> dict[str, Any]:
        """Reply to a specific tweet."""
        return await self.post_tweet(text, in_reply_to=tweet_id)

    async def get_mentions(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get recent mentions of the authenticated user.

        Returns list of mention dicts with id, text, author, created_at.
        """
        if self.mock_mode:
            self._log_mock_action("get_mentions", {"limit": limit})
            return []  # Mock: no real mentions

        client = self._get_client()
        if client is None:
            return []

        try:
            # Get authenticated user ID
            me = client.get_me()
            user_id = me.data.id if me.data else None
            if not user_id:
                return []

            response = client.get_users_mentions(
                id=user_id,
                max_results=min(limit, 100),
                tweet_fields=["created_at", "author_id", "text"],
            )

            mentions = []
            if response.data:
                for tweet in response.data:
                    mentions.append({
                        "id": str(tweet.id),
                        "text": tweet.text,
                        "author_id": str(tweet.author_id),
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else "",
                    })

            logger.info(
                "twitter_mentions_fetched",
                extra={"count": len(mentions)},
            )
            return mentions

        except Exception as e:
            logger.error(
                "twitter_mentions_failed",
                extra={"error": str(e)[:200]},
            )
            return []

    async def get_trending_topics(
        self,
        niche: str = "",
    ) -> list[dict[str, Any]]:
        """
        Get trending topics (mock in v1, real API in v2).

        Args:
            niche: Filter trends by niche keyword.

        Returns:
            List of trending topic dicts.
        """
        if self.mock_mode:
            self._log_mock_action("get_trending", {"niche": niche})
            # Return synthetic trending topics for testing
            return [
                {"topic": f"#{niche.replace(' ', '')}Trends", "tweet_volume": 15000},
                {"topic": "#CyberSecurity", "tweet_volume": 45000},
                {"topic": "#AI", "tweet_volume": 120000},
            ]

        # Real API: Twitter v2 trends are limited, use search as proxy
        client = self._get_client()
        if client is None:
            return []

        try:
            search_query = f"{niche} -is:retweet lang:en" if niche else "trending -is:retweet lang:en"
            response = client.search_recent_tweets(
                query=search_query,
                max_results=10,
                tweet_fields=["public_metrics"],
            )

            topics = []
            if response.data:
                for tweet in response.data:
                    metrics = tweet.public_metrics or {}
                    topics.append({
                        "topic": tweet.text[:100],
                        "tweet_volume": metrics.get("like_count", 0) + metrics.get("retweet_count", 0),
                    })

            return topics

        except Exception as e:
            logger.debug(f"Trending topics fetch failed: {e}")
            return []

    async def _upload_media(self, image_path: str) -> Optional[list[str]]:
        """Upload media for tweet attachment (requires tweepy.API v1.1)."""
        try:
            import tweepy
            if self._api is None:
                auth = tweepy.OAuth1UserHandler(
                    self.api_key,
                    self.api_secret,
                    self.access_token,
                    self.access_secret,
                )
                self._api = tweepy.API(auth)

            media = self._api.media_upload(image_path)
            return [str(media.media_id)]
        except Exception as e:
            logger.warning(f"Media upload failed: {e}")
            return None
