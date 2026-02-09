"""
Social media MCP tools for the Sovereign Venture Engine.

Provides a unified interface for social media operations across
Twitter/X and LinkedIn. All tools respect the sandbox protocol —
mock mode is automatically enabled when API keys are missing.

Tools:
    post_social_update   — Post to Twitter/X or LinkedIn
    get_social_mentions  — Get recent mentions from a platform
    reply_to_post        — Reply to a specific post
    check_trending_topics — Get trending topics for a niche

Usage:
    from core.mcp.tools.social_tools import post_social_update
    result = await post_social_update(
        platform="twitter",
        content="New blog post on zero-trust architecture! 🔒",
        image_path="/path/to/image.png",
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def post_social_update(
    platform: str,
    content: str,
    *,
    image_path: Optional[str] = None,
    link_url: Optional[str] = None,
    link_title: Optional[str] = None,
) -> str:
    """
    Post a social media update to the specified platform.

    Args:
        platform: "twitter" or "linkedin".
        content: Post text content.
        image_path: Optional path to an image to attach.
        link_url: Optional URL to share (LinkedIn only).
        link_title: Optional title for the link (LinkedIn only).

    Returns:
        JSON string with post result.
    """
    logger.info(
        "social_post_update",
        extra={
            "platform": platform,
            "content_length": len(content),
            "has_image": image_path is not None,
        },
    )

    if platform.lower() in ("twitter", "x"):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient.from_env()
        result = await client.post_tweet(content, image_path=image_path)

    elif platform.lower() == "linkedin":
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient.from_env()
        result = await client.post_share(
            content,
            image_url=image_path,
            link_url=link_url,
            link_title=link_title,
        )

    else:
        result = {"error": f"Unsupported platform: {platform}"}

    return json.dumps(result, default=str)


async def get_social_mentions(
    platform: str,
    limit: int = 10,
) -> str:
    """
    Get recent mentions from a social media platform.

    Args:
        platform: "twitter" or "linkedin".
        limit: Maximum number of mentions to return.

    Returns:
        JSON string with list of mentions.
    """
    logger.info(
        "social_get_mentions",
        extra={"platform": platform, "limit": limit},
    )

    if platform.lower() in ("twitter", "x"):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient.from_env()
        mentions = await client.get_mentions(limit=limit)

    elif platform.lower() == "linkedin":
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient.from_env()
        mentions = await client.get_company_updates(limit=limit)

    else:
        mentions = []

    return json.dumps({"platform": platform, "mentions": mentions, "count": len(mentions)})


async def reply_to_post(
    platform: str,
    post_id: str,
    text: str,
) -> str:
    """
    Reply to a specific social media post.

    Args:
        platform: "twitter" or "linkedin".
        post_id: The ID of the post to reply to.
        text: Reply text content.

    Returns:
        JSON string with reply result.
    """
    logger.info(
        "social_reply_to_post",
        extra={
            "platform": platform,
            "post_id": post_id[:20],
            "text_length": len(text),
        },
    )

    if platform.lower() in ("twitter", "x"):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient.from_env()
        result = await client.reply(tweet_id=post_id, text=text)

    elif platform.lower() == "linkedin":
        # LinkedIn comment API is more complex; mock for now
        result = {
            "platform": "linkedin",
            "action": "reply",
            "post_id": post_id,
            "text": text,
            "note": "LinkedIn comment API integration coming in v2",
        }

    else:
        result = {"error": f"Unsupported platform: {platform}"}

    return json.dumps(result, default=str)


async def check_trending_topics(
    niche: str = "",
    platform: str = "twitter",
) -> str:
    """
    Check trending topics for content inspiration.

    Args:
        niche: Topic niche to filter trends (e.g., "cybersecurity").
        platform: Platform to check trends on.

    Returns:
        JSON string with trending topics.
    """
    logger.info(
        "social_check_trends",
        extra={"niche": niche, "platform": platform},
    )

    if platform.lower() in ("twitter", "x"):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient.from_env()
        topics = await client.get_trending_topics(niche=niche)

    else:
        # Generic trending topics (mock)
        topics = [
            {"topic": f"#{niche.replace(' ', '')}", "tweet_volume": 0},
        ]

    return json.dumps({
        "platform": platform,
        "niche": niche,
        "topics": topics,
        "count": len(topics),
    })
