"""
Extracted helpers for the Growth Dashboard page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ─── Proposal Helpers ─────────────────────────────────────────────────

def compute_proposal_stats(proposals: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of proposals.

    Returns:
        Dict with: total, sent, accepted, rejected, pending,
        total_value, win_rate, avg_value
    """
    total = len(proposals)
    sent = sum(1 for p in proposals if p.get("status") == "sent")
    accepted = sum(1 for p in proposals if p.get("status") == "accepted")
    rejected = sum(1 for p in proposals if p.get("status") == "rejected")
    pending = sum(1 for p in proposals if p.get("status") in ("draft", "review", "pending"))

    total_value = sum(float(p.get("pricing_amount", 0) or 0) for p in proposals)
    accepted_value = sum(
        float(p.get("pricing_amount", 0) or 0)
        for p in proposals
        if p.get("status") == "accepted"
    )

    closed = accepted + rejected
    win_rate = (accepted / closed * 100) if closed > 0 else 0.0
    avg_value = total_value / total if total > 0 else 0.0

    return {
        "total": total,
        "sent": sent,
        "accepted": accepted,
        "rejected": rejected,
        "pending": pending,
        "total_value": total_value,
        "accepted_value": accepted_value,
        "win_rate": win_rate,
        "avg_value": avg_value,
    }


def format_proposal_status(status: str) -> tuple[str, str]:
    """
    Return (display_text, color) for a proposal status.
    """
    status_map = {
        "draft": ("Draft", "#8B8B8B"),
        "review": ("In Review", "#E8A838"),
        "pending": ("Pending", "#E8A838"),
        "sent": ("Sent", "#3B82F6"),
        "accepted": ("Accepted", "#10B981"),
        "rejected": ("Rejected", "#EF4444"),
    }
    text, color = status_map.get(status, (status.title(), "#8B8B8B"))
    return text, color


# ─── Social Media Helpers ─────────────────────────────────────────────

def compute_social_stats(posts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of social posts.

    Returns:
        Dict with: total, published, draft, scheduled,
        total_impressions, total_engagements, engagement_rate,
        top_platform
    """
    total = len(posts)
    published = sum(1 for p in posts if p.get("status") == "published")
    draft = sum(1 for p in posts if p.get("status") == "draft")
    scheduled = sum(1 for p in posts if p.get("status") == "scheduled")

    total_impressions = sum(int(p.get("impressions", 0) or 0) for p in posts)
    total_likes = sum(int(p.get("likes", 0) or 0) for p in posts)
    total_shares = sum(int(p.get("shares", 0) or 0) for p in posts)
    total_comments = sum(int(p.get("comments", 0) or 0) for p in posts)
    total_clicks = sum(int(p.get("clicks", 0) or 0) for p in posts)
    total_engagements = total_likes + total_shares + total_comments + total_clicks

    engagement_rate = (
        (total_engagements / total_impressions * 100)
        if total_impressions > 0
        else 0.0
    )

    # Top platform by post count
    platform_counts: dict[str, int] = {}
    for p in posts:
        plat = p.get("platform", "unknown")
        platform_counts[plat] = platform_counts.get(plat, 0) + 1
    top_platform = max(platform_counts, key=platform_counts.get, default="—") if platform_counts else "—"

    return {
        "total": total,
        "published": published,
        "draft": draft,
        "scheduled": scheduled,
        "total_impressions": total_impressions,
        "total_engagements": total_engagements,
        "total_likes": total_likes,
        "total_shares": total_shares,
        "total_comments": total_comments,
        "total_clicks": total_clicks,
        "engagement_rate": engagement_rate,
        "top_platform": top_platform,
    }


def format_platform_icon(platform: str) -> str:
    """Return an emoji icon for a platform."""
    icons = {
        "twitter": "🐦",
        "x": "𝕏",
        "linkedin": "💼",
        "meta": "📘",
        "facebook": "📘",
        "instagram": "📸",
        "google": "🔍",
    }
    return icons.get(platform.lower(), "📱")


# ─── Ads Helpers ──────────────────────────────────────────────────────

def compute_ads_stats(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of ad campaigns.

    Returns:
        Dict with: total, active, paused, total_spend,
        total_impressions, total_clicks, total_conversions,
        avg_ctr, avg_cpa, avg_roas
    """
    total = len(campaigns)
    active = sum(1 for c in campaigns if c.get("status") in ("active", "deployed_shadow"))
    paused = sum(1 for c in campaigns if c.get("status") == "paused")

    total_spend = sum(float(c.get("total_spend", 0) or 0) for c in campaigns)
    total_impressions = sum(int(c.get("impressions", 0) or 0) for c in campaigns)
    total_clicks = sum(int(c.get("clicks", 0) or 0) for c in campaigns)
    total_conversions = sum(int(c.get("conversions", 0) or 0) for c in campaigns)

    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0.0
    avg_cpa = (total_spend / total_conversions) if total_conversions > 0 else 0.0
    avg_roas = (total_conversions * 100 / total_spend) if total_spend > 0 else 0.0  # Simplified ROAS

    return {
        "total": total,
        "active": active,
        "paused": paused,
        "total_spend": total_spend,
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "avg_ctr": avg_ctr,
        "avg_cpa": avg_cpa,
        "avg_roas": avg_roas,
    }


def compute_campaign_health(campaign: dict[str, Any]) -> str:
    """
    Determine campaign health from performance metrics.

    Returns:
        "healthy", "needs_attention", or "critical"
    """
    ctr = float(campaign.get("ctr", 0) or 0)
    cpa = float(campaign.get("cpa", 0) or 0)
    target_cpa = float(campaign.get("target_cpa", 25) or 25)
    impressions = int(campaign.get("impressions", 0) or 0)

    if impressions == 0:
        return "needs_attention"  # No data yet
    if cpa > target_cpa * 1.5:
        return "critical"  # Way over CPA target
    if ctr < 1.0:
        return "needs_attention"  # Low CTR
    return "healthy"


# ─── Content Calendar Helpers ─────────────────────────────────────────

def group_calendar_by_date(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group content calendar entries by their scheduled date.

    Returns:
        Dict mapping date string to list of entries.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        date_str = entry.get("scheduled_date", "unscheduled")
        if date_str not in groups:
            groups[date_str] = []
        groups[date_str].append(entry)
    return groups


def compute_growth_score(
    proposal_stats: dict[str, Any],
    social_stats: dict[str, Any],
    ads_stats: dict[str, Any],
) -> int:
    """
    Compute an overall growth score (0-100) from all three channels.

    The score is a weighted average:
    - Proposals: 40% (revenue impact)
    - Social: 30% (awareness)
    - Ads: 30% (pipeline)

    Each channel scores 0-100 based on activity and performance.
    """
    # Proposal score: based on win rate and volume
    p_score = 0
    if proposal_stats["total"] > 0:
        p_score = min(100, int(
            proposal_stats["win_rate"] * 0.6 +
            min(proposal_stats["total"], 20) * 2
        ))

    # Social score: based on engagement and volume
    s_score = 0
    if social_stats["total"] > 0:
        s_score = min(100, int(
            social_stats["engagement_rate"] * 10 +
            min(social_stats["published"], 30) * 2
        ))

    # Ads score: based on CTR and conversions
    a_score = 0
    if ads_stats["total"] > 0:
        a_score = min(100, int(
            ads_stats["avg_ctr"] * 10 +
            min(ads_stats["total_conversions"], 50) * 1.5
        ))

    weighted = int(p_score * 0.4 + s_score * 0.3 + a_score * 0.3)
    return min(100, max(0, weighted))
