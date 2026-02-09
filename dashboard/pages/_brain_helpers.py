"""
Extracted helpers for the Brain Dashboard page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


# ─── Insight Feed Helpers ────────────────────────────────────

def compute_insight_stats(insights: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of shared insights.

    Returns:
        Dict with: total, by_type, by_agent, avg_confidence,
        high_confidence_count, topics_covered
    """
    total = len(insights)

    by_type: dict[str, int] = defaultdict(int)
    by_agent: dict[str, int] = defaultdict(int)
    topics: set[str] = set()
    confidences: list[float] = []

    for ins in insights:
        by_type[ins.get("insight_type", "unknown")] += 1
        by_agent[ins.get("source_agent_id", "unknown")] += 1

        metadata = ins.get("metadata", {})
        if isinstance(metadata, str):
            try:
                import json
                metadata = json.loads(metadata)
            except (ValueError, TypeError):
                metadata = {}

        topic = metadata.get("topic", ins.get("insight_type", ""))
        if topic:
            topics.add(topic)

        conf = float(ins.get("confidence_score", 0) or 0)
        if conf > 0:
            confidences.append(conf)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    high_confidence = sum(1 for c in confidences if c >= 0.8)

    return {
        "total": total,
        "by_type": dict(by_type),
        "by_agent": dict(by_agent),
        "topics_covered": len(topics),
        "avg_confidence": avg_confidence,
        "high_confidence_count": high_confidence,
    }


def format_insight_type(insight_type: str) -> tuple[str, str]:
    """Return (display_text, color) for an insight type."""
    type_map = {
        "email_performance": ("Email", "#3B82F6"),
        "audience_response": ("Audience", "#8B5CF6"),
        "objection_handling": ("Objections", "#F59E0B"),
        "content_performance": ("Content", "#10B981"),
        "keyword_trends": ("Keywords", "#06B6D4"),
        "social_engagement": ("Social", "#EC4899"),
        "ad_performance": ("Ads", "#F97316"),
        "deal_patterns": ("Deals", "#10B981"),
        "prospect_signals": ("Signals", "#3B82F6"),
        "meeting_insights": ("Meetings", "#8B5CF6"),
        "client_health": ("Client Health", "#EF4444"),
        "payment_patterns": ("Payments", "#F59E0B"),
        "financial_metrics": ("Finance", "#10B981"),
        "market_signal": ("Market", "#6366F1"),
        "competitive_intel": ("Competition", "#DC2626"),
        "winning_pattern": ("Winning Pattern", "#10B981"),
        "proposal_sent": ("Proposal", "#3B82F6"),
        "social_activity": ("Social Activity", "#EC4899"),
        "campaign_deployed": ("Campaign", "#F97316"),
        "commerce_activity": ("Commerce", "#8B5CF6"),
        "voice_interaction": ("Voice", "#06B6D4"),
        "payment_reminder": ("Payment", "#F59E0B"),
    }
    text, color = type_map.get(insight_type, (insight_type.replace("_", " ").title(), "#8B8B8B"))
    return text, color


def format_confidence(confidence: float) -> tuple[str, str]:
    """Return (display_text, color) for a confidence score."""
    if confidence >= 0.9:
        return "Very High", "#10B981"
    elif confidence >= 0.8:
        return "High", "#3B82F6"
    elif confidence >= 0.7:
        return "Good", "#06B6D4"
    elif confidence >= 0.5:
        return "Moderate", "#F59E0B"
    else:
        return "Low", "#EF4444"


# ─── Experiment Helpers ──────────────────────────────────────

def compute_experiment_stats(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of experiments.

    Returns:
        Dict with: total, active, concluded, total_observations,
        with_winner_count
    """
    total = len(experiments)
    active = sum(1 for e in experiments if e.get("status") == "active")
    concluded = sum(1 for e in experiments if e.get("status") == "concluded")
    paused = sum(1 for e in experiments if e.get("status") == "paused")
    total_obs = sum(int(e.get("total_observations", 0) or 0) for e in experiments)

    return {
        "total": total,
        "active": active,
        "concluded": concluded,
        "paused": paused,
        "total_observations": total_obs,
    }


def format_experiment_status(status: str) -> tuple[str, str]:
    """Return (display_text, color) for an experiment status."""
    status_map = {
        "active": ("Active", "#10B981"),
        "concluded": ("Concluded", "#3B82F6"),
        "paused": ("Paused", "#F59E0B"),
    }
    text, color = status_map.get(status, (status.title(), "#8B8B8B"))
    return text, color


# ─── Lead Score Helpers ──────────────────────────────────────

def compute_score_distribution(scores: list[int]) -> dict[str, Any]:
    """
    Compute distribution stats for lead scores.

    Returns:
        Dict with: total, avg, median, hot, warm, lukewarm, cold
    """
    if not scores:
        return {
            "total": 0,
            "avg": 0,
            "median": 0,
            "hot": 0,
            "warm": 0,
            "lukewarm": 0,
            "cold": 0,
        }

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    median = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    return {
        "total": n,
        "avg": sum(scores) / n,
        "median": median,
        "hot": sum(1 for s in scores if s >= 80),
        "warm": sum(1 for s in scores if 60 <= s < 80),
        "lukewarm": sum(1 for s in scores if 40 <= s < 60),
        "cold": sum(1 for s in scores if s < 40),
    }


def format_score_tier(score: int) -> tuple[str, str]:
    """Return (tier_label, color) for a lead score."""
    if score >= 80:
        return "Hot", "#EF4444"
    elif score >= 60:
        return "Warm", "#F59E0B"
    elif score >= 40:
        return "Lukewarm", "#3B82F6"
    else:
        return "Cold", "#6B7280"


# ─── Knowledge Flow Helpers ──────────────────────────────────

def compute_knowledge_flow(flow_log: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze knowledge flow between agents.

    Returns:
        Dict with: total_flows, publishers, consumers,
        most_active_publisher, most_active_consumer
    """
    publishers: dict[str, int] = defaultdict(int)
    consumers: dict[str, int] = defaultdict(int)

    for entry in flow_log:
        agent = entry.get("agent", "unknown")
        if entry.get("direction") == "publish":
            publishers[agent] += 1
        elif entry.get("direction") == "consume":
            consumers[agent] += 1

    total = len(flow_log)
    most_active_pub = max(publishers, key=publishers.get) if publishers else "none"
    most_active_con = max(consumers, key=consumers.get) if consumers else "none"

    return {
        "total_flows": total,
        "publishers": dict(publishers),
        "consumers": dict(consumers),
        "most_active_publisher": most_active_pub,
        "most_active_consumer": most_active_con,
    }


# ─── Brain Health Score ──────────────────────────────────────

def compute_brain_health(
    insight_stats: dict[str, Any],
    experiment_stats: dict[str, Any],
    score_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute a composite brain health score (0-100).

    Weights:
    - Knowledge depth: 40%  (insight variety and confidence)
    - Experimentation: 30%  (active experiments and observations)
    - Prediction quality: 30%  (lead scoring coverage)

    Returns:
        Dict with: score, grade, knowledge_score, experiment_score,
        prediction_score
    """
    # Knowledge depth (0-100)
    topics = insight_stats.get("topics_covered", 0)
    avg_conf = insight_stats.get("avg_confidence", 0)
    knowledge_score = min(topics * 8, 50) + (avg_conf * 50)

    # Experimentation (0-100)
    active = experiment_stats.get("active", 0)
    total_obs = experiment_stats.get("total_observations", 0)
    experiment_score = min(active * 20, 40) + min(total_obs / 10, 60)

    # Prediction quality (0-100)
    total_scored = score_stats.get("total", 0)
    prediction_score = min(total_scored * 2, 100) if total_scored > 0 else 0

    # Weighted composite
    score = (
        knowledge_score * 0.4
        + experiment_score * 0.3
        + prediction_score * 0.3
    )

    # Grade
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": round(min(score, 100), 1),
        "grade": grade,
        "knowledge_score": round(min(knowledge_score, 100), 1),
        "experiment_score": round(min(experiment_score, 100), 1),
        "prediction_score": round(min(prediction_score, 100), 1),
    }
