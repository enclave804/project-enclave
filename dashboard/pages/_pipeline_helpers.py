"""
Pipeline dashboard helper functions.

Pure utility module — no Streamlit imports — so these can be
unit-tested without running the dashboard server.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


# ─── Pipeline Stage Colors ──────────────────────────────────────────

STAGE_CONFIG = {
    "prospect": {"label": "Prospect", "color": "#6b7280", "emoji": "🔍"},
    "qualified": {"label": "Qualified", "color": "#3b82f6", "emoji": "✅"},
    "proposal": {"label": "Proposal", "color": "#8b5cf6", "emoji": "📝"},
    "negotiation": {"label": "Negotiation", "color": "#f59e0b", "emoji": "🤝"},
    "closed_won": {"label": "Closed Won", "color": "#10b981", "emoji": "🎉"},
    "closed_lost": {"label": "Closed Lost", "color": "#ef4444", "emoji": "❌"},
}


def format_deal_stage(stage: str) -> tuple[str, str]:
    """
    Return (label, color) for a pipeline stage.

    >>> format_deal_stage("qualified")
    ('Qualified', '#3b82f6')
    >>> format_deal_stage("unknown_stage")
    ('Unknown Stage', '#6b7280')
    """
    config = STAGE_CONFIG.get(stage, {})
    label = config.get("label", stage.replace("_", " ").title())
    color = config.get("color", "#6b7280")
    return label, color


def format_pipeline_value(cents: int) -> str:
    """
    Format a value in cents as a dollar string.

    >>> format_pipeline_value(150000)
    '$1,500.00'
    >>> format_pipeline_value(0)
    '$0.00'
    >>> format_pipeline_value(-50000)
    '-$500.00'
    """
    if cents < 0:
        return f"-${abs(cents) / 100:,.2f}"
    return f"${cents / 100:,.2f}"


# ─── Pipeline Metrics ──────────────────────────────────────────────


def compute_pipeline_metrics(
    opportunities: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute pipeline overview metrics from a list of opportunities.

    Returns:
        {
            "total_deals": int,
            "total_value_cents": int,
            "active_deals": int,
            "active_value_cents": int,
            "won_deals": int,
            "won_value_cents": int,
            "lost_deals": int,
            "win_rate": float,  # 0.0-1.0
            "stage_breakdown": {stage: {"count": int, "value_cents": int}},
            "avg_deal_value_cents": float,
        }
    """
    stage_breakdown: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "value_cents": 0}
    )

    total_deals = len(opportunities)
    total_value = 0
    won_deals = 0
    won_value = 0
    lost_deals = 0
    active_deals = 0
    active_value = 0

    for opp in opportunities:
        stage = opp.get("stage", "prospect")
        value = opp.get("value_cents", 0) or 0

        stage_breakdown[stage]["count"] += 1
        stage_breakdown[stage]["value_cents"] += value

        total_value += value

        if stage == "closed_won":
            won_deals += 1
            won_value += value
        elif stage == "closed_lost":
            lost_deals += 1
        else:
            active_deals += 1
            active_value += value

    closed_total = won_deals + lost_deals
    win_rate = won_deals / closed_total if closed_total > 0 else 0.0

    avg_deal_value = total_value / total_deals if total_deals > 0 else 0.0

    return {
        "total_deals": total_deals,
        "total_value_cents": total_value,
        "active_deals": active_deals,
        "active_value_cents": active_value,
        "won_deals": won_deals,
        "won_value_cents": won_value,
        "lost_deals": lost_deals,
        "win_rate": round(win_rate, 4),
        "stage_breakdown": dict(stage_breakdown),
        "avg_deal_value_cents": round(avg_deal_value, 2),
    }


# ─── Sequence Stats ────────────────────────────────────────────────


def compute_sequence_stats(
    sequences: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute follow-up sequence statistics.

    Returns:
        {
            "total_sequences": int,
            "active": int,
            "completed": int,
            "replied": int,
            "paused": int,
            "cancelled": int,
            "avg_steps_completed": float,
            "reply_rate": float,  # 0.0-1.0
        }
    """
    status_counts: dict[str, int] = defaultdict(int)
    total_steps = 0

    for seq in sequences:
        status = seq.get("status", "active")
        status_counts[status] += 1
        total_steps += seq.get("current_step", 0)

    total = len(sequences)
    avg_steps = total_steps / total if total > 0 else 0.0
    replied = status_counts.get("replied", 0)
    completed = status_counts.get("completed", 0)
    finished = replied + completed
    reply_rate = replied / finished if finished > 0 else 0.0

    return {
        "total_sequences": total,
        "active": status_counts.get("active", 0),
        "completed": completed,
        "replied": replied,
        "paused": status_counts.get("paused", 0),
        "cancelled": status_counts.get("cancelled", 0),
        "avg_steps_completed": round(avg_steps, 2),
        "reply_rate": round(reply_rate, 4),
    }


# ─── Meeting Stats ──────────────────────────────────────────────────


def compute_meeting_stats(
    meetings: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute meeting scheduling statistics.

    Returns:
        {
            "total_meetings": int,
            "proposed": int,
            "confirmed": int,
            "completed": int,
            "cancelled": int,
            "no_show": int,
            "confirmation_rate": float,  # 0.0-1.0
            "completion_rate": float,    # 0.0-1.0
            "by_type": {meeting_type: count},
        }
    """
    status_counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, int] = defaultdict(int)

    for meeting in meetings:
        status = meeting.get("status", "proposed")
        status_counts[status] += 1

        meeting_type = meeting.get("meeting_type", "discovery")
        type_counts[meeting_type] += 1

    total = len(meetings)
    proposed = status_counts.get("proposed", 0)
    confirmed = status_counts.get("confirmed", 0)
    completed = status_counts.get("completed", 0)
    cancelled = status_counts.get("cancelled", 0)
    no_show = status_counts.get("no_show", 0)

    # Confirmation rate: confirmed+completed / total proposed
    total_proposed = proposed + confirmed + completed + cancelled + no_show
    confirmed_total = confirmed + completed
    confirmation_rate = (
        confirmed_total / total_proposed if total_proposed > 0 else 0.0
    )

    # Completion rate: completed / (confirmed + completed)
    total_confirmed = confirmed + completed
    completion_rate = completed / total_confirmed if total_confirmed > 0 else 0.0

    return {
        "total_meetings": total,
        "proposed": proposed,
        "confirmed": confirmed,
        "completed": completed,
        "cancelled": cancelled,
        "no_show": no_show,
        "confirmation_rate": round(confirmation_rate, 4),
        "completion_rate": round(completion_rate, 4),
        "by_type": dict(type_counts),
    }
