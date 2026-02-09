"""
Extracted helpers for the Autopilot Dashboard page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


# ─── Autopilot Stats ──────────────────────────────────────────


def compute_autopilot_stats(
    sessions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute aggregate stats from autopilot sessions and actions.

    Returns:
        {
            "total_sessions": int,
            "completed_sessions": int,
            "failed_sessions": int,
            "total_actions": int,
            "successful_actions": int,
            "pending_actions": int,
            "last_run": str or None,
            "avg_actions_per_session": float,
        }
    """
    total_sessions = len(sessions)
    completed = sum(1 for s in sessions if s.get("status") == "completed")
    failed = sum(1 for s in sessions if s.get("status") == "failed")

    total_actions = len(actions)
    successful = sum(1 for a in actions if a.get("result") == "success")
    pending = sum(1 for a in actions if a.get("result") == "pending")

    # Last run
    last_run = None
    if sessions:
        dates = [
            s.get("completed_at") or s.get("created_at", "")
            for s in sessions
        ]
        dates = [d for d in dates if d]
        if dates:
            last_run = max(dates)

    avg_actions = (
        round(total_actions / total_sessions, 1)
        if total_sessions > 0 else 0.0
    )

    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed,
        "failed_sessions": failed,
        "total_actions": total_actions,
        "successful_actions": successful,
        "pending_actions": pending,
        "last_run": last_run,
        "avg_actions_per_session": avg_actions,
    }


# ─── Healing Log ──────────────────────────────────────────────


def compute_healing_log(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a displayable healing log from optimization actions.

    Filters for healing-related actions and formats them for display.

    Returns:
        [
            {
                "timestamp": str,
                "agent_id": str,
                "action_type": str,
                "description": str,
                "result": str,
                "result_color": str,
            }
        ]
    """
    healing_types = {"config_fix", "agent_restart", "agent_disable"}

    log = []
    for action in actions:
        action_type = action.get("action_type", "")
        if action_type not in healing_types:
            continue

        result = action.get("result", "unknown")
        result_color = {
            "success": "#10B981",
            "failed": "#EF4444",
            "pending": "#F59E0B",
            "rejected": "#6B7280",
        }.get(result, "#6B7280")

        params = action.get("parameters", {})
        if isinstance(params, str):
            try:
                import json
                params = json.loads(params)
            except (ValueError, TypeError):
                params = {}

        description = params.get("reasoning", action_type.replace("_", " ").title())

        log.append({
            "timestamp": action.get("created_at", ""),
            "agent_id": action.get("target", "unknown"),
            "action_type": action_type,
            "description": description[:100],
            "result": result,
            "result_color": result_color,
        })

    return log


# ─── Budget Overview ──────────────────────────────────────────


def compute_budget_overview(
    snapshots: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute budget overview from snapshots and budget actions.

    Returns:
        {
            "latest_spend": float,
            "latest_revenue": float,
            "latest_roas": float,
            "spend_trend": list[dict],  # [{period, spend, revenue, roas}]
            "reallocation_count": int,
            "total_shifted": float,
        }
    """
    # Latest snapshot
    latest_spend = 0.0
    latest_revenue = 0.0
    latest_roas = 0.0

    if snapshots:
        latest = snapshots[0]  # Already ordered by period desc
        latest_spend = float(latest.get("total_spend", 0))
        latest_revenue = float(latest.get("total_revenue", 0))
        latest_roas = float(latest.get("roas", 0))

    # Spend trend
    trend = []
    for snap in reversed(snapshots):  # Oldest first for charts
        trend.append({
            "period": snap.get("period", ""),
            "spend": float(snap.get("total_spend", 0)),
            "revenue": float(snap.get("total_revenue", 0)),
            "roas": float(snap.get("roas", 0)),
        })

    # Budget reallocation stats
    budget_actions = [
        a for a in actions
        if a.get("action_type") == "budget_reallocation"
    ]
    reallocation_count = len(budget_actions)

    total_shifted = 0.0
    for a in budget_actions:
        params = a.get("parameters", {})
        if isinstance(params, str):
            try:
                import json
                params = json.loads(params)
            except (ValueError, TypeError):
                params = {}
        total_shifted += abs(float(params.get("delta", 0)))

    return {
        "latest_spend": round(latest_spend, 2),
        "latest_revenue": round(latest_revenue, 2),
        "latest_roas": round(latest_roas, 2),
        "spend_trend": trend,
        "reallocation_count": reallocation_count,
        "total_shifted": round(total_shifted, 2),
    }


# ─── Strategy Pipeline ───────────────────────────────────────


def compute_strategy_pipeline(
    sessions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute strategy pipeline overview from sessions and actions.

    Returns:
        {
            "pending_proposals": list[dict],
            "active_experiments": list[dict],
            "completed_strategies": int,
            "actions_by_type": dict[str, int],
        }
    """
    # Pending proposals from recent sessions
    pending_proposals = []
    for session in sessions:
        strategy = session.get("strategy_output", {})
        if isinstance(strategy, str):
            try:
                import json
                strategy = json.loads(strategy)
            except (ValueError, TypeError):
                strategy = {}

        if session.get("status") == "running":
            pending_proposals.append({
                "session_id": session.get("id", ""),
                "type": session.get("session_type", ""),
                "started_at": session.get("started_at", ""),
                "summary": strategy.get("summary", "Running..."),
            })

    # Actions by type
    actions_by_type: dict[str, int] = defaultdict(int)
    for action in actions:
        actions_by_type[action.get("action_type", "unknown")] += 1

    # Completed strategies
    completed = sum(
        1 for s in sessions
        if s.get("status") == "completed"
        and s.get("session_type") == "full_analysis"
    )

    # Experiment-related actions
    experiment_actions = [
        a for a in actions
        if a.get("action_type") == "experiment_launched"
    ]

    return {
        "pending_proposals": pending_proposals,
        "active_experiments": experiment_actions,
        "completed_strategies": completed,
        "actions_by_type": dict(actions_by_type),
    }


# ─── Formatting Helpers ──────────────────────────────────────


def format_health_score(score: float) -> tuple[str, str]:
    """
    Return (label, color) for a health score.

    Args:
        score: 0.0-1.0 health score.

    Returns:
        (label, hex_color)
    """
    if score >= 0.8:
        return "Healthy", "#10B981"
    elif score >= 0.6:
        return "Good", "#3B82F6"
    elif score >= 0.4:
        return "Degraded", "#F59E0B"
    elif score >= 0.2:
        return "Critical", "#EF4444"
    else:
        return "Down", "#991B1B"


def format_roas(roas: float) -> tuple[str, str]:
    """
    Return (formatted_string, color) for ROAS display.

    Args:
        roas: Return on Ad Spend multiplier.

    Returns:
        (display_string, hex_color)
    """
    if roas >= 3.0:
        return f"{roas:.1f}x", "#10B981"
    elif roas >= 1.5:
        return f"{roas:.1f}x", "#3B82F6"
    elif roas >= 1.0:
        return f"{roas:.1f}x", "#F59E0B"
    else:
        return f"{roas:.1f}x", "#EF4444"


def format_action_type(action_type: str) -> tuple[str, str]:
    """Return (display_text, color) for an action type."""
    type_map = {
        "config_fix": ("Config Fix", "#3B82F6"),
        "agent_restart": ("Restart", "#8B5CF6"),
        "agent_disable": ("Disabled", "#EF4444"),
        "budget_reallocation": ("Budget Shift", "#F59E0B"),
        "experiment_launched": ("Experiment", "#10B981"),
        "schedule_adjustment": ("Schedule", "#06B6D4"),
        "alert_sent": ("Alert", "#EC4899"),
    }
    return type_map.get(action_type, (action_type.replace("_", " ").title(), "#6B7280"))


def format_session_type(session_type: str) -> tuple[str, str]:
    """Return (display_text, color) for a session type."""
    type_map = {
        "full_analysis": ("Full Analysis", "#3B82F6"),
        "healing": ("Healing", "#10B981"),
        "budget": ("Budget", "#F59E0B"),
        "strategy": ("Strategy", "#8B5CF6"),
    }
    return type_map.get(session_type, (session_type.title(), "#6B7280"))


def format_session_status(status: str) -> tuple[str, str]:
    """Return (display_text, color) for a session status."""
    status_map = {
        "running": ("Running", "#3B82F6"),
        "completed": ("Completed", "#10B981"),
        "failed": ("Failed", "#EF4444"),
        "cancelled": ("Cancelled", "#6B7280"),
    }
    return status_map.get(status, (status.title(), "#6B7280"))
