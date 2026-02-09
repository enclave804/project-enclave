"""
Extracted helpers for the Command Center page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from typing import Any


def compute_agent_status(agent: dict[str, Any]) -> str:
    """
    Determine an agent's status from its record.

    Returns:
        One of: "active", "paused", "shadow", "circuit_breaker"
    """
    enabled = agent.get("enabled", True)
    shadow_mode = agent.get("shadow_mode", False)
    config = agent.get("config", {}) or {}
    consecutive_errors = config.get("consecutive_errors", 0)
    max_errors = config.get("max_consecutive_errors", 5)

    if not enabled:
        return "paused"
    if shadow_mode:
        return "shadow"
    if consecutive_errors >= max_errors:
        return "circuit_breaker"
    return "active"


def compute_fleet_summary(agents: list[dict[str, Any]]) -> dict[str, int]:
    """
    Compute fleet-wide summary counts.

    Returns:
        Dict with keys: active, paused, shadow, tripped, total
    """
    active = 0
    paused = 0
    shadow = 0
    tripped = 0

    for agent in agents:
        status = compute_agent_status(agent)
        if status == "active":
            active += 1
        elif status == "paused":
            paused += 1
        elif status == "shadow":
            shadow += 1
        elif status == "circuit_breaker":
            tripped += 1

    return {
        "active": active,
        "paused": paused,
        "shadow": shadow,
        "tripped": tripped,
        "total": len(agents),
    }


def compute_health_status(agents: list[dict[str, Any]], failed_tasks: int = 0) -> str:
    """
    Determine overall system health.

    Returns:
        "operational", "degraded", or "critical"
    """
    summary = compute_fleet_summary(agents)

    if summary["total"] == 0:
        return "operational"  # No agents = nothing to be degraded

    if summary["paused"] > 0 or summary["tripped"] > 0 or failed_tasks > 5:
        return "degraded"

    return "operational"


def compute_success_rate(agent_stats: list[dict[str, Any]]) -> float:
    """
    Compute overall success rate from agent stats.

    Returns:
        Percentage (0-100).
    """
    total_runs = sum(s.get("total_runs", 0) for s in agent_stats)
    total_success = sum(s.get("success_runs", 0) for s in agent_stats)

    if total_runs == 0:
        return 0.0

    return (total_success / total_runs) * 100


def format_run_description(run: dict[str, Any]) -> str:
    """
    Format a run record into a human-readable description.

    Returns:
        Description string like "Completed in 450ms" or "Failed: timeout"
    """
    status = run.get("status", "?")
    error_msg = run.get("error_message") or ""
    duration = run.get("duration_ms", 0) or 0
    agent_type = run.get("agent_type", "")

    if status == "failed":
        return f"Failed: {error_msg[:80]}" if error_msg else "Failed"
    elif status == "completed":
        return f"Completed in {duration}ms"
    elif status == "started":
        return f"Started ({agent_type})"
    else:
        return status


def compute_pipeline_value(opportunities: int, avg_ticket: int = 6000) -> int:
    """Compute estimated pipeline value."""
    return opportunities * avg_ticket


def group_content_by_status(
    items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group content items by their status for Kanban display.

    Returns:
        Dict mapping status to list of items.
    """
    groups: dict[str, list[dict[str, Any]]] = {
        "draft": [],
        "review": [],
        "approved": [],
        "published": [],
    }

    for item in items:
        status = item.get("status", "draft")
        if status in groups:
            groups[status].append(item)

    return groups


def format_credential_status(env_var: str, label: str) -> dict[str, Any]:
    """
    Check if a credential is set and return status info.

    NOTE: This function checks os.environ. In production,
    it should also check st.secrets.

    Returns:
        Dict with: env_var, label, is_set, status_text
    """
    import os

    is_set = bool(os.environ.get(env_var, ""))

    return {
        "env_var": env_var,
        "label": label,
        "is_set": is_set,
        "status_text": "configured" if is_set else "missing",
    }
