"""
Sovereign Cockpit — Autopilot Control Center

The self-driving layer: autonomous healing, budget optimization,
and strategic recommendations at a glance.

- System Overview: health gauges, session history, action counts
- Self-Healing Log: recent crash diagnoses and config fixes
- Budget Autopilot: ROAS trends, reallocation history, spend breakdown
- Strategy Pipeline: pending proposals, experiment pipeline, next steps
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


st.set_page_config(
    page_title="Autopilot — Sovereign Cockpit",
    page_icon="◆",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Auth + Theme + Sidebar
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()

from dashboard.theme import (
    COLORS, inject_theme_css, page_header,
    section_header, status_badge, render_empty_state,
    render_divider, render_timestamp, render_stat_grid,
)

inject_theme_css()

from dashboard.sidebar import render_sidebar

vertical_id = render_sidebar()


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db(vid: str):
    from core.integrations.supabase_client import EnclaveDB
    return EnclaveDB(vid)


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Helper imports
# ---------------------------------------------------------------------------

from dashboard.pages._autopilot_helpers import (
    compute_autopilot_stats,
    compute_healing_log,
    compute_budget_overview,
    compute_strategy_pipeline,
    format_health_score,
    format_roas,
    format_action_type,
    format_session_type,
    format_session_status,
)


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    "Autopilot Control Center",
    "Self-driving optimization — healing, budget, and strategy.",
)


# ---------------------------------------------------------------------------
# Load data (with graceful fallback)
# ---------------------------------------------------------------------------

db = get_db(vertical_id)

# Autopilot sessions
sessions = _safe_call(
    lambda: db.client.table("autopilot_sessions")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(50)
    .execute().data,
    [],
)

# Optimization actions
actions = _safe_call(
    lambda: db.client.table("optimization_actions")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(100)
    .execute().data,
    [],
)

# Budget snapshots
budget_snapshots = _safe_call(
    lambda: db.client.table("budget_snapshots")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("period", desc=True)
    .limit(12)
    .execute().data,
    [],
)

# Compute stats
stats = compute_autopilot_stats(sessions or [], actions or [])
healing_log = compute_healing_log(actions or [])
budget_overview = compute_budget_overview(budget_snapshots or [], actions or [])
strategy_pipeline = compute_strategy_pipeline(sessions or [], actions or [])


# ---------------------------------------------------------------------------
# System Overview (top banner)
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sessions",
        stats["total_sessions"],
        delta=None,
    )

with col2:
    st.metric(
        "Actions Taken",
        stats["total_actions"],
        delta=f"{stats['successful_actions']} successful",
    )

with col3:
    roas_str, roas_color = format_roas(budget_overview["latest_roas"])
    st.metric(
        "Current ROAS",
        roas_str,
        delta=f"${budget_overview['latest_spend']:,.0f} spend",
    )

with col4:
    if stats["last_run"]:
        st.metric(
            "Last Run",
            stats["last_run"][:10] if stats["last_run"] else "Never",
            delta=f"{stats['completed_sessions']} completed",
        )
    else:
        st.metric("Last Run", "Never", delta="No runs yet")


render_divider()


# ---------------------------------------------------------------------------
# Self-Healing Log
# ---------------------------------------------------------------------------

section_header("Self-Healing Log", "Recent crash diagnoses and automated config fixes")

if healing_log:
    for entry in healing_log[:10]:
        result_label = entry["result"].title()
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 2, 4, 1])
            with c1:
                st.caption(entry["timestamp"][:19] if entry["timestamp"] else "—")
            with c2:
                st.markdown(f"**{entry['agent_id']}**")
            with c3:
                st.text(entry["description"])
            with c4:
                st.markdown(
                    f"<span style='color: {entry['result_color']}; font-weight: 600;'>"
                    f"{result_label}</span>",
                    unsafe_allow_html=True,
                )
else:
    render_empty_state(
        "No Healing Actions Yet",
        "The self-healer will appear here when agents need attention.",
    )


render_divider()


# ---------------------------------------------------------------------------
# Budget Autopilot
# ---------------------------------------------------------------------------

section_header("Budget Autopilot", "ROAS tracking, spend allocation, and reallocation history")

if budget_overview["latest_spend"] > 0:
    bc1, bc2, bc3 = st.columns(3)

    with bc1:
        st.metric("Total Spend", f"${budget_overview['latest_spend']:,.0f}")

    with bc2:
        st.metric("Total Revenue", f"${budget_overview['latest_revenue']:,.0f}")

    with bc3:
        st.metric(
            "Reallocations",
            budget_overview["reallocation_count"],
            delta=f"${budget_overview['total_shifted']:,.0f} shifted",
        )

    # Spend trend chart
    if budget_overview["spend_trend"]:
        import pandas as pd

        df = pd.DataFrame(budget_overview["spend_trend"])
        if not df.empty and "period" in df.columns:
            st.subheader("Spend & Revenue Trend")
            chart_df = df[["period", "spend", "revenue"]].set_index("period")
            st.line_chart(chart_df)

            st.subheader("ROAS Trend")
            roas_df = df[["period", "roas"]].set_index("period")
            st.line_chart(roas_df)
else:
    render_empty_state(
        "No Budget Data Yet",
        "Budget snapshots will appear here once the Autopilot analyzes ad spend.",
    )


render_divider()


# ---------------------------------------------------------------------------
# Strategy Pipeline
# ---------------------------------------------------------------------------

section_header("Strategy Pipeline", "Pending proposals, active experiments, and optimization history")

if strategy_pipeline["pending_proposals"]:
    st.subheader("Pending Proposals")
    for proposal in strategy_pipeline["pending_proposals"]:
        type_label, type_color = format_session_type(proposal.get("type", ""))
        with st.container():
            st.markdown(
                f"<div style='border-left: 3px solid {type_color}; "
                f"padding-left: 1rem; margin-bottom: 0.5rem;'>"
                f"<strong>{type_label}</strong> — "
                f"{proposal.get('summary', 'No summary')[:120]}"
                f"<br><span style='color: {COLORS['text_secondary']}; font-size: 0.8rem;'>"
                f"Started: {proposal.get('started_at', '')[:19]}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# Action breakdown
if strategy_pipeline["actions_by_type"]:
    st.subheader("Actions by Type")
    for action_type, count in sorted(
        strategy_pipeline["actions_by_type"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        label, color = format_action_type(action_type)
        st.markdown(
            f"<span style='color: {color}; font-weight: 600;'>{label}</span>: "
            f"{count} actions",
            unsafe_allow_html=True,
        )

# Completed strategies
completed = strategy_pipeline["completed_strategies"]
if completed > 0:
    st.markdown(
        f"**Completed Strategies:** {completed} full analysis sessions"
    )

if (
    not strategy_pipeline["pending_proposals"]
    and not strategy_pipeline["actions_by_type"]
):
    render_empty_state(
        "No Strategy Activity Yet",
        "The Strategist will propose optimizations after analyzing performance data.",
    )


render_divider()


# ---------------------------------------------------------------------------
# Recent Sessions (table view)
# ---------------------------------------------------------------------------

section_header("Session History", "All autopilot runs with status and summary")

if sessions:
    for session in (sessions or [])[:15]:
        status = session.get("status", "unknown")
        status_label, status_color = format_session_status(status)
        type_label, type_color = format_session_type(
            session.get("session_type", ""),
        )

        strategy = session.get("strategy_output", {})
        if isinstance(strategy, str):
            try:
                import json
                strategy = json.loads(strategy)
            except (ValueError, TypeError):
                strategy = {}
        summary = strategy.get("summary", "—")[:100]

        issues = session.get("detected_issues", [])
        issue_count = len(issues) if isinstance(issues, list) else 0

        created = session.get("created_at", "")[:19]
        completed = session.get("completed_at", "")
        completed_str = completed[:19] if completed else "—"

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 3, 1, 1.5])
            with c1:
                st.caption(created)
            with c2:
                st.markdown(
                    f"<span style='color: {type_color};'>{type_label}</span>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.text(summary)
            with c4:
                if issue_count:
                    st.markdown(f"**{issue_count}** issues")
                else:
                    st.text("—")
            with c5:
                st.markdown(
                    f"<span style='color: {status_color}; font-weight: 600;'>"
                    f"{status_label}</span>",
                    unsafe_allow_html=True,
                )
else:
    render_empty_state(
        "No Sessions Yet",
        "Run the Autopilot agent to see session history here.",
    )
