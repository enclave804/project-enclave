"""
Sovereign Cockpit — Command Center

The nerve center of the Sovereign Venture Engine. Provides:
- System health overview with real-time KPIs
- Live activity feed (agent actions as they happen)
- Agent fleet status grid
- Threat map (errors, circuit breakers, failed tasks)
- Training data / RLHF flywheel metrics

Run with: streamlit run dashboard/app.py
Or:       ./dashboard/run.sh
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Command Center — Sovereign Cockpit",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Authentication Gate
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

from dashboard.theme import (
    COLORS, STATUS_CONFIG, inject_theme_css, page_header,
    section_header, status_badge, feed_item, kpi_card,
    render_health_indicator, render_timestamp, sparkline_svg,
    render_empty_state, render_divider,
)

inject_theme_css()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

from dashboard.sidebar import render_sidebar

vertical_id = render_sidebar()


# ---------------------------------------------------------------------------
# DB connection (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db(vid: str):
    """Initialize DB connection (cached across reruns)."""
    from core.integrations.supabase_client import EnclaveDB
    return EnclaveDB(vid)


def _safe_call(fn, default=None):
    """Call a DB function, return default on any error."""
    try:
        return fn()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

db = get_db(vertical_id)

agents_data = _safe_call(lambda: db.list_agent_records(), [])
enabled_agents = [a for a in agents_data if a.get("enabled", True)]
disabled_agents = [a for a in agents_data if not a.get("enabled", True)]
shadow_agents = [a for a in agents_data if a.get("shadow_mode", False)]

pending_tasks = _safe_call(
    lambda: len(db.list_tasks(status="pending", limit=1000) or []), 0
)
failed_tasks = _safe_call(
    lambda: len(db.list_tasks(status="failed", limit=100) or []), 0
)

leads = _safe_call(lambda: db.count_contacts(), 0)
opps = _safe_call(lambda: db.count_opportunities(), 0)
avg_ticket = 6000  # configurable per vertical in future

# Agent stats
agent_stats = _safe_call(lambda: db.get_agent_stats(days=30), [])
stats_by_agent = {s["agent_id"]: s for s in (agent_stats or [])}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

# Auto-refresh: rerun every 30s when enabled
if st.sidebar.checkbox("Auto-refresh (30s)", value=False, key="auto_refresh"):
    st.cache_data.clear()
    import time as _time
    _time.sleep(0)  # Signal intent; Streamlit rerun handled below

header_col, ts_col = st.columns([3, 1])
with header_col:
    page_header(
        "◆ Command Center",
        f"Monitoring {vertical_id.replace('_', ' ').title()} — {len(agents_data)} agents deployed",
    )
with ts_col:
    st.markdown(
        f'<div style="text-align:right; padding-top:12px;">{render_timestamp()}</div>',
        unsafe_allow_html=True,
    )

# Auto-refresh polling
if st.session_state.get("auto_refresh", False):
    import time as _time
    _time.sleep(30)
    st.rerun()


# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------

section_header("KEY METRICS", f"{datetime.now(timezone.utc).strftime('%H:%M')} UTC")

# Fetch daily run data for sparklines (last 7 days)
daily_runs = _safe_call(lambda: db.get_daily_run_counts(days=7), [])
daily_values = [d.get("count", 0) for d in (daily_runs or [])] or [0]

# Fetch daily lead data for sparklines
daily_leads = _safe_call(lambda: db.get_daily_contact_counts(days=7), [])
daily_lead_values = [d.get("count", 0) for d in (daily_leads or [])] or [0]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    spark = sparkline_svg(daily_lead_values, COLORS["status_green"])
    st.markdown(
        f"""<div class="sov-kpi-card">
            <div class="sov-kpi-label">Total Leads</div>
            <div class="sov-kpi-value">{leads:,}</div>
            <div style="margin-top:8px;">{spark}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.metric("Active Agents", f"{len(enabled_agents)}/{len(agents_data)}")

with col3:
    st.metric("Tasks Pending", pending_tasks)

with col4:
    pipeline_value = opps * avg_ticket
    st.metric("Pipeline", f"${pipeline_value:,.0f}")

with col5:
    # Overall success rate
    total_runs = sum(s.get("total_runs", 0) for s in (agent_stats or []))
    total_success = sum(s.get("success_runs", 0) for s in (agent_stats or []))
    rate = (total_success / total_runs * 100) if total_runs > 0 else 0
    spark = sparkline_svg(daily_values, COLORS["accent_primary"])
    st.markdown(
        f"""<div class="sov-kpi-card">
            <div class="sov-kpi-label">Success Rate</div>
            <div class="sov-kpi-value">{rate:.0f}%</div>
            <div style="margin-top:8px;">{spark}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# System Health Bar
# ---------------------------------------------------------------------------

tripped_count = 0
for a in agents_data:
    cfg = a.get("config", {}) or {}
    if cfg.get("consecutive_errors", 0) >= cfg.get("max_consecutive_errors", 5):
        tripped_count += 1

healthy = len(enabled_agents) - tripped_count
degraded = tripped_count + len(shadow_agents)
failed_count = len(disabled_agents)

health_html = render_health_indicator(
    max(healthy, 0),
    max(degraded, 0),
    max(failed_count, 0),
)
st.markdown(health_html, unsafe_allow_html=True)

# Human-in-the-loop gate indicator
gate_html = f"""
<div style="display: flex; align-items: center; gap: 10px; padding: 8px 16px;
             background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.15);
             border-radius: 8px; margin: 8px 0 16px;">
    <span style="font-size: 0.9rem;">🛡</span>
    <span style="font-size: 0.72rem; color: {COLORS['text_secondary']};">
        <strong style="color: {COLORS['status_green']};">Human-in-the-Loop</strong> —
        All outbound actions require approval · {len(shadow_agents)} agent(s) in shadow mode ·
        {failed_tasks} task(s) need attention
    </span>
</div>
"""
st.markdown(gate_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Content: Activity Feed + Agent Fleet (side by side)
# ---------------------------------------------------------------------------

feed_col, fleet_col = st.columns([3, 2])


# ─── Live Activity Feed ──────────────────────────────────
with feed_col:
    section_header("LIVE ACTIVITY FEED", f"{total_runs} runs (30d)")

    runs = _safe_call(lambda: db.get_agent_runs(limit=25), [])

    if runs:
        feed_html = ""
        for r in runs:
            status_raw = r.get("status", "?")
            status_map = {
                "started": "running",
                "completed": "completed",
                "failed": "failed",
            }
            status_key = status_map.get(status_raw, "idle")

            agent_id = r.get("agent_id", "unknown")
            agent_type = r.get("agent_type", "")

            # Build description
            error_msg = r.get("error_message") or ""
            duration = r.get("duration_ms", 0) or 0

            if status_raw == "failed":
                desc = f"Failed: {error_msg[:80]}" if error_msg else "Failed"
            elif status_raw == "completed":
                desc = f"Completed in {duration}ms"
            else:
                desc = f"Started ({agent_type})"

            time_str = (r.get("created_at") or "")[:19].replace("T", " ")

            feed_html += feed_item(
                time_str=time_str,
                agent_name=agent_id,
                text=desc,
                status=status_key,
            )

        st.markdown(
            f'<div style="max-height: 480px; overflow-y: auto; padding-right: 4px;">'
            f'{feed_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            render_empty_state(
                "◌",
                "No agent activity recorded yet",
                "Agent runs will appear here in real-time as they execute.",
                f"python main.py agent run {vertical_id} outreach",
            ),
            unsafe_allow_html=True,
        )


# ─── Agent Fleet Grid ────────────────────────────────────
with fleet_col:
    section_header("AGENT FLEET", f"{len(agents_data)} deployed")

    if agents_data:
        for agent in agents_data:
            agent_id = agent.get("agent_id", "unknown")
            agent_name = agent.get("name", agent_id)
            agent_type = agent.get("agent_type", "?")
            enabled = agent.get("enabled", True)
            shadow_mode = agent.get("shadow_mode", False)
            config = agent.get("config", {}) or {}
            consecutive_errors = config.get("consecutive_errors", 0)
            max_errors = config.get("max_consecutive_errors", 5)

            # Determine status
            if not enabled:
                status_key = "paused"
            elif shadow_mode:
                status_key = "shadow"
            elif consecutive_errors >= max_errors:
                status_key = "circuit_breaker"
            else:
                status_key = "active"

            cfg = STATUS_CONFIG[status_key]
            badge_html = status_badge(status_key)

            # Stats for this agent
            stats = stats_by_agent.get(agent_id, {})
            success_rate = stats.get("success_rate")
            total_agent_runs = stats.get("total_runs", 0)

            sr_text = f"{float(success_rate):.0f}%" if success_rate is not None else "—"

            card_class = {"active": "active", "paused": "paused",
                          "shadow": "shadow", "circuit_breaker": "tripped"}.get(status_key, "")

            st.markdown(
                f"""
                <div class="sov-agent-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                                {agent_name}
                            </div>
                            <div style="font-size: 0.72rem; color: {COLORS['text_tertiary']}; margin-top: 2px;">
                                {agent_type} · {total_agent_runs} runs · {sr_text} success
                            </div>
                        </div>
                        <div>{badge_html}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            render_empty_state(
                "◌",
                "No agents registered",
                "Deploy agent configurations to see your fleet here.",
                f"python main.py agent list {vertical_id}",
            ),
            unsafe_allow_html=True,
        )


st.markdown(render_divider(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Threat Map (Errors & Warnings)
# ---------------------------------------------------------------------------

section_header("THREAT MAP", f"{failed_tasks} issues")

threat_col1, threat_col2, threat_col3 = st.columns(3)

with threat_col1:
    st.markdown(
        f"""
        <div class="sov-card">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                         color: {COLORS['text_secondary']}; font-weight: 600; margin-bottom: 8px;">
                Disabled Agents
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {COLORS['status_red'] if disabled_agents else COLORS['text_primary']};">
                {len(disabled_agents)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if disabled_agents:
        for a in disabled_agents:
            st.caption(f"🔴 {a.get('name', a.get('agent_id', '?'))}")

with threat_col2:
    st.markdown(
        f"""
        <div class="sov-card">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                         color: {COLORS['text_secondary']}; font-weight: 600; margin-bottom: 8px;">
                Circuit Breakers
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {COLORS['status_yellow'] if tripped_count > 0 else COLORS['text_primary']};">
                {tripped_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for a in agents_data:
        cfg = a.get("config", {}) or {}
        errs = cfg.get("consecutive_errors", 0)
        mx = cfg.get("max_consecutive_errors", 5)
        if errs >= mx:
            st.caption(f"⚡ {a.get('name', a.get('agent_id', '?'))} ({errs}/{mx})")

with threat_col3:
    st.markdown(
        f"""
        <div class="sov-card">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                         color: {COLORS['text_secondary']}; font-weight: 600; margin-bottom: 8px;">
                Failed Tasks
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {COLORS['status_red'] if failed_tasks > 5 else COLORS['text_primary']};">
                {failed_tasks}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Training Data / RLHF Flywheel
# ---------------------------------------------------------------------------

st.markdown(render_divider(), unsafe_allow_html=True)

section_header("RLHF FLYWHEEL", "learning loop")

training_stats = _safe_call(
    lambda: db.get_training_stats(vertical_id=vertical_id), []
)

if training_stats:
    rlhf_cols = st.columns(min(len(training_stats), 4))
    for i, stat in enumerate(training_stats):
        col = rlhf_cols[i % len(rlhf_cols)]
        with col:
            agent_id = stat.get("agent_id", "?")
            total = stat.get("total_examples", 0)
            avg = stat.get("avg_score")
            corrected = stat.get("corrected_examples", 0)

            avg_text = f"{avg:.0f}/100" if avg is not None else "—"

            st.markdown(
                f"""
                <div class="sov-card">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                                 color: {COLORS['text_accent']}; font-weight: 600; margin-bottom: 4px;">
                        {agent_id}
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {COLORS['text_primary']};">
                        {total} <span style="font-size: 0.7rem; color: {COLORS['text_tertiary']};">examples</span>
                    </div>
                    <div style="font-size: 0.72rem; color: {COLORS['text_secondary']}; margin-top: 4px;">
                        Score: {avg_text} · {corrected} corrections
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        render_empty_state(
            "◎",
            "No training examples yet",
            "Approve and edit content in the Approvals page to start the RLHF flywheel. "
            "Each human edit becomes a training signal.",
        ),
        unsafe_allow_html=True,
    )
