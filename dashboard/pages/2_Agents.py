"""
Sovereign Cockpit — Agent Barracks

Operational control panel for all agents. Features:
- Agent cards with live status, performance metrics, and health
- Brain View: inspect agent state, memory, recent decisions
- Force Run / Sleep / Kill controls
- Per-agent run history and error log
- Circuit breaker management
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


st.set_page_config(
    page_title="Agent Barracks — Sovereign Cockpit",
    page_icon="◆",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Auth + Theme + Sidebar
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()

from dashboard.theme import (
    COLORS, STATUS_CONFIG, inject_theme_css, page_header,
    section_header, status_badge, status_dot, render_empty_state,
    render_progress_bar, render_divider, render_timestamp,
    render_stat_grid,
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
# Emergency Controls (Sidebar)
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    f'<div style="height: 1px; background: {COLORS["border_subtle"]}; margin: 12px 0;"></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f'<div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; '
    f'color: {COLORS["status_red"]}; font-weight: 600; margin-bottom: 8px;">Emergency Controls</div>',
    unsafe_allow_html=True,
)

ec1, ec2 = st.sidebar.columns(2)

with ec1:
    if st.button("⏸ PAUSE ALL", use_container_width=True, key="pause_all"):
        db = get_db(vertical_id)
        agents = _safe_call(lambda: db.list_agent_records(), [])
        for agent in agents:
            try:
                db.client.table("agents").update({"enabled": False}).eq(
                    "agent_id", agent["agent_id"]
                ).eq("vertical_id", vertical_id).execute()
            except Exception:
                pass
        st.rerun()

with ec2:
    if st.button("▶ RESUME ALL", use_container_width=True, key="resume_all"):
        db = get_db(vertical_id)
        agents = _safe_call(lambda: db.list_agent_records(), [])
        for agent in agents:
            try:
                db.client.table("agents").update({"enabled": True}).eq(
                    "agent_id", agent["agent_id"]
                ).eq("vertical_id", vertical_id).execute()
            except Exception:
                pass
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

db = get_db(vertical_id)
agents = _safe_call(lambda: db.list_agent_records(), [])

# Agent stats (batch fetch)
agent_stats = _safe_call(lambda: db.get_agent_stats(days=30), [])
stats_by_agent = {s["agent_id"]: s for s in (agent_stats or [])}

page_header(
    "Agent Barracks",
    f"{len(agents)} agents registered for {vertical_id.replace('_', ' ').title()}",
)

# Safety notice
shadow_agents = [a for a in agents if a.get("shadow_mode", False)]
if shadow_agents:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 16px;
                     background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.2);
                     border-radius: 8px; margin-bottom: 16px;">
            <span style="font-size: 1.1rem;">◐</span>
            <span style="font-size: 0.78rem; color: {COLORS['text_secondary']};">
                <strong style="color: {COLORS['status_purple']};">{len(shadow_agents)} agent(s) in shadow mode</strong> —
                Actions are simulated and logged but never reach real customers.
                Review results before promoting to live.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not agents:
    st.markdown(
        render_empty_state(
            "◌",
            "No agents registered",
            "Deploy agent YAML configs to populate your fleet.",
            f"python main.py agent list {vertical_id}",
        ),
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Summary Bar
# ---------------------------------------------------------------------------

active_count = sum(1 for a in agents
                   if a.get("enabled", True) and not a.get("shadow_mode", False))
shadow_count = sum(1 for a in agents if a.get("shadow_mode", False))
paused_count = sum(1 for a in agents if not a.get("enabled", True))
tripped_count = 0
for a in agents:
    cfg = a.get("config", {}) or {}
    if cfg.get("consecutive_errors", 0) >= cfg.get("max_consecutive_errors", 5):
        tripped_count += 1

st.markdown(
    render_stat_grid([
        (str(active_count), "Active", COLORS["status_green"]),
        (str(shadow_count), "Shadow", COLORS["status_purple"]),
        (str(paused_count), "Paused", COLORS["status_red"]),
        (str(tripped_count), "Tripped", COLORS["status_yellow"]),
    ]),
    unsafe_allow_html=True,
)

st.markdown(render_divider(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent Cards
# ---------------------------------------------------------------------------

for agent in agents:
    agent_id = agent.get("agent_id", "unknown")
    agent_name = agent.get("name", agent_id)
    agent_type = agent.get("agent_type", "?")
    enabled = agent.get("enabled", True)
    shadow_mode = agent.get("shadow_mode", False)
    shadow_of = agent.get("shadow_of", "")
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

    badge_html = status_badge(status_key)
    cfg_status = STATUS_CONFIG[status_key]

    # Card border color based on status
    card_class = {"active": "active", "paused": "paused",
                  "shadow": "shadow", "circuit_breaker": "tripped"}.get(status_key, "")

    # Stats
    stats = stats_by_agent.get(agent_id, {})
    total_runs = stats.get("total_runs", 0)
    success_rate = stats.get("success_rate")
    avg_duration = stats.get("avg_duration_ms")
    failed_runs = stats.get("failed_runs", 0)

    sr_text = f"{float(success_rate):.0f}%" if success_rate is not None else "—"
    dur_text = f"{int(float(avg_duration))}ms" if avg_duration is not None else "—"

    # Pending tasks
    agent_pending = _safe_call(
        lambda aid=agent_id: db.count_pending_tasks(aid), 0
    )

    # ─── Agent Card ───────────────────────────────────
    with st.expander(
        f"{cfg_status['icon']} {agent_name} — {cfg_status['label']}",
        expanded=not enabled or consecutive_errors > 0,
    ):
        # ─── Info + Stats Row ─────────────────────────
        row1_cols = st.columns([2, 1, 1, 1, 1])

        with row1_cols[0]:
            st.markdown(
                f"""
                <div style="margin-bottom: 4px;">
                    <span style="font-weight: 600; color: {COLORS['text_primary']}; font-size: 0.95rem;">
                        {agent_name}
                    </span>
                    <span style="margin-left: 8px;">{badge_html}</span>
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['text_tertiary']};">
                    ID: <code>{agent_id}</code> · Type: <code>{agent_type}</code>
                    {f' · Shadow of: <code>{shadow_of}</code>' if shadow_of else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row1_cols[1]:
            st.metric("Runs (30d)", total_runs)

        with row1_cols[2]:
            st.metric("Success", sr_text)

        with row1_cols[3]:
            st.metric("Avg Time", dur_text)

        with row1_cols[4]:
            st.metric("Pending", agent_pending)

        # ─── Error Bar ────────────────────────────────
        if consecutive_errors > 0:
            error_pct = min(consecutive_errors / max(max_errors, 1) * 100, 100)
            bar_color = COLORS["status_red"] if error_pct >= 100 else COLORS["status_yellow"]
            st.markdown(
                f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem;
                                 color: {COLORS['text_secondary']};">
                        <span>Error Count</span>
                        <span style="color: {bar_color};">{consecutive_errors}/{max_errors}</span>
                    </div>
                    <div style="height: 4px; background: {COLORS['bg_secondary']}; border-radius: 2px; margin-top: 4px;">
                        <div style="width: {error_pct}%; height: 100%; background: {bar_color};
                                     border-radius: 2px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ─── Control Buttons ──────────────────────────
        st.markdown(
            f'<div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; '
            f'color: {COLORS["text_secondary"]}; font-weight: 600; margin: 12px 0 8px;">Controls</div>',
            unsafe_allow_html=True,
        )

        ctrl_cols = st.columns(4)

        with ctrl_cols[0]:
            if enabled:
                if st.button("⏸ Pause", key=f"pause_{agent_id}", use_container_width=True):
                    try:
                        db.client.table("agents").update(
                            {"enabled": False}
                        ).eq("agent_id", agent_id).eq(
                            "vertical_id", vertical_id
                        ).execute()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                if st.button("▶ Resume", key=f"resume_{agent_id}", use_container_width=True):
                    try:
                        db.client.table("agents").update(
                            {"enabled": True}
                        ).eq("agent_id", agent_id).eq(
                            "vertical_id", vertical_id
                        ).execute()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        with ctrl_cols[1]:
            label = "Exit Shadow" if shadow_mode else "Enter Shadow"
            if st.button(f"{'◐' if shadow_mode else '◑'} {label}", key=f"shadow_{agent_id}", use_container_width=True):
                try:
                    db.client.table("agents").update(
                        {"shadow_mode": not shadow_mode}
                    ).eq("agent_id", agent_id).eq(
                        "vertical_id", vertical_id
                    ).execute()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        with ctrl_cols[2]:
            if consecutive_errors > 0:
                if st.button("⟲ Reset Errors", key=f"reset_{agent_id}", use_container_width=True):
                    try:
                        db.reset_agent_errors(agent_id, vertical_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # ─── Brain View (Agent State) ─────────────────
        brain_tab, runs_tab = st.tabs(["🧠 Brain View", "📋 Run History"])

        with brain_tab:
            st.markdown(
                f'<div style="font-size: 0.72rem; color: {COLORS["text_tertiary"]}; margin-bottom: 8px;">'
                f'Agent configuration and internal state</div>',
                unsafe_allow_html=True,
            )

            # Show config as formatted JSON
            display_config = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "enabled": enabled,
                "shadow_mode": shadow_mode,
                "consecutive_errors": consecutive_errors,
                "max_consecutive_errors": max_errors,
            }

            # Merge in any additional config keys
            for k, v in config.items():
                if k not in ("consecutive_errors", "max_consecutive_errors"):
                    display_config[k] = v

            st.code(json.dumps(display_config, indent=2, default=str), language="json")

            # Recent insights from this agent
            try:
                from core.llm.embeddings import Embedder
                embedder = Embedder()
                query_emb = embedder.embed(f"Agent {agent_id} insights")
                insights = db.search_insights(
                    query_embedding=query_emb,
                    source_agent_id=agent_id,
                    limit=3,
                )
                if insights:
                    st.markdown(
                        f'<div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; '
                        f'color: {COLORS["text_accent"]}; font-weight: 600; margin: 12px 0 8px;">Recent Insights</div>',
                        unsafe_allow_html=True,
                    )
                    for insight in insights:
                        title = insight.get("title", "")
                        content = insight.get("content", "")[:200]
                        confidence = insight.get("confidence_score", 0)
                        st.markdown(
                            f"""
                            <div style="padding: 8px 12px; background: {COLORS['bg_secondary']};
                                         border-radius: 6px; margin-bottom: 6px; font-size: 0.78rem;">
                                <div style="color: {COLORS['text_primary']}; font-weight: 500;">
                                    {title or 'Untitled Insight'}
                                </div>
                                <div style="color: {COLORS['text_secondary']}; margin-top: 2px;">
                                    {content}
                                </div>
                                <div style="color: {COLORS['text_tertiary']}; margin-top: 4px; font-size: 0.7rem;">
                                    Confidence: {confidence:.0%}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass  # Insights are optional

        with runs_tab:
            runs = _safe_call(
                lambda aid=agent_id: db.get_agent_runs(agent_id=aid, limit=15), []
            )

            if runs:
                table_data = []
                for r in runs:
                    run_status = r.get("status", "?")
                    emoji = {"started": "🔄", "completed": "✅", "failed": "❌"}.get(
                        run_status, "❓"
                    )
                    table_data.append(
                        {
                            "Status": f"{emoji} {run_status}",
                            "Run ID": (r.get("run_id") or "")[:12],
                            "Duration": f"{r.get('duration_ms', 0) or 0}ms",
                            "Error": (r.get("error_message") or "—")[:60],
                            "Time": (r.get("created_at") or "")[:19],
                        }
                    )
                st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    render_empty_state(
                        "○",
                        "No runs recorded yet",
                        "This agent hasn't been executed. Start a run to see history.",
                    ),
                    unsafe_allow_html=True,
                )
