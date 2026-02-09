"""
Pipeline Control Center — Sales Pipeline Dashboard.

Visualizes the full revenue pipeline: deal stages, follow-up sequences,
meeting scheduling, and conversion metrics.

Phase 16: The Money Machine.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)
load_dotenv()

# ── Page config (must be first Streamlit command) ──────────────────
st.set_page_config(
    page_title="Pipeline | Sovereign Venture Engine",
    page_icon="🔄",
    layout="wide",
)

# ── Imports ────────────────────────────────────────────────────────
try:
    from dashboard.auth import require_auth
    from dashboard.theme import (
        COLORS,
        inject_theme_css,
        page_header,
        section_header,
        status_badge,
        render_empty_state,
        render_divider,
        render_timestamp,
        render_stat_grid,
    )
    from dashboard.sidebar import render_sidebar
    from dashboard.pages._pipeline_helpers import (
        compute_pipeline_metrics,
        compute_sequence_stats,
        compute_meeting_stats,
        format_deal_stage,
        format_pipeline_value,
        STAGE_CONFIG,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── Auth & Theme ──────────────────────────────────────────────────
require_auth()
inject_theme_css()
vertical_id = render_sidebar()

# ── DB Connection ─────────────────────────────────────────────────


@st.cache_resource
def get_db():
    """Cached DB connection."""
    try:
        from core.integrations.supabase_client import SupabaseClient
        return SupabaseClient()
    except Exception:
        return None


db = get_db()

# ── Page Header ───────────────────────────────────────────────────
page_header(
    title="Pipeline Control Center",
    subtitle="Track deals from first touch to closed-won",
    icon="🔄",
)

if not db:
    render_empty_state(
        "No database connection",
        "Configure SUPABASE_URL and SUPABASE_KEY in your .env file.",
    )
    st.stop()


# ── Load Data ────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_opportunities(_db, vid: str) -> list[dict[str, Any]]:
    try:
        result = _db.client.table("opportunities").select("*").eq(
            "vertical_id", vid
        ).execute()
        return result.data or []
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_sequences(_db, vid: str) -> list[dict[str, Any]]:
    try:
        result = _db.client.table("follow_up_sequences").select("*").eq(
            "vertical_id", vid
        ).execute()
        return result.data or []
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_meetings(_db, vid: str) -> list[dict[str, Any]]:
    try:
        result = _db.client.table("scheduled_meetings").select("*").eq(
            "vertical_id", vid
        ).execute()
        return result.data or []
    except Exception:
        return []


opportunities = load_opportunities(db, vertical_id)
sequences = load_sequences(db, vertical_id)
meetings = load_meetings(db, vertical_id)

# Compute metrics
pipeline_metrics = compute_pipeline_metrics(opportunities)
sequence_stats = compute_sequence_stats(sequences)
meeting_stats = compute_meeting_stats(meetings)

# ═════════════════════════════════════════════════════════════════
# Section 1: Pipeline Overview
# ═════════════════════════════════════════════════════════════════
section_header("Pipeline Overview", icon="📊")

stats = [
    {
        "label": "Active Deals",
        "value": str(pipeline_metrics["active_deals"]),
        "color": COLORS.get("primary", "#3b82f6"),
    },
    {
        "label": "Pipeline Value",
        "value": format_pipeline_value(pipeline_metrics["active_value_cents"]),
        "color": COLORS.get("success", "#10b981"),
    },
    {
        "label": "Won This Period",
        "value": str(pipeline_metrics["won_deals"]),
        "color": COLORS.get("success", "#10b981"),
    },
    {
        "label": "Win Rate",
        "value": f"{pipeline_metrics['win_rate'] * 100:.1f}%",
        "color": COLORS.get("warning", "#f59e0b"),
    },
]
render_stat_grid(stats)

# Stage funnel
render_divider()
st.subheader("Deal Funnel")

stage_order = ["prospect", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"]
breakdown = pipeline_metrics.get("stage_breakdown", {})

for stage in stage_order:
    data = breakdown.get(stage, {"count": 0, "value_cents": 0})
    label, color = format_deal_stage(stage)
    emoji = STAGE_CONFIG.get(stage, {}).get("emoji", "")
    count = data["count"]
    value = format_pipeline_value(data["value_cents"])

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"{emoji} **{label}**")
    with col2:
        st.markdown(f"`{count}` deals")
    with col3:
        st.markdown(f"`{value}`")

# ═════════════════════════════════════════════════════════════════
# Section 2: Follow-Up Sequences
# ═════════════════════════════════════════════════════════════════
render_divider()
section_header("Follow-Up Sequences", icon="📧")

seq_stats = [
    {
        "label": "Active Sequences",
        "value": str(sequence_stats["active"]),
        "color": COLORS.get("primary", "#3b82f6"),
    },
    {
        "label": "Completed",
        "value": str(sequence_stats["completed"]),
        "color": COLORS.get("success", "#10b981"),
    },
    {
        "label": "Reply Rate",
        "value": f"{sequence_stats['reply_rate'] * 100:.1f}%",
        "color": COLORS.get("warning", "#f59e0b"),
    },
    {
        "label": "Avg Steps",
        "value": f"{sequence_stats['avg_steps_completed']:.1f}",
        "color": COLORS.get("muted", "#6b7280"),
    },
]
render_stat_grid(seq_stats)

if sequences:
    with st.expander(f"Active Sequences ({sequence_stats['active']})", expanded=False):
        active_seqs = [s for s in sequences if s.get("status") == "active"]
        if active_seqs:
            for seq in active_seqs[:10]:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**{seq.get('contact_name', 'Unknown')}** ({seq.get('company_name', '')})")
                with col2:
                    st.markdown(f"`Step {seq.get('current_step', 0)}/{seq.get('max_steps', 5)}`")
                with col3:
                    st.markdown(f"📧 {seq.get('contact_email', '')}")
                with col4:
                    status_badge(seq.get("status", "active"))
        else:
            render_empty_state("No active sequences", "Start a follow-up sequence from the outreach pipeline.")

# ═════════════════════════════════════════════════════════════════
# Section 3: Meetings
# ═════════════════════════════════════════════════════════════════
render_divider()
section_header("Meeting Schedule", icon="📅")

mtg_stats = [
    {
        "label": "Proposed",
        "value": str(meeting_stats["proposed"]),
        "color": COLORS.get("warning", "#f59e0b"),
    },
    {
        "label": "Confirmed",
        "value": str(meeting_stats["confirmed"]),
        "color": COLORS.get("primary", "#3b82f6"),
    },
    {
        "label": "Completed",
        "value": str(meeting_stats["completed"]),
        "color": COLORS.get("success", "#10b981"),
    },
    {
        "label": "Confirmation Rate",
        "value": f"{meeting_stats['confirmation_rate'] * 100:.1f}%",
        "color": COLORS.get("success", "#10b981"),
    },
]
render_stat_grid(mtg_stats)

if meetings:
    upcoming = [
        m for m in meetings
        if m.get("status") in ("proposed", "confirmed")
    ]
    if upcoming:
        with st.expander(f"Upcoming Meetings ({len(upcoming)})", expanded=False):
            for meeting in upcoming[:10]:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(
                        f"**{meeting.get('contact_name', 'Unknown')}** "
                        f"({meeting.get('company_name', '')})"
                    )
                with col2:
                    mtype = meeting.get("meeting_type", "discovery")
                    st.markdown(f"`{mtype}`")
                with col3:
                    scheduled = meeting.get("scheduled_at", "TBD")
                    if scheduled and scheduled != "TBD":
                        try:
                            dt = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
                            st.markdown(dt.strftime("%b %d, %H:%M"))
                        except (ValueError, TypeError):
                            st.markdown(str(scheduled)[:16])
                    else:
                        st.markdown("TBD")
                with col4:
                    status_badge(meeting.get("status", "proposed"))

# ═════════════════════════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════════════════════════
render_divider()
render_timestamp()
