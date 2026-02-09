"""
Sovereign Cockpit — Growth Dashboard

Unified view of all growth channels:
- Proposals: pipeline value, win rate, deal flow
- Social Media: engagement metrics, content calendar, platform breakdown
- Ads Strategy: campaign performance, spend tracking, ROI metrics
- Growth Score: composite health of all growth channels
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
    page_title="Growth Dashboard — Sovereign Cockpit",
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

from dashboard.pages._growth_helpers import (
    compute_proposal_stats,
    compute_social_stats,
    compute_ads_stats,
    compute_growth_score,
    format_proposal_status,
    format_platform_icon,
    compute_campaign_health,
    group_calendar_by_date,
)


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

st.markdown(page_header(
    title="Growth Dashboard",
    subtitle="Proposals · Social · Ads — All Growth Channels in One View",
), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load data (with graceful fallbacks)
# ---------------------------------------------------------------------------

db = get_db(vertical_id)

# Try loading from DB, fall back to empty lists
proposals = _safe_call(
    lambda: db.client.table("proposals")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(100)
    .execute()
    .data,
    default=[],
) or []

social_posts = _safe_call(
    lambda: db.client.table("social_posts")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(100)
    .execute()
    .data,
    default=[],
) or []

ad_campaigns = _safe_call(
    lambda: db.client.table("ad_campaigns")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(50)
    .execute()
    .data,
    default=[],
) or []

calendar_entries = _safe_call(
    lambda: db.client.table("content_calendar")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("scheduled_date", desc=False)
    .limit(30)
    .execute()
    .data,
    default=[],
) or []


# ---------------------------------------------------------------------------
# Compute Stats
# ---------------------------------------------------------------------------

p_stats = compute_proposal_stats(proposals)
s_stats = compute_social_stats(social_posts)
a_stats = compute_ads_stats(ad_campaigns)
growth_score = compute_growth_score(p_stats, s_stats, a_stats)


# ---------------------------------------------------------------------------
# Growth Score Banner
# ---------------------------------------------------------------------------

score_color = "#10B981" if growth_score >= 70 else "#E8A838" if growth_score >= 40 else "#EF4444"
score_label = "Thriving" if growth_score >= 70 else "Growing" if growth_score >= 40 else "Needs Fuel"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['bg']} 100%);
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <div>
        <div style="font-size: 14px; color: {COLORS['text_secondary']}; text-transform: uppercase; letter-spacing: 1px;">
            Growth Score
        </div>
        <div style="font-size: 48px; font-weight: 700; color: {score_color}; line-height: 1.1;">
            {growth_score}
        </div>
        <div style="font-size: 14px; color: {COLORS['text_secondary']};">
            {score_label}
        </div>
    </div>
    <div style="display: flex; gap: 40px;">
        <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 600; color: {COLORS['text_primary']};">{p_stats['total']}</div>
            <div style="font-size: 12px; color: {COLORS['text_secondary']};">Proposals</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 600; color: {COLORS['text_primary']};">{s_stats['published']}</div>
            <div style="font-size: 12px; color: {COLORS['text_secondary']};">Posts Published</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 600; color: {COLORS['text_primary']};">{a_stats['active']}</div>
            <div style="font-size: 12px; color: {COLORS['text_secondary']};">Active Campaigns</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 600; color: {score_color};">
                ${p_stats['accepted_value']:,.0f}
            </div>
            <div style="font-size: 12px; color: {COLORS['text_secondary']};">Won Revenue</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------------------------

tab_proposals, tab_social, tab_ads, tab_calendar = st.tabs([
    f"📋 Proposals ({p_stats['total']})",
    f"📱 Social Media ({s_stats['total']})",
    f"📢 Ad Campaigns ({a_stats['total']})",
    f"📅 Content Calendar ({len(calendar_entries)})",
])


# ─── TAB 1: Proposals ────────────────────────────────────────────────

with tab_proposals:
    st.markdown(section_header("Proposal Pipeline"), unsafe_allow_html=True)

    # KPIs
    kpi_cols = st.columns(5)
    kpi_data = [
        ("Total", str(p_stats["total"]), ""),
        ("Win Rate", f"{p_stats['win_rate']:.1f}%", ""),
        ("Accepted", str(p_stats["accepted"]), ""),
        ("Pipeline Value", f"${p_stats['total_value']:,.0f}", ""),
        ("Avg Deal", f"${p_stats['avg_value']:,.0f}", ""),
    ]
    for i, (label, value, _) in enumerate(kpi_data):
        with kpi_cols[i]:
            st.metric(label, value)

    st.markdown(render_divider(), unsafe_allow_html=True)

    if not proposals:
        st.markdown(render_empty_state(
            "No Proposals Yet",
            "Run the Proposal Builder Agent to create your first proposal.",
        ), unsafe_allow_html=True)
    else:
        for p in proposals[:20]:
            status_text, status_color = format_proposal_status(p.get("status", "draft"))
            company = p.get("company_name", "Unknown Company")
            amount = float(p.get("pricing_amount", 0) or 0)
            prop_type = (p.get("proposal_type", "proposal") or "proposal").replace("_", " ").title()
            created = p.get("created_at", "")[:10]

            st.markdown(f"""
            <div style="
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <span style="font-weight: 600; color: {COLORS['text_primary']};">{company}</span>
                    <span style="color: {COLORS['text_secondary']}; margin-left: 12px;">{prop_type}</span>
                </div>
                <div style="display: flex; gap: 20px; align-items: center;">
                    <span style="color: {COLORS['text_secondary']}; font-size: 13px;">{created}</span>
                    <span style="font-weight: 600; color: {COLORS['text_primary']};">${amount:,.0f}</span>
                    <span style="
                        background: {status_color}22;
                        color: {status_color};
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: 500;
                    ">{status_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 2: Social Media ─────────────────────────────────────────────

with tab_social:
    st.markdown(section_header("Social Media Performance"), unsafe_allow_html=True)

    # KPIs
    kpi_cols = st.columns(5)
    social_kpis = [
        ("Published", str(s_stats["published"]), ""),
        ("Impressions", f"{s_stats['total_impressions']:,}", ""),
        ("Engagements", f"{s_stats['total_engagements']:,}", ""),
        ("Eng. Rate", f"{s_stats['engagement_rate']:.2f}%", ""),
        ("Top Platform", f"{format_platform_icon(s_stats['top_platform'])} {s_stats['top_platform'].title()}", ""),
    ]
    for i, (label, value, _) in enumerate(social_kpis):
        with kpi_cols[i]:
            st.metric(label, value)

    st.markdown(render_divider(), unsafe_allow_html=True)

    if not social_posts:
        st.markdown(render_empty_state(
            "No Social Posts Yet",
            "Run the Social Media Agent to create and publish posts.",
        ), unsafe_allow_html=True)
    else:
        # Platform breakdown
        platform_cols = st.columns(2)
        with platform_cols[0]:
            st.markdown(f"**Platform Breakdown**")
            platforms: dict[str, int] = {}
            for p in social_posts:
                plat = p.get("platform", "unknown")
                platforms[plat] = platforms.get(plat, 0) + 1
            for plat, count in sorted(platforms.items(), key=lambda x: -x[1]):
                icon = format_platform_icon(plat)
                st.markdown(f"{icon} **{plat.title()}** — {count} posts")

        with platform_cols[1]:
            st.markdown(f"**Engagement Summary**")
            st.markdown(f"❤️ Likes: **{s_stats['total_likes']:,}**")
            st.markdown(f"🔄 Shares: **{s_stats['total_shares']:,}**")
            st.markdown(f"💬 Comments: **{s_stats['total_comments']:,}**")
            st.markdown(f"🔗 Clicks: **{s_stats['total_clicks']:,}**")

        st.markdown(render_divider(), unsafe_allow_html=True)

        # Recent posts
        st.markdown("**Recent Posts**")
        for p in social_posts[:15]:
            platform = p.get("platform", "?")
            icon = format_platform_icon(platform)
            content = (p.get("content", "") or "")[:120]
            status = p.get("status", "draft")
            impressions = int(p.get("impressions", 0) or 0)

            st.markdown(f"""
            <div style="
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 6px;
            ">
                <div style="display: flex; justify-content: space-between;">
                    <span>{icon} <strong>{platform.title()}</strong> — {content}{'...' if len(p.get('content', '')) > 120 else ''}</span>
                    <span style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        👁 {impressions:,} · {status}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 3: Ad Campaigns ─────────────────────────────────────────────

with tab_ads:
    st.markdown(section_header("Ad Campaign Performance"), unsafe_allow_html=True)

    # KPIs
    kpi_cols = st.columns(5)
    ads_kpis = [
        ("Active", str(a_stats["active"]), ""),
        ("Total Spend", f"${a_stats['total_spend']:,.2f}", ""),
        ("Clicks", f"{a_stats['total_clicks']:,}", ""),
        ("Avg CTR", f"{a_stats['avg_ctr']:.2f}%", ""),
        ("Avg CPA", f"${a_stats['avg_cpa']:.2f}", ""),
    ]
    for i, (label, value, _) in enumerate(ads_kpis):
        with kpi_cols[i]:
            st.metric(label, value)

    st.markdown(render_divider(), unsafe_allow_html=True)

    if not ad_campaigns:
        st.markdown(render_empty_state(
            "No Ad Campaigns Yet",
            "Run the Ads Strategy Agent to design your first campaign.",
        ), unsafe_allow_html=True)
    else:
        for c in ad_campaigns[:15]:
            name = c.get("campaign_name", "Unnamed Campaign")
            platform = c.get("platform", "?")
            icon = format_platform_icon(platform)
            status = c.get("status", "draft")
            spend = float(c.get("total_spend", 0) or 0)
            clicks = int(c.get("clicks", 0) or 0)
            conversions = int(c.get("conversions", 0) or 0)
            health = compute_campaign_health(c)
            health_color = {"healthy": "#10B981", "needs_attention": "#E8A838", "critical": "#EF4444"}.get(health, "#8B8B8B")

            st.markdown(f"""
            <div style="
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-left: 3px solid {health_color};
                border-radius: 8px;
                padding: 14px 18px;
                margin-bottom: 8px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">
                            {icon} {name}
                        </span>
                        <span style="color: {COLORS['text_secondary']}; margin-left: 8px; font-size: 13px;">
                            {status.replace('_', ' ').title()}
                        </span>
                    </div>
                    <div style="display: flex; gap: 24px; font-size: 13px;">
                        <span>💰 ${spend:,.2f}</span>
                        <span>🖱️ {clicks:,} clicks</span>
                        <span>🎯 {conversions} conv.</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 4: Content Calendar ─────────────────────────────────────────

with tab_calendar:
    st.markdown(section_header("Content Calendar"), unsafe_allow_html=True)

    if not calendar_entries:
        st.markdown(render_empty_state(
            "Calendar Empty",
            "Social and Ads agents will populate the content calendar as they create content.",
        ), unsafe_allow_html=True)
    else:
        grouped = group_calendar_by_date(calendar_entries)
        for date_str, entries in sorted(grouped.items()):
            st.markdown(f"**📅 {date_str}**")
            for entry in entries:
                platform = entry.get("platform", "?")
                icon = format_platform_icon(platform)
                topic = entry.get("topic", entry.get("content_type", "Post"))
                status = entry.get("status", "planned")
                post_type = entry.get("post_type", "")

                status_colors = {
                    "planned": "#3B82F6",
                    "draft": "#8B8B8B",
                    "ready": "#10B981",
                    "published": "#10B981",
                    "cancelled": "#EF4444",
                }
                s_color = status_colors.get(status, "#8B8B8B")

                st.markdown(f"""
                <div style="
                    background: {COLORS['surface']};
                    border: 1px solid {COLORS['border_subtle']};
                    border-radius: 6px;
                    padding: 10px 14px;
                    margin-bottom: 4px;
                    margin-left: 20px;
                    display: flex;
                    justify-content: space-between;
                ">
                    <span>{icon} {topic} {f'({post_type})' if post_type else ''}</span>
                    <span style="color: {s_color}; font-size: 12px; font-weight: 500;">{status.title()}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")  # Spacer


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(render_divider(), unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align: center; color: {COLORS["text_secondary"]}; font-size: 12px;">'
    f"Growth Dashboard · {vertical_id} · "
    f"{render_timestamp(datetime.now(timezone.utc).isoformat())}"
    f"</div>",
    unsafe_allow_html=True,
)
