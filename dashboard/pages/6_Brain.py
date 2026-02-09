"""
Sovereign Cockpit — Brain Dashboard

The Hive Mind: cross-agent shared intelligence at a glance.
- Insight Feed: all shared learnings across agents
- Experiments: A/B tests running across the system
- Lead Scoring: predictive model performance
- Knowledge Flow: who's publishing/consuming what
- Brain Health: composite intelligence score
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
    page_title="Brain Dashboard — Sovereign Cockpit",
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

from dashboard.pages._brain_helpers import (
    compute_insight_stats,
    compute_experiment_stats,
    compute_score_distribution,
    compute_knowledge_flow,
    compute_brain_health,
    format_insight_type,
    format_confidence,
    format_experiment_status,
    format_score_tier,
)


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header("Brain Dashboard", "The Hive Mind: cross-agent shared intelligence, experiments, and predictions.")


# ---------------------------------------------------------------------------
# Load data (with graceful fallback)
# ---------------------------------------------------------------------------

db = get_db(vertical_id)

# Insights from shared_insights table
insights = _safe_call(
    lambda: db.client.table("shared_insights")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(100)
    .execute().data,
    [],
)

# Experiments
experiments = _safe_call(
    lambda: db.client.table("experiments")
    .select("*")
    .eq("vertical_id", vertical_id)
    .order("created_at", desc=True)
    .limit(50)
    .execute().data,
    [],
)

# Lead scores (from pipeline_runs or dedicated scoring runs)
lead_scores_raw = _safe_call(
    lambda: db.client.table("lead_scores")
    .select("score")
    .eq("vertical_id", vertical_id)
    .limit(500)
    .execute().data,
    [],
)
lead_scores = [int(r.get("score", 0)) for r in (lead_scores_raw or []) if r.get("score") is not None]


# Compute stats
insight_stats = compute_insight_stats(insights or [])
experiment_stats = compute_experiment_stats(experiments or [])
score_stats = compute_score_distribution(lead_scores)
brain_health = compute_brain_health(insight_stats, experiment_stats, score_stats)


# ---------------------------------------------------------------------------
# Brain Health Score (top banner)
# ---------------------------------------------------------------------------

score_color = (
    COLORS["success"] if brain_health["score"] >= 80
    else COLORS["warning"] if brain_health["score"] >= 60
    else COLORS["error"]
)

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {COLORS['card_bg']}, {COLORS['surface']});
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
">
    <div style="text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: {score_color};">
            {brain_health['score']}
        </div>
        <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; text-transform: uppercase;">
            Brain Health
        </div>
    </div>
    <div style="flex: 1; display: flex; gap: 2rem;">
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Knowledge Depth</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {COLORS['text_primary']};">
                {brain_health['knowledge_score']}%
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Experimentation</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {COLORS['text_primary']};">
                {brain_health['experiment_score']}%
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Prediction Quality</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {COLORS['text_primary']};">
                {brain_health['prediction_score']}%
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Grade</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {score_color};">
                {brain_health['grade']}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs: Insights | Experiments | Lead Scoring | Knowledge Flow
# ---------------------------------------------------------------------------

tab_insights, tab_experiments, tab_scoring, tab_flow = st.tabs([
    "Insight Feed", "Experiments", "Lead Scoring", "Knowledge Flow",
])


# ═══════════════════════════════════════════════════════════════
# INSIGHT FEED TAB
# ═══════════════════════════════════════════════════════════════

with tab_insights:

    # ── Insight KPIs ───────────────────────────────────────────

    section_header("Shared Intelligence")

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        render_stat_grid([{"label": "Total Insights", "value": str(insight_stats["total"])}])
    with kpi_cols[1]:
        render_stat_grid([{"label": "Topics Covered", "value": str(insight_stats["topics_covered"])}])
    with kpi_cols[2]:
        render_stat_grid([{"label": "Avg Confidence", "value": f"{insight_stats['avg_confidence']:.0%}"}])
    with kpi_cols[3]:
        render_stat_grid([{"label": "High Confidence", "value": str(insight_stats["high_confidence_count"])}])
    with kpi_cols[4]:
        render_stat_grid([{"label": "Agent Sources", "value": str(len(insight_stats["by_agent"]))}])

    render_divider()

    # ── Insights by Type breakdown ─────────────────────────────

    if insight_stats["by_type"]:
        section_header("Insights by Type")
        type_cols = st.columns(min(len(insight_stats["by_type"]), 6))
        for idx, (itype, count) in enumerate(sorted(
            insight_stats["by_type"].items(), key=lambda x: x[1], reverse=True
        )):
            text, color = format_insight_type(itype)
            with type_cols[idx % len(type_cols)]:
                st.markdown(f"""
                <div style="
                    text-align: center; padding: 0.75rem;
                    background: {color}15; border: 1px solid {color}30;
                    border-radius: 8px; margin-bottom: 0.5rem;
                ">
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{count}</div>
                    <div style="font-size: 0.75rem; color: {COLORS['text_secondary']};">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        render_divider()

    # ── Insight Feed ───────────────────────────────────────────

    section_header("Recent Insights")

    if not insights:
        render_empty_state(
            "No shared insights yet",
            "Agents will publish learnings as they work. Run some agents to populate the Hive Mind.",
        )
    else:
        for ins in (insights or [])[:15]:
            itype = ins.get("insight_type", "unknown")
            type_text, type_color = format_insight_type(itype)
            conf = float(ins.get("confidence_score", 0) or 0)
            conf_text, conf_color = format_confidence(conf)
            agent = ins.get("source_agent_id", "unknown")
            title = ins.get("title", "")[:80]
            content = ins.get("content", "")[:150]
            created = str(ins.get("created_at", ""))[:10]
            usage = int(ins.get("usage_count", 0) or 0)

            st.markdown(f"""
            <div style="
                padding: 0.75rem 1rem;
                background: {COLORS['card_bg']};
                border: 1px solid {COLORS['border']};
                border-left: 3px solid {type_color};
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem;">
                    <span style="
                        padding: 2px 8px; border-radius: 4px;
                        font-size: 0.7rem; font-weight: 600;
                        background: {type_color}20; color: {type_color};
                    ">{type_text}</span>
                    <span style="
                        padding: 2px 8px; border-radius: 4px;
                        font-size: 0.7rem; font-weight: 600;
                        background: {conf_color}20; color: {conf_color};
                    ">{conf_text}</span>
                    <span style="font-size: 0.75rem; color: {COLORS['text_secondary']};">
                        by {agent} · {created} · used {usage}x
                    </span>
                </div>
                <div style="font-weight: 600; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                    {title}
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-top: 0.2rem;">
                    {content}{'...' if len(ins.get('content', '')) > 150 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS TAB
# ═══════════════════════════════════════════════════════════════

with tab_experiments:

    # ── Experiment KPIs ────────────────────────────────────────

    section_header("Experiment Lab")

    exp_cols = st.columns(4)
    with exp_cols[0]:
        render_stat_grid([{"label": "Total Experiments", "value": str(experiment_stats["total"])}])
    with exp_cols[1]:
        render_stat_grid([{"label": "Active", "value": str(experiment_stats["active"])}])
    with exp_cols[2]:
        render_stat_grid([{"label": "Concluded", "value": str(experiment_stats["concluded"])}])
    with exp_cols[3]:
        render_stat_grid([{"label": "Total Observations", "value": str(experiment_stats["total_observations"])}])

    render_divider()

    # ── Experiment List ────────────────────────────────────────

    section_header("Experiments")

    if not experiments:
        render_empty_state(
            "No experiments yet",
            "Agents run A/B tests automatically. The Experiment Engine tracks variants and declares winners.",
        )
    else:
        for exp in (experiments or [])[:15]:
            status = exp.get("status", "active")
            status_text, status_color = format_experiment_status(status)
            name = exp.get("name", "Unnamed Experiment")
            agent = exp.get("agent_id", "unknown")
            metric = exp.get("metric", "conversion")
            variants = exp.get("variants", [])
            if isinstance(variants, str):
                import json as _json
                try:
                    variants = _json.loads(variants)
                except (ValueError, TypeError):
                    variants = []
            variant_count = len(variants) if isinstance(variants, list) else 0
            total_obs = int(exp.get("total_observations", 0) or 0)
            created = str(exp.get("created_at", ""))[:10]

            st.markdown(f"""
            <div style="
                padding: 0.75rem 1rem;
                background: {COLORS['card_bg']};
                border: 1px solid {COLORS['border']};
                border-left: 3px solid {status_color};
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem;">
                    <span style="
                        padding: 2px 8px; border-radius: 4px;
                        font-size: 0.7rem; font-weight: 600;
                        background: {status_color}20; color: {status_color};
                    ">{status_text}</span>
                    <span style="font-size: 0.75rem; color: {COLORS['text_secondary']};">
                        by {agent} · {metric} · {variant_count} variants · {total_obs} obs · {created}
                    </span>
                </div>
                <div style="font-weight: 600; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                    {name}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# LEAD SCORING TAB
# ═══════════════════════════════════════════════════════════════

with tab_scoring:

    # ── Score Distribution KPIs ────────────────────────────────

    section_header("Lead Score Distribution")

    score_cols = st.columns(6)
    with score_cols[0]:
        render_stat_grid([{"label": "Total Scored", "value": str(score_stats["total"])}])
    with score_cols[1]:
        render_stat_grid([{"label": "Average", "value": f"{score_stats['avg']:.0f}"}])
    with score_cols[2]:
        render_stat_grid([{"label": "Median", "value": f"{score_stats['median']:.0f}"}])
    with score_cols[3]:
        tier_text, tier_color = format_score_tier(80)
        render_stat_grid([{"label": "Hot (80+)", "value": str(score_stats["hot"])}])
    with score_cols[4]:
        tier_text, tier_color = format_score_tier(60)
        render_stat_grid([{"label": "Warm (60-79)", "value": str(score_stats["warm"])}])
    with score_cols[5]:
        render_stat_grid([{"label": "Cold (<40)", "value": str(score_stats["cold"])}])

    render_divider()

    # ── Score Tier Breakdown (visual) ──────────────────────────

    if score_stats["total"] > 0:
        section_header("Score Tiers")

        total = score_stats["total"]
        tiers = [
            ("Hot", score_stats["hot"], "#EF4444"),
            ("Warm", score_stats["warm"], "#F59E0B"),
            ("Lukewarm", score_stats["lukewarm"], "#3B82F6"),
            ("Cold", score_stats["cold"], "#6B7280"),
        ]

        for label, count, color in tiers:
            pct = (count / total * 100) if total > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; font-weight: 600; color: {color};">{label}</span>
                    <span style="font-size: 0.85rem; color: {COLORS['text_secondary']};">{count} ({pct:.0f}%)</span>
                </div>
                <div style="
                    height: 8px; border-radius: 4px;
                    background: {COLORS['surface']};
                    overflow: hidden;
                ">
                    <div style="
                        height: 100%; width: {pct}%;
                        background: {color};
                        border-radius: 4px;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        render_empty_state(
            "No lead scores yet",
            "The Lead Scorer will assign scores as leads flow through the pipeline.",
        )


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE FLOW TAB
# ═══════════════════════════════════════════════════════════════

with tab_flow:

    section_header("Cross-Agent Knowledge Flow")

    # ── Agent Contributions ────────────────────────────────────

    if insight_stats["by_agent"]:
        section_header("Agent Contributions")

        agent_cols = st.columns(min(len(insight_stats["by_agent"]), 6))
        for idx, (agent, count) in enumerate(sorted(
            insight_stats["by_agent"].items(), key=lambda x: x[1], reverse=True
        )):
            with agent_cols[idx % len(agent_cols)]:
                st.markdown(f"""
                <div style="
                    text-align: center; padding: 0.75rem;
                    background: {COLORS['card_bg']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px; margin-bottom: 0.5rem;
                ">
                    <div style="font-size: 1.5rem; font-weight: 700; color: {COLORS['accent']};">{count}</div>
                    <div style="font-size: 0.75rem; color: {COLORS['text_secondary']};">{agent}</div>
                </div>
                """, unsafe_allow_html=True)

        render_divider()

    # ── Insight Types Summary ──────────────────────────────────

    if insight_stats["by_type"]:
        section_header("Knowledge Categories")

        for itype, count in sorted(insight_stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            type_text, type_color = format_insight_type(itype)
            bar_width = min(count / max(insight_stats["by_type"].values()) * 100, 100) if insight_stats["by_type"] else 0

            st.markdown(f"""
            <div style="
                display: flex; align-items: center; gap: 1rem;
                padding: 0.5rem 0;
            ">
                <div style="width: 120px; font-size: 0.85rem; font-weight: 600; color: {type_color};">{type_text}</div>
                <div style="flex: 1; height: 8px; border-radius: 4px; background: {COLORS['surface']}; overflow: hidden;">
                    <div style="height: 100%; width: {bar_width}%; background: {type_color}; border-radius: 4px;"></div>
                </div>
                <div style="width: 40px; text-align: right; font-size: 0.85rem; color: {COLORS['text_secondary']};">{count}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        render_empty_state(
            "No knowledge flow data",
            "As agents share insights through the Hive Mind, the knowledge network will appear here.",
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

render_divider()
render_timestamp()
