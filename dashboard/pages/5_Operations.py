"""
Sovereign Cockpit — Operations Dashboard

Unified view of all post-sale operations:
- Finance: invoices, payments, overdue tracking, P&L
- Customer Success: client health, onboarding, churn risk
- Operations Score: composite health of all operations
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
    page_title="Operations Dashboard — Sovereign Cockpit",
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

from dashboard.pages._operations_helpers import (
    compute_invoice_stats,
    compute_reminder_stats,
    compute_client_stats,
    compute_interaction_stats,
    compute_operations_score,
    format_invoice_status,
    format_amount_cents,
    format_reminder_tone,
    format_client_status,
    format_churn_risk,
    format_interaction_type,
)


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header("Operations Center", "Post-sale operations: invoicing, client success, and P&L tracking.")


# ---------------------------------------------------------------------------
# Load data (with graceful fallback)
# ---------------------------------------------------------------------------

db = get_db(vertical_id)

# Finance data
invoices = _safe_call(lambda: db.client.table("invoices").select("*").eq("vertical_id", vertical_id).execute().data, [])
reminders = _safe_call(lambda: db.client.table("payment_reminders").select("*").eq("vertical_id", vertical_id).execute().data, [])

# CS data
clients = _safe_call(lambda: db.client.table("client_records").select("*").eq("vertical_id", vertical_id).execute().data, [])
interactions = _safe_call(lambda: db.client.table("cs_interactions").select("*").eq("vertical_id", vertical_id).execute().data, [])

# Compute stats
invoice_stats = compute_invoice_stats(invoices or [])
reminder_stats = compute_reminder_stats(reminders or [])
client_stats = compute_client_stats(clients or [])
interaction_stats = compute_interaction_stats(interactions or [])
ops_score = compute_operations_score(invoice_stats, client_stats, reminder_stats)


# ---------------------------------------------------------------------------
# Operations Score (top banner)
# ---------------------------------------------------------------------------

score_color = (
    COLORS["success"] if ops_score["score"] >= 80
    else COLORS["warning"] if ops_score["score"] >= 60
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
            {ops_score['score']}
        </div>
        <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; text-transform: uppercase;">
            Operations Score
        </div>
    </div>
    <div style="flex: 1; display: flex; gap: 2rem;">
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Collection Rate</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {COLORS['text_primary']};">
                {ops_score['collection_score']}%
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Client Health</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {COLORS['text_primary']};">
                {ops_score['health_score']}%
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Responsiveness</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {COLORS['text_primary']};">
                {ops_score['responsiveness_score']}%
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 4px;">Grade</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {score_color};">
                {ops_score['grade']}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Two-column layout: Finance | Client Success
# ---------------------------------------------------------------------------

tab_finance, tab_cs = st.tabs(["Finance", "Customer Success"])


# ═══════════════════════════════════════════════════════════════
# FINANCE TAB
# ═══════════════════════════════════════════════════════════════

with tab_finance:

    # ── Invoice KPIs ──────────────────────────────────────────

    section_header("Invoice Overview")

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        render_stat_grid([{"label": "Total Invoices", "value": str(invoice_stats["total"])}])
    with kpi_cols[1]:
        render_stat_grid([{"label": "Paid", "value": str(invoice_stats["paid"])}])
    with kpi_cols[2]:
        render_stat_grid([{"label": "Open", "value": str(invoice_stats["open"])}])
    with kpi_cols[3]:
        render_stat_grid([{"label": "Overdue", "value": str(invoice_stats["overdue"])}])
    with kpi_cols[4]:
        render_stat_grid([{"label": "Collection Rate", "value": f"{invoice_stats['collection_rate']:.0f}%"}])

    render_divider()

    # ── Revenue Summary ───────────────────────────────────────

    section_header("Revenue Summary")

    rev_cols = st.columns(4)
    with rev_cols[0]:
        render_stat_grid([{
            "label": "Total Invoiced",
            "value": format_amount_cents(invoice_stats["total_amount"]),
        }])
    with rev_cols[1]:
        render_stat_grid([{
            "label": "Collected",
            "value": format_amount_cents(invoice_stats["paid_amount"]),
        }])
    with rev_cols[2]:
        render_stat_grid([{
            "label": "Outstanding",
            "value": format_amount_cents(invoice_stats["open_amount"]),
        }])
    with rev_cols[3]:
        render_stat_grid([{
            "label": "Overdue Amount",
            "value": format_amount_cents(invoice_stats["overdue_amount"]),
        }])

    render_divider()

    # ── Invoice List ──────────────────────────────────────────

    section_header("Recent Invoices")

    if not invoices:
        render_empty_state("No invoices yet", "Finance Agent will create invoices from accepted proposals.")
    else:
        for inv in sorted(invoices, key=lambda x: x.get("created_at", ""), reverse=True)[:10]:
            status_text, status_color = format_invoice_status(inv.get("status", "draft"))
            amount = format_amount_cents(int(inv.get("amount_cents", 0) or 0))
            company = inv.get("company_name", "Unknown")
            created = inv.get("created_at", "")[:10]

            st.markdown(f"""
            <div style="
                display: flex; align-items: center; gap: 1rem;
                padding: 0.75rem 1rem;
                background: {COLORS['card_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <div style="flex: 2; font-weight: 600; color: {COLORS['text_primary']};">{company}</div>
                <div style="flex: 1; color: {COLORS['text_primary']};">{amount}</div>
                <div style="flex: 1;">
                    <span style="
                        padding: 2px 8px; border-radius: 4px;
                        font-size: 0.75rem; font-weight: 600;
                        background: {status_color}20; color: {status_color};
                    ">{status_text}</span>
                </div>
                <div style="flex: 1; color: {COLORS['text_secondary']}; font-size: 0.85rem;">{created}</div>
            </div>
            """, unsafe_allow_html=True)

    render_divider()

    # ── Payment Reminders ─────────────────────────────────────

    section_header("Payment Reminders")

    rem_cols = st.columns(4)
    with rem_cols[0]:
        render_stat_grid([{"label": "Total Reminders", "value": str(reminder_stats["total"])}])
    with rem_cols[1]:
        render_stat_grid([{"label": "Sent", "value": str(reminder_stats["sent"])}])
    with rem_cols[2]:
        render_stat_grid([{"label": "Pending Review", "value": str(reminder_stats["pending"])}])
    with rem_cols[3]:
        tone_summary = " / ".join(
            f"{v} {k}" for k, v in reminder_stats["by_tone"].items() if v > 0
        ) or "None"
        render_stat_grid([{"label": "By Tone", "value": tone_summary}])


# ═══════════════════════════════════════════════════════════════
# CUSTOMER SUCCESS TAB
# ═══════════════════════════════════════════════════════════════

with tab_cs:

    # ── Client KPIs ───────────────────────────────────────────

    section_header("Client Overview")

    cs_cols = st.columns(5)
    with cs_cols[0]:
        render_stat_grid([{"label": "Total Clients", "value": str(client_stats["total"])}])
    with cs_cols[1]:
        render_stat_grid([{"label": "Active", "value": str(client_stats["active"])}])
    with cs_cols[2]:
        render_stat_grid([{"label": "Onboarding", "value": str(client_stats["onboarding"])}])
    with cs_cols[3]:
        render_stat_grid([{"label": "At Risk", "value": str(client_stats["at_risk"])}])
    with cs_cols[4]:
        risk_text, risk_color = format_churn_risk(client_stats["avg_churn_risk"])
        render_stat_grid([{"label": "Avg Churn Risk", "value": f"{risk_text} ({client_stats['avg_churn_risk']:.0%})"}])

    render_divider()

    # ── Client List ───────────────────────────────────────────

    section_header("Client Portfolio")

    if not clients:
        render_empty_state("No clients yet", "Customer Success Agent tracks clients from accepted proposals.")
    else:
        for client in sorted(clients, key=lambda x: float(x.get("churn_risk", 0) or 0), reverse=True)[:10]:
            status_text, status_color = format_client_status(client.get("status", "active"))
            risk = float(client.get("churn_risk", 0) or 0)
            risk_text, risk_color = format_churn_risk(risk)
            company = client.get("company_name", "Unknown")
            contact = client.get("contact_name", "")
            last_contact = client.get("last_contact_at", "")
            if last_contact:
                last_contact = last_contact[:10]

            st.markdown(f"""
            <div style="
                display: flex; align-items: center; gap: 1rem;
                padding: 0.75rem 1rem;
                background: {COLORS['card_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <div style="flex: 2;">
                    <div style="font-weight: 600; color: {COLORS['text_primary']};">{company}</div>
                    <div style="font-size: 0.8rem; color: {COLORS['text_secondary']};">{contact}</div>
                </div>
                <div style="flex: 1;">
                    <span style="
                        padding: 2px 8px; border-radius: 4px;
                        font-size: 0.75rem; font-weight: 600;
                        background: {status_color}20; color: {status_color};
                    ">{status_text}</span>
                </div>
                <div style="flex: 1;">
                    <span style="
                        padding: 2px 8px; border-radius: 4px;
                        font-size: 0.75rem; font-weight: 600;
                        background: {risk_color}20; color: {risk_color};
                    ">{risk_text}</span>
                </div>
                <div style="flex: 1; color: {COLORS['text_secondary']}; font-size: 0.85rem;">{last_contact or 'No contact'}</div>
            </div>
            """, unsafe_allow_html=True)

    render_divider()

    # ── CS Interactions ───────────────────────────────────────

    section_header("Recent Interactions")

    int_cols = st.columns(4)
    with int_cols[0]:
        render_stat_grid([{"label": "Total Interactions", "value": str(interaction_stats["total"])}])
    with int_cols[1]:
        render_stat_grid([{"label": "Sent", "value": str(interaction_stats["sent"])}])
    with int_cols[2]:
        render_stat_grid([{"label": "Pending", "value": str(interaction_stats["pending"])}])
    with int_cols[3]:
        type_summary = " / ".join(
            f"{v} {k.replace('_', ' ')}"
            for k, v in interaction_stats["by_type"].items()
            if v > 0
        ) or "None"
        render_stat_grid([{"label": "By Type", "value": type_summary}])


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

render_divider()
render_timestamp()
