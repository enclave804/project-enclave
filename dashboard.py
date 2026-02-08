"""
Project Enclave — Sales Pipeline Dashboard

Streamlit dashboard for monitoring pipeline health, outreach performance,
company intelligence, and the RAG knowledge base.

Run with: streamlit run dashboard.py
Or:       python main.py dashboard enclave_guard
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env", override=True)

from core.integrations.supabase_client import EnclaveDB

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enclave Dashboard",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("\U0001f6e1\ufe0f Enclave Dashboard")

# Vertical selector — for now only enclave_guard exists
vertical_options = {"Enclave Guard": "enclave_guard"}
vertical_label = st.sidebar.selectbox("Vertical", list(vertical_options.keys()))
vertical_id = vertical_options[vertical_label]

page = st.sidebar.radio(
    "Navigate",
    [
        "\U0001f4ca Overview",
        "\u2699\ufe0f Pipeline Activity",
        "\U0001f3e2 Companies & Contacts",
        "\U0001f4e7 Outreach Performance",
        "\U0001f9e0 Knowledge Base",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Project Enclave v0.1.0")

# ---------------------------------------------------------------------------
# DB connection (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db(vid: str) -> EnclaveDB:
    return EnclaveDB(vid)


db = get_db(vertical_id)


# ===================================================================
# PAGE: Overview
# ===================================================================

def page_overview():
    st.title(f"\U0001f4ca {vertical_label} — Overview")

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies", db.count_companies())
    with col2:
        st.metric("Contacts", db.count_contacts())
    with col3:
        st.metric("Outreach Events", db.count_outreach_events())
    with col4:
        st.metric("Opportunities", db.count_opportunities())

    st.markdown("---")

    # Outreach performance
    st.subheader("\U0001f4e8 Outreach Performance (Last 30 Days)")
    stats = _get_outreach_stats(vertical_id, 30)

    if stats:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sent", stats.get("total_sent", 0))
        with c2:
            open_rate = stats.get("open_rate", 0) or 0
            st.metric("Open Rate", f"{open_rate:.1%}")
        with c3:
            reply_rate = stats.get("reply_rate", 0) or 0
            st.metric("Reply Rate", f"{reply_rate:.1%}")
        with c4:
            meeting_rate = stats.get("meeting_rate", 0) or 0
            st.metric("Meeting Rate", f"{meeting_rate:.1%}")
    else:
        st.info("No outreach data yet. Run `python main.py test-run enclave_guard` to generate data.")

    st.markdown("---")

    # RAG learning loop
    st.subheader("\U0001f9e0 RAG Learning Loop")
    event_count = db.count_outreach_events()
    threshold = 100
    progress = min(event_count / threshold, 1.0)
    st.progress(progress, text=f"{event_count} / {threshold} outreach events (activates at {threshold})")
    if event_count >= threshold:
        st.success("Learning loop is ACTIVE! Pattern extraction is running.")
    else:
        remaining = threshold - event_count
        st.info(f"{remaining} more outreach events needed to activate the learning loop.")


# ===================================================================
# PAGE: Pipeline Activity
# ===================================================================

def page_pipeline():
    st.title("\u2699\ufe0f Pipeline Activity")

    runs = _get_pipeline_runs(vertical_id, 100)
    if not runs:
        st.info("No pipeline runs yet.")
        return

    # Recent runs table
    st.subheader("Recent Pipeline Runs")
    table_data = []
    for r in runs[:50]:
        table_data.append({
            "Lead ID": (r.get("lead_id") or "")[:8] + "...",
            "Node": r.get("node_name", ""),
            "Status": r.get("status", ""),
            "Duration (ms)": r.get("duration_ms", ""),
            "Error": (r.get("error_message") or "")[:60],
            "Time": (r.get("created_at") or "")[:19],
        })
    st.dataframe(table_data, use_container_width=True)

    st.markdown("---")

    # Node stats
    st.subheader("Node Performance")
    node_stats: dict[str, dict] = {}
    for r in runs:
        node = r.get("node_name", "unknown")
        if node not in node_stats:
            node_stats[node] = {"total": 0, "completed": 0, "failed": 0, "total_ms": 0}
        node_stats[node]["total"] += 1
        if r.get("status") == "completed":
            node_stats[node]["completed"] += 1
        elif r.get("status") == "failed":
            node_stats[node]["failed"] += 1
        node_stats[node]["total_ms"] += r.get("duration_ms") or 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Success Rate by Node**")
        chart_data = {}
        for node, s in node_stats.items():
            rate = s["completed"] / s["total"] if s["total"] > 0 else 0
            chart_data[node] = rate
        st.bar_chart(chart_data)

    with col2:
        st.markdown("**Avg Duration (ms) by Node**")
        dur_data = {}
        for node, s in node_stats.items():
            avg = s["total_ms"] / s["total"] if s["total"] > 0 else 0
            dur_data[node] = round(avg)
        st.bar_chart(dur_data)


# ===================================================================
# PAGE: Companies & Contacts
# ===================================================================

def page_companies():
    st.title("\U0001f3e2 Companies & Contacts")

    companies = db.list_companies(limit=100)
    if not companies:
        st.info("No companies in the database yet.")
        return

    # Companies table
    st.subheader(f"Companies ({len(companies)})")
    table_data = []
    for c in companies:
        tech = c.get("tech_stack") or {}
        tech_list = list(tech.keys())[:5] if isinstance(tech, dict) else []
        table_data.append({
            "Name": c.get("name", ""),
            "Domain": c.get("domain", ""),
            "Industry": c.get("industry", ""),
            "Employees": c.get("employee_count", ""),
            "Tech Stack": ", ".join(tech_list),
            "Enriched": "\u2705" if c.get("enriched_at") else "\u274c",
        })
    st.dataframe(table_data, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Industry distribution
    with col1:
        st.subheader("Industry Distribution")
        industries = [c.get("industry", "unknown") for c in companies if c.get("industry")]
        if industries:
            industry_counts = dict(Counter(industries))
            st.bar_chart(industry_counts)
        else:
            st.caption("No industry data")

    # Tech stack frequency
    with col2:
        st.subheader("Top Technologies")
        tech_counter: Counter = Counter()
        for c in companies:
            tech = c.get("tech_stack") or {}
            if isinstance(tech, dict):
                for t in tech.keys():
                    tech_counter[t] += 1
        if tech_counter:
            top_tech = dict(tech_counter.most_common(15))
            st.bar_chart(top_tech)
        else:
            st.caption("No tech stack data")

    # Contacts section
    st.markdown("---")
    st.subheader("Contacts")

    company_names = {c["id"]: c.get("name", c.get("domain", "")) for c in companies}
    selected_company_name = st.selectbox(
        "Filter by company",
        ["All"] + list(company_names.values()),
    )

    if selected_company_name == "All":
        # Show all contacts (limited)
        contacts_data = []
        for c in companies[:20]:
            contacts = db.get_contacts_for_company(c["id"])
            for ct in contacts:
                contacts_data.append({
                    "Name": ct.get("name", ""),
                    "Title": ct.get("title", ""),
                    "Email": ct.get("email", ""),
                    "Company": c.get("name", ""),
                    "Seniority": ct.get("seniority", ""),
                })
        if contacts_data:
            st.dataframe(contacts_data, use_container_width=True)
        else:
            st.caption("No contacts found")
    else:
        # Find company ID by name
        target_id = None
        for cid, cname in company_names.items():
            if cname == selected_company_name:
                target_id = cid
                break
        if target_id:
            contacts = db.get_contacts_for_company(target_id)
            contacts_data = [{
                "Name": ct.get("name", ""),
                "Title": ct.get("title", ""),
                "Email": ct.get("email", ""),
                "Seniority": ct.get("seniority", ""),
            } for ct in contacts]
            if contacts_data:
                st.dataframe(contacts_data, use_container_width=True)
            else:
                st.caption("No contacts for this company")


# ===================================================================
# PAGE: Outreach Performance
# ===================================================================

def page_outreach():
    st.title("\U0001f4e7 Outreach Performance")

    # Time window selector
    days = st.selectbox("Time Window", [7, 30, 90], index=1)
    stats = _get_outreach_stats(vertical_id, days)

    if stats:
        # Metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Sent", stats.get("total_sent", 0))
        with c2:
            st.metric("Opened", stats.get("total_opened", 0))
        with c3:
            st.metric("Replied", stats.get("total_replied", 0))
        with c4:
            st.metric("Bounced", stats.get("total_bounced", 0))
        with c5:
            st.metric("Meetings", stats.get("total_meetings", 0))

        st.markdown("---")

        # Rate metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            open_rate = stats.get("open_rate", 0) or 0
            st.metric("Open Rate", f"{open_rate:.1%}")
        with c2:
            reply_rate = stats.get("reply_rate", 0) or 0
            st.metric("Reply Rate", f"{reply_rate:.1%}")
        with c3:
            bounce_rate = stats.get("bounce_rate", 0) or 0
            st.metric("Bounce Rate", f"{bounce_rate:.1%}")
        with c4:
            meeting_rate = stats.get("meeting_rate", 0) or 0
            st.metric("Meeting Rate", f"{meeting_rate:.1%}")
    else:
        st.info("No outreach data for this period.")

    st.markdown("---")

    # Recent outreach events
    st.subheader("Recent Outreach Events")
    events = _get_recent_outreach(vertical_id, 20)
    if events:
        table_data = []
        for e in events:
            contact = e.get("contacts") or {}
            status = e.get("status", "")
            status_emoji = {
                "sent": "\U0001f4e4",
                "recorded": "\U0001f4dd",
                "opened": "\U0001f440",
                "replied": "\U0001f4ac",
                "bounced": "\u274c",
                "send_failed": "\u26a0\ufe0f",
            }.get(status, "\u2753")
            table_data.append({
                "Status": f"{status_emoji} {status}",
                "Contact": contact.get("name", "N/A"),
                "Email": contact.get("email", "N/A"),
                "Subject": (e.get("subject") or "")[:50],
                "Approach": e.get("sequence_name", ""),
                "Sent": (e.get("sent_at") or "")[:19],
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.caption("No outreach events yet")


# ===================================================================
# PAGE: Knowledge Base
# ===================================================================

def page_knowledge():
    st.title("\U0001f9e0 Knowledge Base")

    knowledge_stats = _get_knowledge_stats(vertical_id)
    total_chunks = sum(knowledge_stats.values()) if knowledge_stats else 0

    st.metric("Total Knowledge Chunks", total_chunks)

    st.markdown("---")

    if knowledge_stats:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Chunks by Type")
            # Friendly names
            friendly = {
                "vulnerability_knowledge": "\U0001f6e1\ufe0f Vulnerability",
                "winning_pattern": "\U0001f3af Winning Pattern",
                "objection_handling": "\U0001f4ac Objection Handling",
                "company_intel": "\U0001f3e2 Company Intel",
                "outreach_result": "\U0001f4e7 Outreach Result",
            }
            chart_data = {}
            for chunk_type, count in knowledge_stats.items():
                label = friendly.get(chunk_type, chunk_type)
                chart_data[label] = count
            st.bar_chart(chart_data)

        with col2:
            st.subheader("Distribution")
            for chunk_type, count in sorted(knowledge_stats.items(), key=lambda x: x[1], reverse=True):
                label = friendly.get(chunk_type, chunk_type)
                pct = count / total_chunks * 100 if total_chunks > 0 else 0
                st.markdown(f"**{label}**: {count} ({pct:.0f}%)")
    else:
        st.info(
            "No knowledge chunks yet. Run `python main.py seed-knowledge enclave_guard` "
            "to load seed data."
        )


# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _get_outreach_stats(_vid: str, days: int) -> dict:
    _db = get_db(_vid)
    try:
        return _db.get_outreach_stats(days=days)
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _get_pipeline_runs(_vid: str, limit: int) -> list[dict]:
    _db = get_db(_vid)
    try:
        return _db.get_pipeline_runs(limit=limit)
    except Exception:
        return []


@st.cache_data(ttl=60)
def _get_knowledge_stats(_vid: str) -> dict[str, int]:
    _db = get_db(_vid)
    try:
        return _db.get_knowledge_stats()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _get_recent_outreach(_vid: str, limit: int) -> list[dict]:
    _db = get_db(_vid)
    try:
        return _db.get_recent_outreach(limit=limit)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if "\U0001f4ca Overview" in page:
    page_overview()
elif "\u2699\ufe0f Pipeline" in page:
    page_pipeline()
elif "\U0001f3e2 Companies" in page:
    page_companies()
elif "\U0001f4e7 Outreach" in page:
    page_outreach()
elif "\U0001f9e0 Knowledge" in page:
    page_knowledge()
