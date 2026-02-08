"""
Sovereign Cockpit — Mission Control Dashboard

The command center for the Sovereign Venture Engine. Provides:
- System health overview with real-time KPIs
- Agent status monitoring
- Task queue depth
- Pipeline value estimation

Run with: streamlit run dashboard/app.py
Or:       ./dashboard/run.sh
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
    page_title="Sovereign Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Authentication Gate (must be AFTER set_page_config, BEFORE any content)
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ Sovereign Cockpit")

vertical_options = {"Enclave Guard": "enclave_guard"}
vertical_label = st.sidebar.selectbox("Vertical", list(vertical_options.keys()))
vertical_id = vertical_options[vertical_label]

st.sidebar.markdown("---")
st.sidebar.caption("Sovereign Venture Engine v0.2.0")


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
# System Status
# ---------------------------------------------------------------------------

st.title("🎯 Mission Control")

# Global system status indicator
agents_data = _safe_call(lambda: get_db(vertical_id).list_agent_records(), [])
enabled_agents = [a for a in agents_data if a.get("enabled", True)]
disabled_agents = [a for a in agents_data if not a.get("enabled", True)]

# Check for critical issues
pending_tasks = _safe_call(
    lambda: len(get_db(vertical_id).list_tasks(status="pending", limit=1000)), 0
)
failed_tasks = _safe_call(
    lambda: len(get_db(vertical_id).list_tasks(status="failed", limit=100)), 0
)

has_critical = len(disabled_agents) > 0 or failed_tasks > 5

status_col, detail_col = st.columns([1, 3])

with status_col:
    if has_critical:
        st.error("⚠️ SYSTEM STATUS: DEGRADED")
    else:
        st.success("✅ SYSTEM STATUS: OPERATIONAL")

with detail_col:
    if disabled_agents:
        names = ", ".join(a.get("name", a.get("agent_id", "?")) for a in disabled_agents)
        st.warning(f"Disabled agents: {names}")
    if failed_tasks > 5:
        st.warning(f"{failed_tasks} failed tasks in queue — review recommended")

st.markdown("---")


# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    leads = _safe_call(lambda: get_db(vertical_id).count_contacts(), 0)
    st.metric("Total Leads", leads)

with col2:
    st.metric("Active Agents", len(enabled_agents))

with col3:
    st.metric("Tasks Pending", pending_tasks)

with col4:
    # Estimated pipeline value: opportunities * avg ticket
    opps = _safe_call(lambda: get_db(vertical_id).count_opportunities(), 0)
    avg_ticket = 6000  # mid-range for enclave_guard ($5k-$7k)
    pipeline_value = opps * avg_ticket
    st.metric("Est. Pipeline Value", f"${pipeline_value:,.0f}")

st.markdown("---")


# ---------------------------------------------------------------------------
# Agent Overview Grid
# ---------------------------------------------------------------------------

st.subheader("🤖 Agent Fleet")

if agents_data:
    agent_cols = st.columns(min(len(agents_data), 4))
    for i, agent in enumerate(agents_data):
        col = agent_cols[i % len(agent_cols)]
        with col:
            enabled = agent.get("enabled", True)
            shadow = agent.get("shadow_mode", False)
            status_icon = "🟢" if enabled else "🔴"
            shadow_badge = " 👻" if shadow else ""

            st.markdown(
                f"### {status_icon} {agent.get('name', agent.get('agent_id', '?'))}{shadow_badge}"
            )
            st.caption(f"Type: `{agent.get('agent_type', '?')}`")

            # Agent-specific pending tasks
            agent_id = agent.get("agent_id", "")
            agent_pending = _safe_call(
                lambda aid=agent_id: get_db(vertical_id).count_pending_tasks(aid), 0
            )
            st.caption(f"Pending tasks: {agent_pending}")
else:
    st.info(
        "No agents registered yet. Run `python main.py agent list enclave_guard` "
        "to discover agents."
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Recent Agent Runs
# ---------------------------------------------------------------------------

st.subheader("📜 Recent Agent Activity")

runs = _safe_call(lambda: get_db(vertical_id).get_agent_runs(limit=20), [])

if runs:
    table_data = []
    for r in runs:
        status = r.get("status", "?")
        status_emoji = {"started": "🔄", "completed": "✅", "failed": "❌"}.get(
            status, "❓"
        )
        table_data.append(
            {
                "Status": f"{status_emoji} {status}",
                "Agent": r.get("agent_id", ""),
                "Type": r.get("agent_type", ""),
                "Duration": f"{r.get('duration_ms', 0) or 0}ms",
                "Error": (r.get("error_message") or "")[:80],
                "Time": (r.get("created_at") or "")[:19],
            }
        )
    st.dataframe(table_data, use_container_width=True)
else:
    st.caption("No agent runs recorded yet.")


# ---------------------------------------------------------------------------
# Training Data / RLHF Stats
# ---------------------------------------------------------------------------

st.subheader("🧠 Training Data (RLHF Flywheel)")

training_stats = _safe_call(
    lambda: get_db(vertical_id).get_training_stats(vertical_id=vertical_id), []
)

if training_stats:
    train_cols = st.columns(min(len(training_stats), 4))
    for i, stat in enumerate(training_stats):
        col = train_cols[i % len(train_cols)]
        with col:
            st.metric(
                f"{stat.get('agent_id', '?')} Examples",
                stat.get("total_examples", 0),
            )
            avg = stat.get("avg_score")
            if avg is not None:
                st.caption(f"Avg score: {avg}/100")
            corrected = stat.get("corrected_examples", 0)
            if corrected:
                st.caption(f"Human corrections: {corrected}")
else:
    st.caption("No training examples collected yet. Approve/edit content to start the flywheel.")
