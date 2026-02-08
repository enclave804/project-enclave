"""
Sovereign Cockpit — Agent Command Center

Operational control panel for all agents. Provides:
- Agent status cards with health indicators
- Pause / Resume / Shadow Mode toggle controls
- Per-agent run history and log console
- Circuit breaker status and error tracking
- Performance analytics (success rate, avg duration, throughput)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


st.set_page_config(
    page_title="Agent Command — Sovereign Cockpit",
    page_icon="🤖",
    layout="wide",
)


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
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 Agent Command")

vertical_options = {"Enclave Guard": "enclave_guard"}
vertical_label = st.sidebar.selectbox("Vertical", list(vertical_options.keys()))
vertical_id = vertical_options[vertical_label]

st.sidebar.markdown("---")

# Emergency controls
st.sidebar.subheader("🚨 Emergency Controls")

if st.sidebar.button("⏸️ Pause ALL Agents", use_container_width=True):
    db = get_db(vertical_id)
    agents = _safe_call(lambda: db.list_agent_records(), [])
    for agent in agents:
        try:
            db.client.table("agents").update({"enabled": False}).eq(
                "agent_id", agent["agent_id"]
            ).eq("vertical_id", vertical_id).execute()
        except Exception:
            pass
    st.sidebar.error("All agents PAUSED.")
    st.rerun()

if st.sidebar.button("▶️ Resume ALL Agents", use_container_width=True):
    db = get_db(vertical_id)
    agents = _safe_call(lambda: db.list_agent_records(), [])
    for agent in agents:
        try:
            db.client.table("agents").update({"enabled": True}).eq(
                "agent_id", agent["agent_id"]
            ).eq("vertical_id", vertical_id).execute()
        except Exception:
            pass
    st.sidebar.success("All agents RESUMED.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Changes take effect on next agent run cycle.")


# ---------------------------------------------------------------------------
# Agent Grid
# ---------------------------------------------------------------------------

st.title("🤖 Agent Command Center")

db = get_db(vertical_id)
agents = _safe_call(lambda: db.list_agent_records(), [])

if not agents:
    st.info("No agents registered. Agents are registered when first discovered via YAML configs.")
    st.stop()

# Agent stats (batch fetch)
agent_stats = _safe_call(lambda: db.get_agent_stats(days=30), [])
stats_by_agent = {s["agent_id"]: s for s in (agent_stats or [])}


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

    # Errors from config
    consecutive_errors = config.get("consecutive_errors", 0)
    max_errors = config.get("max_consecutive_errors", 5)

    # Status indicators
    if not enabled:
        status_icon = "🔴"
        status_text = "PAUSED"
    elif shadow_mode:
        status_icon = "👻"
        status_text = f"SHADOW (of {shadow_of})"
    elif consecutive_errors >= max_errors:
        status_icon = "🟠"
        status_text = "CIRCUIT BREAKER TRIPPED"
    else:
        status_icon = "🟢"
        status_text = "ACTIVE"

    with st.expander(
        f"{status_icon} {agent_name} — {status_text}",
        expanded=not enabled or consecutive_errors > 0,
    ):
        # ─── Info Row ─────────────────────────────────────────
        info_cols = st.columns(5)
        with info_cols[0]:
            st.markdown(f"**Agent ID:** `{agent_id}`")
        with info_cols[1]:
            st.markdown(f"**Type:** `{agent_type}`")
        with info_cols[2]:
            st.markdown(f"**Shadow:** {'Yes' if shadow_mode else 'No'}")
        with info_cols[3]:
            st.markdown(f"**Errors:** {consecutive_errors}/{max_errors}")
        with info_cols[4]:
            pending = _safe_call(
                lambda aid=agent_id: db.count_pending_tasks(aid), 0
            )
            st.markdown(f"**Pending Tasks:** {pending}")

        # ─── Performance Stats ────────────────────────────────
        stats = stats_by_agent.get(agent_id, {})
        if stats:
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Total Runs", stats.get("total_runs", 0))
            with stat_cols[1]:
                success_rate = stats.get("success_rate")
                if success_rate is not None:
                    st.metric("Success Rate", f"{float(success_rate):.0%}")
            with stat_cols[2]:
                avg_dur = stats.get("avg_duration_ms")
                if avg_dur is not None:
                    st.metric("Avg Duration", f"{int(float(avg_dur))}ms")
            with stat_cols[3]:
                failed = stats.get("failed_runs", 0)
                st.metric("Failed Runs", failed)

        # ─── Control Buttons ─────────────────────────────────
        st.markdown("**Controls:**")
        ctrl_cols = st.columns(4)

        with ctrl_cols[0]:
            if enabled:
                if st.button("⏸️ Pause", key=f"pause_{agent_id}"):
                    try:
                        db.client.table("agents").update(
                            {"enabled": False}
                        ).eq("agent_id", agent_id).eq(
                            "vertical_id", vertical_id
                        ).execute()
                        st.warning(f"Paused {agent_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                if st.button("▶️ Resume", key=f"resume_{agent_id}"):
                    try:
                        db.client.table("agents").update(
                            {"enabled": True}
                        ).eq("agent_id", agent_id).eq(
                            "vertical_id", vertical_id
                        ).execute()
                        st.success(f"Resumed {agent_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        with ctrl_cols[1]:
            shadow_label = "🌞 Exit Shadow" if shadow_mode else "👻 Enter Shadow"
            if st.button(shadow_label, key=f"shadow_{agent_id}"):
                try:
                    db.client.table("agents").update(
                        {"shadow_mode": not shadow_mode}
                    ).eq("agent_id", agent_id).eq(
                        "vertical_id", vertical_id
                    ).execute()
                    mode = "exited" if shadow_mode else "entered"
                    st.info(f"Shadow mode {mode} for {agent_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        with ctrl_cols[2]:
            if consecutive_errors > 0:
                if st.button("🔄 Reset Errors", key=f"reset_{agent_id}"):
                    try:
                        db.reset_agent_errors(agent_id, vertical_id)
                        st.success(f"Error counter reset for {agent_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # ─── Recent Runs (Console) ───────────────────────────
        st.markdown("**Recent Runs (Last 10):**")

        runs = _safe_call(
            lambda aid=agent_id: db.get_agent_runs(agent_id=aid, limit=10), []
        )

        if runs:
            log_data = []
            for r in runs:
                run_status = r.get("status", "?")
                emoji = {"started": "🔄", "completed": "✅", "failed": "❌"}.get(
                    run_status, "❓"
                )
                log_data.append(
                    {
                        "Status": f"{emoji} {run_status}",
                        "Run ID": (r.get("run_id") or "")[:12] + "...",
                        "Duration": f"{r.get('duration_ms', 0) or 0}ms",
                        "Error": (r.get("error_message") or "")[:60],
                        "Time": (r.get("created_at") or "")[:19],
                    }
                )
            st.dataframe(log_data, use_container_width=True, hide_index=True)
        else:
            st.caption("No runs recorded yet.")

    st.markdown("")  # spacing
