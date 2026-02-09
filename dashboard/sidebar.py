"""
Sovereign Cockpit — Navigation Sidebar

The nerve center sidebar provides:
- Vertical selector (multi-tenant switching)
- Navigation with page indicators
- Global health badge
- Emergency controls
- System version

Usage:
    from dashboard.sidebar import render_sidebar

    vertical_id = render_sidebar()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VERSION = "v1.0.0"

# Track session start time
import time as _time
_SESSION_START = _time.time()


def get_vertical_options() -> dict[str, str]:
    """
    Dynamically discover verticals with valid config.yaml files.

    Returns:
        Dict mapping display names to vertical IDs.
        Falls back to {"Enclave Guard": "enclave_guard"} if discovery fails.
    """
    try:
        from core.config.loader import list_available_verticals, load_vertical_config

        vertical_ids = list_available_verticals()
        if not vertical_ids:
            return {"Enclave Guard": "enclave_guard"}

        options = {}
        for vid in vertical_ids:
            try:
                cfg = load_vertical_config(vid)
                display_name = getattr(cfg, "vertical_name", vid.replace("_", " ").title())
                options[display_name] = vid
            except Exception:
                # Config exists but is invalid — still show it
                options[vid.replace("_", " ").title()] = vid

        return options if options else {"Enclave Guard": "enclave_guard"}

    except Exception:
        return {"Enclave Guard": "enclave_guard"}


def _get_health_summary(db: Any) -> dict[str, Any]:
    """Compute a quick health summary for the sidebar badge."""
    try:
        agents = db.list_agent_records() or []
        enabled = sum(1 for a in agents if a.get("enabled", True))
        total = len(agents)
        disabled = total - enabled

        # Check for circuit breaker trips
        tripped = 0
        for a in agents:
            cfg = a.get("config", {}) or {}
            errs = cfg.get("consecutive_errors", 0)
            max_e = cfg.get("max_consecutive_errors", 5)
            if errs >= max_e:
                tripped += 1

        shadow_count = sum(1 for a in agents if a.get("shadow_mode", False))

        # Quick task check
        try:
            failed_tasks = len(db.list_tasks(status="failed", limit=100) or [])
        except Exception:
            failed_tasks = 0

        if disabled > 0 or tripped > 0 or failed_tasks > 5:
            status = "degraded"
        else:
            status = "operational"

        return {
            "status": status,
            "agents_total": total,
            "agents_enabled": enabled,
            "agents_shadow": shadow_count,
            "agents_tripped": tripped,
            "failed_tasks": failed_tasks,
        }
    except Exception:
        return {"status": "unknown", "agents_total": 0, "agents_enabled": 0,
                "agents_shadow": 0, "agents_tripped": 0, "failed_tasks": 0}


def render_sidebar(
    show_version: bool = True,
    show_health: bool = True,
) -> str:
    """
    Render the Sovereign Cockpit sidebar.

    Returns:
        The selected vertical_id string.
    """
    from dashboard.theme import COLORS

    # ─── Logo / Brand ─────────────────────────────────────
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding: 8px 0 16px 0;">
            <div style="font-size: 1.6rem; font-weight: 700; color: {COLORS['text_primary']}; letter-spacing: -0.02em;">
                ◆ SOVEREIGN
            </div>
            <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.2em;
                         color: {COLORS['text_tertiary']}; margin-top: 2px;">
                Venture Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Vertical Selector ────────────────────────────────
    st.sidebar.markdown(
        f'<div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; '
        f'color: {COLORS["text_tertiary"]}; font-weight: 600; margin-bottom: 4px;">Active Vertical</div>',
        unsafe_allow_html=True,
    )

    vertical_options = get_vertical_options()
    vertical_label = st.sidebar.selectbox(
        "Vertical",
        list(vertical_options.keys()),
        label_visibility="collapsed",
    )
    vertical_id = vertical_options[vertical_label]

    # Store in session state for cross-page access
    st.session_state["active_vertical_id"] = vertical_id

    st.sidebar.markdown(
        f'<div style="height: 1px; background: {COLORS["border_subtle"]}; margin: 12px 0;"></div>',
        unsafe_allow_html=True,
    )

    # ─── Health Badge ─────────────────────────────────────
    if show_health:
        try:
            @st.cache_resource
            def _get_db(vid: str):
                from core.integrations.supabase_client import EnclaveDB
                return EnclaveDB(vid)

            db = _get_db(vertical_id)
            health = _get_health_summary(db)
        except Exception:
            health = {"status": "unknown", "agents_total": 0, "agents_enabled": 0,
                       "agents_shadow": 0, "agents_tripped": 0, "failed_tasks": 0}

        status = health["status"]
        if status == "operational":
            dot_color = COLORS["status_green"]
            label = "ALL SYSTEMS OPERATIONAL"
        elif status == "degraded":
            dot_color = COLORS["status_yellow"]
            label = "SYSTEM DEGRADED"
        else:
            dot_color = COLORS["status_gray"]
            label = "STATUS UNKNOWN"

        st.sidebar.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px;
                         background: rgba(0,0,0,0.2); border-radius: 8px; margin-bottom: 12px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: {dot_color};
                            box-shadow: 0 0 6px {dot_color};"></div>
                <span style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em;
                             color: {COLORS['text_secondary']}; font-weight: 600;">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick stats
        st.sidebar.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.72rem; margin-bottom: 8px;">
                <div style="color: {COLORS['text_tertiary']};">Agents</div>
                <div style="color: {COLORS['text_primary']}; text-align: right;">{health['agents_enabled']}/{health['agents_total']}</div>
                <div style="color: {COLORS['text_tertiary']};">Shadow</div>
                <div style="color: {COLORS['status_purple']}; text-align: right;">{health['agents_shadow']}</div>
                <div style="color: {COLORS['text_tertiary']};">Tripped</div>
                <div style="color: {COLORS['status_yellow'] if health['agents_tripped'] > 0 else COLORS['text_primary']}; text-align: right;">{health['agents_tripped']}</div>
                <div style="color: {COLORS['text_tertiary']};">Failed Tasks</div>
                <div style="color: {COLORS['status_red'] if health['failed_tasks'] > 5 else COLORS['text_primary']}; text-align: right;">{health['failed_tasks']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown(
            f'<div style="height: 1px; background: {COLORS["border_subtle"]}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )

    # ─── Version Footer + Session Timer ──────────────────
    if show_version:
        # Session duration
        elapsed = int(_time.time() - _SESSION_START)
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            session_str = f"{hrs}h {mins}m"
        elif mins > 0:
            session_str = f"{mins}m {secs}s"
        else:
            session_str = f"{secs}s"

        st.sidebar.markdown(
            f"""
            <div style="position: fixed; bottom: 16px; font-size: 0.65rem;
                         color: {COLORS['text_tertiary']};">
                <div>Sovereign Engine {VERSION}</div>
                <div style="margin-top: 2px; font-family: 'JetBrains Mono', monospace;">
                    Session: {session_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return vertical_id
