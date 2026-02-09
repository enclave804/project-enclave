"""
Shared sidebar configuration for the Sovereign Cockpit.

Provides dynamic vertical discovery so new verticals appear in the
dropdown automatically — no hardcoded dict needed.

Usage:
    from dashboard.sidebar import render_sidebar

    vertical_id = render_sidebar(title="🛡️ Sovereign Cockpit")
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def render_sidebar(
    title: str = "🛡️ Sovereign Cockpit",
    show_version: bool = True,
) -> str:
    """
    Render the shared sidebar and return the selected vertical_id.

    Args:
        title: Sidebar title text.
        show_version: Whether to show version footer.

    Returns:
        The selected vertical_id string.
    """
    st.sidebar.title(title)

    vertical_options = get_vertical_options()
    vertical_label = st.sidebar.selectbox("Vertical", list(vertical_options.keys()))
    vertical_id = vertical_options[vertical_label]

    st.sidebar.markdown("---")

    if show_version:
        st.sidebar.caption("Sovereign Venture Engine v0.3.0")

    return vertical_id
