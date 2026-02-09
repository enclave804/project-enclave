"""
Sovereign Cockpit — Design System & Theme

Palantir-inspired dark UI theme for the Sovereign Venture Engine.
Provides CSS injection, color palette, and reusable UI components.

Design principles:
    1. Dark mode — OLED-friendly backgrounds, crisp text
    2. Information dense — max data per pixel
    3. Action-oriented — controls are prominent and obvious
    4. Status-driven — every element communicates health at a glance
    5. Glassmorphism depth — layered translucency for visual hierarchy
    6. Micro-interactions — skeleton loaders, hover transitions, glow effects
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

COLORS = {
    # Backgrounds
    "bg_primary": "#0A0E17",       # Near-black
    "bg_secondary": "#111827",     # Dark navy
    "bg_card": "#1A1F2E",          # Card surface
    "bg_elevated": "#232A3B",      # Elevated surface
    "bg_input": "#0F1420",         # Input fields
    "bg_hover": "#2A3148",         # Hover state

    # Accents
    "accent_primary": "#6366F1",   # Indigo
    "accent_secondary": "#818CF8", # Light indigo
    "accent_tertiary": "#4F46E5",  # Deep indigo
    "accent_glow": "rgba(99, 102, 241, 0.15)",

    # Status
    "status_green": "#10B981",     # Active / success
    "status_yellow": "#F59E0B",    # Warning / pending
    "status_red": "#EF4444",       # Error / critical
    "status_blue": "#3B82F6",      # Info / running
    "status_purple": "#8B5CF6",    # Shadow mode
    "status_gray": "#6B7280",      # Disabled / inactive

    # Text
    "text_primary": "#F1F5F9",     # Main text
    "text_secondary": "#94A3B8",   # Muted text
    "text_tertiary": "#64748B",    # Very muted
    "text_accent": "#A5B4FC",      # Accent text

    # Borders
    "border_subtle": "rgba(255, 255, 255, 0.06)",
    "border_default": "rgba(255, 255, 255, 0.1)",
    "border_focus": "rgba(99, 102, 241, 0.5)",
}


# ---------------------------------------------------------------------------
# Status Indicators
# ---------------------------------------------------------------------------

STATUS_CONFIG = {
    "active": {"icon": "●", "color": COLORS["status_green"], "label": "ACTIVE", "glow": True},
    "paused": {"icon": "●", "color": COLORS["status_red"], "label": "PAUSED", "glow": False},
    "shadow": {"icon": "◐", "color": COLORS["status_purple"], "label": "SHADOW", "glow": True},
    "circuit_breaker": {"icon": "⚡", "color": COLORS["status_yellow"], "label": "TRIPPED", "glow": True},
    "idle": {"icon": "○", "color": COLORS["status_gray"], "label": "IDLE", "glow": False},
    "running": {"icon": "◉", "color": COLORS["status_blue"], "label": "RUNNING", "glow": True},
    "completed": {"icon": "✓", "color": COLORS["status_green"], "label": "DONE", "glow": False},
    "failed": {"icon": "✕", "color": COLORS["status_red"], "label": "FAILED", "glow": False},
    "pending": {"icon": "◌", "color": COLORS["status_yellow"], "label": "PENDING", "glow": False},
}


# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------

def inject_theme_css() -> None:
    """Inject the Sovereign Cockpit CSS into the Streamlit page."""
    st.markdown(f"""
    <style>
    /* ─── Global ────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {{
        background: {COLORS["bg_primary"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* ─── Selection ─────────────────────────────────────────── */
    ::selection {{
        background: {COLORS["accent_primary"]};
        color: white;
    }}

    /* ─── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {COLORS["bg_secondary"]};
        border-right: 1px solid {COLORS["border_subtle"]};
    }}

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}

    /* ─── KPI Cards ─────────────────────────────────────────── */
    div[data-testid="stMetric"] {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 20px {COLORS["accent_glow"]};
        transform: translateY(-1px);
    }}

    div[data-testid="stMetric"] label {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }}

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 700;
        font-size: 1.8rem;
    }}

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: 0.75rem;
    }}

    /* ─── Buttons ───────────────────────────────────────────── */
    .stButton > button {{
        border: 1px solid {COLORS["border_default"]};
        border-radius: 8px;
        background: {COLORS["bg_card"]};
        color: {COLORS["text_primary"]};
        font-weight: 500;
        font-size: 0.82rem;
        padding: 0.4rem 1rem;
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }}

    .stButton > button:hover {{
        background: {COLORS["bg_hover"]};
        border-color: {COLORS["accent_primary"]};
        box-shadow: 0 0 12px {COLORS["accent_glow"]};
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS["accent_primary"]}, {COLORS["accent_tertiary"]});
        border-color: {COLORS["accent_primary"]};
        color: white;
    }}

    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 0 24px {COLORS["accent_glow"]};
    }}

    /* ─── Dataframes / Tables ───────────────────────────────── */
    .stDataFrame {{
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ─── Expanders ─────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border_subtle"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text_primary"]} !important;
        font-weight: 500;
        transition: all 0.15s ease;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS["border_focus"]} !important;
        box-shadow: 0 0 12px {COLORS["accent_glow"]};
    }}

    /* ─── Tabs ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background: {COLORS["bg_secondary"]};
        border-radius: 10px;
        padding: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {COLORS["text_secondary"]};
        font-weight: 500;
        font-size: 0.82rem;
        transition: all 0.15s ease;
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {COLORS["bg_card"]};
        color: {COLORS["text_primary"]};
    }}

    /* ─── Dividers ──────────────────────────────────────────── */
    hr {{
        border-color: {COLORS["border_subtle"]} !important;
        margin: 1rem 0 !important;
    }}

    /* ─── Text Areas & Inputs ───────────────────────────────── */
    .stTextInput input, .stTextArea textarea {{
        background: {COLORS["bg_input"]} !important;
        border: 1px solid {COLORS["border_default"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text_primary"]} !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.15s ease;
    }}

    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: {COLORS["accent_primary"]} !important;
        box-shadow: 0 0 12px {COLORS["accent_glow"]} !important;
    }}

    /* ─── Selectbox ─────────────────────────────────────────── */
    .stSelectbox [data-baseweb="select"] {{
        background: {COLORS["bg_input"]};
        border-radius: 8px;
    }}

    /* ─── Code Blocks ───────────────────────────────────────── */
    .stCodeBlock {{
        border-radius: 8px !important;
        border: 1px solid {COLORS["border_subtle"]} !important;
    }}

    code {{
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.82rem;
    }}

    /* ─── Custom Components ─────────────────────────────────── */
    .sov-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
    }}

    .sov-card:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 24px {COLORS["accent_glow"]};
        transform: translateY(-1px);
    }}

    .sov-card-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }}

    /* ─── Glassmorphism Card (elevated variant) ─────────────── */
    .sov-glass {{
        background: rgba(26, 31, 46, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .sov-glass:hover {{
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
    }}

    .sov-status-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }}

    .sov-status-dot.glow {{
        box-shadow: 0 0 8px currentColor;
        animation: dotPulse 2s ease-in-out infinite;
    }}

    @keyframes dotPulse {{
        0%, 100% {{ box-shadow: 0 0 4px currentColor; }}
        50% {{ box-shadow: 0 0 12px currentColor, 0 0 24px currentColor; }}
    }}

    .sov-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }}

    .sov-kpi-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 12px;
        padding: 16px 20px;
        text-align: left;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .sov-kpi-card:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 20px {COLORS["accent_glow"]};
        transform: translateY(-1px);
    }}

    .sov-kpi-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS["text_secondary"]};
        font-weight: 600;
        margin-bottom: 4px;
    }}

    .sov-kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        line-height: 1.2;
    }}

    .sov-kpi-delta {{
        font-size: 0.75rem;
        margin-top: 2px;
    }}

    /* ─── Sparkline in KPI ──────────────────────────────────── */
    .sov-sparkline {{
        position: absolute;
        bottom: 0;
        right: 0;
        width: 60%;
        height: 40px;
        opacity: 0.15;
    }}

    .sov-feed-item {{
        padding: 12px 16px;
        border-left: 3px solid {COLORS["border_subtle"]};
        margin-bottom: 8px;
        background: {COLORS["bg_secondary"]};
        border-radius: 0 8px 8px 0;
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .sov-feed-item:hover {{
        background: {COLORS["bg_hover"]};
        border-left-color: {COLORS["accent_primary"]};
        transform: translateX(2px);
    }}

    .sov-feed-time {{
        font-size: 0.7rem;
        color: {COLORS["text_tertiary"]};
        font-family: 'JetBrains Mono', monospace;
    }}

    .sov-feed-agent {{
        font-size: 0.75rem;
        color: {COLORS["text_accent"]};
        font-weight: 600;
    }}

    .sov-feed-text {{
        font-size: 0.82rem;
        color: {COLORS["text_primary"]};
        margin-top: 2px;
    }}

    .sov-section-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid {COLORS["border_subtle"]};
    }}

    .sov-section-title {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS["text_secondary"]};
        font-weight: 600;
    }}

    .sov-badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 10px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        transition: all 0.15s ease;
    }}

    .sov-badge-green {{ background: rgba(16, 185, 129, 0.15); color: {COLORS["status_green"]}; }}
    .sov-badge-yellow {{ background: rgba(245, 158, 11, 0.15); color: {COLORS["status_yellow"]}; }}
    .sov-badge-red {{ background: rgba(239, 68, 68, 0.15); color: {COLORS["status_red"]}; }}
    .sov-badge-blue {{ background: rgba(59, 130, 246, 0.15); color: {COLORS["status_blue"]}; }}
    .sov-badge-purple {{ background: rgba(139, 92, 246, 0.15); color: {COLORS["status_purple"]}; }}
    .sov-badge-gray {{ background: rgba(107, 114, 128, 0.15); color: {COLORS["status_gray"]}; }}

    /* ─── Scrollbar ──────────────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS["bg_primary"]};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS["bg_hover"]};
        border-radius: 3px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS["text_tertiary"]};
    }}

    /* ─── Page Title Styling ─────────────────────────────────── */
    .sov-page-header {{
        margin-bottom: 24px;
    }}

    .sov-page-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        margin: 0;
        padding: 0;
        letter-spacing: -0.02em;
    }}

    .sov-page-subtitle {{
        font-size: 0.82rem;
        color: {COLORS["text_tertiary"]};
        margin-top: 4px;
    }}

    /* ─── Agent Cards ────────────────────────────────────────── */
    .sov-agent-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .sov-agent-card:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 24px {COLORS["accent_glow"]};
        transform: translateY(-1px);
    }}

    .sov-agent-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        transition: width 0.2s ease;
    }}

    .sov-agent-card:hover::before {{
        width: 6px;
    }}

    .sov-agent-card.active::before {{ background: {COLORS["status_green"]}; }}
    .sov-agent-card.paused::before {{ background: {COLORS["status_red"]}; }}
    .sov-agent-card.shadow::before {{ background: {COLORS["status_purple"]}; }}
    .sov-agent-card.tripped::before {{ background: {COLORS["status_yellow"]}; }}

    /* ─── Kanban Board ───────────────────────────────────────── */
    .sov-kanban-col {{
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 12px;
        padding: 16px;
        min-height: 200px;
    }}

    .sov-kanban-header {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS["text_secondary"]};
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .sov-kanban-count {{
        background: {COLORS["bg_hover"]};
        padding: 2px 8px;
        border-radius: 100px;
        font-size: 0.7rem;
    }}

    .sov-kanban-item {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .sov-kanban-item:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 12px {COLORS["accent_glow"]};
        transform: translateY(-1px);
    }}

    /* ─── Toast / Notifications ──────────────────────────────── */
    .sov-toast {{
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 9999;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS["border_subtle"]};
    }}

    .sov-toast-success {{ background: rgba(16, 185, 129, 0.9); color: white; }}
    .sov-toast-error {{ background: rgba(239, 68, 68, 0.9); color: white; }}
    .sov-toast-info {{ background: rgba(99, 102, 241, 0.9); color: white; }}

    @keyframes slideIn {{
        from {{ transform: translateX(100px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}

    @keyframes fadeOut {{
        from {{ opacity: 1; }}
        to {{ opacity: 0; transform: translateY(-10px); }}
    }}

    /* ─── Pulse animation for active status ──────────────────── */
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
        100% {{ opacity: 1; }}
    }}

    .sov-pulse {{
        animation: pulse 2s infinite;
    }}

    /* ─── Skeleton Loader ───────────────────────────────────── */
    .sov-skeleton {{
        background: linear-gradient(
            90deg,
            {COLORS["bg_card"]} 25%,
            {COLORS["bg_elevated"]} 50%,
            {COLORS["bg_card"]} 75%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s ease-in-out infinite;
        border-radius: 8px;
    }}

    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    /* ─── Progress Bar ──────────────────────────────────────── */
    .sov-progress {{
        height: 4px;
        background: {COLORS["bg_secondary"]};
        border-radius: 2px;
        overflow: hidden;
        margin: 8px 0;
    }}

    .sov-progress-bar {{
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* ─── Tooltip ────────────────────────────────────────────── */
    .sov-tooltip {{
        position: relative;
        cursor: help;
    }}

    .sov-tooltip::after {{
        content: attr(data-tooltip);
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        padding: 6px 12px;
        background: {COLORS["bg_elevated"]};
        border: 1px solid {COLORS["border_default"]};
        border-radius: 6px;
        font-size: 0.72rem;
        color: {COLORS["text_primary"]};
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease;
        z-index: 1000;
    }}

    .sov-tooltip:hover::after {{
        opacity: 1;
    }}

    /* ─── Quick Actions Bar ─────────────────────────────────── */
    .sov-quick-actions {{
        display: flex;
        gap: 8px;
        padding: 8px 0;
        margin-bottom: 16px;
        overflow-x: auto;
    }}

    .sov-quick-action {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border_subtle"]};
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
        color: {COLORS["text_secondary"]};
        cursor: pointer;
        transition: all 0.15s ease;
        white-space: nowrap;
    }}

    .sov-quick-action:hover {{
        background: {COLORS["bg_hover"]};
        border-color: {COLORS["accent_primary"]};
        color: {COLORS["text_primary"]};
    }}

    /* ─── Last Updated Timestamp ────────────────────────────── */
    .sov-timestamp {{
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        color: {COLORS["text_tertiary"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* ─── Stat Grid (compact metrics) ───────────────────────── */
    .sov-stat-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 8px;
    }}

    .sov-stat {{
        text-align: center;
        padding: 12px 8px;
        background: {COLORS["bg_secondary"]};
        border-radius: 8px;
        border: 1px solid {COLORS["border_subtle"]};
    }}

    .sov-stat-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        line-height: 1;
    }}

    .sov-stat-label {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLORS["text_tertiary"]};
        margin-top: 4px;
    }}

    /* ─── Hotkey Hint ───────────────────────────────────────── */
    .sov-hotkey {{
        display: inline-flex;
        align-items: center;
        padding: 1px 6px;
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["border_default"]};
        border-radius: 4px;
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        color: {COLORS["text_tertiary"]};
        line-height: 1.4;
    }}

    /* ─── Breadcrumb / Hierarchy ─────────────────────────────── */
    .sov-breadcrumb {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem;
        color: {COLORS["text_tertiary"]};
        margin-bottom: 16px;
    }}

    .sov-breadcrumb-sep {{
        color: {COLORS["text_tertiary"]};
        opacity: 0.5;
    }}

    .sov-breadcrumb-active {{
        color: {COLORS["text_primary"]};
        font-weight: 500;
    }}

    /* ─── Empty State ───────────────────────────────────────── */
    .sov-empty {{
        text-align: center;
        padding: 48px 24px;
        color: {COLORS["text_tertiary"]};
    }}

    .sov-empty-icon {{
        font-size: 2.5rem;
        margin-bottom: 12px;
        opacity: 0.5;
    }}

    .sov-empty-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {COLORS["text_secondary"]};
        margin-bottom: 8px;
    }}

    .sov-empty-desc {{
        font-size: 0.78rem;
        max-width: 360px;
        margin: 0 auto;
        line-height: 1.5;
    }}

    /* ─── Divider with label ─────────────────────────────────── */
    .sov-divider {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 20px 0;
    }}

    .sov-divider-line {{
        flex: 1;
        height: 1px;
        background: {COLORS["border_subtle"]};
    }}

    .sov-divider-label {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLORS["text_tertiary"]};
        font-weight: 600;
        white-space: nowrap;
    }}

    /* ─── Fade-in animation ─────────────────────────────────── */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .sov-fade-in {{
        animation: fadeIn 0.3s ease-out;
    }}

    /* Hide Streamlit elements for cleaner look ───────────────── */
    #MainMenu {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable Components
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page header."""
    html = f'<div class="sov-page-header"><h1 class="sov-page-title">{title}</h1>'
    if subtitle:
        html += f'<p class="sov-page-subtitle">{subtitle}</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, badge: str = "") -> None:
    """Render a section header with optional badge."""
    html = f"""
    <div class="sov-section-header">
        <span class="sov-section-title">{title}</span>
        {f'<span class="sov-badge sov-badge-gray">{badge}</span>' if badge else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    config = STATUS_CONFIG.get(status, STATUS_CONFIG["idle"])
    badge_class = {
        COLORS["status_green"]: "sov-badge-green",
        COLORS["status_yellow"]: "sov-badge-yellow",
        COLORS["status_red"]: "sov-badge-red",
        COLORS["status_blue"]: "sov-badge-blue",
        COLORS["status_purple"]: "sov-badge-purple",
        COLORS["status_gray"]: "sov-badge-gray",
    }.get(config["color"], "sov-badge-gray")

    return f'<span class="sov-badge {badge_class}">{config["icon"]} {config["label"]}</span>'


def status_dot(status: str, size: int = 10) -> str:
    """Return HTML for a status indicator dot."""
    config = STATUS_CONFIG.get(status, STATUS_CONFIG["idle"])
    glow = "glow" if config.get("glow") else ""
    return (
        f'<span class="sov-status-dot {glow}" '
        f'style="background:{config["color"]}; color:{config["color"]}; '
        f'width:{size}px; height:{size}px;"></span>'
    )


def feed_item(time_str: str, agent_name: str, text: str, status: str = "active") -> str:
    """Return HTML for a live feed item."""
    config = STATUS_CONFIG.get(status, STATUS_CONFIG["idle"])
    return f"""
    <div class="sov-feed-item" style="border-left-color: {config['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="sov-feed-agent">{agent_name}</span>
            <span class="sov-feed-time">{time_str}</span>
        </div>
        <div class="sov-feed-text">{text}</div>
    </div>
    """


def kpi_card(label: str, value: str, delta: str = "", delta_color: str = "") -> str:
    """Return HTML for a custom KPI card."""
    delta_html = ""
    if delta:
        color = delta_color or COLORS["status_green"]
        delta_html = f'<div class="sov-kpi-delta" style="color: {color};">{delta}</div>'

    return f"""
    <div class="sov-kpi-card">
        <div class="sov-kpi-label">{label}</div>
        <div class="sov-kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def render_health_indicator(
    operational: int,
    degraded: int,
    failed: int,
) -> str:
    """
    Render a compact health bar showing system status.

    Returns HTML string.
    """
    total = max(operational + degraded + failed, 1)
    op_pct = (operational / total) * 100
    dg_pct = (degraded / total) * 100
    fl_pct = (failed / total) * 100

    return f"""
    <div style="display: flex; gap: 2px; height: 6px; border-radius: 3px; overflow: hidden; margin: 8px 0;">
        <div style="width: {op_pct}%; background: {COLORS['status_green']};"></div>
        <div style="width: {dg_pct}%; background: {COLORS['status_yellow']};"></div>
        <div style="width: {fl_pct}%; background: {COLORS['status_red']};"></div>
    </div>
    """


def sparkline_svg(values: list[int | float], color: str = "", width: int = 120, height: int = 32) -> str:
    """
    Generate an inline SVG sparkline from a list of values.

    Returns HTML string with an SVG element.
    """
    if not values or len(values) < 2:
        return ""

    color = color or COLORS["accent_primary"]
    n = len(values)
    v_min = min(values)
    v_max = max(values)
    v_range = max(v_max - v_min, 1)

    # Build polyline points
    points = []
    for i, v in enumerate(values):
        x = (i / (n - 1)) * width
        y = height - ((v - v_min) / v_range) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    # Area fill points (add bottom corners)
    area_points = polyline + f" {width:.1f},{height} 0,{height}"

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
         style="display:block;" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.3"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0.02"/>
            </linearGradient>
        </defs>
        <polygon points="{area_points}" fill="url(#sparkFill)"/>
        <polyline points="{polyline}" fill="none" stroke="{color}"
                  stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """


def render_skeleton(height: int = 16, width: str = "100%", count: int = 1) -> str:
    """Render placeholder skeleton loading animation. Returns HTML string."""
    lines = []
    for _ in range(count):
        lines.append(
            f'<div class="sov-skeleton" '
            f'style="height:{height}px; width:{width}; margin-bottom:8px;"></div>'
        )
    return "\n".join(lines)


def render_empty_state(icon: str, title: str, description: str = "", action_hint: str = "") -> str:
    """Render a styled empty state placeholder. Returns HTML string."""
    html = f"""
    <div class="sov-empty">
        <div class="sov-empty-icon">{icon}</div>
        <div class="sov-empty-title">{title}</div>
        <div class="sov-empty-desc">{description}</div>
    """
    if action_hint:
        html += f"""
        <div style="margin-top:12px; font-size:0.72rem; color:{COLORS['text_tertiary']};">
            <code style="font-size:0.72rem;">{action_hint}</code>
        </div>
        """
    html += "</div>"
    return html


def render_progress_bar(value: float, max_value: float = 100, color: str = "") -> str:
    """
    Render a thin progress bar. Returns HTML string.

    Args:
        value: Current value.
        max_value: Maximum value (default 100).
        color: Bar color (defaults to accent_primary).
    """
    color = color or COLORS["accent_primary"]
    pct = min((value / max(max_value, 1)) * 100, 100)
    return f"""
    <div class="sov-progress">
        <div class="sov-progress-bar" style="width:{pct:.1f}%; background:{color};"></div>
    </div>
    """


def render_breadcrumb(items: list[str]) -> str:
    """Render a breadcrumb navigation trail. Returns HTML string."""
    parts = []
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        cls = "sov-breadcrumb-active" if is_last else ""
        parts.append(f'<span class="{cls}">{item}</span>')
        if not is_last:
            parts.append('<span class="sov-breadcrumb-sep">›</span>')

    return f'<div class="sov-breadcrumb">{"".join(parts)}</div>'


def render_stat_grid(stats: list[tuple[str, str, str]]) -> str:
    """
    Render a compact stat grid. Returns HTML string.

    Args:
        stats: List of (value, label, color) tuples.
    """
    items = []
    for value, label, color in stats:
        items.append(f"""
        <div class="sov-stat">
            <div class="sov-stat-value" style="color:{color or COLORS['text_primary']};">{value}</div>
            <div class="sov-stat-label">{label}</div>
        </div>
        """)
    return f'<div class="sov-stat-grid">{"".join(items)}</div>'


def render_divider(label: str = "") -> str:
    """Render a horizontal divider with optional label. Returns HTML string."""
    if label:
        return f"""
        <div class="sov-divider">
            <div class="sov-divider-line"></div>
            <span class="sov-divider-label">{label}</span>
            <div class="sov-divider-line"></div>
        </div>
        """
    return f'<div style="height:1px; background:{COLORS["border_subtle"]}; margin:20px 0;"></div>'


def render_timestamp(label: str = "Last updated") -> str:
    """Render a last-updated timestamp. Returns HTML string."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    return f"""
    <div class="sov-timestamp">
        {label} · {now}
    </div>
    """


def render_toast(message: str, variant: str = "info") -> None:
    """Display a toast notification. Renders directly via st.markdown."""
    toast_class = f"sov-toast-{variant}"
    st.markdown(
        f"""
        <div class="sov-toast {toast_class}" style="animation: slideIn 0.3s ease-out, fadeOut 0.3s ease-in 3s forwards;">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
