"""
Sovereign Cockpit — Design System & Theme

Palantir-inspired dark UI theme for the Sovereign Venture Engine.
Provides CSS injection, color palette, and reusable UI components.

Design principles:
    1. Dark mode — OLED-friendly backgrounds, crisp text
    2. Information dense — max data per pixel
    3. Action-oriented — controls are prominent and obvious
    4. Status-driven — every element communicates health at a glance
"""

from __future__ import annotations

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {{
        background: {COLORS["bg_primary"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
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
        transition: all 0.2s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 20px {COLORS["accent_glow"]};
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
        transition: all 0.15s ease;
    }}

    .stButton > button:hover {{
        background: {COLORS["bg_hover"]};
        border-color: {COLORS["accent_primary"]};
        box-shadow: 0 0 12px {COLORS["accent_glow"]};
    }}

    .stButton > button[kind="primary"] {{
        background: {COLORS["accent_primary"]};
        border-color: {COLORS["accent_primary"]};
        color: white;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: {COLORS["accent_tertiary"]};
        box-shadow: 0 0 20px {COLORS["accent_glow"]};
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
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS["border_focus"]} !important;
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
        transition: all 0.2s ease;
    }}

    .sov-card:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 24px {COLORS["accent_glow"]};
    }}

    .sov-card-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
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

    .sov-feed-item {{
        padding: 12px 16px;
        border-left: 3px solid {COLORS["border_subtle"]};
        margin-bottom: 8px;
        background: {COLORS["bg_secondary"]};
        border-radius: 0 8px 8px 0;
        transition: all 0.15s ease;
    }}

    .sov-feed-item:hover {{
        background: {COLORS["bg_hover"]};
        border-left-color: {COLORS["accent_primary"]};
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
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }}

    .sov-agent-card:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 24px {COLORS["accent_glow"]};
    }}

    .sov-agent-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
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
        transition: all 0.15s ease;
    }}

    .sov-kanban-item:hover {{
        border-color: {COLORS["border_focus"]};
        box-shadow: 0 0 12px {COLORS["accent_glow"]};
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
        animation: slideIn 0.3s ease-out;
    }}

    @keyframes slideIn {{
        from {{ transform: translateX(100px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
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
