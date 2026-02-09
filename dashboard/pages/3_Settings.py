"""
Sovereign Cockpit — Settings & Configuration

Provides:
- YAML config viewer/editor for vertical configuration
- Agent YAML config viewer for each registered agent
- Config validation via Pydantic schemas
- Credential status display
- System information
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


st.set_page_config(
    page_title="Settings — Sovereign Cockpit",
    page_icon="◆",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Auth + Theme + Sidebar
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()

from dashboard.theme import (
    COLORS, inject_theme_css, page_header, section_header,
)

inject_theme_css()

from dashboard.sidebar import render_sidebar

vertical_id = render_sidebar()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

page_header(
    "Settings",
    f"Configuration for {vertical_id.replace('_', ' ').title()}",
)


# ---------------------------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------------------------

config_tab, agents_tab, creds_tab, system_tab = st.tabs([
    "⚙ Vertical Config",
    "🤖 Agent Configs",
    "🔑 Credentials",
    "📊 System Info",
])


# ---------------------------------------------------------------------------
# Tab 1: Vertical Config
# ---------------------------------------------------------------------------

with config_tab:
    section_header("VERTICAL CONFIGURATION", vertical_id)

    config_path = PROJECT_ROOT / "verticals" / vertical_id / "config.yaml"

    if config_path.exists():
        raw_yaml = config_path.read_text()

        # Display YAML
        st.code(raw_yaml, language="yaml", line_numbers=True)

        # Validate button
        if st.button("✓ Validate Config", key="validate_config", use_container_width=False):
            try:
                from core.config.loader import load_vertical_config
                cfg = load_vertical_config(vertical_id)
                st.success(
                    f"Config is valid. "
                    f"Vertical: {getattr(cfg, 'vertical_name', vertical_id)} | "
                    f"Personas: {len(getattr(getattr(cfg, 'targeting', None), 'personas', []))} | "
                    f"Channels: {len(getattr(getattr(cfg, 'outreach', None), 'channels', []))}"
                )
            except Exception as e:
                st.error(f"Validation failed: {e}")

        # Config path info
        st.markdown(
            f'<div style="font-size: 0.7rem; color: {COLORS["text_tertiary"]}; margin-top: 12px;">'
            f'Path: <code>{config_path}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="text-align: center; padding: 40px; color: {COLORS["text_tertiary"]};">'
            f'No config.yaml found at <code>{config_path}</code></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 2: Agent Configs
# ---------------------------------------------------------------------------

with agents_tab:
    section_header("AGENT CONFIGURATIONS", vertical_id)

    agents_dir = PROJECT_ROOT / "verticals" / vertical_id / "agents"

    if agents_dir.exists():
        yaml_files = sorted(agents_dir.glob("*.yaml"))

        if yaml_files:
            for yaml_file in yaml_files:
                agent_name = yaml_file.stem
                raw_yaml = yaml_file.read_text()

                with st.expander(f"🤖 {agent_name}.yaml", expanded=False):
                    st.code(raw_yaml, language="yaml", line_numbers=True)

                    if st.button(f"Validate {agent_name}", key=f"val_{agent_name}"):
                        try:
                            import yaml
                            from core.config.agent_schema import AgentInstanceConfig
                            data = yaml.safe_load(raw_yaml)
                            config = AgentInstanceConfig(**data)
                            st.success(
                                f"Valid. Type: {config.agent_type} | "
                                f"Tools: {len(config.tools)} | "
                                f"Enabled: {config.enabled}"
                            )
                        except Exception as e:
                            st.error(f"Validation failed: {e}")

                    st.markdown(
                        f'<div style="font-size: 0.7rem; color: {COLORS["text_tertiary"]}; margin-top: 4px;">'
                        f'Path: <code>{yaml_file}</code></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f'<div style="text-align: center; padding: 40px; color: {COLORS["text_tertiary"]};">'
                f'No agent YAML files found in <code>{agents_dir}</code></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="text-align: center; padding: 40px; color: {COLORS["text_tertiary"]};">'
            f'No agents directory found at <code>{agents_dir}</code></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: Credentials
# ---------------------------------------------------------------------------

with creds_tab:
    section_header("API CREDENTIALS", "status")

    # Define all credentials to check
    credential_groups = {
        "Platform (Required)": [
            ("ANTHROPIC_API_KEY", "Anthropic Claude API"),
            ("OPENAI_API_KEY", "OpenAI (Embeddings)"),
            ("SUPABASE_URL", "Supabase URL"),
            ("SUPABASE_KEY", "Supabase API Key"),
        ],
        "Outreach": [
            ("APOLLO_API_KEY", "Apollo (Lead Data)"),
            ("N8N_WEBHOOK_URL", "n8n Webhooks"),
            ("SENDGRID_API_KEY", "SendGrid (Email)"),
        ],
        "Voice & Phone": [
            ("TWILIO_ACCOUNT_SID", "Twilio Account SID"),
            ("TWILIO_AUTH_TOKEN", "Twilio Auth Token"),
            ("TWILIO_PHONE_NUMBER", "Twilio Phone Number"),
            ("ELEVENLABS_API_KEY", "ElevenLabs (TTS)"),
        ],
        "Commerce": [
            ("STRIPE_API_KEY", "Stripe"),
            ("SHOPIFY_API_KEY", "Shopify"),
        ],
        "Monitoring": [
            ("LANGFUSE_HOST", "LangFuse Host"),
            ("LANGFUSE_PUBLIC_KEY", "LangFuse Public Key"),
            ("LANGFUSE_SECRET_KEY", "LangFuse Secret Key"),
            ("TELEGRAM_BOT_TOKEN", "Telegram Notifications"),
        ],
        "Dashboard": [
            ("DASHBOARD_PASSWORD", "Dashboard Password"),
        ],
    }

    total_set = 0
    total_creds = 0

    for group_name, creds in credential_groups.items():
        st.markdown(
            f'<div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; '
            f'color: {COLORS["text_accent"]}; font-weight: 600; margin: 16px 0 8px;">{group_name}</div>',
            unsafe_allow_html=True,
        )

        for env_var, label in creds:
            total_creds += 1
            is_set = bool(os.environ.get(env_var, ""))
            if is_set:
                total_set += 1

            # Try st.secrets too
            if not is_set:
                try:
                    is_set = bool(st.secrets.get(env_var))
                    if is_set:
                        total_set += 1
                except Exception:
                    pass

            dot_color = COLORS["status_green"] if is_set else COLORS["status_red"]
            status_text = "configured" if is_set else "missing"

            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;
                             border-bottom: 1px solid {COLORS['border_subtle']};">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: {dot_color};
                                flex-shrink: 0;"></div>
                    <div style="flex: 1;">
                        <span style="font-size: 0.82rem; color: {COLORS['text_primary']};">{label}</span>
                        <span style="font-size: 0.7rem; color: {COLORS['text_tertiary']}; margin-left: 8px;">
                            {env_var}
                        </span>
                    </div>
                    <span style="font-size: 0.7rem; color: {dot_color}; font-weight: 600;
                                  text-transform: uppercase; letter-spacing: 0.04em;">
                        {status_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Summary
    st.markdown(
        f"""
        <div style="margin-top: 16px; padding: 12px 16px; background: {COLORS['bg_card']};
                     border-radius: 8px; border: 1px solid {COLORS['border_subtle']};">
            <span style="font-size: 0.82rem; color: {COLORS['text_primary']};">
                {total_set}/{total_creds} credentials configured
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 4: System Info
# ---------------------------------------------------------------------------

with system_tab:
    section_header("SYSTEM INFORMATION")

    # Python / packages
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            f"""
            <div class="sov-card">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                             color: {COLORS['text_secondary']}; font-weight: 600; margin-bottom: 8px;">
                    Runtime
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['text_primary']}; line-height: 1.8;">
                    Python: {sys.version.split()[0]}<br>
                    Platform: {sys.platform}<br>
                    Working Dir: {os.getcwd()}<br>
                    Env: {os.environ.get('ENCLAVE_ENV', 'development')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col2:
        # Package versions
        packages = {}
        try:
            import anthropic
            packages["anthropic"] = getattr(anthropic, "__version__", "?")
        except ImportError:
            packages["anthropic"] = "not installed"

        try:
            import langgraph
            packages["langgraph"] = getattr(langgraph, "__version__", "?")
        except ImportError:
            packages["langgraph"] = "not installed"

        try:
            import streamlit as st_mod
            packages["streamlit"] = getattr(st_mod, "__version__", "?")
        except ImportError:
            packages["streamlit"] = "not installed"

        try:
            import pydantic
            packages["pydantic"] = getattr(pydantic, "__version__", "?")
        except ImportError:
            packages["pydantic"] = "not installed"

        try:
            import httpx
            packages["httpx"] = getattr(httpx, "__version__", "?")
        except ImportError:
            packages["httpx"] = "not installed"

        pkg_lines = "<br>".join(f"{k}: {v}" for k, v in packages.items())

        st.markdown(
            f"""
            <div class="sov-card">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                             color: {COLORS['text_secondary']}; font-weight: 600; margin-bottom: 8px;">
                    Key Packages
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['text_primary']}; line-height: 1.8;">
                    {pkg_lines}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Vertical discovery
    section_header("DISCOVERED VERTICALS")

    try:
        from core.config.loader import list_available_verticals
        verticals = list_available_verticals()
        if verticals:
            for vid in verticals:
                config_exists = (PROJECT_ROOT / "verticals" / vid / "config.yaml").exists()
                agents_dir = PROJECT_ROOT / "verticals" / vid / "agents"
                agent_count = len(list(agents_dir.glob("*.yaml"))) if agents_dir.exists() else 0

                dot_color = COLORS["status_green"] if config_exists else COLORS["status_red"]

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 10px; padding: 8px 0;
                                 border-bottom: 1px solid {COLORS['border_subtle']};">
                        <div style="width: 8px; height: 8px; border-radius: 50%; background: {dot_color};"></div>
                        <span style="font-size: 0.82rem; color: {COLORS['text_primary']}; font-weight: 500;">
                            {vid}
                        </span>
                        <span style="font-size: 0.7rem; color: {COLORS['text_tertiary']};">
                            {agent_count} agent configs
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No verticals discovered.")
    except Exception:
        st.caption("Could not discover verticals.")
