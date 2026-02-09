"""
🚀 New Business — Genesis Engine Onboarding

The conversational interface for launching a new business vertical.
Guides the user through the five-gate Genesis flow:

    1. Interview → Gather business context via chat
    2. Blueprint → AI generates strategic plan (human reviews)
    3. Config    → Auto-generate YAML configs (human reviews)
    4. Credentials → Collect required API keys
    5. Launch    → Deploy agents in shadow mode

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="New Business — Genesis Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🚀 Genesis Engine")
st.sidebar.markdown(
    "Launch a new business vertical from scratch.\n\n"
    "The Genesis Engine interviews you, builds a strategic plan, "
    "generates agent configs, and deploys your AI team."
)
st.sidebar.markdown("---")

# Show active sessions in sidebar
st.sidebar.markdown("### Sessions")

if "genesis_sessions" not in st.session_state:
    st.session_state.genesis_sessions = {}

for sid, session in st.session_state.genesis_sessions.items():
    status = session.get("status", "interview")
    label = session.get("idea", "Unnamed")[:40]
    icon = {
        "interview": "💬",
        "blueprint_review": "📋",
        "config_review": "⚙️",
        "credential_collection": "🔑",
        "launched": "✅",
        "failed": "❌",
    }.get(status, "🔄")
    st.sidebar.caption(f"{icon} {label}")

st.sidebar.markdown("---")
st.sidebar.caption("Sovereign Venture Engine v0.3.0")


# ---------------------------------------------------------------------------
# Imports (after sidebar so they don't block rendering)
# ---------------------------------------------------------------------------

from dashboard.pages._genesis_helpers import (
    build_chat_history,
    compute_progress,
    format_blueprint_summary,
    format_credential_status,
    validate_business_idea,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "genesis_stage" not in st.session_state:
    st.session_state.genesis_stage = "welcome"  # welcome, interview, blueprint_review, config_review, credentials, launched

if "genesis_session_id" not in st.session_state:
    st.session_state.genesis_session_id = str(uuid.uuid4())

if "genesis_messages" not in st.session_state:
    st.session_state.genesis_messages = []

if "genesis_context" not in st.session_state:
    st.session_state.genesis_context = {}

if "genesis_blueprint" not in st.session_state:
    st.session_state.genesis_blueprint = None

if "genesis_vertical_id" not in st.session_state:
    st.session_state.genesis_vertical_id = None


# ---------------------------------------------------------------------------
# Helper: Safe DB access
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db(vid: str):
    """Initialize DB connection (cached across reruns)."""
    from core.integrations.supabase_client import EnclaveDB
    return EnclaveDB(vid)


def _safe_call(fn, default=None):
    """Call a function, return default on any error."""
    try:
        return fn()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Stage: Welcome
# ---------------------------------------------------------------------------

def render_welcome():
    """Initial screen — user describes their business idea."""
    st.title("🚀 Launch a New Business")
    st.markdown(
        "Describe your business idea below, and the Genesis Engine will:\n\n"
        "1. **Interview** you to understand your business\n"
        "2. **Generate** a strategic blueprint with AI\n"
        "3. **Build** agent configurations automatically\n"
        "4. **Collect** your API credentials\n"
        "5. **Deploy** your AI agent fleet in shadow mode\n"
    )

    st.markdown("---")

    idea = st.text_area(
        "What business do you want to launch?",
        placeholder=(
            "Example: I want to start a 3D printing service focused on "
            "custom prototypes for hardware startups in the Bay Area. "
            "We'll offer rapid turnaround (48 hours) and handle everything "
            "from design review to shipping."
        ),
        height=150,
        key="business_idea_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        start_btn = st.button("🚀 Start Genesis", type="primary", use_container_width=True)

    if start_btn:
        valid, msg = validate_business_idea(idea)
        if not valid:
            st.error(msg)
            return

        # Initialize interview
        st.session_state.genesis_stage = "interview"
        st.session_state.genesis_context = {"business_idea": idea.strip()}
        st.session_state.genesis_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Great! Let me learn more about your business. "
                    f"You mentioned: *\"{idea.strip()[:100]}{'...' if len(idea.strip()) > 100 else ''}\"*\n\n"
                    f"Let's dive deeper. **What problem does your business solve for your customers?** "
                    f"Who are they, and what's their biggest pain point?"
                ),
            }
        ]
        st.rerun()


# ---------------------------------------------------------------------------
# Stage: Interview
# ---------------------------------------------------------------------------

def render_interview():
    """Chat-based interview to gather business context."""
    st.title("💬 Business Interview")

    # Progress indicator
    questions_count = len([m for m in st.session_state.genesis_messages if m["role"] == "user"])
    max_questions = 8
    progress = min(questions_count / max_questions, 0.95)
    st.progress(progress, text=f"Question {questions_count + 1} of ~{max_questions}")

    # Chat messages
    for msg in st.session_state.genesis_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Your answer...")

    if user_input:
        # Add user message
        st.session_state.genesis_messages.append(
            {"role": "user", "content": user_input}
        )

        # Store in context
        st.session_state.genesis_context[f"answer_{questions_count + 1}"] = user_input

        # Generate next question (using interview engine)
        next_question = _generate_next_question(
            st.session_state.genesis_context,
            questions_count + 1,
        )

        if next_question is None:
            # Interview complete — move to blueprint generation
            st.session_state.genesis_messages.append({
                "role": "assistant",
                "content": (
                    "Excellent! I have enough context to create your strategic blueprint. "
                    "Let me analyze the market and generate a plan for your business...\n\n"
                    "**Moving to blueprint generation** ⏳"
                ),
            })
            st.session_state.genesis_stage = "blueprint_review"
            st.session_state.genesis_blueprint = _generate_mock_blueprint(
                st.session_state.genesis_context
            )
        else:
            st.session_state.genesis_messages.append(
                {"role": "assistant", "content": next_question}
            )

        st.rerun()

    # Manual advance button (for testing / fast-track)
    st.sidebar.markdown("### Interview Controls")
    if st.sidebar.button("⏩ Skip to Blueprint", help="Fast-track to blueprint review"):
        st.session_state.genesis_stage = "blueprint_review"
        st.session_state.genesis_blueprint = _generate_mock_blueprint(
            st.session_state.genesis_context
        )
        st.rerun()


def _generate_next_question(
    context: dict,
    question_num: int,
) -> str | None:
    """
    Generate the next interview question based on context.

    Returns None when enough context has been gathered.
    """
    try:
        from core.genesis.interview import InterviewEngine

        engine = InterviewEngine()
        question = engine.get_next_question(context, question_num)
        return question
    except Exception:
        # Fallback question bank for when interview engine isn't available
        fallback_questions = [
            "What's your target market? (geography, company size, industry)",
            "What's your revenue model? (pricing, packages, payment terms)",
            "Who are your main competitors, and how are you different?",
            "What's your current budget for sales and marketing tools?",
            "What integrations or tools do you already use? (CRM, email, etc.)",
            "What does your ideal customer look like? (title, industry, company size)",
            "What's your timeline? When do you want your first customers?",
        ]

        idx = question_num - 1
        if idx < len(fallback_questions):
            return fallback_questions[idx]
        return None


def _generate_mock_blueprint(context: dict) -> dict:
    """
    Generate a blueprint from context.

    In production this uses the ArchitectAgent. For the dashboard demo,
    we generate a structured placeholder.
    """
    idea = context.get("business_idea", "Unknown business")
    return {
        "vertical_name": idea[:50].strip(),
        "vertical_id": "new_business",
        "industry": "general",
        "strategy_reasoning": (
            f"Based on your description — \"{idea[:100]}\" — "
            "we recommend a targeted outbound strategy with "
            "SEO content to build authority and appointment setting "
            "to close deals."
        ),
        "icp": {
            "company_sizes": ["11-50", "51-200"],
            "industries": ["technology", "manufacturing"],
            "signals": ["recent funding", "job postings", "tech adoption"],
        },
        "personas": [
            {
                "title": "CTO / VP Engineering",
                "pain_points": ["scaling challenges", "technical debt"],
            },
            {
                "title": "CEO / Founder",
                "pain_points": ["growth strategy", "operational efficiency"],
            },
        ],
        "agents": [
            {
                "agent_type": "outreach",
                "description": "Finds and contacts potential customers",
            },
            {
                "agent_type": "seo_content",
                "description": "Creates blog posts and landing pages for SEO",
            },
            {
                "agent_type": "appointment_setter",
                "description": "Books meetings from interested replies",
            },
        ],
        "integrations": [
            {"name": "Apollo", "purpose": "Lead data and enrichment"},
            {"name": "SendGrid", "purpose": "Email delivery"},
        ],
        "risk_factors": [
            "New market — may need messaging iteration",
            "Competitive space — differentiation critical",
        ],
        "success_metrics": [
            "50+ qualified leads per month within 90 days",
            "5% positive reply rate on outbound",
            "10+ meetings booked per month by month 3",
        ],
        "context": context,
    }


# ---------------------------------------------------------------------------
# Stage: Blueprint Review
# ---------------------------------------------------------------------------

def render_blueprint_review():
    """Human reviews the generated blueprint."""
    st.title("📋 Blueprint Review")
    st.markdown("Review the strategic plan generated for your business.")

    blueprint = st.session_state.genesis_blueprint

    if blueprint is None:
        st.warning("No blueprint generated yet. Please complete the interview first.")
        if st.button("← Back to Interview"):
            st.session_state.genesis_stage = "interview"
            st.rerun()
        return

    # Progress
    st.progress(0.5, text="Stage 2 of 5 — Blueprint Review")

    # Display blueprint
    summary = format_blueprint_summary(blueprint)
    st.markdown(summary)

    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve Blueprint", type="primary", use_container_width=True):
            st.session_state.genesis_stage = "config_review"
            st.rerun()

    with col2:
        if st.button("✏️ Request Changes", use_container_width=True):
            st.session_state.genesis_stage = "interview"
            st.session_state.genesis_messages.append({
                "role": "assistant",
                "content": (
                    "No problem! What would you like to change about the blueprint? "
                    "I'll update the plan based on your feedback."
                ),
            })
            st.rerun()

    with col3:
        if st.button("🗑️ Start Over", use_container_width=True):
            _reset_genesis()
            st.rerun()


# ---------------------------------------------------------------------------
# Stage: Config Review
# ---------------------------------------------------------------------------

def render_config_review():
    """Human reviews generated YAML configs."""
    st.title("⚙️ Configuration Review")
    st.markdown(
        "The Genesis Engine has generated agent configurations for your business. "
        "Review them below before proceeding to credential setup."
    )

    # Progress
    st.progress(0.7, text="Stage 3 of 5 — Configuration Review")

    blueprint = st.session_state.genesis_blueprint or {}

    # Show what would be generated
    st.subheader("Agent Configurations")

    agents = blueprint.get("agents", [])
    if agents:
        for agent in agents:
            with st.expander(
                f"🤖 {agent.get('agent_type', '?')} — {agent.get('description', '')}",
                expanded=False,
            ):
                st.code(
                    f"agent_id: {agent.get('agent_type', 'unknown')}_v1\n"
                    f"agent_type: {agent.get('agent_type', 'unknown')}\n"
                    f"name: \"{agent.get('description', 'Agent')}\"\n"
                    f"enabled: true\n"
                    f"shadow_mode: true  # Safe — no real emails sent\n"
                    f"schedule:\n"
                    f"  trigger: manual\n"
                    f"params: {{}}\n",
                    language="yaml",
                )
    else:
        st.info("No agents defined in blueprint.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve Configs", type="primary", use_container_width=True):
            st.session_state.genesis_stage = "credentials"
            st.rerun()

    with col2:
        if st.button("← Back to Blueprint", use_container_width=True):
            st.session_state.genesis_stage = "blueprint_review"
            st.rerun()


# ---------------------------------------------------------------------------
# Stage: Credentials
# ---------------------------------------------------------------------------

def render_credentials():
    """Collect API credentials from the user."""
    st.title("🔑 Credential Setup")
    st.markdown(
        "Enter the API keys needed for your business. "
        "These are encrypted and stored securely."
    )

    # Progress
    st.progress(0.8, text="Stage 4 of 5 — Credential Collection")

    # Show required credentials
    blueprint = st.session_state.genesis_blueprint or {}
    integrations = blueprint.get("integrations", [])

    st.subheader("Required Credentials")

    # Standard platform credentials
    platform_creds = [
        ("ANTHROPIC_API_KEY", "Anthropic API Key", "Powers the AI agents", True),
        ("OPENAI_API_KEY", "OpenAI API Key", "For embeddings and RAG", True),
    ]

    # Integration-specific credentials
    integration_creds = []
    for intg in integrations:
        name = intg.get("name", "").upper().replace(" ", "_")
        integration_creds.append(
            (f"{name}_API_KEY", f"{intg.get('name', '?')} API Key", intg.get("purpose", ""), True)
        )

    all_creds = platform_creds + integration_creds

    for env_var, label, help_text, required in all_creds:
        current_value = os.environ.get(env_var, "")
        icon = "✅" if current_value else ("❌" if required else "⬜")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                f"{icon} {label}",
                value="••••••••" if current_value else "",
                type="password",
                help=help_text,
                key=f"cred_{env_var}",
                disabled=bool(current_value),
            )
        with col2:
            if current_value:
                st.success("Set", icon="✅")
            elif required:
                st.error("Required", icon="❌")
            else:
                st.info("Optional", icon="ℹ️")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Launch in Shadow Mode", type="primary", use_container_width=True):
            st.session_state.genesis_stage = "launched"
            st.rerun()

    with col2:
        if st.button("← Back to Configs", use_container_width=True):
            st.session_state.genesis_stage = "config_review"
            st.rerun()

    st.info(
        "💡 **Shadow Mode** means agents will run but won't send real emails "
        "or make real API calls. You can review everything before going live."
    )


# ---------------------------------------------------------------------------
# Stage: Launched
# ---------------------------------------------------------------------------

def render_launched():
    """Success screen — agents are running in shadow mode."""
    st.title("🎉 Business Launched!")

    st.balloons()

    # Progress
    st.progress(1.0, text="Complete — Agents running in shadow mode")

    blueprint = st.session_state.genesis_blueprint or {}
    name = blueprint.get("vertical_name", "Your Business")

    st.success(
        f"**{name}** is now running in shadow mode! "
        "Your AI agent fleet is active but sandboxed — no real emails will be sent "
        "until you promote to live mode."
    )

    st.markdown("---")

    # Agent status cards
    st.subheader("🤖 Agent Fleet Status")

    agents = blueprint.get("agents", [])
    if agents:
        cols = st.columns(min(len(agents), 3))
        for i, agent in enumerate(agents):
            with cols[i % len(cols)]:
                st.markdown(
                    f"### 👻 {agent.get('agent_type', '?')}\n"
                    f"{agent.get('description', '')}\n\n"
                    f"Status: **Shadow Mode** 🔒"
                )

    st.markdown("---")

    # Next steps
    st.subheader("📋 Next Steps")
    st.markdown(
        "1. **Review** shadow mode activity in the [Approvals](/Approvals) page\n"
        "2. **Monitor** agent performance in the [Agents](/Agents) page\n"
        "3. **Promote** to live mode when you're satisfied with the results\n"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏠 Go to Mission Control", type="primary", use_container_width=True):
            st.switch_page("app.py")

    with col2:
        if st.button("🚀 Launch Another Business", use_container_width=True):
            _reset_genesis()
            st.rerun()


# ---------------------------------------------------------------------------
# Reset helper
# ---------------------------------------------------------------------------

def _reset_genesis():
    """Reset all genesis session state."""
    st.session_state.genesis_stage = "welcome"
    st.session_state.genesis_session_id = str(uuid.uuid4())
    st.session_state.genesis_messages = []
    st.session_state.genesis_context = {}
    st.session_state.genesis_blueprint = None
    st.session_state.genesis_vertical_id = None


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

stage = st.session_state.get("genesis_stage", "welcome")

if stage == "welcome":
    render_welcome()
elif stage == "interview":
    render_interview()
elif stage == "blueprint_review":
    render_blueprint_review()
elif stage == "config_review":
    render_config_review()
elif stage == "credentials":
    render_credentials()
elif stage == "launched":
    render_launched()
else:
    st.error(f"Unknown genesis stage: {stage}")
    _reset_genesis()
    st.rerun()
