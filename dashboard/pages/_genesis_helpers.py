"""
Extracted helpers for the New Business (Genesis) page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Interview conversation management
# ---------------------------------------------------------------------------

def build_chat_history(
    questions_asked: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Convert genesis session Q&A into a flat chat history.

    Args:
        questions_asked: List of {"question": str, "answer": str} dicts
                         from the genesis session.

    Returns:
        List of {"role": "assistant"|"user", "content": str} messages.
    """
    messages: list[dict[str, str]] = []

    for qa in questions_asked:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        if q:
            messages.append({"role": "assistant", "content": q})
        if a:
            messages.append({"role": "user", "content": a})

    return messages


def compute_progress(status: str) -> tuple[float, str]:
    """
    Convert a genesis session status to a progress bar value and label.

    Returns:
        (fraction 0.0-1.0, human-readable label)
    """
    progress_map: dict[str, tuple[float, str]] = {
        "interview": (0.15, "Interviewing — gathering your business context"),
        "market_analysis": (0.30, "Analyzing market and competition"),
        "blueprint_generation": (0.45, "Generating strategic blueprint"),
        "blueprint_review": (0.50, "Blueprint ready for your review"),
        "config_generation": (0.65, "Generating agent configurations"),
        "config_review": (0.70, "Configs ready for your review"),
        "credential_collection": (0.80, "Waiting for API credentials"),
        "launching": (0.90, "Deploying agent fleet"),
        "launched": (1.0, "✅ Agents running in shadow mode!"),
        "failed": (0.0, "❌ An error occurred"),
        "cancelled": (0.0, "Cancelled"),
    }
    return progress_map.get(status, (0.0, f"Unknown status: {status}"))


def format_blueprint_summary(blueprint_data: dict[str, Any]) -> str:
    """
    Format a BusinessBlueprint dict into a human-readable Markdown summary.

    Args:
        blueprint_data: The serialized BusinessBlueprint.

    Returns:
        Markdown string for display.
    """
    parts = []

    name = blueprint_data.get("vertical_name", "Unknown Business")
    industry = blueprint_data.get("industry", "")
    parts.append(f"## {name}")
    if industry:
        parts.append(f"**Industry:** {industry}\n")

    # Strategy reasoning
    reasoning = blueprint_data.get("strategy_reasoning", "")
    if reasoning:
        parts.append(f"### Strategy\n{reasoning}\n")

    # ICP
    icp = blueprint_data.get("icp", {})
    if icp:
        parts.append("### Ideal Customer Profile")
        if icp.get("company_sizes"):
            parts.append(f"- **Company sizes:** {', '.join(icp['company_sizes'])}")
        if icp.get("industries"):
            parts.append(f"- **Target industries:** {', '.join(icp['industries'])}")
        if icp.get("signals"):
            parts.append(f"- **Buying signals:** {', '.join(icp['signals'])}")
        parts.append("")

    # Personas
    personas = blueprint_data.get("personas", [])
    if personas:
        parts.append("### Target Personas")
        for p in personas:
            title = p.get("title", "?")
            pain = p.get("pain_points", [])
            parts.append(f"- **{title}**" + (f": {', '.join(pain)}" if pain else ""))
        parts.append("")

    # Agents
    agents = blueprint_data.get("agents", [])
    if agents:
        parts.append("### Agent Fleet")
        for a in agents:
            agent_type = a.get("agent_type", "?")
            desc = a.get("description", "")
            parts.append(f"- **{agent_type}** — {desc}")
        parts.append("")

    # Integrations
    integrations = blueprint_data.get("integrations", [])
    if integrations:
        parts.append("### Required Integrations")
        for i in integrations:
            name = i.get("name", "?")
            purpose = i.get("purpose", "")
            parts.append(f"- **{name}** — {purpose}")
        parts.append("")

    # Risk factors
    risks = blueprint_data.get("risk_factors", [])
    if risks:
        parts.append("### Risk Factors")
        for r in risks:
            parts.append(f"- ⚠️ {r}")
        parts.append("")

    # Success metrics
    metrics = blueprint_data.get("success_metrics", [])
    if metrics:
        parts.append("### Success Metrics")
        for m in metrics:
            parts.append(f"- 📊 {m}")
        parts.append("")

    return "\n".join(parts)


def format_credential_status(
    credentials: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Format credential data into display-ready dicts.

    Args:
        credentials: List of credential status dicts from CredentialManager.

    Returns:
        List of {name, env_var, required, status, icon} dicts.
    """
    formatted = []
    for c in credentials:
        is_set = c.get("is_set", False)
        required = c.get("required", True)

        if is_set:
            icon = "✅"
            status = "Set"
        elif required:
            icon = "❌"
            status = "Missing (required)"
        else:
            icon = "⬜"
            status = "Not set (optional)"

        formatted.append({
            "name": c.get("credential_name", c.get("env_var_name", "?")),
            "env_var": c.get("env_var_name", "?"),
            "required": "Required" if required else "Optional",
            "status": status,
            "icon": icon,
        })
    return formatted


def validate_business_idea(idea: str) -> tuple[bool, str]:
    """
    Quick validation of a user's business idea input.

    Returns:
        (is_valid, message)
    """
    if not idea or not idea.strip():
        return False, "Please describe your business idea."

    stripped = idea.strip()
    if len(stripped) < 10:
        return False, "Please provide a bit more detail about your business."

    if len(stripped) > 5000:
        return False, "Please keep your description under 5,000 characters."

    return True, ""
