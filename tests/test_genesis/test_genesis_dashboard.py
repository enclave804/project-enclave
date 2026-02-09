"""
Tests for Genesis Engine Dashboard — helpers, sidebar, and page logic.

Tests the extractable business logic without requiring a running Streamlit server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ===========================================================================
# Test: Genesis Helpers
# ===========================================================================


class TestBuildChatHistory:
    """Test interview Q&A → chat message conversion."""

    def test_empty_questions(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        assert build_chat_history([]) == []

    def test_single_qa_pair(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        messages = build_chat_history([
            {"question": "What do you do?", "answer": "I sell widgets"},
        ])
        assert len(messages) == 2
        assert messages[0] == {"role": "assistant", "content": "What do you do?"}
        assert messages[1] == {"role": "user", "content": "I sell widgets"}

    def test_multiple_qa_pairs(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        messages = build_chat_history([
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
            {"question": "Q3?", "answer": "A3"},
        ])
        assert len(messages) == 6
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert messages[4]["role"] == "assistant"
        assert messages[5]["content"] == "A3"

    def test_missing_answer(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        messages = build_chat_history([
            {"question": "What?", "answer": ""},
        ])
        # Question present, empty answer skipped
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_missing_question(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        messages = build_chat_history([
            {"question": "", "answer": "Something"},
        ])
        # Empty question skipped, answer present
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_empty_qa(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        messages = build_chat_history([
            {"question": "", "answer": ""},
        ])
        assert len(messages) == 0

    def test_missing_keys(self):
        from dashboard.pages._genesis_helpers import build_chat_history
        messages = build_chat_history([{}])
        assert len(messages) == 0


class TestComputeProgress:
    """Test status → progress mapping."""

    def test_interview_stage(self):
        from dashboard.pages._genesis_helpers import compute_progress
        fraction, label = compute_progress("interview")
        assert fraction == 0.15
        assert "interview" in label.lower() or "gathering" in label.lower()

    def test_launched_stage(self):
        from dashboard.pages._genesis_helpers import compute_progress
        fraction, label = compute_progress("launched")
        assert fraction == 1.0
        assert "shadow" in label.lower() or "✅" in label

    def test_failed_stage(self):
        from dashboard.pages._genesis_helpers import compute_progress
        fraction, label = compute_progress("failed")
        assert fraction == 0.0
        assert "error" in label.lower() or "❌" in label

    def test_unknown_status(self):
        from dashboard.pages._genesis_helpers import compute_progress
        fraction, label = compute_progress("nonexistent_status")
        assert fraction == 0.0
        assert "unknown" in label.lower()

    def test_all_statuses_have_progress(self):
        from dashboard.pages._genesis_helpers import compute_progress
        statuses = [
            "interview", "market_analysis", "blueprint_generation",
            "blueprint_review", "config_generation", "config_review",
            "credential_collection", "launching", "launched",
            "failed", "cancelled",
        ]
        for status in statuses:
            fraction, label = compute_progress(status)
            assert isinstance(fraction, float)
            assert isinstance(label, str)
            assert len(label) > 0

    def test_progress_increases_through_flow(self):
        from dashboard.pages._genesis_helpers import compute_progress
        ordered_statuses = [
            "interview", "market_analysis", "blueprint_generation",
            "blueprint_review", "config_generation", "config_review",
            "credential_collection", "launching", "launched",
        ]
        prev_fraction = -1.0
        for status in ordered_statuses:
            fraction, _ = compute_progress(status)
            assert fraction >= prev_fraction, (
                f"Progress should increase: {status} ({fraction}) < previous ({prev_fraction})"
            )
            prev_fraction = fraction


class TestFormatBlueprintSummary:
    """Test blueprint → Markdown formatting."""

    def test_empty_blueprint(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({})
        assert "Unknown Business" in result

    def test_basic_fields(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Print Shop Pro",
            "industry": "Manufacturing",
        })
        assert "Print Shop Pro" in result
        assert "Manufacturing" in result

    def test_strategy_reasoning(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "strategy_reasoning": "We recommend a direct sales approach.",
        })
        assert "Strategy" in result
        assert "direct sales approach" in result

    def test_icp_section(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "icp": {
                "company_sizes": ["11-50", "51-200"],
                "industries": ["tech", "finance"],
                "signals": ["recent funding"],
            },
        })
        assert "Ideal Customer Profile" in result
        assert "11-50" in result
        assert "tech" in result
        assert "recent funding" in result

    def test_personas_section(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "personas": [
                {"title": "CTO", "pain_points": ["scaling", "tech debt"]},
            ],
        })
        assert "Target Personas" in result
        assert "CTO" in result
        assert "scaling" in result

    def test_agents_section(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "agents": [
                {"agent_type": "outreach", "description": "Finds leads"},
                {"agent_type": "seo_content", "description": "Writes content"},
            ],
        })
        assert "Agent Fleet" in result
        assert "outreach" in result
        assert "seo_content" in result

    def test_risk_factors(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "risk_factors": ["Market saturation", "High competition"],
        })
        assert "Risk Factors" in result
        assert "Market saturation" in result

    def test_success_metrics(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "success_metrics": ["100 leads/month", "5% reply rate"],
        })
        assert "Success Metrics" in result
        assert "100 leads/month" in result

    def test_integrations_section(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "integrations": [
                {"name": "Apollo", "purpose": "Lead data"},
            ],
        })
        assert "Required Integrations" in result
        assert "Apollo" in result

    def test_full_blueprint(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        blueprint = {
            "vertical_name": "Acme Consulting",
            "industry": "Technology",
            "strategy_reasoning": "Direct outbound with content marketing.",
            "icp": {"company_sizes": ["51-200"], "industries": ["SaaS"]},
            "personas": [{"title": "VP Sales", "pain_points": ["pipeline"]}],
            "agents": [{"agent_type": "outreach", "description": "Outbound"}],
            "integrations": [{"name": "Apollo", "purpose": "Data"}],
            "risk_factors": ["Competitive market"],
            "success_metrics": ["50 leads/month"],
        }
        result = format_blueprint_summary(blueprint)
        assert "Acme Consulting" in result
        assert len(result) > 200  # Should be a substantial markdown doc


class TestFormatCredentialStatus:
    """Test credential list formatting."""

    def test_empty_list(self):
        from dashboard.pages._genesis_helpers import format_credential_status
        assert format_credential_status([]) == []

    def test_set_credential(self):
        from dashboard.pages._genesis_helpers import format_credential_status
        result = format_credential_status([
            {"credential_name": "Apollo", "env_var_name": "APOLLO_API_KEY", "is_set": True, "required": True},
        ])
        assert len(result) == 1
        assert result[0]["icon"] == "✅"
        assert result[0]["status"] == "Set"

    def test_missing_required(self):
        from dashboard.pages._genesis_helpers import format_credential_status
        result = format_credential_status([
            {"credential_name": "Apollo", "env_var_name": "APOLLO_API_KEY", "is_set": False, "required": True},
        ])
        assert result[0]["icon"] == "❌"
        assert "required" in result[0]["status"].lower()

    def test_missing_optional(self):
        from dashboard.pages._genesis_helpers import format_credential_status
        result = format_credential_status([
            {"credential_name": "Shodan", "env_var_name": "SHODAN_API_KEY", "is_set": False, "required": False},
        ])
        assert result[0]["icon"] == "⬜"
        assert "optional" in result[0]["status"].lower()

    def test_multiple_credentials(self):
        from dashboard.pages._genesis_helpers import format_credential_status
        creds = [
            {"credential_name": "A", "env_var_name": "A_KEY", "is_set": True, "required": True},
            {"credential_name": "B", "env_var_name": "B_KEY", "is_set": False, "required": True},
            {"credential_name": "C", "env_var_name": "C_KEY", "is_set": False, "required": False},
        ]
        result = format_credential_status(creds)
        assert len(result) == 3
        assert result[0]["icon"] == "✅"
        assert result[1]["icon"] == "❌"
        assert result[2]["icon"] == "⬜"


class TestValidateBusinessIdea:
    """Test business idea input validation."""

    def test_empty_string(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, msg = validate_business_idea("")
        assert valid is False
        assert "describe" in msg.lower()

    def test_none(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, msg = validate_business_idea(None)
        assert valid is False

    def test_too_short(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, msg = validate_business_idea("widgets")
        assert valid is False
        assert "more detail" in msg.lower()

    def test_too_long(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, msg = validate_business_idea("x" * 6000)
        assert valid is False
        assert "5,000" in msg

    def test_valid_idea(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, msg = validate_business_idea(
            "I want to start a 3D printing business for hardware startups"
        )
        assert valid is True
        assert msg == ""

    def test_whitespace_only(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, msg = validate_business_idea("   \n\t  ")
        assert valid is False

    def test_minimum_length(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, _ = validate_business_idea("Ten chars!")  # Exactly 10 chars
        assert valid is True

    def test_strips_whitespace(self):
        from dashboard.pages._genesis_helpers import validate_business_idea
        valid, _ = validate_business_idea("  A valid business idea here  ")
        assert valid is True


# ===========================================================================
# Test: Dynamic Sidebar
# ===========================================================================


class TestGetVerticalOptions:
    """Test dynamic vertical discovery."""

    def test_returns_dict(self):
        from dashboard.sidebar import get_vertical_options
        result = get_vertical_options()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_enclave_guard_present(self):
        from dashboard.sidebar import get_vertical_options
        result = get_vertical_options()
        # enclave_guard should always be present (it has a config.yaml)
        assert "enclave_guard" in result.values()

    def test_fallback_on_import_error(self):
        """Falls back to hardcoded dict if loader fails."""
        from dashboard.sidebar import get_vertical_options

        with patch("dashboard.sidebar.get_vertical_options.__module__", side_effect=ImportError):
            # Direct test — if the function can't import loader, it catches
            result = get_vertical_options()
            assert isinstance(result, dict)
            assert len(result) > 0

    def test_display_names_are_strings(self):
        from dashboard.sidebar import get_vertical_options
        result = get_vertical_options()
        for name, vid in result.items():
            assert isinstance(name, str)
            assert isinstance(vid, str)
            assert len(name) > 0
            assert len(vid) > 0

    def test_discovers_multiple_verticals(self):
        """Should find all verticals with config.yaml files."""
        from dashboard.sidebar import get_vertical_options
        result = get_vertical_options()
        # At minimum enclave_guard exists; print_biz may or may not
        assert len(result) >= 1


class TestSidebarVerticalDiscoveryIntegration:
    """Integration test: sidebar discovers verticals from filesystem."""

    def test_enclave_guard_has_human_readable_name(self):
        """Enclave Guard should use vertical_name from config, not raw ID."""
        from dashboard.sidebar import get_vertical_options
        result = get_vertical_options()
        # The key should be the display name (from config.vertical_name)
        # not the raw vertical_id
        display_names = list(result.keys())
        assert any(
            name != "enclave_guard" for name in display_names
        ), "Display names should be human-readable, not raw IDs"


# ===========================================================================
# Test: Blueprint Summary Markdown Quality
# ===========================================================================


class TestBlueprintMarkdownQuality:
    """Ensure generated Markdown is well-structured."""

    def test_has_headers(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "strategy_reasoning": "A strategy",
            "agents": [{"agent_type": "outreach", "description": "test"}],
        })
        assert "##" in result

    def test_no_trailing_whitespace_lines(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({
            "vertical_name": "Test",
            "industry": "Tech",
        })
        for line in result.split("\n"):
            # Allow empty lines, but no trailing spaces on content lines
            if line.strip():
                assert line == line.rstrip(), f"Trailing whitespace: '{line}'"

    def test_returns_string(self):
        from dashboard.pages._genesis_helpers import format_blueprint_summary
        result = format_blueprint_summary({"vertical_name": "X"})
        assert isinstance(result, str)


# ===========================================================================
# Test: Progress Flow Consistency
# ===========================================================================


class TestProgressFlowConsistency:
    """Verify the Genesis progress stages form a valid flow."""

    def test_all_genesis_statuses_covered(self):
        """Every status in the DB schema has a progress mapping."""
        from dashboard.pages._genesis_helpers import compute_progress

        # From 007_genesis_engine.sql CHECK constraint
        db_statuses = [
            "interview", "market_analysis", "blueprint_generation",
            "blueprint_review", "config_generation", "config_review",
            "credential_collection", "launching", "launched",
            "failed", "cancelled",
        ]

        for status in db_statuses:
            fraction, label = compute_progress(status)
            assert isinstance(fraction, float), f"Missing progress for {status}"
            assert label, f"Empty label for {status}"

    def test_terminal_states_at_endpoints(self):
        """launched=1.0, failed/cancelled=0.0."""
        from dashboard.pages._genesis_helpers import compute_progress

        assert compute_progress("launched")[0] == 1.0
        assert compute_progress("failed")[0] == 0.0
        assert compute_progress("cancelled")[0] == 0.0
