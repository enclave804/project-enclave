"""
Unit tests for agent state TypedDicts.

Validates that all agent state types can be created with proper defaults
and that field inheritance works correctly.
"""

import pytest

from core.agents.state import (
    BaseAgentState,
    OutreachAgentState,
    SEOContentAgentState,
    AppointmentAgentState,
)


class TestBaseAgentState:
    """Tests for the BaseAgentState TypedDict."""

    def test_create_minimal_state(self):
        """Should create state with only required fields."""
        state: BaseAgentState = {
            "agent_id": "test_agent",
            "vertical_id": "enclave_guard",
            "run_id": "run-123",
        }
        assert state["agent_id"] == "test_agent"
        assert state["vertical_id"] == "enclave_guard"
        assert state["run_id"] == "run-123"

    def test_create_full_state(self):
        """Should create state with all fields populated."""
        state: BaseAgentState = {
            "agent_id": "test_agent",
            "vertical_id": "enclave_guard",
            "run_id": "run-123",
            "task_input": {"key": "value"},
            "task_type": "process_lead",
            "current_node": "enrich",
            "error": None,
            "error_node": None,
            "retry_count": 0,
            "started_at": "2025-01-01T00:00:00Z",
            "completed_at": None,
            "requires_human_approval": False,
            "human_approval_status": None,
            "human_feedback": None,
            "rag_context": [],
            "knowledge_written": False,
        }
        assert state["current_node"] == "enrich"
        assert state["retry_count"] == 0
        assert state["requires_human_approval"] is False

    def test_state_is_dict_subclass(self):
        """TypedDicts are regular dicts at runtime."""
        state: BaseAgentState = {"agent_id": "x", "vertical_id": "y", "run_id": "z"}
        assert isinstance(state, dict)

    def test_state_allows_extra_fields_at_runtime(self):
        """Python TypedDicts don't enforce at runtime — just for type checking."""
        state: BaseAgentState = {"agent_id": "x", "vertical_id": "y", "run_id": "z"}
        state["custom_field"] = "allowed"
        assert state["custom_field"] == "allowed"


class TestOutreachAgentState:
    """Tests for the OutreachAgentState TypedDict."""

    def test_inherits_base_fields(self):
        """Should include all BaseAgentState fields."""
        state: OutreachAgentState = {
            "agent_id": "outreach",
            "vertical_id": "enclave_guard",
            "run_id": "run-456",
            "lead_id": "lead-001",
            "company_name": "Acme Corp",
            "contact_email": "alice@acme.com",
        }
        assert state["agent_id"] == "outreach"
        assert state["lead_id"] == "lead-001"
        assert state["company_name"] == "Acme Corp"

    def test_outreach_specific_fields(self):
        """Should support all outreach-specific fields."""
        state: OutreachAgentState = {
            "agent_id": "outreach",
            "vertical_id": "enclave_guard",
            "run_id": "run-789",
            "qualification_score": 0.85,
            "qualified": True,
            "matching_signals": ["uses_php", "small_team"],
            "draft_email_subject": "Security Assessment",
            "email_sent": False,
            "compliance_passed": True,
        }
        assert state["qualification_score"] == 0.85
        assert state["qualified"] is True
        assert len(state["matching_signals"]) == 2


class TestSEOContentAgentState:
    """Tests for the SEOContentAgentState TypedDict."""

    def test_seo_specific_fields(self):
        """Should support SEO content fields."""
        state: SEOContentAgentState = {
            "agent_id": "seo_content",
            "vertical_id": "enclave_guard",
            "run_id": "run-seo-1",
            "target_keywords": ["penetration testing", "SOC 2"],
            "content_type": "blog_post",
            "draft_title": "Top 5 Pen Testing Tools",
            "seo_score": 0.92,
            "content_approved": False,
        }
        assert state["content_type"] == "blog_post"
        assert state["seo_score"] == 0.92
        assert len(state["target_keywords"]) == 2


class TestAppointmentAgentState:
    """Tests for the AppointmentAgentState TypedDict."""

    def test_appointment_specific_fields(self):
        """Should support appointment scheduling fields."""
        state: AppointmentAgentState = {
            "agent_id": "appointment_setter",
            "vertical_id": "enclave_guard",
            "run_id": "run-appt-1",
            "reply_intent": "interested",
            "reply_sentiment": "positive",
            "meeting_booked": False,
            "proposed_times": ["2025-02-01T10:00:00Z", "2025-02-01T14:00:00Z"],
            "follow_up_sequence_step": 1,
        }
        assert state["reply_intent"] == "interested"
        assert state["meeting_booked"] is False
        assert len(state["proposed_times"]) == 2
