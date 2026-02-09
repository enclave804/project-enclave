"""
Tests for MeetingSchedulerAgent — Phase 16: Sales Pipeline.

Covers:
    - MeetingSchedulerAgentState TypedDict
    - MeetingSchedulerAgent registration, construction, state class
    - Initial state preparation
    - Constants (MODES, MEETING_TYPES, MEETING_STATUSES)
    - All 5 nodes: check_requests, propose_times, human_review,
      send_invites, report
    - Graph construction and routing
    - System prompt
    - YAML config (meeting_scheduler.yaml)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
# State Tests
# ══════════════════════════════════════════════════════════════════════


class TestMeetingSchedulerState:
    """Tests for MeetingSchedulerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import MeetingSchedulerAgentState
        assert MeetingSchedulerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import MeetingSchedulerAgentState
        state: MeetingSchedulerAgentState = {
            "agent_id": "meeting_scheduler_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "meeting_scheduler_v1"

    def test_create_full(self):
        from core.agents.state import MeetingSchedulerAgentState
        state: MeetingSchedulerAgentState = {
            "agent_id": "meeting_scheduler_v1",
            "vertical_id": "enclave_guard",
            "pending_requests": [{"contact_email": "jane@acme.com"}],
            "total_requests": 1,
            "proposed_meetings": [],
            "meetings_approved": False,
            "human_edits": [],
            "invites_sent": 0,
            "meetings_confirmed": 0,
            "meetings_cancelled": 0,
            "calendar_links": [],
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["total_requests"] == 1
        assert len(state["pending_requests"]) == 1
        assert state["meetings_approved"] is False

    def test_meeting_tracking_fields(self):
        from core.agents.state import MeetingSchedulerAgentState
        state: MeetingSchedulerAgentState = {
            "invites_sent": 3,
            "meetings_confirmed": 2,
            "meetings_cancelled": 1,
            "calendar_links": [
                "https://calendar.app/meeting/1",
                "https://calendar.app/meeting/2",
            ],
        }
        assert state["invites_sent"] == 3
        assert len(state["calendar_links"]) == 2


# ══════════════════════════════════════════════════════════════════════
# Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestMeetingSchedulerAgent:
    """Tests for MeetingSchedulerAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a MeetingSchedulerAgent with mocked dependencies."""
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="meeting_scheduler_v1",
            agent_type="meeting_scheduler",
            name="Meeting Scheduler Agent",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
                "default_duration": 30,
                "working_hours_start": 9,
                "working_hours_end": 17,
                "timezone": "America/New_York",
            },
            **kwargs,
        )
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()

        return MeetingSchedulerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "meeting_scheduler" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent
        assert MeetingSchedulerAgent.agent_type == "meeting_scheduler"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import MeetingSchedulerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is MeetingSchedulerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        repr_str = repr(agent)
        assert "MeetingSchedulerAgent" in repr_str
        assert "meeting_scheduler_v1" in repr_str

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "check_pending", "meeting_requests": []},
            "run-123",
        )
        assert state["pending_requests"] == []
        assert state["total_requests"] == 0
        assert state["proposed_meetings"] == []
        assert state["meetings_approved"] is False
        assert state["invites_sent"] == 0
        assert state["meetings_confirmed"] == 0
        assert state["meetings_cancelled"] == 0
        assert state["calendar_links"] == []
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_requests(self):
        agent = self._make_agent()
        requests = [
            {"contact_email": "jane@acme.com", "meeting_type": "discovery"},
        ]
        state = agent._prepare_initial_state(
            {"meeting_requests": requests}, "run-1"
        )
        assert state["pending_requests"] == requests
        assert state["total_requests"] == 1

    # ─── Constants ──────────────────────────────────────────────────

    def test_constants_modes(self):
        from core.agents.implementations import meeting_agent
        assert "check_pending" in meeting_agent.MODES
        assert "schedule_new" in meeting_agent.MODES
        assert "follow_up_unconfirmed" in meeting_agent.MODES

    def test_constants_meeting_types(self):
        from core.agents.implementations import meeting_agent
        assert "discovery" in meeting_agent.MEETING_TYPES
        assert "demo" in meeting_agent.MEETING_TYPES
        assert "follow_up" in meeting_agent.MEETING_TYPES
        assert "negotiation" in meeting_agent.MEETING_TYPES
        assert "kickoff" in meeting_agent.MEETING_TYPES

    def test_constants_meeting_statuses(self):
        from core.agents.implementations import meeting_agent
        for status in ["proposed", "confirmed", "completed", "cancelled", "no_show", "rescheduled"]:
            assert status in meeting_agent.MEETING_STATUSES

    def test_system_prompt_template(self):
        from core.agents.implementations import meeting_agent
        prompt = meeting_agent.MEETING_SYSTEM_PROMPT
        assert "{company_name}" in prompt

    # ─── Node 1: Check Requests ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_check_requests_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        result = await agent._node_check_requests(state)

        assert result["current_node"] == "check_requests"
        assert result["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_node_check_requests_with_pending(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "mtg_1",
                "contact_email": "jane@acme.com",
                "contact_name": "Jane Doe",
                "company_name": "Acme Corp",
                "meeting_type": "discovery",
                "status": "proposed",
            },
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        result = await agent._node_check_requests(state)

        assert result["total_requests"] == 1
        assert result["pending_requests"][0]["contact_email"] == "jane@acme.com"

    @pytest.mark.asyncio
    async def test_node_check_requests_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.side_effect = Exception("DB error")

        state = agent._prepare_initial_state(
            {"meeting_requests": [{"contact_email": "test@test.com"}]},
            "run-1",
        )
        result = await agent._node_check_requests(state)

        # Should still have the task-provided requests
        assert result["current_node"] == "check_requests"
        assert result["total_requests"] >= 1

    # ─── Node 2: Propose Times ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_propose_times_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["pending_requests"] = []

        result = await agent._node_propose_times(state)

        assert result["current_node"] == "propose_times"
        assert result["proposed_meetings"] == []

    @pytest.mark.asyncio
    async def test_node_propose_times_with_request(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text="Hi Jane, I'd love to set up a discovery call...")]
        ))

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["pending_requests"] = [
            {
                "contact_email": "jane@acme.com",
                "contact_name": "Jane Doe",
                "company_name": "Acme Corp",
                "meeting_type": "discovery",
            },
        ]

        result = await agent._node_propose_times(state)

        assert result["current_node"] == "propose_times"
        assert len(result["proposed_meetings"]) == 1
        proposal = result["proposed_meetings"][0]
        assert proposal["contact_email"] == "jane@acme.com"
        assert len(proposal["proposed_times"]) >= 1
        assert proposal["meeting_type"] == "discovery"

    @pytest.mark.asyncio
    async def test_node_propose_times_llm_failure(self):
        """On LLM failure, uses fallback email template."""
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(
            side_effect=Exception("LLM error")
        )

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["pending_requests"] = [
            {
                "contact_email": "bob@acme.com",
                "contact_name": "Bob",
                "company_name": "Acme",
                "meeting_type": "demo",
            },
        ]

        result = await agent._node_propose_times(state)

        assert len(result["proposed_meetings"]) == 1
        assert "Bob" in result["proposed_meetings"][0]["body"]

    @pytest.mark.asyncio
    async def test_node_propose_times_generates_business_day_slots(self):
        """Proposed times should skip weekends."""
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text="Meeting invite body")]
        ))

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["pending_requests"] = [
            {
                "contact_email": "test@test.com",
                "contact_name": "Test",
                "company_name": "Test Co",
                "meeting_type": "discovery",
            },
        ]

        result = await agent._node_propose_times(state)

        proposal = result["proposed_meetings"][0]
        for time_str in proposal["proposed_times"]:
            dt = datetime.fromisoformat(time_str)
            assert dt.weekday() < 5  # Monday=0 ... Friday=4

    # ─── Node 3: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "proposed_meetings": [{"contact_email": "jane@acme.com"}],
        }
        result = await agent._node_human_review(state)

        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 4: Send Invites ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_send_invites_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["proposed_meetings"] = []

        result = await agent._node_send_invites(state)

        assert result["current_node"] == "send_invites"
        assert result["invites_sent"] == 0

    @pytest.mark.asyncio
    async def test_node_send_invites_with_proposals(self):
        agent = self._make_agent()
        mock_execute = MagicMock()
        agent.db.client.table.return_value.insert.return_value.execute = mock_execute

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["proposed_meetings"] = [
            {
                "contact_email": "jane@acme.com",
                "contact_name": "Jane Doe",
                "company_name": "Acme Corp",
                "meeting_type": "discovery",
                "duration_minutes": 30,
                "proposed_times": ["2025-02-10T10:00:00+00:00"],
                "subject": "Discovery Call",
                "body": "Hi Jane...",
            },
        ]

        result = await agent._node_send_invites(state)

        assert result["invites_sent"] == 1
        assert result["meetings_approved"] is True
        assert len(result["calendar_links"]) == 1
        assert result["knowledge_written"] is True

    @pytest.mark.asyncio
    async def test_node_send_invites_db_error(self):
        """DB errors are handled gracefully."""
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB error")

        state = agent._prepare_initial_state(
            {"meeting_requests": []}, "run-1"
        )
        state["proposed_meetings"] = [
            {
                "contact_email": "test@test.com",
                "contact_name": "Test",
                "company_name": "Test Co",
                "meeting_type": "discovery",
                "proposed_times": ["2025-02-10T10:00:00+00:00"],
                "subject": "Test",
                "body": "Test",
            },
        ]

        result = await agent._node_send_invites(state)

        # Still counts as sent (logging occurred)
        assert result["invites_sent"] == 1

    # ─── Node 5: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "total_requests": 3,
            "proposed_meetings": [{"contact": "a"}, {"contact": "b"}],
            "invites_sent": 2,
            "meetings_confirmed": 1,
            "meetings_cancelled": 0,
            "calendar_links": ["https://cal.app/1", "https://cal.app/2"],
        }
        result = await agent._node_report(state)

        assert result["current_node"] == "report"
        assert result["report_summary"] != ""
        assert result["report_generated_at"] != ""
        assert "Meeting Scheduler Report" in result["report_summary"]

    @pytest.mark.asyncio
    async def test_node_report_no_links(self):
        agent = self._make_agent()
        state = {
            "total_requests": 0,
            "proposed_meetings": [],
            "invites_sent": 0,
            "meetings_confirmed": 0,
            "meetings_cancelled": 0,
            "calendar_links": [],
        }
        result = await agent._node_report(state)

        assert result["current_node"] == "report"
        assert "Calendar Links" not in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent
        assert MeetingSchedulerAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent
        assert MeetingSchedulerAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent
        assert MeetingSchedulerAgent._route_after_review({}) == "approved"

    # ─── System Prompt ───────────────────────────────────────────────

    def test_system_prompt_default(self):
        agent = self._make_agent()
        prompt = agent._get_system_prompt()
        assert "meeting" in prompt.lower() or "scheduler" in prompt.lower()

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
# YAML Config Tests
# ══════════════════════════════════════════════════════════════════════


class TestMeetingSchedulerYAML:
    """Tests for meeting_scheduler.yaml configuration."""

    def _load_config(self):
        import yaml
        config_path = (
            Path(__file__).parent.parent.parent
            / "verticals"
            / "enclave_guard"
            / "agents"
            / "meeting_scheduler.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_agent_id(self):
        config = self._load_config()
        assert config["agent_id"] == "meeting_scheduler_v1"

    def test_agent_type(self):
        config = self._load_config()
        assert config["agent_type"] == "meeting_scheduler"

    def test_enabled(self):
        config = self._load_config()
        assert config["enabled"] is True

    def test_model_provider(self):
        config = self._load_config()
        assert config["model"]["provider"] == "anthropic"

    def test_human_gates_enabled(self):
        config = self._load_config()
        assert config["human_gates"]["enabled"] is True

    def test_schedule_trigger(self):
        config = self._load_config()
        assert config["schedule"]["trigger"] == "event"

    def test_params_default_duration(self):
        config = self._load_config()
        assert config["params"]["default_duration"] == 30

    def test_params_company_name(self):
        config = self._load_config()
        assert config["params"]["company_name"] == "Enclave Guard"
