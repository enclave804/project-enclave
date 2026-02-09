"""
Tests for FollowUpAgent — Phase 16: Sales Pipeline.

Covers:
    - FollowUpAgentState TypedDict
    - FollowUpAgent registration, construction, state class
    - Initial state preparation
    - Constants (MODES, SEQUENCE_STATUSES, DEFAULT_SUBJECTS)
    - All 5 nodes: load_sequences, generate_followups, human_review,
      send_followups, report
    - Graph construction and routing
    - System prompt
    - YAML config (followup.yaml)
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


class TestFollowUpState:
    """Tests for FollowUpAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import FollowUpAgentState
        assert FollowUpAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import FollowUpAgentState
        state: FollowUpAgentState = {
            "agent_id": "followup_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "followup_v1"

    def test_create_full(self):
        from core.agents.state import FollowUpAgentState
        state: FollowUpAgentState = {
            "agent_id": "followup_v1",
            "vertical_id": "enclave_guard",
            "active_sequences": [{"id": "seq_1"}],
            "due_sequences": [{"id": "seq_2"}],
            "total_sequences": 2,
            "draft_followups": [{"step": 1}],
            "followups_approved": False,
            "human_edits": [],
            "followups_sent": 0,
            "sequences_completed": 0,
            "sequences_paused": 0,
            "reply_detected": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["total_sequences"] == 2
        assert len(state["active_sequences"]) == 1
        assert state["followups_approved"] is False

    def test_sequence_tracking_fields(self):
        from core.agents.state import FollowUpAgentState
        state: FollowUpAgentState = {
            "followups_sent": 5,
            "sequences_completed": 2,
            "sequences_paused": 1,
            "reply_detected": True,
        }
        assert state["followups_sent"] == 5
        assert state["reply_detected"] is True

    def test_draft_followups_field(self):
        from core.agents.state import FollowUpAgentState
        state: FollowUpAgentState = {
            "draft_followups": [
                {"step": 1, "subject": "Following up", "body": "Hi there"},
                {"step": 2, "subject": "Quick check-in", "body": "Just checking"},
            ],
        }
        assert len(state["draft_followups"]) == 2
        assert state["draft_followups"][0]["step"] == 1


# ══════════════════════════════════════════════════════════════════════
# Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestFollowUpAgent:
    """Tests for FollowUpAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a FollowUpAgent with mocked dependencies."""
        from core.agents.implementations.followup_agent import FollowUpAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="followup_v1",
            agent_type="followup",
            name="Follow-Up Agent",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
                "max_steps": 5,
                "interval_days": 3,
                "min_confidence": 0.6,
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

        return FollowUpAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.followup_agent import FollowUpAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "followup" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.followup_agent import FollowUpAgent
        assert FollowUpAgent.agent_type == "followup"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import FollowUpAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is FollowUpAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        repr_str = repr(agent)
        assert "FollowUpAgent" in repr_str
        assert "followup_v1" in repr_str

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "check_due"}, "run-123"
        )
        assert state["active_sequences"] == []
        assert state["due_sequences"] == []
        assert state["total_sequences"] == 0
        assert state["draft_followups"] == []
        assert state["followups_approved"] is False
        assert state["followups_sent"] == 0
        assert state["sequences_completed"] == 0
        assert state["sequences_paused"] == 0
        assert state["reply_detected"] is False
        assert state["report_summary"] == ""

    def test_prepare_initial_state_human_edits(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        assert state["human_edits"] == []

    # ─── Constants ──────────────────────────────────────────────────

    def test_constants_modes(self):
        from core.agents.implementations import followup_agent
        assert "check_due" in followup_agent.MODES
        assert "create_sequence" in followup_agent.MODES
        assert "pause_all" in followup_agent.MODES
        assert "resume" in followup_agent.MODES

    def test_constants_statuses(self):
        from core.agents.implementations import followup_agent
        assert "active" in followup_agent.SEQUENCE_STATUSES
        assert "paused" in followup_agent.SEQUENCE_STATUSES
        assert "completed" in followup_agent.SEQUENCE_STATUSES
        assert "cancelled" in followup_agent.SEQUENCE_STATUSES
        assert "replied" in followup_agent.SEQUENCE_STATUSES

    def test_default_subjects(self):
        from core.agents.implementations import followup_agent
        subjects = followup_agent.DEFAULT_SUBJECTS
        assert len(subjects) == 5
        assert isinstance(subjects[0], str)

    def test_system_prompt_template(self):
        from core.agents.implementations import followup_agent
        prompt = followup_agent.FOLLOWUP_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{step}" in prompt
        assert "{max_steps}" in prompt

    # ─── Node 1: Load Sequences ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_sequences_empty(self):
        agent = self._make_agent()
        # Mock DB returning empty sequences
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        result = await agent._node_load_sequences(state)

        assert result["current_node"] == "load_sequences"
        assert result["active_sequences"] == []
        assert result["due_sequences"] == []
        assert result["total_sequences"] == 0

    @pytest.mark.asyncio
    async def test_node_load_sequences_with_due(self):
        agent = self._make_agent()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "seq_1",
                "status": "active",
                "current_step": 1,
                "next_send_at": past,
                "contact_email": "jane@acme.com",
            },
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        result = await agent._node_load_sequences(state)

        assert result["total_sequences"] == 1
        assert len(result["due_sequences"]) == 1

    @pytest.mark.asyncio
    async def test_node_load_sequences_new_sequence(self):
        """New sequence with step 0 and no next_send_at is due immediately."""
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "seq_new",
                "status": "active",
                "current_step": 0,
                "next_send_at": None,
                "contact_email": "new@acme.com",
            },
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        result = await agent._node_load_sequences(state)

        assert len(result["due_sequences"]) == 1

    @pytest.mark.asyncio
    async def test_node_load_sequences_db_error(self):
        """Database error is handled gracefully."""
        agent = self._make_agent()
        agent.db.client.table.side_effect = Exception("DB error")

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        result = await agent._node_load_sequences(state)

        assert result["current_node"] == "load_sequences"
        assert result["total_sequences"] == 0

    # ─── Node 2: Generate Follow-Ups ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_followups_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["due_sequences"] = []

        result = await agent._node_generate_followups(state)

        assert result["current_node"] == "generate_followups"
        assert result["draft_followups"] == []

    @pytest.mark.asyncio
    async def test_node_generate_followups_with_llm(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text="Hi Jane, just following up on our conversation...")]
        ))

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["due_sequences"] = [
            {
                "id": "seq_1",
                "current_step": 1,
                "max_steps": 5,
                "contact_name": "Jane",
                "contact_email": "jane@acme.com",
                "company_name": "Acme Corp",
            },
        ]

        result = await agent._node_generate_followups(state)

        assert result["current_node"] == "generate_followups"
        assert len(result["draft_followups"]) == 1
        assert result["draft_followups"][0]["contact_email"] == "jane@acme.com"
        assert result["draft_followups"][0]["step"] == 2

    @pytest.mark.asyncio
    async def test_node_generate_followups_skips_max_step(self):
        """Sequences at max_steps should be skipped."""
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["due_sequences"] = [
            {
                "id": "seq_done",
                "current_step": 5,
                "max_steps": 5,
                "contact_name": "Done",
                "contact_email": "done@acme.com",
                "company_name": "Acme",
            },
        ]

        result = await agent._node_generate_followups(state)

        assert result["draft_followups"] == []

    @pytest.mark.asyncio
    async def test_node_generate_followups_llm_failure_fallback(self):
        """On LLM failure, uses fallback template."""
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(
            side_effect=Exception("LLM unavailable")
        )

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["due_sequences"] = [
            {
                "id": "seq_1",
                "current_step": 0,
                "max_steps": 5,
                "contact_name": "Bob",
                "contact_email": "bob@acme.com",
                "company_name": "Acme",
            },
        ]

        result = await agent._node_generate_followups(state)

        assert len(result["draft_followups"]) == 1
        assert "Bob" in result["draft_followups"][0]["body"]

    # ─── Node 3: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "draft_followups": [{"step": 1, "subject": "Test"}],
        }
        result = await agent._node_human_review(state)

        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_empty_drafts(self):
        agent = self._make_agent()
        state = {"draft_followups": []}
        result = await agent._node_human_review(state)

        assert result["requires_human_approval"] is True

    # ─── Node 4: Send Follow-Ups ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_send_followups_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["draft_followups"] = []

        result = await agent._node_send_followups(state)

        assert result["current_node"] == "send_followups"
        assert result["followups_sent"] == 0
        assert result["followups_approved"] is True

    @pytest.mark.asyncio
    async def test_node_send_followups_with_drafts(self):
        agent = self._make_agent()
        mock_execute = MagicMock()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["draft_followups"] = [
            {
                "sequence_id": "seq_1",
                "step": 2,
                "subject": "Quick check-in",
                "body": "Hi, following up...",
                "contact_email": "jane@acme.com",
            },
        ]

        result = await agent._node_send_followups(state)

        assert result["followups_sent"] == 1
        assert result["knowledge_written"] is True

    @pytest.mark.asyncio
    async def test_node_send_followups_completes_sequence(self):
        """When step reaches max_steps, sequence is marked completed."""
        agent = self._make_agent()
        mock_execute = MagicMock()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["draft_followups"] = [
            {
                "sequence_id": "seq_1",
                "step": 5,  # max_steps default is 5
                "subject": "Last follow-up",
                "body": "Final message...",
                "contact_email": "jane@acme.com",
            },
        ]

        result = await agent._node_send_followups(state)

        assert result["sequences_completed"] == 1

    @pytest.mark.asyncio
    async def test_node_send_followups_db_error(self):
        """Database errors are handled gracefully."""
        agent = self._make_agent()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("DB write fail")

        state = agent._prepare_initial_state({"mode": "check_due"}, "run-1")
        state["draft_followups"] = [
            {
                "sequence_id": "seq_1",
                "step": 2,
                "subject": "Test",
                "body": "Test",
                "contact_email": "test@test.com",
            },
        ]

        result = await agent._node_send_followups(state)

        # Should still count as sent (logging happened)
        assert result["followups_sent"] == 1

    # ─── Node 5: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "total_sequences": 5,
            "due_sequences": [{"id": "s1"}, {"id": "s2"}],
            "draft_followups": [{"step": 1}],
            "followups_sent": 1,
            "sequences_completed": 0,
            "sequences_paused": 0,
            "reply_detected": False,
        }
        result = await agent._node_report(state)

        assert result["current_node"] == "report"
        assert result["report_summary"] != ""
        assert result["report_generated_at"] != ""
        assert "Follow-Up Report" in result["report_summary"]

    @pytest.mark.asyncio
    async def test_node_report_with_reply(self):
        agent = self._make_agent()
        state = {
            "total_sequences": 1,
            "due_sequences": [],
            "draft_followups": [],
            "followups_sent": 0,
            "sequences_completed": 0,
            "sequences_paused": 0,
            "reply_detected": True,
        }
        result = await agent._node_report(state)

        assert "Reply" in result["report_summary"] or "reply" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.followup_agent import FollowUpAgent
        assert FollowUpAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.followup_agent import FollowUpAgent
        assert FollowUpAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        """Missing human_approval_status defaults to approved."""
        from core.agents.implementations.followup_agent import FollowUpAgent
        assert FollowUpAgent._route_after_review({}) == "approved"

    # ─── System Prompt ───────────────────────────────────────────────

    def test_system_prompt_default(self):
        agent = self._make_agent()
        prompt = agent._get_system_prompt()
        assert "follow-up" in prompt.lower() or "follow up" in prompt.lower()

    def test_write_knowledge_exists(self):
        """write_knowledge method exists (no-op)."""
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
# YAML Config Tests
# ══════════════════════════════════════════════════════════════════════


class TestFollowUpYAML:
    """Tests for followup.yaml configuration."""

    def _load_config(self):
        import yaml
        config_path = (
            Path(__file__).parent.parent.parent
            / "verticals"
            / "enclave_guard"
            / "agents"
            / "followup.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_agent_id(self):
        config = self._load_config()
        assert config["agent_id"] == "followup_v1"

    def test_agent_type(self):
        config = self._load_config()
        assert config["agent_type"] == "followup"

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
        assert config["schedule"]["trigger"] == "scheduled"

    def test_params_max_steps(self):
        config = self._load_config()
        assert config["params"]["max_steps"] == 5

    def test_params_interval_days(self):
        config = self._load_config()
        assert config["params"]["interval_days"] == 3

    def test_params_company_name(self):
        config = self._load_config()
        assert config["params"]["company_name"] == "Enclave Guard"
