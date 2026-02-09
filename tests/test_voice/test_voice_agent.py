"""
Tests for the Voice Agent — The Receptionist / Sales Rep.

Covers:
- Agent registration as "voice" type
- LangGraph build with 10 nodes
- Node implementations (receive, transcribe, classify, handle_*, draft, review, execute, report)
- Intent classification with keyword matching
- Routing by intent and approval
- Human review gate
- Knowledge writing
- State preparation
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from core.agents.implementations.voice_agent import (
    INTENT_SALES,
    INTENT_SUPPORT,
    INTENT_UNKNOWN,
    INTENT_URGENT,
    SALES_KEYWORDS,
    SUPPORT_KEYWORDS,
    URGENT_KEYWORDS,
    VoiceAgent,
)
from core.agents.registry import AGENT_IMPLEMENTATIONS
from core.agents.state import VoiceAgentState
from core.config.agent_schema import AgentInstanceConfig, HumanGateConfig


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_config(**overrides):
    """Create a minimal agent config for testing."""
    defaults = dict(
        agent_id="voice_test",
        agent_type="voice",
        name="Test Voice Agent",
        tools=[],
        human_gates=HumanGateConfig(enabled=False, gate_before=[]),
    )
    defaults.update(overrides)
    config = AgentInstanceConfig(**defaults)
    config.vertical_id = "test_vertical"
    return config


def _make_agent(**config_overrides):
    """Create a VoiceAgent with mock dependencies (no LLM to avoid mock leakage)."""
    config = _make_config(**config_overrides)
    mock_db = MagicMock()
    mock_embedder = MagicMock()
    agent = VoiceAgent(
        config=config,
        db=mock_db,
        embedder=mock_embedder,
        anthropic_client=None,  # No LLM — tests keyword-based logic
    )
    return agent


# ═══════════════════════════════════════════════════════════════════════
# 1. Registration
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceAgentRegistration:
    """Tests that VoiceAgent registers correctly."""

    def test_registered_as_voice_type(self):
        assert "voice" in AGENT_IMPLEMENTATIONS
        assert AGENT_IMPLEMENTATIONS["voice"] is VoiceAgent

    def test_agent_type_attribute(self):
        assert VoiceAgent.agent_type == "voice"


# ═══════════════════════════════════════════════════════════════════════
# 2. Graph Construction
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceAgentGraph:
    """Tests for the LangGraph state machine."""

    def test_builds_graph_successfully(self):
        agent = _make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_graph_has_10_nodes(self):
        agent = _make_agent()
        graph = agent.build_graph()
        node_names = list(graph.nodes.keys())
        # LangGraph adds __start__ and __end__ nodes
        actual_nodes = [n for n in node_names if not n.startswith("__")]
        assert len(actual_nodes) == 10

    def test_graph_contains_expected_nodes(self):
        agent = _make_agent()
        graph = agent.build_graph()
        node_names = list(graph.nodes.keys())
        expected = [
            "receive", "transcribe", "classify_intent",
            "handle_sales", "handle_support", "handle_urgent",
            "draft_response", "human_review", "execute", "report",
        ]
        for node in expected:
            assert node in node_names, f"Missing node: {node}"

    def test_get_tools_returns_empty(self):
        agent = _make_agent()
        assert agent.get_tools() == []

    def test_get_state_class_returns_voice_state(self):
        agent = _make_agent()
        assert agent.get_state_class() is VoiceAgentState


# ═══════════════════════════════════════════════════════════════════════
# 3. Node: Receive
# ═══════════════════════════════════════════════════════════════════════


class TestNodeReceive:
    """Tests for the receive node."""

    def test_receive_voicemail(self):
        agent = _make_agent()
        state = VoiceAgentState(
            task_input={
                "mode": "process_voicemail",
                "caller": "+15551234567",
                "recording_url": "https://api.twilio.com/recording.mp3",
                "call_sid": "CA_123",
            },
        )
        result = _run(agent._node_receive(state))
        assert result["channel"] == "voicemail"
        assert result["caller_number"] == "+15551234567"
        assert result["recording_url"] == "https://api.twilio.com/recording.mp3"
        assert result["call_sid"] == "CA_123"

    def test_receive_sms(self):
        agent = _make_agent()
        state = VoiceAgentState(
            task_input={
                "mode": "process_sms",
                "from_number": "+15559876543",
                "body": "I want a quote",
                "message_sid": "SM_456",
            },
        )
        result = _run(agent._node_receive(state))
        assert result["channel"] == "sms"
        assert result["sms_body"] == "I want a quote"
        assert result["transcript"] == "I want a quote"  # SMS body becomes transcript

    def test_receive_unknown_channel(self):
        agent = _make_agent()
        state = VoiceAgentState(task_input={})
        result = _run(agent._node_receive(state))
        assert result["channel"] == "unknown"

    def test_receive_empty_task_input(self):
        agent = _make_agent()
        state = VoiceAgentState(task_input="not_a_dict")
        result = _run(agent._node_receive(state))
        assert result["channel"] == "unknown"


# ═══════════════════════════════════════════════════════════════════════
# 4. Node: Transcribe
# ═══════════════════════════════════════════════════════════════════════


class TestNodeTranscribe:
    """Tests for the transcribe node."""

    def test_skips_transcription_for_sms(self):
        agent = _make_agent()
        state = VoiceAgentState(channel="sms", transcript="I need help")
        result = _run(agent._node_transcribe(state))
        assert result["transcription_source"] == "sms_body"

    def test_returns_error_without_recording_url(self):
        agent = _make_agent()
        state = VoiceAgentState(channel="voicemail", recording_url="")
        result = _run(agent._node_transcribe(state))
        assert result["transcription_source"] == "none"
        assert "No recording URL" in result.get("transcription_error", "")

    def test_transcribes_recording(self):
        agent = _make_agent()
        state = VoiceAgentState(
            channel="voicemail",
            recording_url="https://api.twilio.com/rec.mp3",
        )

        mock_result = json.dumps({"text": "Hello, I need help", "source": "whisper"})
        with patch("core.mcp.tools.voice_tools.transcribe_audio", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            result = _run(agent._node_transcribe(state))

        assert result["transcript"] == "Hello, I need help"
        assert result["transcription_source"] == "whisper"

    def test_handles_transcription_failure(self):
        agent = _make_agent()
        state = VoiceAgentState(
            channel="voicemail",
            recording_url="https://api.twilio.com/rec.mp3",
        )

        with patch("core.mcp.tools.voice_tools.transcribe_audio", new_callable=AsyncMock) as mock_fn:
            mock_fn.side_effect = Exception("API timeout")
            result = _run(agent._node_transcribe(state))

        assert result["transcript"] == ""
        assert result["transcription_source"] == "failed"
        assert "API timeout" in result.get("transcription_error", "")


# ═══════════════════════════════════════════════════════════════════════
# 5. Node: Classify Intent
# ═══════════════════════════════════════════════════════════════════════


class TestNodeClassifyIntent:
    """Tests for keyword-based intent classification."""

    def test_empty_transcript_returns_unknown(self):
        agent = _make_agent()
        state = VoiceAgentState(transcript="")
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_UNKNOWN
        assert result["intent_confidence"] == 0.0

    def test_urgent_keywords_detected(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="This is urgent, we have a security incident and need help immediately"
        )
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_URGENT
        assert result["intent_confidence"] > 0.5

    def test_sales_keywords_detected(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="I'm interested in your services and would like to learn more about pricing"
        )
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_SALES

    def test_support_keywords_detected(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="I have an issue with my order tracking and need help with a return"
        )
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_SUPPORT

    def test_urgent_takes_priority_over_sales(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="This is urgent, I'm interested in purchasing but we've been hacked"
        )
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_URGENT

    def test_no_keywords_returns_unknown(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="The weather is nice today"
        )
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_UNKNOWN

    def test_confidence_caps_at_0_9(self):
        agent = _make_agent()
        # Many urgent keywords
        transcript = " ".join(URGENT_KEYWORDS)
        state = VoiceAgentState(transcript=transcript)
        result = _run(agent._node_classify_intent(state))
        assert result["intent_confidence"] <= 0.9

    def test_case_insensitive(self):
        agent = _make_agent()
        state = VoiceAgentState(transcript="I AM INTERESTED IN YOUR SERVICES")
        result = _run(agent._node_classify_intent(state))
        assert result["classified_intent"] == INTENT_SALES


# ═══════════════════════════════════════════════════════════════════════
# 6. Routing Functions
# ═══════════════════════════════════════════════════════════════════════


class TestRouting:
    """Tests for intent and approval routing."""

    def test_route_by_intent_sales(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_SALES)
        assert agent._route_by_intent(state) == INTENT_SALES

    def test_route_by_intent_support(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_SUPPORT)
        assert agent._route_by_intent(state) == INTENT_SUPPORT

    def test_route_by_intent_urgent(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_URGENT)
        assert agent._route_by_intent(state) == INTENT_URGENT

    def test_route_by_intent_unknown_default(self):
        agent = _make_agent()
        state = VoiceAgentState()
        assert agent._route_by_intent(state) == INTENT_UNKNOWN

    def test_route_by_approval_approved(self):
        agent = _make_agent()
        state = VoiceAgentState(actions_approved=True)
        assert agent._route_by_approval(state) == "approved"

    def test_route_by_approval_rejected(self):
        agent = _make_agent()
        state = VoiceAgentState(actions_approved=False)
        assert agent._route_by_approval(state) == "rejected"


# ═══════════════════════════════════════════════════════════════════════
# 7. Handler Nodes
# ═══════════════════════════════════════════════════════════════════════


class TestHandlerNodes:
    """Tests for handle_sales, handle_support, handle_urgent nodes."""

    def test_handle_sales_plans_sms_and_lead(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="I want a demo",
            caller_number="+15551234567",
            caller_name="Jane Doe",
            caller_company="Acme Corp",
        )
        result = _run(agent._node_handle_sales(state))
        assert result["response_type"] == "sms_reply"
        assert len(result["actions_planned"]) == 2

        actions = [a["action"] for a in result["actions_planned"]]
        assert "send_sms_reply" in actions
        assert "create_lead" in actions

        # Both require approval
        for action in result["actions_planned"]:
            assert action["requires_approval"] is True

    def test_handle_support_plans_sms_only(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="My order is broken",
            caller_number="+15551234567",
        )
        result = _run(agent._node_handle_support(state))
        assert result["response_type"] == "sms_reply"
        assert len(result["actions_planned"]) == 1
        assert result["actions_planned"][0]["action"] == "send_sms_reply"
        assert result["actions_planned"][0]["requires_approval"] is True

    def test_handle_urgent_plans_sms_and_escalation(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="Security breach!",
            caller_number="+15551234567",
        )
        result = _run(agent._node_handle_urgent(state))
        assert result["response_type"] == "escalate"
        assert result["urgency_level"] == 5
        assert len(result["actions_planned"]) == 2

        actions = {a["action"]: a for a in result["actions_planned"]}
        assert "send_sms_reply" in actions
        assert "escalate_to_overseer" in actions

        # SMS requires approval, escalation does NOT
        assert actions["send_sms_reply"]["requires_approval"] is True
        assert actions["escalate_to_overseer"]["requires_approval"] is False


# ═══════════════════════════════════════════════════════════════════════
# 8. Node: Draft Response
# ═══════════════════════════════════════════════════════════════════════


class TestNodeDraftResponse:
    """Tests for SMS response drafting."""

    def test_sales_template(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_SALES)
        result = _run(agent._node_draft_response(state))
        assert "interest" in result["draft_sms_reply"].lower() or "thank" in result["draft_sms_reply"].lower()
        assert "STOP" in result["draft_sms_reply"]

    def test_support_template(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_SUPPORT)
        result = _run(agent._node_draft_response(state))
        assert "support" in result["draft_sms_reply"].lower() or "looking into" in result["draft_sms_reply"].lower()

    def test_urgent_template(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_URGENT)
        result = _run(agent._node_draft_response(state))
        assert "escalated" in result["draft_sms_reply"].lower() or "30 minutes" in result["draft_sms_reply"].lower()

    def test_unknown_template(self):
        agent = _make_agent()
        state = VoiceAgentState(classified_intent=INTENT_UNKNOWN)
        result = _run(agent._node_draft_response(state))
        assert "STOP" in result["draft_sms_reply"]

    def test_personalized_greeting_with_name(self):
        agent = _make_agent()
        state = VoiceAgentState(
            classified_intent=INTENT_SUPPORT,
            caller_name="Jane Doe",
        )
        result = _run(agent._node_draft_response(state))
        assert "Hi Jane" in result["draft_sms_reply"]

    def test_generic_greeting_without_name(self):
        agent = _make_agent()
        state = VoiceAgentState(
            classified_intent=INTENT_SUPPORT,
            caller_name="",
        )
        result = _run(agent._node_draft_response(state))
        assert result["draft_sms_reply"].startswith("Hi,")


# ═══════════════════════════════════════════════════════════════════════
# 9. Node: Human Review
# ═══════════════════════════════════════════════════════════════════════


class TestNodeHumanReview:
    """Tests for the human review gate."""

    def test_passes_approval_status_through(self):
        agent = _make_agent()
        state = VoiceAgentState(actions_approved=True, actions_planned=[])
        result = _run(agent._node_human_review(state))
        assert result["actions_approved"] is True

    def test_not_approved_by_default(self):
        agent = _make_agent()
        state = VoiceAgentState(
            actions_planned=[{"action": "send_sms_reply", "requires_approval": True}]
        )
        result = _run(agent._node_human_review(state))
        assert result["actions_approved"] is False


# ═══════════════════════════════════════════════════════════════════════
# 10. Node: Execute
# ═══════════════════════════════════════════════════════════════════════


class TestNodeExecute:
    """Tests for action execution."""

    def test_executes_sms_reply(self):
        agent = _make_agent()
        state = VoiceAgentState(
            actions_planned=[{
                "action": "send_sms_reply",
                "target": "+15551234567",
                "requires_approval": True,
            }],
            draft_sms_reply="Thank you for calling!",
        )
        result = _run(agent._node_execute(state))
        assert len(result["actions_executed"]) == 1
        assert result["actions_executed"][0]["status"] == "dispatched"
        assert result["actions_executed"][0]["sms_body"] == "Thank you for calling!"

    def test_executes_create_lead(self):
        agent = _make_agent()
        state = VoiceAgentState(
            actions_planned=[{
                "action": "create_lead",
                "target": "outreach_agent",
                "details": {"caller_number": "+1", "source": "inbound_call"},
                "requires_approval": True,
            }],
            draft_sms_reply="",
        )
        result = _run(agent._node_execute(state))
        assert len(result["actions_executed"]) == 1
        assert result["actions_executed"][0]["status"] == "dispatched"

    def test_executes_escalation(self):
        agent = _make_agent()
        state = VoiceAgentState(
            actions_planned=[{
                "action": "escalate_to_overseer",
                "target": "overseer_agent",
                "details": {"urgency_level": 5},
                "requires_approval": False,
            }],
            draft_sms_reply="",
        )
        result = _run(agent._node_execute(state))
        assert len(result["actions_executed"]) == 1
        assert result["actions_executed"][0]["status"] == "dispatched"

    def test_skips_unknown_action(self):
        agent = _make_agent()
        state = VoiceAgentState(
            actions_planned=[{"action": "unknown_action"}],
            draft_sms_reply="",
        )
        result = _run(agent._node_execute(state))
        assert result["actions_executed"][0]["status"] == "skipped"

    def test_multiple_actions(self):
        agent = _make_agent()
        state = VoiceAgentState(
            actions_planned=[
                {"action": "send_sms_reply", "target": "+1", "requires_approval": True},
                {"action": "create_lead", "target": "outreach", "details": {}, "requires_approval": True},
            ],
            draft_sms_reply="Hi!",
        )
        result = _run(agent._node_execute(state))
        assert len(result["actions_executed"]) == 2

    def test_empty_actions_list(self):
        agent = _make_agent()
        state = VoiceAgentState(actions_planned=[], draft_sms_reply="")
        result = _run(agent._node_execute(state))
        assert result["actions_executed"] == []
        assert result["actions_failed"] == []


# ═══════════════════════════════════════════════════════════════════════
# 11. Node: Report
# ═══════════════════════════════════════════════════════════════════════


class TestNodeReport:
    """Tests for report generation."""

    def test_generates_markdown_report(self):
        agent = _make_agent()
        state = VoiceAgentState(
            channel="voicemail",
            caller_number="+15551234567",
            classified_intent=INTENT_SALES,
            intent_confidence=0.85,
            intent_reasoning="Sales keywords detected",
            transcript="I want to buy your product",
            actions_executed=[{"action": "send_sms_reply", "status": "dispatched"}],
            actions_failed=[],
        )
        result = _run(agent._node_report(state))
        assert "Voice Agent Report" in result["report_summary"]
        assert "voicemail" in result["report_summary"]
        assert "sales" in result["report_summary"]
        assert result["report_generated_at"] != ""

    def test_includes_transcript_preview(self):
        agent = _make_agent()
        state = VoiceAgentState(
            transcript="Hello this is a test transcript",
            classified_intent=INTENT_SUPPORT,
        )
        result = _run(agent._node_report(state))
        assert "Hello this is a test" in result["report_summary"]

    def test_includes_failed_actions(self):
        agent = _make_agent()
        state = VoiceAgentState(
            classified_intent=INTENT_UNKNOWN,
            actions_failed=[{"action": "send_sms_reply", "error": "Timeout"}],
        )
        result = _run(agent._node_report(state))
        assert "send_sms_reply" in result["report_summary"]


# ═══════════════════════════════════════════════════════════════════════
# 12. State Preparation
# ═══════════════════════════════════════════════════════════════════════


class TestPrepareState:
    """Tests for initial state preparation."""

    def test_prepare_state_sets_defaults(self):
        agent = _make_agent()
        state = agent.prepare_state({"mode": "process_sms", "body": "Hello"})
        assert state["agent_id"] == "voice_test"
        assert state["vertical_id"] == "test_vertical"
        assert state["channel"] == "unknown"
        assert state["classified_intent"] == "unknown"
        assert state["actions_approved"] is False
        assert state["actions_planned"] == []

    def test_prepare_state_includes_task_input(self):
        agent = _make_agent()
        task = {"mode": "process_voicemail", "recording_url": "https://example.com/rec.mp3"}
        state = agent.prepare_state(task)
        assert state["task_input"] == task


# ═══════════════════════════════════════════════════════════════════════
# 13. Knowledge Writing
# ═══════════════════════════════════════════════════════════════════════


class TestWriteKnowledge:
    """Tests for shared brain knowledge writing."""

    def test_writes_insight_for_meaningful_transcript(self):
        agent = _make_agent()
        # Mock the store_insight method
        agent.store_insight = MagicMock()

        state = VoiceAgentState(
            classified_intent=INTENT_SALES,
            transcript="I want to learn more about your services and pricing",
            channel="voicemail",
            caller_number="+15551234567",
            response_type="sms_reply",
            urgency_level=1,
        )
        agent.write_knowledge(state)
        agent.store_insight.assert_called_once()

        insight = agent.store_insight.call_args[0][0]
        assert insight.insight_type == "voice_interaction"
        assert "sales" in insight.title.lower()
        assert "voicemail" in insight.title.lower()

    def test_skips_short_transcript(self):
        agent = _make_agent()
        agent.store_insight = MagicMock()

        state = VoiceAgentState(
            classified_intent=INTENT_UNKNOWN,
            transcript="hi",
            channel="sms",
        )
        agent.write_knowledge(state)
        agent.store_insight.assert_not_called()

    def test_skips_empty_transcript(self):
        agent = _make_agent()
        agent.store_insight = MagicMock()

        state = VoiceAgentState(
            classified_intent=INTENT_SUPPORT,
            transcript="",
        )
        agent.write_knowledge(state)
        agent.store_insight.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# 14. Constants
# ═══════════════════════════════════════════════════════════════════════


class TestConstants:
    """Tests for intent constants and keyword lists."""

    def test_intent_constants_are_strings(self):
        assert isinstance(INTENT_SALES, str)
        assert isinstance(INTENT_SUPPORT, str)
        assert isinstance(INTENT_URGENT, str)
        assert isinstance(INTENT_UNKNOWN, str)

    def test_keyword_lists_are_populated(self):
        assert len(SALES_KEYWORDS) >= 10
        assert len(SUPPORT_KEYWORDS) >= 10
        assert len(URGENT_KEYWORDS) >= 5

    def test_keywords_are_lowercase(self):
        for kw in SALES_KEYWORDS + SUPPORT_KEYWORDS + URGENT_KEYWORDS:
            assert kw == kw.lower()
