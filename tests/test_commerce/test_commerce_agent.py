"""
Tests for the Commerce Agent — The Store Manager.

Covers:
- Agent registration and discovery
- Graph construction (8 nodes, conditional routing)
- Node behavior (monitor, triage, VIP, low stock, refund, review, execute, report)
- Triage priority routing (VIP > Refund > Low Stock > Routine)
- Human-in-the-loop gates (refunds ALWAYS require approval)
- Knowledge writing (commerce insights to shared brain)
- State preparation
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agents.implementations.commerce_agent import (
    ACTION_RESTOCK_ALERT,
    ACTION_REFUND_REVIEW,
    ACTION_ROUTINE,
    ACTION_VIP_FOLLOWUP,
    CommerceAgent,
    VIP_THRESHOLD,
    LOW_STOCK_THRESHOLD,
)
from core.agents.state import CommerceAgentState


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_config(**overrides):
    """Create a mock AgentInstanceConfig."""
    config = MagicMock()
    config.agent_id = overrides.get("agent_id", "commerce_test")
    config.agent_type = "commerce"
    config.name = "Commerce Agent Test"
    config.vertical_id = overrides.get("vertical_id", "test_vertical")
    config.model = MagicMock()
    config.model.name = "claude-sonnet-4-20250514"
    config.model.temperature = 0.0
    config.model.max_tokens = 4096
    config.tools = []
    config.browser_enabled = False
    config.human_gates = MagicMock()
    config.human_gates.enabled = True
    config.human_gates.gate_before = ["human_review"]
    config.schedule = MagicMock()
    config.schedule.trigger = "manual"
    config.system_prompt_path = None
    config.params = {
        "vip_threshold": 500.0,
        "low_stock_threshold": 5,
    }
    return config


def _make_agent(**kwargs):
    """Create a CommerceAgent with mock dependencies."""
    config = _mock_config(**kwargs)
    agent = CommerceAgent(
        config=config,
        db=MagicMock(),
        embedder=MagicMock(),
        anthropic_client=None,
        checkpointer=None,
        browser_tool=None,
        mcp_tools=None,
    )
    return agent


# ═══════════════════════════════════════════════════════════════════════
# 1. Agent Registration
# ═══════════════════════════════════════════════════════════════════════


class TestCommerceAgentRegistration:
    """Tests for agent type registration and discovery."""

    def test_registered_as_commerce(self):
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "commerce" in AGENT_IMPLEMENTATIONS
        assert AGENT_IMPLEMENTATIONS["commerce"] is CommerceAgent

    def test_agent_type_attribute(self):
        assert CommerceAgent.agent_type == "commerce"

    def test_instantiation(self):
        agent = _make_agent()
        assert agent.agent_id == "commerce_test"
        assert agent.vertical_id == "test_vertical"


# ═══════════════════════════════════════════════════════════════════════
# 2. Graph Construction
# ═══════════════════════════════════════════════════════════════════════


class TestCommerceAgentGraph:
    """Tests for LangGraph state machine construction."""

    def test_builds_graph(self):
        agent = _make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        agent = _make_agent()
        graph = agent.build_graph()
        # LangGraph compiled graph exposes nodes
        node_names = set(graph.nodes.keys())
        expected = {
            "monitor", "triage", "handle_vip", "handle_low_stock",
            "handle_refund", "human_review", "execute", "report",
            "__start__",
        }
        assert expected.issubset(node_names)

    def test_get_tools_returns_empty(self):
        """Commerce tools are accessed via MCP, not directly."""
        agent = _make_agent()
        assert agent.get_tools() == []

    def test_get_state_class(self):
        agent = _make_agent()
        assert agent.get_state_class() is CommerceAgentState


# ═══════════════════════════════════════════════════════════════════════
# 3. State Preparation
# ═══════════════════════════════════════════════════════════════════════


class TestCommerceAgentState:
    """Tests for initial state preparation."""

    def test_prepare_state_defaults(self):
        agent = _make_agent()
        state = agent.prepare_state({"mode": "full_check"})

        assert state["agent_id"] == "commerce_test"
        assert state["vertical_id"] == "test_vertical"
        assert state["current_node"] == "monitor"
        assert state["order_count"] == 0
        assert state["total_revenue"] == 0.0
        assert state["vip_count"] == 0
        assert state["low_stock_count"] == 0
        assert state["actions_approved"] is False
        assert state["triage_action"] == "routine"

    def test_prepare_state_with_refund(self):
        agent = _make_agent()
        state = agent.prepare_state({
            "mode": "refund_review",
            "order_id": "ORD-123",
            "amount": 49.99,
            "reason": "Defective product",
        })
        assert state["task_input"]["mode"] == "refund_review"
        assert state["task_input"]["order_id"] == "ORD-123"

    def test_prepare_state_has_empty_lists(self):
        agent = _make_agent()
        state = agent.prepare_state({})
        assert state["recent_orders"] == []
        assert state["vip_orders"] == []
        assert state["products"] == []
        assert state["low_stock_alerts"] == []
        assert state["actions_planned"] == []
        assert state["actions_executed"] == []
        assert state["actions_failed"] == []


# ═══════════════════════════════════════════════════════════════════════
# 4. Triage Node
# ═══════════════════════════════════════════════════════════════════════


class TestTriageNode:
    """Tests for the triage decision node."""

    def test_vip_takes_priority(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test",
            vertical_id="v",
            run_id="r",
            task_input={"mode": "full_check"},
            current_node="triage",
            vip_count=1,
            vip_orders=[{
                "customer_name": "Whale Corp",
                "total_price": 9999.0,
            }],
            low_stock_count=3,  # Even with low stock, VIP takes priority
        )

        result = _run(agent._node_triage(state))
        assert result["triage_action"] == ACTION_VIP_FOLLOWUP
        assert "VIP" in result["triage_reasoning"]

    def test_low_stock_when_no_vip(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test",
            vertical_id="v",
            run_id="r",
            task_input={"mode": "full_check"},
            current_node="triage",
            vip_count=0,
            low_stock_count=2,
        )

        result = _run(agent._node_triage(state))
        assert result["triage_action"] == ACTION_RESTOCK_ALERT
        assert "stock" in result["triage_reasoning"].lower()

    def test_routine_when_nothing_urgent(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test",
            vertical_id="v",
            run_id="r",
            task_input={"mode": "full_check"},
            current_node="triage",
            vip_count=0,
            low_stock_count=0,
        )

        result = _run(agent._node_triage(state))
        assert result["triage_action"] == ACTION_ROUTINE
        assert "normal" in result["triage_reasoning"].lower()

    def test_refund_review_mode(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test",
            vertical_id="v",
            run_id="r",
            task_input={"mode": "refund_review"},
            current_node="triage",
            vip_count=2,  # Even VIP doesn't override explicit refund request
        )

        result = _run(agent._node_triage(state))
        assert result["triage_action"] == ACTION_REFUND_REVIEW


# ═══════════════════════════════════════════════════════════════════════
# 5. Routing Functions
# ═══════════════════════════════════════════════════════════════════════


class TestRoutingFunctions:
    """Tests for graph routing functions."""

    def test_route_by_triage_vip(self):
        agent = _make_agent()
        state = {"triage_action": ACTION_VIP_FOLLOWUP}
        assert agent._route_by_triage(state) == ACTION_VIP_FOLLOWUP

    def test_route_by_triage_restock(self):
        agent = _make_agent()
        state = {"triage_action": ACTION_RESTOCK_ALERT}
        assert agent._route_by_triage(state) == ACTION_RESTOCK_ALERT

    def test_route_by_triage_refund(self):
        agent = _make_agent()
        state = {"triage_action": ACTION_REFUND_REVIEW}
        assert agent._route_by_triage(state) == ACTION_REFUND_REVIEW

    def test_route_by_triage_default_routine(self):
        agent = _make_agent()
        state = {}  # No triage_action set
        assert agent._route_by_triage(state) == ACTION_ROUTINE

    def test_route_by_approval_approved(self):
        agent = _make_agent()
        state = {"actions_approved": True}
        assert agent._route_by_approval(state) == "approved"

    def test_route_by_approval_rejected(self):
        agent = _make_agent()
        state = {"actions_approved": False}
        assert agent._route_by_approval(state) == "rejected"

    def test_route_by_approval_default_rejected(self):
        agent = _make_agent()
        state = {}
        assert agent._route_by_approval(state) == "rejected"


# ═══════════════════════════════════════════════════════════════════════
# 6. VIP Handler Node
# ═══════════════════════════════════════════════════════════════════════


class TestHandleVipNode:
    """Tests for the VIP follow-up handler."""

    def test_drafts_vip_email(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test",
            vertical_id="v",
            run_id="r",
            task_input={},
            current_node="handle_vip",
            vip_orders=[{
                "customer_email": "ceo@bigcorp.com",
                "customer_name": "Alice Johnson",
                "total_price": 9999.0,
            }],
        )

        result = _run(agent._node_handle_vip(state))
        assert result["vip_customer_email"] == "ceo@bigcorp.com"
        assert result["vip_customer_name"] == "Alice Johnson"
        assert result["vip_order_total"] == 9999.0
        assert len(result["actions_planned"]) == 1
        assert result["actions_planned"][0]["action"] == "send_vip_email"
        assert result["actions_planned"][0]["requires_approval"] is True

    def test_email_subject_includes_name(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_vip",
            vip_orders=[{
                "customer_email": "bob@corp.com",
                "customer_name": "Bob Smith",
                "total_price": 1000.0,
            }],
        )

        result = _run(agent._node_handle_vip(state))
        assert "Bob" in result["vip_email_subject"]

    def test_handles_no_vip_orders(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_vip",
            vip_orders=[],
        )

        result = _run(agent._node_handle_vip(state))
        assert result["actions_planned"] == []

    def test_vip_email_always_requires_approval(self):
        """Human-in-the-loop: VIP emails MUST require approval."""
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_vip",
            vip_orders=[{
                "customer_email": "whale@co.com",
                "customer_name": "Whale",
                "total_price": 50000.0,
            }],
        )

        result = _run(agent._node_handle_vip(state))
        for action in result["actions_planned"]:
            assert action["requires_approval"] is True, \
                "VIP emails must ALWAYS require human approval"


# ═══════════════════════════════════════════════════════════════════════
# 7. Low Stock Handler Node
# ═══════════════════════════════════════════════════════════════════════


class TestHandleLowStockNode:
    """Tests for the low stock restock handler."""

    def test_prepares_restock_actions(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_low_stock",
            low_stock_alerts=[
                {
                    "variant_id": "var_001c",
                    "product": "Classic Tee",
                    "variant": "Medium",
                    "quantity": 3,
                    "sku": "TEE-M",
                },
            ],
        )

        result = _run(agent._node_handle_low_stock(state))
        assert len(result["actions_planned"]) == 1
        action = result["actions_planned"][0]
        assert action["action"] == "restock"
        assert action["target"] == "var_001c"
        assert action["requires_approval"] is True
        assert action["details"]["recommended_quantity"] >= 50

    def test_restock_recommends_at_least_50(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_low_stock",
            low_stock_alerts=[
                {"variant_id": "v1", "product": "P", "variant": "V", "quantity": 2},
            ],
        )

        result = _run(agent._node_handle_low_stock(state))
        recommended = result["actions_planned"][0]["details"]["recommended_quantity"]
        assert recommended >= 50

    def test_multiple_alerts_multiple_actions(self):
        agent = _make_agent()
        alerts = [
            {"variant_id": f"v_{i}", "product": f"P{i}", "variant": "S", "quantity": i}
            for i in range(3)
        ]
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_low_stock",
            low_stock_alerts=alerts,
        )

        result = _run(agent._node_handle_low_stock(state))
        assert len(result["actions_planned"]) == 3

    def test_restock_always_requires_approval(self):
        """Human-in-the-loop: restocks require human sign-off."""
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="handle_low_stock",
            low_stock_alerts=[
                {"variant_id": "v1", "product": "P", "variant": "V", "quantity": 1},
            ],
        )

        result = _run(agent._node_handle_low_stock(state))
        for action in result["actions_planned"]:
            assert action["requires_approval"] is True


# ═══════════════════════════════════════════════════════════════════════
# 8. Refund Handler Node
# ═══════════════════════════════════════════════════════════════════════


class TestHandleRefundNode:
    """Tests for the refund handling node."""

    def test_drafts_refund(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r",
            task_input={
                "mode": "refund_review",
                "order_id": "ORD-999",
                "amount": 49.99,
                "reason": "Defective product",
                "payment_intent_id": "pi_refund123",
            },
            current_node="handle_refund",
        )

        result = _run(agent._node_handle_refund(state))
        assert result["refund_order_id"] == "ORD-999"
        assert result["refund_amount"] == 49.99
        assert result["refund_reason"] == "Defective product"
        assert len(result["actions_planned"]) == 1

    def test_refund_always_requires_approval(self):
        """CRITICAL: Refunds ALWAYS require human approval."""
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r",
            task_input={
                "mode": "refund_review",
                "order_id": "ORD-1",
                "amount": 100.0,
                "payment_intent_id": "pi_test",
            },
            current_node="handle_refund",
        )

        result = _run(agent._node_handle_refund(state))
        for action in result["actions_planned"]:
            assert action["requires_approval"] is True, \
                "Refunds must ALWAYS require human approval — no exceptions"

    def test_refund_with_missing_fields(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r",
            task_input={},  # No refund info — should use defaults
            current_node="handle_refund",
        )

        result = _run(agent._node_handle_refund(state))
        assert result["refund_order_id"] == ""
        assert result["refund_amount"] == 0.0
        assert "Customer requested" in result["refund_reason"]


# ═══════════════════════════════════════════════════════════════════════
# 9. Human Review Node
# ═══════════════════════════════════════════════════════════════════════


class TestHumanReviewNode:
    """Tests for the human review gate."""

    def test_not_auto_approved(self):
        """By default, actions are NOT approved."""
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="human_review",
            actions_planned=[
                {"action": "process_refund", "requires_approval": True}
            ],
            actions_approved=False,
        )

        result = _run(agent._node_human_review(state))
        assert result["actions_approved"] is False

    def test_pre_approved_passes_through(self):
        """If actions are pre-approved (e.g., test harness), pass through."""
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="human_review",
            actions_planned=[{"action": "restock", "requires_approval": True}],
            actions_approved=True,
        )

        result = _run(agent._node_human_review(state))
        assert result["actions_approved"] is True


# ═══════════════════════════════════════════════════════════════════════
# 10. Execute Node
# ═══════════════════════════════════════════════════════════════════════


class TestExecuteNode:
    """Tests for action execution."""

    def test_executes_vip_email(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="execute",
            actions_planned=[{
                "action": "send_vip_email",
                "target": "whale@corp.com",
                "details": {
                    "customer_name": "Whale",
                    "subject": "Thank you!",
                    "body": "You're amazing.",
                },
            }],
        )

        result = _run(agent._node_execute(state))
        assert len(result["actions_executed"]) == 1
        assert result["actions_executed"][0]["status"] == "dispatched"
        assert result["actions_failed"] == []

    def test_executes_restock(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="execute",
            actions_planned=[{
                "action": "restock",
                "target": "var_001c",
                "details": {"recommended_quantity": 50},
            }],
        )

        result = _run(agent._node_execute(state))
        assert result["actions_executed"][0]["status"] == "recommended"

    def test_executes_refund(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="execute",
            actions_planned=[{
                "action": "process_refund",
                "target": "pi_refund",
                "details": {"amount": 49.99},
            }],
        )

        result = _run(agent._node_execute(state))
        assert result["actions_executed"][0]["status"] == "dispatched"

    def test_skips_unknown_action(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="execute",
            actions_planned=[{
                "action": "launch_missiles",
                "target": "nowhere",
            }],
        )

        result = _run(agent._node_execute(state))
        assert result["actions_executed"][0]["status"] == "skipped"

    def test_multiple_actions(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="execute",
            actions_planned=[
                {"action": "send_vip_email", "target": "a@b.com", "details": {}},
                {"action": "restock", "target": "v1", "details": {}},
            ],
        )

        result = _run(agent._node_execute(state))
        assert len(result["actions_executed"]) == 2


# ═══════════════════════════════════════════════════════════════════════
# 11. Report Node
# ═══════════════════════════════════════════════════════════════════════


class TestReportNode:
    """Tests for the commerce status report generator."""

    def test_generates_markdown_report(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            order_count=5,
            total_revenue=1234.56,
            vip_count=1,
            low_stock_count=2,
            triage_action="vip_followup",
            triage_reasoning="VIP detected",
        )

        result = _run(agent._node_report(state))
        report = result["report_summary"]
        assert "Commerce Status Report" in report
        assert "$1,234.56" in report
        assert "5" in report
        assert result["report_generated_at"] != ""

    def test_report_includes_actions(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            order_count=1,
            total_revenue=100.0,
            actions_executed=[{"action": "send_vip_email", "status": "dispatched"}],
            actions_failed=[{"action": "restock", "error": "API timeout"}],
        )

        result = _run(agent._node_report(state))
        report = result["report_summary"]
        assert "✅" in report
        assert "❌" in report

    def test_report_includes_low_stock_details(self):
        agent = _make_agent()
        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            low_stock_count=1,
            low_stock_alerts=[{
                "product": "Classic Tee",
                "variant": "Medium",
                "quantity": 3,
            }],
        )

        result = _run(agent._node_report(state))
        report = result["report_summary"]
        assert "Classic Tee" in report
        assert "3 remaining" in report


# ═══════════════════════════════════════════════════════════════════════
# 12. Knowledge Writing
# ═══════════════════════════════════════════════════════════════════════


class TestKnowledgeWriting:
    """Tests for commerce insight storage."""

    def test_writes_when_revenue_positive(self):
        agent = _make_agent()
        agent.store_insight = MagicMock()

        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            total_revenue=500.0,
            order_count=3,
            vip_count=0,
            low_stock_count=0,
            triage_action="routine",
        )

        agent.write_knowledge(state)
        agent.store_insight.assert_called_once()
        insight = agent.store_insight.call_args[0][0]
        assert insight.insight_type == "commerce_activity"
        assert "$500.00" in insight.title

    def test_writes_when_vip_detected(self):
        agent = _make_agent()
        agent.store_insight = MagicMock()

        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            total_revenue=0.0,
            order_count=0,
            vip_count=1,
            triage_action="vip_followup",
        )

        agent.write_knowledge(state)
        agent.store_insight.assert_called_once()

    def test_skips_when_no_activity(self):
        agent = _make_agent()
        agent.store_insight = MagicMock()

        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            total_revenue=0.0,
            order_count=0,
            vip_count=0,
        )

        agent.write_knowledge(state)
        agent.store_insight.assert_not_called()

    def test_insight_has_metadata(self):
        agent = _make_agent()
        agent.store_insight = MagicMock()

        state = CommerceAgentState(
            agent_id="test", vertical_id="v", run_id="r", task_input={},
            current_node="report",
            total_revenue=1000.0,
            order_count=5,
            vip_count=2,
            triage_action="vip_followup",
        )

        agent.write_knowledge(state)
        insight = agent.store_insight.call_args[0][0]
        assert insight.metadata["order_count"] == 5
        assert insight.metadata["revenue"] == 1000.0
        assert insight.metadata["vip_count"] == 2
        assert insight.metadata["triage_action"] == "vip_followup"


# ═══════════════════════════════════════════════════════════════════════
# 13. Constants & Safety Invariants
# ═══════════════════════════════════════════════════════════════════════


class TestSafetyInvariants:
    """Tests that critical safety rules cannot be violated."""

    def test_vip_threshold_is_500(self):
        assert VIP_THRESHOLD == 500.0

    def test_low_stock_threshold_is_5(self):
        assert LOW_STOCK_THRESHOLD == 5

    def test_refund_action_names(self):
        assert ACTION_REFUND_REVIEW == "refund_review"

    def test_all_handler_nodes_require_approval(self):
        """Verify that all handler nodes produce actions with requires_approval=True."""
        agent = _make_agent()

        # VIP handler
        vip_state = CommerceAgentState(
            agent_id="t", vertical_id="v", run_id="r", task_input={},
            current_node="handle_vip",
            vip_orders=[{
                "customer_email": "a@b.com",
                "customer_name": "Test",
                "total_price": 1000.0,
            }],
        )
        vip_result = _run(agent._node_handle_vip(vip_state))
        for a in vip_result.get("actions_planned", []):
            assert a["requires_approval"] is True

        # Low stock handler
        stock_state = CommerceAgentState(
            agent_id="t", vertical_id="v", run_id="r", task_input={},
            current_node="handle_low_stock",
            low_stock_alerts=[
                {"variant_id": "v1", "product": "P", "variant": "V", "quantity": 1}
            ],
        )
        stock_result = _run(agent._node_handle_low_stock(stock_state))
        for a in stock_result.get("actions_planned", []):
            assert a["requires_approval"] is True

        # Refund handler
        refund_state = CommerceAgentState(
            agent_id="t", vertical_id="v", run_id="r",
            task_input={
                "mode": "refund_review",
                "order_id": "O1",
                "amount": 10,
                "payment_intent_id": "pi_x",
            },
            current_node="handle_refund",
        )
        refund_result = _run(agent._node_handle_refund(refund_state))
        for a in refund_result.get("actions_planned", []):
            assert a["requires_approval"] is True
