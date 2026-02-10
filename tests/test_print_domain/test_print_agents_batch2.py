"""
Tests for PrintBiz Domain Agents — Batch 2 (5 agents, ~100 tests).

Covers:
    1. PrintManagerAgent   (print_manager)
    2. PostProcessAgent    (post_process)
    3. QCInspectorAgent    (qc_inspector)
    4. CADAdvisorAgent     (cad_advisor)
    5. LogisticsAgent      (logistics)

Each agent is tested for:
    - State TypedDict construction
    - Registration in AGENT_IMPLEMENTATIONS
    - Construction with mocked deps
    - State class accessor
    - Initial state preparation
    - Constants / system prompt
    - Every node (async)
    - Graph construction and node keys
    - Routing function
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
#  1. PrintManagerAgent
# ══════════════════════════════════════════════════════════════════════


class TestPrintManagerState:
    """Tests for PrintManagerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import PrintManagerAgentState
        assert PrintManagerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import PrintManagerAgentState
        state: PrintManagerAgentState = {
            "agent_id": "pm_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "pm_v1"

    def test_create_full(self):
        from core.agents.state import PrintManagerAgentState
        state: PrintManagerAgentState = {
            "agent_id": "pm_v1",
            "vertical_id": "print_biz",
            "pending_jobs": [{"id": "j1"}],
            "available_printers": [{"printer_id": "p1"}],
            "active_prints": [],
            "queue_depth": 1,
            "job_assignments": [],
            "assignment_conflicts": [],
            "assignments_approved": False,
            "jobs_started": 0,
            "jobs_failed_to_start": 0,
            "throughput_summary": {},
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["queue_depth"] == 1
        assert state["assignments_approved"] is False
        assert len(state["pending_jobs"]) == 1

    def test_queue_depth_field(self):
        from core.agents.state import PrintManagerAgentState
        state: PrintManagerAgentState = {"queue_depth": 42}
        assert state["queue_depth"] == 42

    def test_throughput_summary_field(self):
        from core.agents.state import PrintManagerAgentState
        state: PrintManagerAgentState = {
            "throughput_summary": {"jobs_assigned": 5, "printer_utilization_pct": 80.0},
        }
        assert state["throughput_summary"]["jobs_assigned"] == 5


class TestPrintManagerAgent:
    """Tests for PrintManagerAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.print_manager_agent import PrintManagerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="pm_v1",
            agent_type="print_manager",
            name="Print Manager",
            vertical_id="print_biz",
            params={"company_name": "PrintBiz"},
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

        return PrintManagerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ──────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.print_manager_agent import PrintManagerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "print_manager" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.print_manager_agent import PrintManagerAgent
        assert PrintManagerAgent.agent_type == "print_manager"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import PrintManagerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is PrintManagerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "PrintManagerAgent" in r
        assert "pm_v1" in r

    # ─── Initial State ────────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["pending_jobs"] == []
        assert state["available_printers"] == []
        assert state["active_prints"] == []
        assert state["queue_depth"] == 0
        assert state["job_assignments"] == []
        assert state["assignment_conflicts"] == []
        assert state["assignments_approved"] is False
        assert state["jobs_started"] == 0
        assert state["jobs_failed_to_start"] == 0
        assert state["throughput_summary"] == {}
        assert state["report_summary"] == ""
        assert state["report_generated_at"] == ""

    # ─── Constants ────────────────────────────────────────────────────

    def test_constants_printer_types(self):
        from core.agents.implementations import print_manager_agent
        assert "fdm_standard" in print_manager_agent.PRINTER_TYPES
        assert "sla_standard" in print_manager_agent.PRINTER_TYPES
        assert "sls_industrial" in print_manager_agent.PRINTER_TYPES

    def test_constants_job_priorities(self):
        from core.agents.implementations import print_manager_agent
        assert print_manager_agent.JOB_PRIORITIES["rush"] == 1
        assert print_manager_agent.JOB_PRIORITIES["low"] == 4

    def test_constants_speed_multipliers(self):
        from core.agents.implementations import print_manager_agent
        assert print_manager_agent.SPEED_MULTIPLIERS["fast"] == 0.7
        assert print_manager_agent.SPEED_MULTIPLIERS["standard"] == 1.0
        assert print_manager_agent.SPEED_MULTIPLIERS["slow"] == 1.4

    def test_system_prompt(self):
        from core.agents.implementations import print_manager_agent
        assert "{pending_jobs}" in print_manager_agent.PRINT_MANAGER_SYSTEM_PROMPT
        assert "{available_printers}" in print_manager_agent.PRINT_MANAGER_SYSTEM_PROMPT

    # ─── Node 1: scan_queue ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_queue_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"scan_scope": "all"}, "run-1")
        result = await agent._node_scan_queue(state)
        assert result["current_node"] == "scan_queue"
        assert result["pending_jobs"] == []
        assert result["queue_depth"] == 0

    @pytest.mark.asyncio
    async def test_node_scan_queue_with_jobs(self):
        agent = self._make_agent()
        mock_pending = MagicMock()
        mock_pending.data = [
            {"id": "j1", "priority": "rush", "material": "PLA"},
            {"id": "j2", "priority": "normal", "material": "PETG"},
        ]
        mock_active = MagicMock()
        mock_active.data = [{"id": "j3", "status": "printing"}]

        agent.db.client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_pending
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_active

        state = agent._prepare_initial_state({}, "run-2")
        result = await agent._node_scan_queue(state)
        assert result["queue_depth"] == 2
        assert len(result["pending_jobs"]) == 2

    @pytest.mark.asyncio
    async def test_node_scan_queue_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-3")
        result = await agent._node_scan_queue(state)
        assert result["pending_jobs"] == []
        assert result["queue_depth"] == 0

    # ─── Node 2: assign_jobs ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_assign_jobs_no_pending(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["pending_jobs"] = []
        state["available_printers"] = [{"printer_id": "p1", "materials": ["PLA"], "build_volume_mm": [220, 220, 250], "speed": "standard"}]
        result = await agent._node_assign_jobs(state)
        assert result["current_node"] == "assign_jobs"
        assert result["job_assignments"] == []
        assert result["assignment_conflicts"] == []

    @pytest.mark.asyncio
    async def test_node_assign_jobs_no_printers(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["pending_jobs"] = [{"id": "j1", "material": "PLA", "priority": "normal"}]
        state["available_printers"] = []
        result = await agent._node_assign_jobs(state)
        assert len(result["assignment_conflicts"]) == 1
        assert result["job_assignments"] == []

    @pytest.mark.asyncio
    async def test_node_assign_jobs_successful_match(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-3")
        state["pending_jobs"] = [
            {"id": "j1", "material": "PLA", "dimensions": {"x": 100, "y": 100, "z": 50}, "priority": "rush", "volume_cm3": 30.0},
        ]
        state["available_printers"] = [
            {"printer_id": "p1", "type": "fdm_standard", "materials": ["PLA", "PETG"], "build_volume_mm": [220, 220, 250], "speed": "standard", "status": "idle"},
        ]
        state["active_prints"] = []
        result = await agent._node_assign_jobs(state)
        assert len(result["job_assignments"]) == 1
        assert result["job_assignments"][0]["job_id"] == "j1"
        assert result["job_assignments"][0]["printer_id"] == "p1"

    # ─── Node 3: human_review ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"job_assignments": [{"job_id": "j1"}], "assignment_conflicts": []}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 4: execute_assignments ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_execute_assignments_success(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = MagicMock()

        state = agent._prepare_initial_state({}, "run-1")
        state["job_assignments"] = [
            {"job_id": "j1", "printer_id": "p1", "estimated_hours": 3.5},
        ]
        result = await agent._node_execute_assignments(state)
        assert result["current_node"] == "execute_assignments"
        assert result["assignments_approved"] is True
        assert result["jobs_started"] == 1
        assert result["jobs_failed_to_start"] == 0

    @pytest.mark.asyncio
    async def test_node_execute_assignments_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-2")
        state["job_assignments"] = [
            {"job_id": "j1", "printer_id": "p1", "estimated_hours": 2.0},
        ]
        result = await agent._node_execute_assignments(state)
        assert result["jobs_started"] == 0
        assert result["jobs_failed_to_start"] == 1

    # ─── Node 5: report ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "job_assignments": [
                {"job_id": "j1", "printer_id": "p1", "material": "PLA", "estimated_hours": 3.0},
            ],
            "assignment_conflicts": [],
            "jobs_started": 1,
            "jobs_failed_to_start": 0,
            "queue_depth": 1,
            "active_prints": [],
            "available_printers": [{"printer_id": "p1"}],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Print Farm Manager Report" in result["report_summary"]
        assert result["report_generated_at"] != ""
        assert result["throughput_summary"]["jobs_assigned"] == 1

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.print_manager_agent import PrintManagerAgent
        state = {"human_approval_status": "approved"}
        result = PrintManagerAgent._route_after_review(state)
        assert result == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.print_manager_agent import PrintManagerAgent
        state = {"human_approval_status": "rejected"}
        result = PrintManagerAgent._route_after_review(state)
        assert result == "rejected"

    def test_route_default(self):
        from core.agents.implementations.print_manager_agent import PrintManagerAgent
        result = PrintManagerAgent._route_after_review({})
        assert result == "approved"

    # ─── Graph ────────────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "scan_queue", "assign_jobs", "human_review",
            "execute_assignments", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  2. PostProcessAgent
# ══════════════════════════════════════════════════════════════════════


class TestPostProcessState:
    """Tests for PostProcessAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import PostProcessAgentState
        assert PostProcessAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import PostProcessAgentState
        state: PostProcessAgentState = {
            "agent_id": "pp_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "pp_v1"

    def test_create_full(self):
        from core.agents.state import PostProcessAgentState
        state: PostProcessAgentState = {
            "agent_id": "pp_v1",
            "vertical_id": "print_biz",
            "print_job_id": "job_123",
            "print_technology": "FDM",
            "material": "PLA",
            "finish_requirement": "premium",
            "recommended_steps": [],
            "total_estimated_minutes": 90,
            "finish_level_score": 3,
            "work_order": {},
            "work_order_id": "wo_abc",
            "work_order_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["total_estimated_minutes"] == 90
        assert state["finish_level_score"] == 3

    def test_finish_requirement_field(self):
        from core.agents.state import PostProcessAgentState
        state: PostProcessAgentState = {"finish_requirement": "exhibition"}
        assert state["finish_requirement"] == "exhibition"

    def test_work_order_id_field(self):
        from core.agents.state import PostProcessAgentState
        state: PostProcessAgentState = {"work_order_id": "wo_xyz", "work_order_saved": True}
        assert state["work_order_id"] == "wo_xyz"
        assert state["work_order_saved"] is True


class TestPostProcessAgent:
    """Tests for PostProcessAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.post_process_agent import PostProcessAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="pp_v1",
            agent_type="post_process",
            name="Post Process",
            vertical_id="print_biz",
            params={"company_name": "PrintBiz"},
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

        return PostProcessAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ──────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.post_process_agent import PostProcessAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "post_process" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.post_process_agent import PostProcessAgent
        assert PostProcessAgent.agent_type == "post_process"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import PostProcessAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is PostProcessAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "PostProcessAgent" in r
        assert "pp_v1" in r

    # ─── Initial State ────────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["print_job_id"] == ""
        assert state["print_technology"] == ""
        assert state["material"] == ""
        assert state["finish_requirement"] == "standard"
        assert state["recommended_steps"] == []
        assert state["total_estimated_minutes"] == 0
        assert state["finish_level_score"] == 0
        assert state["work_order"] == {}
        assert state["work_order_id"] == ""
        assert state["work_order_saved"] is False
        assert state["report_summary"] == ""

    # ─── Constants ────────────────────────────────────────────────────

    def test_constants_finishing_steps(self):
        from core.agents.implementations import post_process_agent
        assert "FDM" in post_process_agent.FINISHING_STEPS
        assert "SLA" in post_process_agent.FINISHING_STEPS
        assert "SLS" in post_process_agent.FINISHING_STEPS

    def test_constants_finish_levels(self):
        from core.agents.implementations import post_process_agent
        assert post_process_agent.FINISH_LEVELS["raw"] == 0
        assert post_process_agent.FINISH_LEVELS["exhibition"] == 4

    def test_constants_finish_level_descriptions(self):
        from core.agents.implementations import post_process_agent
        descs = post_process_agent.FINISH_LEVEL_DESCRIPTIONS
        assert 0 in descs
        assert 4 in descs
        assert "Raw" in descs[0]
        assert "Exhibition" in descs[4]

    def test_system_prompt(self):
        from core.agents.implementations import post_process_agent
        assert "{technology}" in post_process_agent.POST_PROCESS_SYSTEM_PROMPT
        assert "{material}" in post_process_agent.POST_PROCESS_SYSTEM_PROMPT

    # ─── Node 1: load_job ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_job_from_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"id": "job_123", "technology": "SLA", "material": "SLA_Resin", "finish_requirement": "premium"}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"print_job_id": "job_123"}, "run-1")
        result = await agent._node_load_job(state)
        assert result["current_node"] == "load_job"
        assert result["print_job_id"] == "job_123"
        assert result["print_technology"] == "SLA"
        assert result["material"] == "SLA_Resin"

    @pytest.mark.asyncio
    async def test_node_load_job_from_task(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"print_job_id": "", "technology": "SLS", "material": "Nylon_SLS"}, "run-2")
        result = await agent._node_load_job(state)
        assert result["print_technology"] == "SLS"
        assert result["material"] == "Nylon_SLS"

    @pytest.mark.asyncio
    async def test_node_load_job_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({"print_job_id": "bad_id", "technology": "FDM"}, "run-3")
        result = await agent._node_load_job(state)
        assert result["print_technology"] == "FDM"

    # ─── Node 2: recommend_finishing ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_recommend_finishing_fdm_raw(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["print_technology"] = "FDM"
        state["material"] = "PLA"
        state["finish_requirement"] = "raw"
        state["finish_level_score"] = 0
        result = await agent._node_recommend_finishing(state)
        assert result["current_node"] == "recommend_finishing"
        # Raw: only required steps
        required_steps = [s for s in result["recommended_steps"] if s["required"]]
        optional_steps = [s for s in result["recommended_steps"] if not s["required"]]
        assert len(required_steps) >= 1
        assert len(optional_steps) == 0

    @pytest.mark.asyncio
    async def test_node_recommend_finishing_sla_standard(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["print_technology"] = "SLA"
        state["material"] = "SLA_Resin"
        state["finish_requirement"] = "standard"
        state["finish_level_score"] = 2
        result = await agent._node_recommend_finishing(state)
        step_names = [s["step"] for s in result["recommended_steps"]]
        assert "washing" in step_names
        assert "uv_curing" in step_names
        assert result["total_estimated_minutes"] > 0

    @pytest.mark.asyncio
    async def test_node_recommend_finishing_exhibition(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({"recommended_steps": [], "total_minutes": 120, "special_considerations": "Handle with care"})
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-3")
        state["print_technology"] = "FDM"
        state["material"] = "PLA"
        state["finish_requirement"] = "exhibition"
        state["finish_level_score"] = 4
        result = await agent._node_recommend_finishing(state)
        # Exhibition includes all optional steps with 1.5x times
        assert len(result["recommended_steps"]) >= 3
        assert result["total_estimated_minutes"] > 0

    # ─── Node 3: generate_work_order ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_work_order(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()

        state = agent._prepare_initial_state({}, "run-1")
        state["print_job_id"] = "job_123"
        state["print_technology"] = "FDM"
        state["material"] = "PLA"
        state["finish_requirement"] = "standard"
        state["recommended_steps"] = [
            {"step": "support_removal", "est_minutes": 15, "required": True, "description": "Remove supports"},
            {"step": "sanding", "est_minutes": 30, "required": False, "description": "Sand surfaces"},
        ]
        state["total_estimated_minutes"] = 45
        result = await agent._node_generate_work_order(state)
        assert result["current_node"] == "generate_work_order"
        assert result["work_order_id"].startswith("wo_")
        assert result["work_order"]["total_estimated_minutes"] == 45
        assert len(result["work_order"]["steps"]) == 2

    # ─── Node 4: human_review ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"work_order": {"work_order_id": "wo_abc", "total_estimated_minutes": 60}}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: report ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "print_job_id": "job_123",
            "print_technology": "FDM",
            "material": "PLA",
            "finish_requirement": "standard",
            "recommended_steps": [
                {"step": "support_removal", "est_minutes": 15, "required": True},
                {"step": "sanding", "est_minutes": 30, "required": False},
            ],
            "total_estimated_minutes": 45,
            "work_order": {"work_order_id": "wo_abc", "status": "pending_approval"},
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Post-Processing Work Order Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.post_process_agent import PostProcessAgent
        state = {"human_approval_status": "approved"}
        result = PostProcessAgent._route_after_review(state)
        assert result == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.post_process_agent import PostProcessAgent
        state = {"human_approval_status": "rejected"}
        result = PostProcessAgent._route_after_review(state)
        assert result == "rejected"

    def test_route_default(self):
        from core.agents.implementations.post_process_agent import PostProcessAgent
        result = PostProcessAgent._route_after_review({})
        assert result == "approved"

    # ─── Graph ────────────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "load_job", "recommend_finishing", "generate_work_order",
            "human_review", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  3. QCInspectorAgent
# ══════════════════════════════════════════════════════════════════════


class TestQCInspectorState:
    """Tests for QCInspectorAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import QCInspectorAgentState
        assert QCInspectorAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import QCInspectorAgentState
        state: QCInspectorAgentState = {
            "agent_id": "qc_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "qc_v1"

    def test_create_full(self):
        from core.agents.state import QCInspectorAgentState
        state: QCInspectorAgentState = {
            "agent_id": "qc_v1",
            "vertical_id": "print_biz",
            "print_job_id": "job_456",
            "file_analysis_id": "fa_789",
            "expected_dimensions": {"x": 100, "y": 80, "z": 50},
            "expected_material": "PLA",
            "expected_technology": "FDM",
            "measured_dimensions": {"x": 100.2, "y": 79.8, "z": 50.1},
            "dimensional_deviations": [],
            "defects_found": [],
            "defect_count": 0,
            "dimensional_accuracy_score": 95.0,
            "surface_quality_score": 80.0,
            "structural_integrity_score": 85.0,
            "visual_appearance_score": 75.0,
            "overall_qc_score": 84.0,
            "qc_pass": True,
            "disposition": "ship",
            "disposition_reasoning": "All checks passed",
            "inspection_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["overall_qc_score"] == 84.0
        assert state["qc_pass"] is True
        assert state["disposition"] == "ship"

    def test_qc_pass_field(self):
        from core.agents.state import QCInspectorAgentState
        state: QCInspectorAgentState = {"qc_pass": False, "overall_qc_score": 45.0}
        assert state["qc_pass"] is False

    def test_disposition_field(self):
        from core.agents.state import QCInspectorAgentState
        state: QCInspectorAgentState = {"disposition": "rework", "disposition_reasoning": "Surface defects fixable"}
        assert state["disposition"] == "rework"


class TestQCInspectorAgent:
    """Tests for QCInspectorAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.qc_inspector_agent import QCInspectorAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="qc_v1",
            agent_type="qc_inspector",
            name="QC Inspector",
            vertical_id="print_biz",
            params={"company_name": "PrintBiz"},
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

        return QCInspectorAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ──────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.qc_inspector_agent import QCInspectorAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "qc_inspector" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.qc_inspector_agent import QCInspectorAgent
        assert QCInspectorAgent.agent_type == "qc_inspector"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import QCInspectorAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is QCInspectorAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "QCInspectorAgent" in r
        assert "qc_v1" in r

    # ─── Initial State ────────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["print_job_id"] == ""
        assert state["file_analysis_id"] == ""
        assert state["expected_dimensions"] == {}
        assert state["defects_found"] == []
        assert state["defect_count"] == 0
        assert state["overall_qc_score"] == 0.0
        assert state["qc_pass"] is False
        assert state["disposition"] == ""
        assert state["inspection_saved"] is False

    # ─── Constants ────────────────────────────────────────────────────

    def test_constants_qc_criteria(self):
        from core.agents.implementations import qc_inspector_agent
        criteria = qc_inspector_agent.QC_CRITERIA
        assert "dimensional_accuracy" in criteria
        assert "surface_quality" in criteria
        assert "structural_integrity" in criteria
        assert "visual_appearance" in criteria
        total_weight = sum(c["weight"] for c in criteria.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_constants_defect_types(self):
        from core.agents.implementations import qc_inspector_agent
        assert "warping" in qc_inspector_agent.DEFECT_TYPES
        assert "layer_separation" in qc_inspector_agent.DEFECT_TYPES
        assert qc_inspector_agent.DEFECT_TYPES["layer_separation"]["severity"] == "critical"

    def test_constants_qc_pass_threshold(self):
        from core.agents.implementations import qc_inspector_agent
        assert qc_inspector_agent.QC_PASS_THRESHOLD == 70.0

    def test_constants_disposition_rules(self):
        from core.agents.implementations import qc_inspector_agent
        rules = qc_inspector_agent.QC_DISPOSITION_RULES
        assert "ship" in rules
        assert "rework" in rules
        assert "reprint" in rules
        assert "scrap" in rules
        assert rules["ship"]["min_score"] == 70

    def test_system_prompt(self):
        from core.agents.implementations import qc_inspector_agent
        assert "{technology}" in qc_inspector_agent.QC_SYSTEM_PROMPT
        assert "{material}" in qc_inspector_agent.QC_SYSTEM_PROMPT

    # ─── Node 1: load_specs ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_specs_from_db(self):
        agent = self._make_agent()
        mock_job = MagicMock()
        mock_job.data = [{"id": "job_456", "material": "PLA", "technology": "FDM", "dimensions": {"x": 100, "y": 80, "z": 50}, "file_analysis_id": "fa_789"}]
        mock_analysis = MagicMock()
        mock_analysis.data = [{"id": "fa_789", "bounding_box": {"x": 100, "y": 80, "z": 50}}]

        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = [mock_job, mock_analysis]

        state = agent._prepare_initial_state({"print_job_id": "job_456"}, "run-1")
        result = await agent._node_load_specs(state)
        assert result["current_node"] == "load_specs"
        assert result["print_job_id"] == "job_456"
        assert result["expected_material"] == "PLA"

    @pytest.mark.asyncio
    async def test_node_load_specs_from_task(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({
            "print_job_id": "",
            "material": "PETG",
            "technology": "FDM",
            "expected_dimensions": {"x": 200, "y": 150, "z": 100},
        }, "run-2")
        result = await agent._node_load_specs(state)
        assert result["expected_material"] == "PETG"
        assert result["expected_technology"] == "FDM"

    @pytest.mark.asyncio
    async def test_node_load_specs_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({"print_job_id": "bad_id", "material": "ABS"}, "run-3")
        result = await agent._node_load_specs(state)
        assert result["expected_material"] == "ABS"

    # ─── Node 2: run_inspection ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_run_inspection_no_defects(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "defects": [],
            "surface_quality_score": 90,
            "structural_integrity_score": 92,
            "visual_appearance_score": 88,
            "inspector_notes": "Clean part",
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["expected_dimensions"] = {"x": 100, "y": 80, "z": 50}
        state["measured_dimensions"] = {"x": 100.1, "y": 80.0, "z": 50.2}
        state["expected_material"] = "PLA"
        state["expected_technology"] = "FDM"
        result = await agent._node_run_inspection(state)
        assert result["current_node"] == "run_inspection"
        assert result["surface_quality_score"] == 90
        assert result["defect_count"] == 0

    @pytest.mark.asyncio
    async def test_node_run_inspection_with_defects(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "defects": [
                {"type": "warping", "severity": "high", "description": "Corner warping", "location": "base", "remediation": "Reprint with brim"},
            ],
            "surface_quality_score": 60,
            "structural_integrity_score": 70,
            "visual_appearance_score": 55,
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-2")
        state["expected_dimensions"] = {"x": 100, "y": 80, "z": 50}
        state["measured_dimensions"] = {"x": 100, "y": 80, "z": 50}
        state["expected_material"] = "PLA"
        state["expected_technology"] = "FDM"
        result = await agent._node_run_inspection(state)
        assert result["defect_count"] >= 1
        types = [d["type"] for d in result["defects_found"]]
        assert "warping" in types

    @pytest.mark.asyncio
    async def test_node_run_inspection_dimensional_deviation(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "defects": [],
            "surface_quality_score": 85,
            "structural_integrity_score": 90,
            "visual_appearance_score": 80,
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-3")
        state["expected_dimensions"] = {"x": 100, "y": 80, "z": 50}
        state["measured_dimensions"] = {"x": 102, "y": 78, "z": 50}
        state["expected_material"] = "PLA"
        state["expected_technology"] = "FDM"
        result = await agent._node_run_inspection(state)
        # x deviation=2mm and y deviation=2mm, both > 0.5mm tolerance
        deviation_defects = [d for d in result["defects_found"] if d["type"] == "dimensional_deviation"]
        assert len(deviation_defects) >= 1

    # ─── Node 3: score_quality ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_score_quality_pass(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["dimensional_deviations"] = [
            {"axis": "x", "deviation_mm": 0.2, "within_tolerance": True},
            {"axis": "y", "deviation_mm": 0.1, "within_tolerance": True},
        ]
        state["defects_found"] = []
        state["surface_quality_score"] = 85.0
        state["structural_integrity_score"] = 90.0
        state["visual_appearance_score"] = 80.0
        result = await agent._node_score_quality(state)
        assert result["current_node"] == "score_quality"
        assert result["qc_pass"] is True
        assert result["disposition"] == "ship"
        assert result["overall_qc_score"] >= 70.0

    @pytest.mark.asyncio
    async def test_node_score_quality_fail_critical_defect(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["dimensional_deviations"] = []
        state["defects_found"] = [
            {"type": "layer_separation", "severity": "critical"},
        ]
        state["surface_quality_score"] = 60.0
        state["structural_integrity_score"] = 50.0
        state["visual_appearance_score"] = 55.0
        result = await agent._node_score_quality(state)
        assert result["qc_pass"] is False
        assert result["disposition"] in ("reprint", "scrap")

    # ─── Node 4: human_review ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"overall_qc_score": 75.0, "disposition": "ship", "defect_count": 1}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: report ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = MagicMock()

        state = {
            "print_job_id": "job_456",
            "expected_technology": "FDM",
            "expected_material": "PLA",
            "defects_found": [{"type": "stringing", "severity": "low", "description": "Minor stringing"}],
            "dimensional_deviations": [{"axis": "x", "expected_mm": 100, "measured_mm": 100.2, "deviation_mm": 0.2, "within_tolerance": True}],
            "overall_qc_score": 82.0,
            "qc_pass": True,
            "disposition": "ship",
            "disposition_reasoning": "Score above threshold",
            "dimensional_accuracy_score": 95.0,
            "surface_quality_score": 80.0,
            "structural_integrity_score": 85.0,
            "visual_appearance_score": 75.0,
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Quality Control Inspection Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.qc_inspector_agent import QCInspectorAgent
        state = {"human_approval_status": "approved"}
        result = QCInspectorAgent._route_after_review(state)
        assert result == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.qc_inspector_agent import QCInspectorAgent
        state = {"human_approval_status": "rejected"}
        result = QCInspectorAgent._route_after_review(state)
        assert result == "rejected"

    def test_route_default(self):
        from core.agents.implementations.qc_inspector_agent import QCInspectorAgent
        result = QCInspectorAgent._route_after_review({})
        assert result == "approved"

    # ─── Graph ────────────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "load_specs", "run_inspection", "score_quality",
            "human_review", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  4. CADAdvisorAgent
# ══════════════════════════════════════════════════════════════════════


class TestCADAdvisorState:
    """Tests for CADAdvisorAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import CADAdvisorAgentState
        assert CADAdvisorAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import CADAdvisorAgentState
        state: CADAdvisorAgentState = {
            "agent_id": "cad_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "cad_v1"

    def test_create_full(self):
        from core.agents.state import CADAdvisorAgentState
        state: CADAdvisorAgentState = {
            "agent_id": "cad_v1",
            "vertical_id": "print_biz",
            "print_job_id": "job_789",
            "file_analysis_id": "fa_123",
            "design_file_name": "building.stl",
            "consultation_notes": "Fine facade detail needed",
            "target_technology": "SLA",
            "target_scale": "1:100",
            "printability_issues": [],
            "design_warnings": [],
            "printability_score": 85.0,
            "advisory_report": "",
            "suggestions": [],
            "architecture_tips_applied": ["base_stability"],
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["printability_score"] == 85.0
        assert state["target_technology"] == "SLA"
        assert "base_stability" in state["architecture_tips_applied"]

    def test_printability_score_field(self):
        from core.agents.state import CADAdvisorAgentState
        state: CADAdvisorAgentState = {"printability_score": 72.5}
        assert state["printability_score"] == 72.5

    def test_suggestions_field(self):
        from core.agents.state import CADAdvisorAgentState
        state: CADAdvisorAgentState = {
            "suggestions": [{"category": "geometry", "suggestion": "Hollow the base"}],
        }
        assert len(state["suggestions"]) == 1


class TestCADAdvisorAgent:
    """Tests for CADAdvisorAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.cad_advisor_agent import CADAdvisorAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="cad_v1",
            agent_type="cad_advisor",
            name="CAD Advisor",
            vertical_id="print_biz",
            params={"company_name": "PrintBiz"},
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

        return CADAdvisorAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ──────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.cad_advisor_agent import CADAdvisorAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "cad_advisor" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.cad_advisor_agent import CADAdvisorAgent
        assert CADAdvisorAgent.agent_type == "cad_advisor"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import CADAdvisorAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is CADAdvisorAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "CADAdvisorAgent" in r
        assert "cad_v1" in r

    # ─── Initial State ────────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["print_job_id"] == ""
        assert state["file_analysis_id"] == ""
        assert state["design_file_name"] == ""
        assert state["consultation_notes"] == ""
        assert state["target_technology"] == "FDM"
        assert state["target_scale"] == "1:100"
        assert state["printability_issues"] == []
        assert state["design_warnings"] == []
        assert state["printability_score"] == 0.0
        assert state["advisory_report"] == ""
        assert state["suggestions"] == []
        assert state["architecture_tips_applied"] == []

    # ─── Constants ────────────────────────────────────────────────────

    def test_constants_printability_rules(self):
        from core.agents.implementations import cad_advisor_agent
        rules = cad_advisor_agent.PRINTABILITY_RULES
        assert "min_wall_thickness_mm" in rules
        assert "max_overhang_angle" in rules
        assert rules["min_wall_thickness_mm"]["value"] == 0.8

    def test_constants_architecture_tips(self):
        from core.agents.implementations import cad_advisor_agent
        tips = cad_advisor_agent.ARCHITECTURE_TIPS
        assert "detachable_features" in tips
        assert "hollow_sections" in tips
        assert "base_stability" in tips
        assert "drainage_holes" in tips

    def test_constants_technology_constraints(self):
        from core.agents.implementations import cad_advisor_agent
        constraints = cad_advisor_agent.TECHNOLOGY_CONSTRAINTS
        assert "FDM" in constraints
        assert "SLA" in constraints
        assert "SLS" in constraints
        assert "MJF" in constraints
        assert constraints["SLA"]["min_wall_mm"] == 0.5

    def test_system_prompt(self):
        from core.agents.implementations import cad_advisor_agent
        assert "{file_name}" in cad_advisor_agent.CAD_ADVISOR_SYSTEM_PROMPT
        assert "{technology}" in cad_advisor_agent.CAD_ADVISOR_SYSTEM_PROMPT

    # ─── Node 1: load_consultation ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_consultation_from_db(self):
        agent = self._make_agent()
        mock_analysis = MagicMock()
        mock_analysis.data = [{"id": "fa_123", "file_name": "tower.stl"}]
        mock_job = MagicMock()
        mock_job.data = [{"id": "job_789", "technology": "SLA", "scale": "1:50"}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = [mock_analysis, mock_job]

        state = agent._prepare_initial_state({
            "print_job_id": "job_789",
            "file_analysis_id": "fa_123",
            "consultation_notes": "Need facade detail",
        }, "run-1")
        result = await agent._node_load_consultation(state)
        assert result["current_node"] == "load_consultation"
        assert result["design_file_name"] == "tower.stl"
        assert result["target_technology"] == "SLA"
        assert result["consultation_notes"] == "Need facade detail"

    @pytest.mark.asyncio
    async def test_node_load_consultation_from_task_only(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({
            "file_name": "house.stl",
            "technology": "FDM",
            "scale": "1:200",
        }, "run-2")
        result = await agent._node_load_consultation(state)
        assert result["design_file_name"] == "house.stl"
        assert result["target_technology"] == "FDM"
        assert result["target_scale"] == "1:200"

    @pytest.mark.asyncio
    async def test_node_load_consultation_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({
            "file_analysis_id": "fa_bad",
            "technology": "SLS",
        }, "run-3")
        result = await agent._node_load_consultation(state)
        assert result["target_technology"] == "SLS"

    # ─── Node 2: analyze_design ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_design_no_issues(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"min_wall_thickness_mm": 1.5, "min_detail_mm": 0.6, "overhang_percentage": 10, "is_manifold": True, "is_watertight": True, "volume_cm3": 100}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["file_analysis_id"] = "fa_123"
        state["target_technology"] = "FDM"
        state["target_scale"] = "1:100"
        state["design_file_name"] = "good.stl"
        state["consultation_notes"] = ""
        result = await agent._node_analyze_design(state)
        assert result["current_node"] == "analyze_design"
        assert result["printability_score"] > 50.0

    @pytest.mark.asyncio
    async def test_node_analyze_design_thin_walls(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"min_wall_thickness_mm": 0.3, "min_detail_mm": 0.1, "overhang_percentage": 50, "is_manifold": False, "is_watertight": False, "volume_cm3": 50}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-2")
        state["file_analysis_id"] = "fa_456"
        state["target_technology"] = "FDM"
        state["target_scale"] = "1:100"
        state["design_file_name"] = "bad.stl"
        state["consultation_notes"] = ""
        result = await agent._node_analyze_design(state)
        assert len(result["printability_issues"]) >= 3
        rules_violated = [i["rule"] for i in result["printability_issues"]]
        assert "min_wall_thickness_mm" in rules_violated
        assert "mesh_integrity" in rules_violated

    @pytest.mark.asyncio
    async def test_node_analyze_design_facade_detail(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"volume_cm3": 800, "min_wall_thickness_mm": 1.0, "is_manifold": True, "is_watertight": True}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-3")
        state["file_analysis_id"] = "fa_789"
        state["target_technology"] = "SLA"
        state["target_scale"] = "1:100"
        state["design_file_name"] = "facade.stl"
        state["consultation_notes"] = "Detailed facade texture needed"
        result = await agent._node_analyze_design(state)
        assert "layer_orientation" in result["architecture_tips_applied"]
        assert "hollow_sections" in result["architecture_tips_applied"]
        assert "drainage_holes" in result["architecture_tips_applied"]

    # ─── Node 3: generate_advisory ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_advisory(self):
        agent = self._make_agent()
        mock_analysis = MagicMock()
        mock_analysis.data = [{"id": "fa_123"}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_analysis

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "printability_issues": [],
            "design_warnings": [{"category": "material", "description": "PLA may warp", "impact": "Low", "mitigation": "Use enclosure"}],
            "printability_score": 80,
            "architecture_tips": ["base_stability"],
            "suggestions": [{"category": "orientation", "suggestion": "Rotate 15 degrees", "impact": "Better surface quality", "priority": "medium"}],
            "executive_summary": "Design is generally printable.",
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["target_technology"] = "FDM"
        state["target_scale"] = "1:100"
        state["design_file_name"] = "test.stl"
        state["consultation_notes"] = ""
        state["printability_issues"] = []
        state["architecture_tips_applied"] = ["base_stability"]
        state["file_analysis_id"] = "fa_123"
        state["printability_score"] = 90.0
        result = await agent._node_generate_advisory(state)
        assert result["current_node"] == "generate_advisory"
        assert len(result["suggestions"]) >= 1
        assert len(result["design_warnings"]) >= 1
        assert "Design Advisory Report" in result["advisory_report"]

    # ─── Node 4: human_review ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"printability_score": 75.0, "printability_issues": [{"rule": "test"}], "suggestions": []}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: report ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()

        state = {
            "print_job_id": "job_789",
            "file_analysis_id": "fa_123",
            "design_file_name": "building.stl",
            "target_technology": "FDM",
            "target_scale": "1:100",
            "printability_issues": [{"rule": "min_wall_thickness_mm", "severity": "high"}],
            "suggestions": [{"category": "geometry", "suggestion": "Thicken walls"}],
            "printability_score": 70.0,
            "advisory_report": "# Advisory\nDetails here",
            "architecture_tips_applied": ["base_stability", "scale_features"],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "CAD Advisory Summary" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.cad_advisor_agent import CADAdvisorAgent
        state = {"human_approval_status": "approved"}
        result = CADAdvisorAgent._route_after_review(state)
        assert result == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.cad_advisor_agent import CADAdvisorAgent
        state = {"human_approval_status": "rejected"}
        result = CADAdvisorAgent._route_after_review(state)
        assert result == "rejected"

    def test_route_default(self):
        from core.agents.implementations.cad_advisor_agent import CADAdvisorAgent
        result = CADAdvisorAgent._route_after_review({})
        assert result == "approved"

    # ─── Graph ────────────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "load_consultation", "analyze_design", "generate_advisory",
            "human_review", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  5. LogisticsAgent
# ══════════════════════════════════════════════════════════════════════


class TestLogisticsState:
    """Tests for LogisticsAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import LogisticsAgentState
        assert LogisticsAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import LogisticsAgentState
        state: LogisticsAgentState = {
            "agent_id": "log_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "log_v1"

    def test_create_full(self):
        from core.agents.state import LogisticsAgentState
        state: LogisticsAgentState = {
            "agent_id": "log_v1",
            "vertical_id": "print_biz",
            "print_job_id": "pj_abc",
            "customer_id": "cust_1",
            "customer_name": "John Doe",
            "customer_email": "john@example.com",
            "shipping_address": {"city": "Austin", "state": "TX", "zip": "78701"},
            "package_weight_kg": 2.5,
            "package_dimensions": {"length": 30, "width": 20, "height": 15},
            "is_fragile": True,
            "selected_packaging": "padded_box",
            "packaging_cost_cents": 550,
            "packaging_notes": "Fragile item",
            "carrier_options": [],
            "selected_carrier": "UPS Ground",
            "shipping_cost_cents": 1499,
            "estimated_delivery_days": 5,
            "tracking_number": "1Z999AA10123456784",
            "shipment_status": "shipped",
            "shipped_at": "2025-01-15T10:00:00Z",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["package_weight_kg"] == 2.5
        assert state["is_fragile"] is True
        assert state["selected_carrier"] == "UPS Ground"

    def test_shipping_address_field(self):
        from core.agents.state import LogisticsAgentState
        state: LogisticsAgentState = {
            "shipping_address": {"city": "NYC", "state": "NY", "zip": "10001"},
        }
        assert state["shipping_address"]["city"] == "NYC"

    def test_shipment_status_field(self):
        from core.agents.state import LogisticsAgentState
        state: LogisticsAgentState = {"shipment_status": "pending", "tracking_number": ""}
        assert state["shipment_status"] == "pending"


class TestLogisticsAgent:
    """Tests for LogisticsAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.logistics_agent import LogisticsAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="log_v1",
            agent_type="logistics",
            name="Logistics",
            vertical_id="print_biz",
            params={"company_name": "PrintBiz"},
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

        return LogisticsAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ──────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.logistics_agent import LogisticsAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "logistics" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.logistics_agent import LogisticsAgent
        assert LogisticsAgent.agent_type == "logistics"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import LogisticsAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is LogisticsAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "LogisticsAgent" in r
        assert "log_v1" in r

    # ─── Initial State ────────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["print_job_id"] == ""
        assert state["customer_id"] == ""
        assert state["customer_name"] == ""
        assert state["customer_email"] == ""
        assert state["shipping_address"] == {}
        assert state["package_weight_kg"] == 0.0
        assert state["package_dimensions"] == {}
        assert state["is_fragile"] is False
        assert state["selected_packaging"] == ""
        assert state["packaging_cost_cents"] == 0
        assert state["carrier_options"] == []
        assert state["selected_carrier"] == ""
        assert state["shipping_cost_cents"] == 0
        assert state["estimated_delivery_days"] == 0
        assert state["tracking_number"] == ""
        assert state["shipment_status"] == "pending"
        assert state["shipped_at"] == ""

    # ─── Constants ────────────────────────────────────────────────────

    def test_constants_packaging_options(self):
        from core.agents.implementations import logistics_agent
        opts = logistics_agent.PACKAGING_OPTIONS
        assert "standard_box" in opts
        assert "padded_box" in opts
        assert "custom_crate" in opts
        assert "envelope_mailer" in opts
        assert "rigid_mailer" in opts

    def test_constants_carrier_options(self):
        from core.agents.implementations import logistics_agent
        carriers = logistics_agent.CARRIER_OPTIONS
        assert len(carriers) >= 5
        carrier_names = [c["name"] for c in carriers]
        assert "USPS" in carrier_names
        assert "UPS" in carrier_names
        assert "FedEx" in carrier_names

    def test_constants_fragility_map(self):
        from core.agents.implementations import logistics_agent
        fmap = logistics_agent.FRAGILITY_MAP
        assert fmap["SLA"] == "high"
        assert fmap["FDM"] == "medium"
        assert fmap["SLS"] == "low"

    def test_system_prompt(self):
        from core.agents.implementations import logistics_agent
        assert "{print_technology}" in logistics_agent.LOGISTICS_SYSTEM_PROMPT
        assert "{weight_kg}" in logistics_agent.LOGISTICS_SYSTEM_PROMPT
        assert "{destination}" in logistics_agent.LOGISTICS_SYSTEM_PROMPT

    # ─── Node 1: load_shipment ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_shipment_from_task(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({
            "print_job_id": "",
            "customer_name": "Jane",
            "customer_email": "jane@example.com",
            "shipping_address": {"city": "Austin", "state": "TX"},
            "package_weight_kg": 1.5,
            "print_technology": "FDM",
        }, "run-1")
        result = await agent._node_load_shipment(state)
        assert result["current_node"] == "load_shipment"
        assert result["customer_name"] == "Jane"
        assert result["package_weight_kg"] == 1.5
        assert result["is_fragile"] is False  # FDM is medium fragility

    @pytest.mark.asyncio
    async def test_node_load_shipment_from_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{
            "id": "pj_abc",
            "customer_id": "cust_1",
            "customer_name": "Bob",
            "customer_email": "bob@example.com",
            "shipping_address": {"city": "SF", "state": "CA"},
            "package_weight_kg": 3.0,
            "print_technology": "SLA",
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"print_job_id": "pj_abc"}, "run-2")
        result = await agent._node_load_shipment(state)
        assert result["customer_name"] == "Bob"
        assert result["is_fragile"] is True  # SLA is high fragility

    @pytest.mark.asyncio
    async def test_node_load_shipment_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({
            "print_job_id": "bad_id",
            "customer_name": "Fallback",
            "print_technology": "SLS",
        }, "run-3")
        result = await agent._node_load_shipment(state)
        assert result["customer_name"] == "Fallback"
        assert result["is_fragile"] is False  # SLS is low fragility

    # ─── Node 2: plan_packaging ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_plan_packaging_fragile_light(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["package_weight_kg"] = 2.0
        state["is_fragile"] = True
        state["package_dimensions"] = {"length": 20, "width": 15, "height": 10}
        result = await agent._node_plan_packaging(state)
        assert result["current_node"] == "plan_packaging"
        assert result["selected_packaging"] == "padded_box"
        assert result["packaging_cost_cents"] > 0

    @pytest.mark.asyncio
    async def test_node_plan_packaging_fragile_heavy(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["package_weight_kg"] = 8.0
        state["is_fragile"] = True
        state["package_dimensions"] = {"length": 40, "width": 30, "height": 25}
        result = await agent._node_plan_packaging(state)
        assert result["selected_packaging"] == "custom_crate"

    @pytest.mark.asyncio
    async def test_node_plan_packaging_small_light(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-3")
        state["package_weight_kg"] = 0.3
        state["is_fragile"] = False
        state["package_dimensions"] = {"length": 10, "width": 8, "height": 3}
        result = await agent._node_plan_packaging(state)
        assert result["selected_packaging"] == "envelope_mailer"

    @pytest.mark.asyncio
    async def test_node_plan_packaging_standard(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-4")
        state["package_weight_kg"] = 4.0
        state["is_fragile"] = False
        state["package_dimensions"] = {"length": 30, "width": 25, "height": 20}
        result = await agent._node_plan_packaging(state)
        assert result["selected_packaging"] == "standard_box"

    # ─── Node 3: select_carrier ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_select_carrier_standard(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"shipping_priority": "standard"}, "run-1")
        state["package_weight_kg"] = 2.0
        result = await agent._node_select_carrier(state)
        assert result["current_node"] == "select_carrier"
        assert len(result["carrier_options"]) >= 1
        assert result["selected_carrier"] != ""
        assert result["shipping_cost_cents"] > 0
        assert result["estimated_delivery_days"] > 0

    @pytest.mark.asyncio
    async def test_node_select_carrier_express(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"shipping_priority": "express"}, "run-2")
        state["package_weight_kg"] = 1.0
        result = await agent._node_select_carrier(state)
        # Express prioritizes speed; best option should be fast
        assert result["estimated_delivery_days"] <= 5

    @pytest.mark.asyncio
    async def test_node_select_carrier_economy(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"shipping_priority": "economy"}, "run-3")
        state["package_weight_kg"] = 3.0
        result = await agent._node_select_carrier(state)
        assert result["shipping_cost_cents"] > 0

    # ─── Node 4: human_review ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "selected_carrier": "USPS Priority Mail",
            "selected_packaging": "padded_box",
            "shipping_cost_cents": 1200,
            "packaging_cost_cents": 550,
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: report ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "selected_carrier": "UPS Ground",
            "selected_packaging": "standard_box",
            "shipping_cost_cents": 1299,
            "packaging_cost_cents": 300,
            "packaging_notes": "Standard packaging selected",
            "estimated_delivery_days": 5,
            "shipment_status": "pending",
            "shipping_address": {"city": "Austin", "state": "TX", "zip": "78701"},
            "package_weight_kg": 2.5,
            "is_fragile": False,
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Logistics Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_no_address(self):
        agent = self._make_agent()
        state = {
            "selected_carrier": "FedEx Express",
            "selected_packaging": "custom_crate",
            "shipping_cost_cents": 3299,
            "packaging_cost_cents": 1800,
            "packaging_notes": "Custom crate for fragile item",
            "estimated_delivery_days": 2,
            "shipment_status": "pending",
            "shipping_address": {},
            "package_weight_kg": 5.0,
            "is_fragile": True,
        }
        result = await agent._node_report(state)
        assert "Logistics Report" in result["report_summary"]

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.logistics_agent import LogisticsAgent
        state = {"human_approval_status": "approved"}
        result = LogisticsAgent._route_after_review(state)
        assert result == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.logistics_agent import LogisticsAgent
        state = {"human_approval_status": "rejected"}
        result = LogisticsAgent._route_after_review(state)
        assert result == "rejected"

    def test_route_default(self):
        from core.agents.implementations.logistics_agent import LogisticsAgent
        result = LogisticsAgent._route_after_review({})
        assert result == "approved"

    # ─── Graph ────────────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "load_shipment", "plan_packaging", "select_carrier",
            "human_review", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")
