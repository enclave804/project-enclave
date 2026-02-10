# tests/test_print_domain/__init__.py must exist
"""
Tests for PrintBiz Domain Expert Agents — Batch 1.

Covers the first 5 PrintBiz 3D-printing domain agents:
    1. FileAnalystAgent (file_analyst)
    2. MeshRepairAgent (mesh_repair)
    3. ScaleOptimizerAgent (scale_optimizer)
    4. MaterialAdvisorAgent (material_advisor)
    5. QuoteEngineAgent (quote_engine)

Each agent tests:
    - State TypedDict import and creation
    - Agent registration, construction, state class
    - Initial state preparation
    - Module-level constants
    - All graph nodes (async, mocked DB/LLM)
    - Graph construction and routing
    - System prompt
    - __repr__ and write_knowledge
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
#  1.  FileAnalystAgent
# ══════════════════════════════════════════════════════════════════════


class TestFileAnalystState:
    """Tests for FileAnalystAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import FileAnalystAgentState
        assert FileAnalystAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import FileAnalystAgentState
        state: FileAnalystAgentState = {
            "agent_id": "fa_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "fa_v1"

    def test_create_full(self):
        from core.agents.state import FileAnalystAgentState
        state: FileAnalystAgentState = {
            "agent_id": "fa_v1",
            "vertical_id": "print_biz",
            "print_job_id": "pj_001",
            "file_name": "model.stl",
            "file_format": "STL",
            "file_size_bytes": 1024000,
            "file_url": "https://cdn.example.com/model.stl",
            "customer_id": "cust_1",
            "customer_name": "Alice",
            "vertex_count": 12480,
            "face_count": 24960,
            "edge_count": 37438,
            "is_manifold": True,
            "is_watertight": True,
            "has_inverted_normals": False,
            "bounding_box": {"x": 120.0, "y": 80.0, "z": 95.0},
            "volume_cm3": 456.2,
            "surface_area_cm2": 892.5,
            "center_of_mass": {"x": 60.0, "y": 40.0, "z": 47.5},
            "printability_score": 85.0,
            "issues": [],
            "warnings": [],
            "min_wall_thickness_mm": 0.8,
            "min_detail_mm": 0.3,
            "overhang_percentage": 15.0,
            "file_analysis_id": "fa_abc",
            "analysis_saved": True,
            "all_findings": [],
            "report_summary": "Report",
            "report_generated_at": "2025-01-01T00:00:00Z",
        }
        assert state["file_name"] == "model.stl"
        assert state["printability_score"] == 85.0
        assert state["analysis_saved"] is True

    def test_geometry_fields(self):
        from core.agents.state import FileAnalystAgentState
        state: FileAnalystAgentState = {
            "vertex_count": 5000,
            "face_count": 10000,
            "edge_count": 15000,
            "is_manifold": False,
            "is_watertight": False,
        }
        assert state["vertex_count"] == 5000
        assert state["is_manifold"] is False

    def test_collections_fields(self):
        from core.agents.state import FileAnalystAgentState
        state: FileAnalystAgentState = {
            "issues": [{"issue": "Non-manifold", "severity": "critical"}],
            "warnings": [{"warning": "High overhang"}],
            "all_findings": [{"finding_type": "issue"}],
        }
        assert len(state["issues"]) == 1
        assert state["warnings"][0]["warning"] == "High overhang"

    def test_numeric_fields(self):
        from core.agents.state import FileAnalystAgentState
        state: FileAnalystAgentState = {
            "volume_cm3": 123.45,
            "surface_area_cm2": 678.9,
            "min_wall_thickness_mm": 0.5,
            "overhang_percentage": 22.5,
        }
        assert state["volume_cm3"] == 123.45
        assert state["min_wall_thickness_mm"] == 0.5


class TestFileAnalystAgent:
    """Tests for FileAnalystAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a FileAnalystAgent with mocked dependencies."""
        from core.agents.implementations.file_analyst_agent import FileAnalystAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="fa_v1",
            agent_type="file_analyst",
            name="File Analyst",
            vertical_id="print_biz",
            params={
                "company_name": "PrintBiz",
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

        return FileAnalystAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.file_analyst_agent import FileAnalystAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "file_analyst" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.file_analyst_agent import FileAnalystAgent
        assert FileAnalystAgent.agent_type == "file_analyst"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import FileAnalystAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is FileAnalystAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "FileAnalystAgent" in r
        assert "fa_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_001", "file_name": "test.stl"}, "run-1"
        )
        assert state["print_job_id"] == ""
        assert state["file_name"] == ""
        assert state["file_format"] == ""
        assert state["file_size_bytes"] == 0
        assert state["vertex_count"] == 0
        assert state["face_count"] == 0
        assert state["is_manifold"] is False
        assert state["is_watertight"] is False
        assert state["has_inverted_normals"] is False
        assert state["issues"] == []
        assert state["warnings"] == []
        assert state["printability_score"] == 0.0
        assert state["analysis_saved"] is False
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_input(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_002", "file_name": "gear.obj"}, "run-2"
        )
        assert state["task_input"]["print_job_id"] == "pj_002"
        assert state["task_input"]["file_name"] == "gear.obj"

    # ─── Constants ──────────────────────────────────────────────────

    def test_supported_formats(self):
        from core.agents.implementations import file_analyst_agent
        fmts = file_analyst_agent.SUPPORTED_FORMATS
        assert "STL" in fmts
        assert "OBJ" in fmts
        assert "STEP" in fmts
        assert "3MF" in fmts
        assert len(fmts) == 4

    def test_printability_thresholds(self):
        from core.agents.implementations import file_analyst_agent
        t = file_analyst_agent.PRINTABILITY_THRESHOLDS
        assert t["min_wall_thickness_mm"] == 0.5
        assert t["min_detail_mm"] == 0.2
        assert t["max_overhang_degrees"] == 45

    def test_severity_weights(self):
        from core.agents.implementations import file_analyst_agent
        w = file_analyst_agent.SEVERITY_WEIGHTS
        assert w["critical"] == 30
        assert w["high"] == 20
        assert w["medium"] == 10
        assert w["low"] == 5
        assert w["info"] == 1

    def test_system_prompt_template(self):
        from core.agents.implementations import file_analyst_agent
        prompt = file_analyst_agent.FILE_ANALYST_SYSTEM_PROMPT
        assert "{file_name}" in prompt
        assert "{file_format}" in prompt
        assert "{vertex_count}" in prompt
        assert "{is_manifold}" in prompt

    def test_system_prompt(self):
        from core.agents.implementations.file_analyst_agent import FILE_ANALYST_SYSTEM_PROMPT
        assert isinstance(FILE_ANALYST_SYSTEM_PROMPT, str)
        assert len(FILE_ANALYST_SYSTEM_PROMPT) > 50

    # ─── Node 1: Load File ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_file_basic(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_001", "file_name": "model.stl", "file_format": "STL"}, "run-1"
        )

        mock_result = MagicMock()
        mock_result.data = [{"id": "pj_001", "file_name": "model.stl", "file_format": "STL",
                             "file_size_bytes": 50000, "file_url": "https://cdn.example.com/model.stl",
                             "customer_id": "cust_1", "customer_name": "Alice"}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        result = await agent._node_load_file(state)
        assert result["current_node"] == "load_file"
        assert result["file_name"] == "model.stl"
        assert result["file_format"] == "STL"
        assert result["customer_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_node_load_file_no_job_id(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"file_name": "cube.obj", "file_format": "OBJ"}, "run-1"
        )
        result = await agent._node_load_file(state)
        assert result["current_node"] == "load_file"
        assert result["file_name"] == "cube.obj"
        assert result["file_format"] == "OBJ"

    @pytest.mark.asyncio
    async def test_node_load_file_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.side_effect = Exception("DB fail")
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_bad", "file_name": "test.stl"}, "run-1"
        )
        result = await agent._node_load_file(state)
        assert result["current_node"] == "load_file"
        assert result["file_name"] == "test.stl"

    # ─── Node 2: Analyze Geometry ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_geometry_clean(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "issues": [],
            "warnings": [],
            "printability_score": 95,
            "summary": "Clean geometry",
        })
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state(
            {
                "print_job_id": "pj_001",
                "file_name": "clean.stl",
                "file_format": "STL",
                "is_manifold": True,
                "is_watertight": True,
                "has_inverted_normals": False,
                "min_wall_thickness_mm": 1.5,
                "min_detail_mm": 0.5,
                "overhang_percentage": 10.0,
            }, "run-1"
        )
        state["file_name"] = "clean.stl"
        state["file_format"] = "STL"

        result = await agent._node_analyze_geometry(state)
        assert result["current_node"] == "analyze_geometry"
        assert result["is_manifold"] is True
        assert result["is_watertight"] is True
        assert result["printability_score"] > 0

    @pytest.mark.asyncio
    async def test_node_analyze_geometry_with_issues(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM fail")

        state = agent._prepare_initial_state(
            {
                "print_job_id": "pj_002",
                "file_name": "broken.stl",
                "file_format": "STL",
                "is_manifold": False,
                "is_watertight": False,
                "has_inverted_normals": True,
                "min_wall_thickness_mm": 0.3,
                "min_detail_mm": 0.1,
                "overhang_percentage": 40.0,
            }, "run-1"
        )
        state["file_name"] = "broken.stl"
        state["file_format"] = "STL"

        result = await agent._node_analyze_geometry(state)
        assert result["current_node"] == "analyze_geometry"
        assert len(result["issues"]) >= 3  # non-manifold, not watertight, inverted normals, thin wall
        assert result["printability_score"] < 50

    # ─── Node 3: Human Review ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"issues": [{"severity": "critical"}], "warnings": [], "printability_score": 40.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_empty(self):
        agent = self._make_agent()
        state = {"issues": [], "warnings": [], "printability_score": 100.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ─── Node 5: Report ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "file_name": "model.stl",
            "file_format": "STL",
            "vertex_count": 12480,
            "face_count": 24960,
            "is_manifold": True,
            "is_watertight": True,
            "volume_cm3": 456.2,
            "surface_area_cm2": 892.5,
            "bounding_box": {"x": 120.0, "y": 80.0, "z": 95.0},
            "printability_score": 85.0,
            "issues": [{"issue": "Thin wall", "severity": "high", "description": "test", "recommendation": "fix"}],
            "warnings": [{"warning": "High overhang", "impact": "more supports", "suggestion": "reorient"}],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "3D File Analysis Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_no_issues(self):
        agent = self._make_agent()
        state = {
            "file_name": "perfect.stl",
            "file_format": "STL",
            "vertex_count": 1000,
            "face_count": 2000,
            "is_manifold": True,
            "is_watertight": True,
            "volume_cm3": 100.0,
            "surface_area_cm2": 200.0,
            "bounding_box": {"x": 50.0, "y": 50.0, "z": 50.0},
            "printability_score": 100.0,
            "issues": [],
            "warnings": [],
        }
        result = await agent._node_report(state)
        assert "3D File Analysis Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.file_analyst_agent import FileAnalystAgent
        assert FileAnalystAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.file_analyst_agent import FileAnalystAgent
        assert FileAnalystAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.file_analyst_agent import FileAnalystAgent
        assert FileAnalystAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"load_file", "analyze_geometry", "human_review", "save_analysis", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  2.  MeshRepairAgent
# ══════════════════════════════════════════════════════════════════════


class TestMeshRepairState:
    """Tests for MeshRepairAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import MeshRepairAgentState
        assert MeshRepairAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import MeshRepairAgentState
        state: MeshRepairAgentState = {
            "agent_id": "mr_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "mr_v1"

    def test_create_full(self):
        from core.agents.state import MeshRepairAgentState
        state: MeshRepairAgentState = {
            "agent_id": "mr_v1",
            "vertical_id": "print_biz",
            "file_analysis_id": "fa_abc",
            "print_job_id": "pj_001",
            "file_name": "model.stl",
            "original_issues": [{"issue": "Non-manifold", "severity": "critical"}],
            "repair_plan": [{"strategy": "non_manifold"}],
            "repair_plan_summary": "1 repair planned",
            "repairs_applied": [{"strategy": "non_manifold", "result": "success"}],
            "repairs_failed": [],
            "repaired_vertex_count": 120,
            "repaired_face_count": 216,
            "post_repair_manifold": True,
            "post_repair_watertight": True,
            "issues_resolved": 1,
            "issues_remaining": 0,
            "repair_success_rate": 1.0,
            "report_summary": "All repairs successful",
            "report_generated_at": "2025-01-01T00:00:00Z",
        }
        assert state["file_analysis_id"] == "fa_abc"
        assert state["repair_success_rate"] == 1.0
        assert state["post_repair_manifold"] is True

    def test_repair_collections(self):
        from core.agents.state import MeshRepairAgentState
        state: MeshRepairAgentState = {
            "original_issues": [{"issue": "Holes"}, {"issue": "Non-manifold"}],
            "repairs_applied": [{"result": "success"}],
            "repairs_failed": [{"result": "failed"}],
            "repair_plan": [{"strategy": "holes"}],
        }
        assert len(state["original_issues"]) == 2
        assert len(state["repairs_applied"]) == 1
        assert len(state["repairs_failed"]) == 1

    def test_boolean_fields(self):
        from core.agents.state import MeshRepairAgentState
        state: MeshRepairAgentState = {
            "post_repair_manifold": False,
            "post_repair_watertight": True,
        }
        assert state["post_repair_manifold"] is False
        assert state["post_repair_watertight"] is True


class TestMeshRepairAgent:
    """Tests for MeshRepairAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a MeshRepairAgent with mocked dependencies."""
        from core.agents.implementations.mesh_repair_agent import MeshRepairAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="mr_v1",
            agent_type="mesh_repair",
            name="Mesh Repair",
            vertical_id="print_biz",
            params={
                "company_name": "PrintBiz",
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

        return MeshRepairAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.mesh_repair_agent import MeshRepairAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "mesh_repair" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.mesh_repair_agent import MeshRepairAgent
        assert MeshRepairAgent.agent_type == "mesh_repair"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import MeshRepairAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is MeshRepairAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "MeshRepairAgent" in r
        assert "mr_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_001", "print_job_id": "pj_001"}, "run-1"
        )
        assert state["file_analysis_id"] == ""
        assert state["print_job_id"] == ""
        assert state["file_name"] == ""
        assert state["original_issues"] == []
        assert state["repair_plan"] == []
        assert state["repairs_applied"] == []
        assert state["repairs_failed"] == []
        assert state["repaired_vertex_count"] == 0
        assert state["post_repair_manifold"] is False
        assert state["post_repair_watertight"] is False
        assert state["repair_success_rate"] == 0.0
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_input(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_002", "print_job_id": "pj_002"}, "run-2"
        )
        assert state["task_input"]["file_analysis_id"] == "fa_002"
        assert state["task_input"]["print_job_id"] == "pj_002"

    # ─── Constants ──────────────────────────────────────────────────

    def test_repair_strategies(self):
        from core.agents.implementations import mesh_repair_agent
        strats = mesh_repair_agent.REPAIR_STRATEGIES
        assert "non_manifold" in strats
        assert "inverted_normals" in strats
        assert "holes" in strats
        assert "degenerate_faces" in strats
        assert "self_intersecting" in strats
        assert "thin_walls" in strats
        assert "duplicate_faces" in strats
        assert len(strats) == 7

    def test_repair_strategy_fields(self):
        from core.agents.implementations import mesh_repair_agent
        strat = mesh_repair_agent.REPAIR_STRATEGIES["non_manifold"]
        assert "label" in strat
        assert "description" in strat
        assert "complexity" in strat
        assert "risk" in strat
        assert "typical_success_rate" in strat
        assert "steps" in strat
        assert strat["typical_success_rate"] == 0.95

    def test_system_prompt_template(self):
        from core.agents.implementations import mesh_repair_agent
        prompt = mesh_repair_agent.MESH_REPAIR_SYSTEM_PROMPT
        assert "{file_name}" in prompt
        assert "{issues_text}" in prompt
        assert "{strategies_text}" in prompt

    def test_system_prompt(self):
        from core.agents.implementations.mesh_repair_agent import MESH_REPAIR_SYSTEM_PROMPT
        assert isinstance(MESH_REPAIR_SYSTEM_PROMPT, str)
        assert len(MESH_REPAIR_SYSTEM_PROMPT) > 50

    # ─── Node 1: Load Issues ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_issues_from_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{
            "id": "fa_001",
            "file_name": "model.stl",
            "print_job_id": "pj_001",
            "issues": json.dumps([{"issue": "Non-manifold", "severity": "critical"}]),
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_001", "print_job_id": "pj_001"}, "run-1"
        )
        result = await agent._node_load_issues(state)
        assert result["current_node"] == "load_issues"
        assert result["file_name"] == "model.stl"
        assert len(result["original_issues"]) == 1

    @pytest.mark.asyncio
    async def test_node_load_issues_no_analysis_id(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"issues": [{"issue": "Holes", "severity": "high"}]}, "run-1"
        )
        result = await agent._node_load_issues(state)
        assert result["current_node"] == "load_issues"
        assert len(result["original_issues"]) == 1

    @pytest.mark.asyncio
    async def test_node_load_issues_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.side_effect = Exception("DB fail")
        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_bad"}, "run-1"
        )
        result = await agent._node_load_issues(state)
        assert result["current_node"] == "load_issues"

    # ─── Node 2: Plan Repairs ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_plan_repairs_with_llm(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "repair_steps": [
                {"issue": "Non-manifold", "strategy": "non_manifold", "priority": 1,
                 "estimated_impact": "critical", "notes": "Fix edges",
                 "estimated_vertex_changes": 120},
            ],
            "repair_order_reasoning": "Priority-based ordering",
            "risk_assessment": "Low risk",
            "expected_outcome": "Clean mesh",
        })
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["file_name"] = "model.stl"
        state["original_issues"] = [{"issue": "Non-manifold", "severity": "critical",
                                      "description": "Bad edges", "location": "topology"}]

        result = await agent._node_plan_repairs(state)
        assert result["current_node"] == "plan_repairs"
        assert len(result["repair_plan"]) == 1
        assert "non_manifold" in result["repair_plan"][0]["strategy"]

    @pytest.mark.asyncio
    async def test_node_plan_repairs_no_issues(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["file_name"] = "clean.stl"
        state["original_issues"] = []

        result = await agent._node_plan_repairs(state)
        assert result["repair_plan"] == []
        assert "No issues" in result["repair_plan_summary"]

    # ─── Node 3: Execute Repairs ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_execute_repairs(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["file_name"] = "model.stl"
        state["original_issues"] = [{"issue": "Non-manifold geometry detected"}]
        state["repair_plan"] = [{
            "issue": "Non-manifold",
            "strategy": "non_manifold",
            "strategy_label": "Non-Manifold Edge Repair",
            "priority": 1,
            "estimated_impact": "critical",
            "notes": "",
            "estimated_vertex_changes": 120,
            "complexity": "medium",
            "risk": "low",
            "typical_success_rate": 0.95,
            "steps": [],
        }]

        result = await agent._node_execute_repairs(state)
        assert result["current_node"] == "execute_repairs"
        assert len(result["repairs_applied"]) >= 1
        assert result["repair_success_rate"] > 0

    # ─── Node 4: Human Review ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"repairs_applied": [{"result": "success"}], "repairs_failed": [],
                 "repair_success_rate": 1.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_with_failures(self):
        agent = self._make_agent()
        state = {"repairs_applied": [], "repairs_failed": [{"result": "failed"}],
                 "repair_success_rate": 0.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ─── Node 5: Report ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "file_name": "model.stl",
            "original_issues": [{"issue": "Non-manifold", "severity": "critical"}],
            "repairs_applied": [{
                "issue": "Non-manifold",
                "strategy": "non_manifold",
                "strategy_label": "Non-Manifold Edge Repair",
                "result": "success",
                "vertices_modified": 120,
                "notes": "Applied successfully.",
            }],
            "repairs_failed": [],
            "repair_success_rate": 1.0,
            "post_repair_manifold": True,
            "post_repair_watertight": True,
            "repaired_vertex_count": 120,
            "issues_resolved": 1,
            "issues_remaining": 0,
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Mesh Repair Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_no_repairs(self):
        agent = self._make_agent()
        state = {
            "file_name": "clean.stl",
            "original_issues": [],
            "repairs_applied": [],
            "repairs_failed": [],
            "repair_success_rate": 1.0,
            "post_repair_manifold": True,
            "post_repair_watertight": True,
            "repaired_vertex_count": 0,
            "issues_resolved": 0,
            "issues_remaining": 0,
        }
        result = await agent._node_report(state)
        assert "Mesh Repair Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.mesh_repair_agent import MeshRepairAgent
        assert MeshRepairAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.mesh_repair_agent import MeshRepairAgent
        assert MeshRepairAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.mesh_repair_agent import MeshRepairAgent
        assert MeshRepairAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"load_issues", "plan_repairs", "execute_repairs", "human_review", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  3.  ScaleOptimizerAgent
# ══════════════════════════════════════════════════════════════════════


class TestScaleOptimizerState:
    """Tests for ScaleOptimizerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ScaleOptimizerAgentState
        assert ScaleOptimizerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ScaleOptimizerAgentState
        state: ScaleOptimizerAgentState = {
            "agent_id": "so_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "so_v1"

    def test_create_full(self):
        from core.agents.state import ScaleOptimizerAgentState
        state: ScaleOptimizerAgentState = {
            "agent_id": "so_v1",
            "vertical_id": "print_biz",
            "file_analysis_id": "fa_001",
            "print_job_id": "pj_001",
            "file_name": "building.stl",
            "original_dimensions": {"x": 200.0, "y": 150.0, "z": 120.0},
            "target_scale": "1:100",
            "print_technology": "SLA",
            "target_scale_factor": 0.01,
            "scaled_dimensions": {"x": 2.0, "y": 1.5, "z": 1.2},
            "min_feature_size_mm": 1.0,
            "tech_min_feature_mm": 0.2,
            "detail_loss_percentage": 15.0,
            "features_at_risk": [{"feature": "Window frames"}],
            "recommended_scale_factor": 0.02,
            "recommended_technology": "SLA",
            "scale_adjustments": [{"adjustment": "Use larger scale"}],
            "fit_on_build_plate": True,
            "build_plate_utilization": 25.0,
            "report_summary": "Scale report",
            "report_generated_at": "2025-01-01T00:00:00Z",
        }
        assert state["target_scale"] == "1:100"
        assert state["detail_loss_percentage"] == 15.0
        assert state["fit_on_build_plate"] is True

    def test_dimension_fields(self):
        from core.agents.state import ScaleOptimizerAgentState
        state: ScaleOptimizerAgentState = {
            "original_dimensions": {"x": 500.0, "y": 300.0, "z": 400.0},
            "scaled_dimensions": {"x": 5.0, "y": 3.0, "z": 4.0},
            "target_scale_factor": 0.01,
        }
        assert state["original_dimensions"]["x"] == 500.0
        assert state["scaled_dimensions"]["z"] == 4.0

    def test_collections_fields(self):
        from core.agents.state import ScaleOptimizerAgentState
        state: ScaleOptimizerAgentState = {
            "features_at_risk": [{"feature": "Railings"}, {"feature": "Doors"}],
            "scale_adjustments": [{"adjustment": "Increase scale"}],
        }
        assert len(state["features_at_risk"]) == 2
        assert len(state["scale_adjustments"]) == 1


class TestScaleOptimizerAgent:
    """Tests for ScaleOptimizerAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a ScaleOptimizerAgent with mocked dependencies."""
        from core.agents.implementations.scale_optimizer_agent import ScaleOptimizerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="so_v1",
            agent_type="scale_optimizer",
            name="Scale Optimizer",
            vertical_id="print_biz",
            params={
                "company_name": "PrintBiz",
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

        return ScaleOptimizerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.scale_optimizer_agent import ScaleOptimizerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "scale_optimizer" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.scale_optimizer_agent import ScaleOptimizerAgent
        assert ScaleOptimizerAgent.agent_type == "scale_optimizer"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ScaleOptimizerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ScaleOptimizerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "ScaleOptimizerAgent" in r
        assert "so_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_001", "target_scale": "1:50"}, "run-1"
        )
        assert state["file_analysis_id"] == ""
        assert state["print_job_id"] == ""
        assert state["file_name"] == ""
        assert state["original_dimensions"] == {}
        assert state["target_scale"] == "1:100"
        assert state["print_technology"] == "FDM"
        assert state["target_scale_factor"] == 0.01
        assert state["scaled_dimensions"] == {}
        assert state["features_at_risk"] == []
        assert state["scale_adjustments"] == []
        assert state["fit_on_build_plate"] is True
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_input(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_002", "target_scale": "1:50", "print_technology": "SLA"}, "run-2"
        )
        assert state["task_input"]["target_scale"] == "1:50"
        assert state["task_input"]["print_technology"] == "SLA"

    # ─── Constants ──────────────────────────────────────────────────

    def test_scale_presets(self):
        from core.agents.implementations import scale_optimizer_agent
        presets = scale_optimizer_agent.SCALE_PRESETS
        assert "1:10" in presets
        assert "1:100" in presets
        assert "1:1000" in presets
        assert presets["1:100"] == 0.01

    def test_detail_thresholds(self):
        from core.agents.implementations import scale_optimizer_agent
        thresholds = scale_optimizer_agent.DETAIL_THRESHOLDS
        assert "FDM" in thresholds
        assert "SLA" in thresholds
        assert "SLS" in thresholds
        assert "MJF" in thresholds
        assert "DLP" in thresholds
        assert thresholds["FDM"]["min_feature_mm"] == 0.8
        assert thresholds["SLA"]["min_feature_mm"] == 0.2

    def test_build_plates(self):
        from core.agents.implementations import scale_optimizer_agent
        plates = scale_optimizer_agent.BUILD_PLATES
        assert "FDM" in plates
        assert "SLA" in plates
        assert plates["FDM"]["x_mm"] == 250
        assert plates["SLA"]["x_mm"] == 145

    def test_system_prompt_template(self):
        from core.agents.implementations import scale_optimizer_agent
        prompt = scale_optimizer_agent.SCALE_OPTIMIZER_SYSTEM_PROMPT
        assert "{file_name}" in prompt
        assert "{target_scale}" in prompt
        assert "{scale_factor}" in prompt
        assert "{print_tech}" in prompt

    def test_system_prompt(self):
        from core.agents.implementations.scale_optimizer_agent import SCALE_OPTIMIZER_SYSTEM_PROMPT
        assert isinstance(SCALE_OPTIMIZER_SYSTEM_PROMPT, str)
        assert len(SCALE_OPTIMIZER_SYSTEM_PROMPT) > 50

    # ─── Node 1: Load Dimensions ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_dimensions_from_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{
            "id": "fa_001",
            "file_name": "building.stl",
            "print_job_id": "pj_001",
            "bounding_box": json.dumps({"x": 200.0, "y": 150.0, "z": 120.0}),
            "min_detail_mm": 1.5,
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"file_analysis_id": "fa_001", "target_scale": "1:50", "print_technology": "SLA"}, "run-1"
        )
        result = await agent._node_load_dimensions(state)
        assert result["current_node"] == "load_dimensions"
        assert result["file_name"] == "building.stl"
        assert result["original_dimensions"]["x"] == 200.0
        assert result["print_technology"] == "SLA"

    @pytest.mark.asyncio
    async def test_node_load_dimensions_no_analysis(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"target_scale": "1:100", "print_technology": "FDM"}, "run-1"
        )
        result = await agent._node_load_dimensions(state)
        assert result["current_node"] == "load_dimensions"
        # Should use default dimensions
        assert result["original_dimensions"]["x"] == 200.0

    @pytest.mark.asyncio
    async def test_node_load_dimensions_custom_scale(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"target_scale": "1:75", "print_technology": "FDM"}, "run-1"
        )
        result = await agent._node_load_dimensions(state)
        assert result["target_scale"] == "1:75"
        assert abs(result["target_scale_factor"] - 1.0 / 75.0) < 0.001

    # ─── Node 2: Analyze Scale ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_scale(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["original_dimensions"] = {"x": 200.0, "y": 150.0, "z": 120.0}
        state["target_scale_factor"] = 0.01
        state["print_technology"] = "FDM"
        state["min_feature_size_mm"] = 1.0

        result = await agent._node_analyze_scale(state)
        assert result["current_node"] == "analyze_scale"
        assert "x" in result["scaled_dimensions"]
        assert result["scaled_dimensions"]["x"] == 2.0
        assert result["tech_min_feature_mm"] == 0.8
        assert isinstance(result["features_at_risk"], list)

    # ─── Node 3: Recommend Adjustments ──────────────────────────────

    @pytest.mark.asyncio
    async def test_node_recommend_adjustments(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "scale_assessment": "fair",
            "recommendations": [
                {"adjustment": "Switch to SLA", "reason": "Better detail", "impact": "Higher cost", "priority": 1}
            ],
            "optimal_scale_factor": 0.02,
            "optimal_technology": "SLA",
            "summary": "Consider SLA",
        })
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["file_name"] = "building.stl"
        state["original_dimensions"] = {"x": 200.0, "y": 150.0, "z": 120.0}
        state["target_scale"] = "1:100"
        state["target_scale_factor"] = 0.01
        state["print_technology"] = "FDM"
        state["scaled_dimensions"] = {"x": 2.0, "y": 1.5, "z": 1.2}
        state["min_feature_size_mm"] = 1.0
        state["features_at_risk"] = [{"feature": "Windows", "original_mm": 5.0, "scaled_mm": 0.05, "printable": False}]
        state["detail_loss_percentage"] = 50.0
        state["fit_on_build_plate"] = True

        result = await agent._node_recommend_adjustments(state)
        assert result["current_node"] == "recommend_adjustments"
        assert isinstance(result["scale_adjustments"], list)

    # ─── Node 4: Human Review ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"scale_adjustments": [{"adjustment": "test"}],
                 "features_at_risk": [{"feature": "Railings"}],
                 "detail_loss_percentage": 20.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_no_risk(self):
        agent = self._make_agent()
        state = {"scale_adjustments": [], "features_at_risk": [],
                 "detail_loss_percentage": 0.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ─── Node 5: Report ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "file_name": "building.stl",
            "target_scale": "1:100",
            "target_scale_factor": 0.01,
            "print_technology": "FDM",
            "original_dimensions": {"x": 200.0, "y": 150.0, "z": 120.0},
            "scaled_dimensions": {"x": 2.0, "y": 1.5, "z": 1.2},
            "features_at_risk": [{"feature": "Windows", "original_mm": 5.0,
                                  "scaled_mm": 0.05, "printable": False}],
            "scale_adjustments": [{"adjustment": "Switch to SLA", "reason": "Better detail",
                                   "impact": "Higher cost"}],
            "detail_loss_percentage": 50.0,
            "fit_on_build_plate": True,
            "build_plate_utilization": 5.0,
            "recommended_technology": "SLA",
            "recommended_scale_factor": 0.02,
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Scale Optimization Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_no_risks(self):
        agent = self._make_agent()
        state = {
            "file_name": "simple.stl",
            "target_scale": "1:10",
            "target_scale_factor": 0.1,
            "print_technology": "FDM",
            "original_dimensions": {"x": 100.0, "y": 100.0, "z": 100.0},
            "scaled_dimensions": {"x": 10.0, "y": 10.0, "z": 10.0},
            "features_at_risk": [],
            "scale_adjustments": [],
            "detail_loss_percentage": 0.0,
            "fit_on_build_plate": True,
            "build_plate_utilization": 15.0,
            "recommended_technology": "FDM",
            "recommended_scale_factor": 0.1,
        }
        result = await agent._node_report(state)
        assert "Scale Optimization Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.scale_optimizer_agent import ScaleOptimizerAgent
        assert ScaleOptimizerAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.scale_optimizer_agent import ScaleOptimizerAgent
        assert ScaleOptimizerAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.scale_optimizer_agent import ScaleOptimizerAgent
        assert ScaleOptimizerAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"load_dimensions", "analyze_scale", "recommend_adjustments", "human_review", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  4.  MaterialAdvisorAgent
# ══════════════════════════════════════════════════════════════════════


class TestMaterialAdvisorState:
    """Tests for MaterialAdvisorAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import MaterialAdvisorAgentState
        assert MaterialAdvisorAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import MaterialAdvisorAgentState
        state: MaterialAdvisorAgentState = {
            "agent_id": "ma_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "ma_v1"

    def test_create_full(self):
        from core.agents.state import MaterialAdvisorAgentState
        state: MaterialAdvisorAgentState = {
            "agent_id": "ma_v1",
            "vertical_id": "print_biz",
            "print_job_id": "pj_001",
            "file_analysis_id": "fa_001",
            "customer_requirements": {"durability": "high", "budget": "medium"},
            "intended_use": "functional",
            "volume_cm3": 456.2,
            "surface_area_cm2": 892.5,
            "candidate_materials": ["PLA", "ABS", "NYLON_SLS"],
            "material_scores": [{"material": "NYLON_SLS", "score": 90.0}],
            "technology_options": ["FDM", "SLS"],
            "recommended_material": "NYLON_SLS",
            "recommended_technology": "SLS",
            "material_cost_estimate": 6843.0,
            "layer_height_um": 100,
            "detail_rating": "high",
            "recommendation_reasoning": "Best for functional parts",
            "alternative_materials": [{"material": "ABS", "technology": "FDM"}],
            "report_summary": "Material report",
            "report_generated_at": "2025-01-01T00:00:00Z",
        }
        assert state["recommended_material"] == "NYLON_SLS"
        assert state["material_cost_estimate"] == 6843.0
        assert state["intended_use"] == "functional"

    def test_collections_fields(self):
        from core.agents.state import MaterialAdvisorAgentState
        state: MaterialAdvisorAgentState = {
            "candidate_materials": ["PLA", "ABS", "PETG"],
            "material_scores": [{"material": "PLA", "score": 80}],
            "technology_options": ["FDM", "SLA"],
            "alternative_materials": [{"material": "ABS"}],
        }
        assert len(state["candidate_materials"]) == 3
        assert len(state["material_scores"]) == 1
        assert "FDM" in state["technology_options"]

    def test_numeric_fields(self):
        from core.agents.state import MaterialAdvisorAgentState
        state: MaterialAdvisorAgentState = {
            "volume_cm3": 200.0,
            "surface_area_cm2": 500.0,
            "material_cost_estimate": 1500.0,
            "layer_height_um": 50,
        }
        assert state["volume_cm3"] == 200.0
        assert state["layer_height_um"] == 50


class TestMaterialAdvisorAgent:
    """Tests for MaterialAdvisorAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a MaterialAdvisorAgent with mocked dependencies."""
        from core.agents.implementations.material_advisor_agent import MaterialAdvisorAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="ma_v1",
            agent_type="material_advisor",
            name="Material Advisor",
            vertical_id="print_biz",
            params={
                "company_name": "PrintBiz",
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

        return MaterialAdvisorAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.material_advisor_agent import MaterialAdvisorAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "material_advisor" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.material_advisor_agent import MaterialAdvisorAgent
        assert MaterialAdvisorAgent.agent_type == "material_advisor"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import MaterialAdvisorAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is MaterialAdvisorAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "MaterialAdvisorAgent" in r
        assert "ma_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_001", "intended_use": "functional"}, "run-1"
        )
        assert state["print_job_id"] == ""
        assert state["file_analysis_id"] == ""
        assert state["customer_requirements"] == {}
        assert state["intended_use"] == ""
        assert state["volume_cm3"] == 0.0
        assert state["surface_area_cm2"] == 0.0
        assert state["candidate_materials"] == []
        assert state["material_scores"] == []
        assert state["recommended_material"] == ""
        assert state["recommended_technology"] == ""
        assert state["material_cost_estimate"] == 0.0
        assert state["alternative_materials"] == []
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_input(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_002", "intended_use": "display",
             "customer_requirements": {"detail_level": "high"}}, "run-2"
        )
        assert state["task_input"]["intended_use"] == "display"
        assert state["task_input"]["customer_requirements"]["detail_level"] == "high"

    # ─── Constants ──────────────────────────────────────────────────

    def test_materials_db(self):
        from core.agents.implementations import material_advisor_agent
        db = material_advisor_agent.MATERIALS_DB
        assert "PLA" in db
        assert "ABS" in db
        assert "SLA_RESIN" in db
        assert "NYLON_SLS" in db
        assert "PETG" in db
        assert "SANDSTONE" in db
        assert len(db) == 6

    def test_materials_db_fields(self):
        from core.agents.implementations import material_advisor_agent
        pla = material_advisor_agent.MATERIALS_DB["PLA"]
        assert "tech" in pla
        assert "cost_per_cm3" in pla
        assert "detail" in pla
        assert "layer_um" in pla
        assert "strength" in pla
        assert "best_for" in pla
        assert pla["tech"] == "FDM"

    def test_detail_rank(self):
        from core.agents.implementations import material_advisor_agent
        rank = material_advisor_agent.DETAIL_RANK
        assert rank["low"] == 1
        assert rank["very_high"] == 4

    def test_strength_rank(self):
        from core.agents.implementations import material_advisor_agent
        rank = material_advisor_agent.STRENGTH_RANK
        assert rank["low"] == 1
        assert rank["excellent"] == 4

    def test_heat_rank(self):
        from core.agents.implementations import material_advisor_agent
        rank = material_advisor_agent.HEAT_RANK
        assert rank["low"] == 1
        assert rank["high"] == 4

    def test_budget_cost_thresholds(self):
        from core.agents.implementations import material_advisor_agent
        thresholds = material_advisor_agent.BUDGET_COST_THRESHOLDS
        assert thresholds["low"] == 0.06
        assert thresholds["medium"] == 0.12
        assert thresholds["high"] == 0.20
        assert thresholds["unlimited"] == 999.0

    def test_system_prompt_template(self):
        from core.agents.implementations import material_advisor_agent
        prompt = material_advisor_agent.MATERIAL_ADVISOR_SYSTEM_PROMPT
        assert "{intended_use}" in prompt
        assert "{detail_level}" in prompt
        assert "{durability}" in prompt
        assert "{budget}" in prompt
        assert "{volume_cm3}" in prompt

    def test_system_prompt(self):
        from core.agents.implementations.material_advisor_agent import MATERIAL_ADVISOR_SYSTEM_PROMPT
        assert isinstance(MATERIAL_ADVISOR_SYSTEM_PROMPT, str)
        assert len(MATERIAL_ADVISOR_SYSTEM_PROMPT) > 50

    # ─── Node 1: Gather Requirements ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_gather_requirements_from_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{
            "id": "pj_001",
            "intended_use": "functional",
            "volume_cm3": 456.2,
            "surface_area_cm2": 892.5,
            "customer_requirements": json.dumps({"durability": "high", "budget": "medium"}),
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"print_job_id": "pj_001"}, "run-1"
        )
        result = await agent._node_gather_requirements(state)
        assert result["current_node"] == "gather_requirements"
        assert result["intended_use"] == "functional"
        assert result["volume_cm3"] == 456.2

    @pytest.mark.asyncio
    async def test_node_gather_requirements_from_task(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"intended_use": "display", "volume_cm3": 100.0,
             "customer_requirements": {"detail_level": "high"}}, "run-1"
        )
        result = await agent._node_gather_requirements(state)
        assert result["current_node"] == "gather_requirements"
        assert result["intended_use"] == "display"
        assert result["volume_cm3"] == 100.0

    @pytest.mark.asyncio
    async def test_node_gather_requirements_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.side_effect = Exception("DB fail")
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_bad", "intended_use": "prototype"}, "run-1"
        )
        result = await agent._node_gather_requirements(state)
        assert result["current_node"] == "gather_requirements"
        assert result["intended_use"] == "prototype"

    # ─── Node 2: Evaluate Materials ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_evaluate_materials(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["customer_requirements"] = {"detail_level": "high", "durability": "good",
                                           "budget": "medium", "heat_resistance": "low"}
        state["intended_use"] = "functional"
        state["volume_cm3"] = 200.0

        result = await agent._node_evaluate_materials(state)
        assert result["current_node"] == "evaluate_materials"
        assert len(result["candidate_materials"]) == 6
        assert len(result["material_scores"]) == 6
        assert result["material_scores"][0]["score"] >= result["material_scores"][-1]["score"]

    @pytest.mark.asyncio
    async def test_node_evaluate_materials_budget_filter(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["customer_requirements"] = {"budget": "low"}
        state["intended_use"] = "prototype"
        state["volume_cm3"] = 100.0

        result = await agent._node_evaluate_materials(state)
        # PLA should score highest for low budget
        assert result["material_scores"][0]["material"] in ["PLA", "ABS", "PETG"]

    # ─── Node 3: Recommend ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_recommend_with_llm(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "recommended_material": "NYLON_SLS",
            "recommended_technology": "SLS",
            "reasoning": "Best combination of strength and detail for functional use",
            "confidence": 0.85,
            "alternatives": [
                {"material": "ABS", "technology": "FDM", "tradeoff": "Lower cost but less durable"}
            ],
            "warnings": [],
        })
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["intended_use"] = "functional"
        state["customer_requirements"] = {"durability": "excellent"}
        state["volume_cm3"] = 200.0
        state["material_scores"] = [
            {"material": "NYLON_SLS", "technology": "SLS", "score": 90.0,
             "cost_per_cm3": 0.15, "estimated_cost_cents": 3000, "layer_um": 100,
             "detail": "high", "strength": "excellent", "finish": "textured",
             "strengths": ["Strength: excellent"], "weaknesses": []},
            {"material": "ABS", "technology": "FDM", "score": 75.0,
             "cost_per_cm3": 0.05, "estimated_cost_cents": 1000, "layer_um": 200,
             "detail": "medium", "strength": "good", "finish": "matte",
             "strengths": ["Cost effective"], "weaknesses": []},
        ]

        result = await agent._node_recommend(state)
        assert result["current_node"] == "recommend"
        assert result["recommended_material"] == "NYLON_SLS"
        assert result["recommended_technology"] == "SLS"

    @pytest.mark.asyncio
    async def test_node_recommend_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["intended_use"] = "prototype"
        state["customer_requirements"] = {}
        state["volume_cm3"] = 100.0
        state["material_scores"] = [
            {"material": "PLA", "technology": "FDM", "score": 85.0,
             "cost_per_cm3": 0.04, "estimated_cost_cents": 400, "layer_um": 200,
             "detail": "medium", "strength": "moderate", "finish": "matte",
             "strengths": ["Cheap"], "weaknesses": []},
        ]

        result = await agent._node_recommend(state)
        assert result["current_node"] == "recommend"
        assert result["recommended_material"] == "PLA"

    # ─── Node 4: Human Review ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"recommended_material": "NYLON_SLS", "material_cost_estimate": 3000}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_empty(self):
        agent = self._make_agent()
        state = {"recommended_material": "", "material_cost_estimate": 0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ─── Node 5: Report ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "recommended_material": "NYLON_SLS",
            "recommended_technology": "SLS",
            "material_cost_estimate": 3000,
            "layer_height_um": 100,
            "detail_rating": "high",
            "recommendation_reasoning": "Best for functional parts",
            "intended_use": "functional",
            "volume_cm3": 200.0,
            "material_scores": [
                {"material": "NYLON_SLS", "technology": "SLS", "score": 90.0, "estimated_cost_cents": 3000},
                {"material": "ABS", "technology": "FDM", "score": 75.0, "estimated_cost_cents": 1000},
            ],
            "alternative_materials": [
                {"material": "ABS", "technology": "FDM", "tradeoff": "Lower cost but less durable"},
            ],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Material Recommendation Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_minimal(self):
        agent = self._make_agent()
        state = {
            "recommended_material": "PLA",
            "recommended_technology": "FDM",
            "material_cost_estimate": 400,
            "layer_height_um": 200,
            "detail_rating": "medium",
            "recommendation_reasoning": "Budget pick",
            "intended_use": "prototype",
            "volume_cm3": 100.0,
            "material_scores": [],
            "alternative_materials": [],
        }
        result = await agent._node_report(state)
        assert "Material Recommendation Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.material_advisor_agent import MaterialAdvisorAgent
        assert MaterialAdvisorAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.material_advisor_agent import MaterialAdvisorAgent
        assert MaterialAdvisorAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.material_advisor_agent import MaterialAdvisorAgent
        assert MaterialAdvisorAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"gather_requirements", "evaluate_materials", "recommend", "human_review", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  5.  QuoteEngineAgent
# ══════════════════════════════════════════════════════════════════════


class TestQuoteEngineState:
    """Tests for QuoteEngineAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import QuoteEngineAgentState
        assert QuoteEngineAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import QuoteEngineAgentState
        state: QuoteEngineAgentState = {
            "agent_id": "qe_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "qe_v1"

    def test_create_full(self):
        from core.agents.state import QuoteEngineAgentState
        state: QuoteEngineAgentState = {
            "agent_id": "qe_v1",
            "vertical_id": "print_biz",
            "print_job_id": "pj_001",
            "file_analysis_id": "fa_001",
            "customer_id": "cust_1",
            "customer_name": "Bob",
            "customer_email": "bob@example.com",
            "volume_cm3": 456.2,
            "surface_area_cm2": 892.5,
            "bounding_box": {"x": 120.0, "y": 80.0, "z": 95.0},
            "print_technology": "SLS",
            "material": "NYLON_SLS",
            "material_cost_cents": 6843,
            "time_cost_cents": 4560,
            "post_process_costs": [{"process": "sanding", "cost_cents": 1500}],
            "post_process_total_cents": 1500,
            "shipping_cost_cents": 999,
            "subtotal_cents": 14902,
            "markup_percent": 30.0,
            "markup_cents": 4471,
            "total_cents": 19373,
            "estimated_print_hours": 18.2,
            "quote_id": "q_abc",
            "quote_document": "# Quote\n\nTotal: $193.73",
            "quote_valid_days": 30,
            "quote_saved": True,
            "quote_sent": True,
            "sent_at": "2025-01-01T00:00:00Z",
            "report_summary": "Quote report",
            "report_generated_at": "2025-01-01T00:00:00Z",
        }
        assert state["material"] == "NYLON_SLS"
        assert state["total_cents"] == 19373
        assert state["quote_sent"] is True

    def test_cost_fields(self):
        from core.agents.state import QuoteEngineAgentState
        state: QuoteEngineAgentState = {
            "material_cost_cents": 1000,
            "time_cost_cents": 2000,
            "post_process_total_cents": 500,
            "shipping_cost_cents": 899,
            "subtotal_cents": 4399,
            "markup_percent": 40.0,
            "markup_cents": 1760,
            "total_cents": 6159,
        }
        assert state["subtotal_cents"] == 4399
        assert state["markup_percent"] == 40.0

    def test_boolean_fields(self):
        from core.agents.state import QuoteEngineAgentState
        state: QuoteEngineAgentState = {
            "quote_saved": False,
            "quote_sent": False,
        }
        assert state["quote_saved"] is False
        assert state["quote_sent"] is False

    def test_collections_fields(self):
        from core.agents.state import QuoteEngineAgentState
        state: QuoteEngineAgentState = {
            "post_process_costs": [
                {"process": "sanding", "cost_cents": 1500},
                {"process": "painting", "cost_cents": 3500},
            ],
            "bounding_box": {"x": 100.0, "y": 80.0, "z": 60.0},
        }
        assert len(state["post_process_costs"]) == 2
        assert state["bounding_box"]["z"] == 60.0


class TestQuoteEngineAgent:
    """Tests for QuoteEngineAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a QuoteEngineAgent with mocked dependencies."""
        from core.agents.implementations.quote_engine_agent import QuoteEngineAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="qe_v1",
            agent_type="quote_engine",
            name="Quote Engine",
            vertical_id="print_biz",
            params={
                "company_name": "PrintBiz",
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

        return QuoteEngineAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.quote_engine_agent import QuoteEngineAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "quote_engine" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.quote_engine_agent import QuoteEngineAgent
        assert QuoteEngineAgent.agent_type == "quote_engine"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import QuoteEngineAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is QuoteEngineAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "QuoteEngineAgent" in r
        assert "qe_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_001", "material": "NYLON_SLS"}, "run-1"
        )
        assert state["print_job_id"] == ""
        assert state["file_analysis_id"] == ""
        assert state["customer_id"] == ""
        assert state["customer_name"] == ""
        assert state["customer_email"] == ""
        assert state["volume_cm3"] == 0.0
        assert state["material"] == ""
        assert state["material_cost_cents"] == 0
        assert state["time_cost_cents"] == 0
        assert state["post_process_costs"] == []
        assert state["post_process_total_cents"] == 0
        assert state["shipping_cost_cents"] == 0
        assert state["subtotal_cents"] == 0
        assert state["markup_percent"] == 0.0
        assert state["markup_cents"] == 0
        assert state["total_cents"] == 0
        assert state["quote_id"] == ""
        assert state["quote_document"] == ""
        assert state["quote_valid_days"] == 30
        assert state["quote_saved"] is False
        assert state["quote_sent"] is False
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_input(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_002", "material": "PLA", "volume_cm3": 100.0}, "run-2"
        )
        assert state["task_input"]["material"] == "PLA"
        assert state["task_input"]["volume_cm3"] == 100.0

    # ─── Constants ──────────────────────────────────────────────────

    def test_pricing_rates(self):
        from core.agents.implementations import quote_engine_agent
        rates = quote_engine_agent.PRICING_RATES
        assert rates["print_rate_per_hour"] == 15_00
        assert rates["operator_rate_per_hour"] == 25_00
        assert rates["setup_fee"] == 10_00
        assert rates["rush_multiplier"] == 1.5
        assert rates["min_order_cents"] == 25_00
        assert rates["default_markup_percent"] == 40.0
        assert rates["volume_discount_threshold_cm3"] == 500.0
        assert rates["volume_discount_percent"] == 10.0

    def test_post_process_costs(self):
        from core.agents.implementations import quote_engine_agent
        costs = quote_engine_agent.POST_PROCESS_COSTS
        assert "support_removal" in costs
        assert "sanding" in costs
        assert "painting" in costs
        assert "vapor_smoothing" in costs
        assert "metal_plating" in costs
        assert costs["support_removal"] == 10_00
        assert costs["painting"] == 35_00

    def test_print_speeds(self):
        from core.agents.implementations import quote_engine_agent
        speeds = quote_engine_agent.PRINT_SPEEDS
        assert "FDM" in speeds
        assert "SLA" in speeds
        assert "SLS" in speeds
        assert speeds["FDM"] == 15.0
        assert speeds["SLS"] == 25.0

    def test_shipping_rates(self):
        from core.agents.implementations import quote_engine_agent
        rates = quote_engine_agent.SHIPPING_RATES
        assert "standard" in rates
        assert "express" in rates
        assert "overnight" in rates
        assert rates["standard"]["days"] == 5
        assert rates["overnight"]["days"] == 1

    def test_system_prompt_template(self):
        from core.agents.implementations import quote_engine_agent
        prompt = quote_engine_agent.QUOTE_SYSTEM_PROMPT
        assert "{material}" in prompt
        assert "{technology}" in prompt
        assert "{volume_cm3}" in prompt
        assert "{total}" in prompt
        assert "{valid_days}" in prompt

    def test_system_prompt(self):
        from core.agents.implementations.quote_engine_agent import QUOTE_SYSTEM_PROMPT
        assert isinstance(QUOTE_SYSTEM_PROMPT, str)
        assert len(QUOTE_SYSTEM_PROMPT) > 50

    # ─── Node 1: Gather Data ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_gather_data_from_db(self):
        agent = self._make_agent()
        mock_result_job = MagicMock()
        mock_result_job.data = [{
            "id": "pj_001",
            "volume_cm3": 456.2,
            "material": "NYLON_SLS",
            "print_technology": "SLS",
            "customer_id": "cust_1",
            "customer_name": "Bob",
            "customer_email": "bob@example.com",
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result_job

        state = agent._prepare_initial_state(
            {"print_job_id": "pj_001"}, "run-1"
        )
        result = await agent._node_gather_data(state)
        assert result["current_node"] == "gather_data"
        assert result["material"] == "NYLON_SLS"
        assert result["customer_name"] == "Bob"

    @pytest.mark.asyncio
    async def test_node_gather_data_from_task(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"material": "PLA", "volume_cm3": 100.0, "print_technology": "FDM",
             "customer_name": "Alice", "customer_email": "alice@test.com"}, "run-1"
        )
        result = await agent._node_gather_data(state)
        assert result["current_node"] == "gather_data"
        assert result["material"] == "PLA"
        assert result["volume_cm3"] == 100.0

    @pytest.mark.asyncio
    async def test_node_gather_data_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.side_effect = Exception("DB fail")
        state = agent._prepare_initial_state(
            {"print_job_id": "pj_bad", "material": "ABS", "volume_cm3": 50.0}, "run-1"
        )
        result = await agent._node_gather_data(state)
        assert result["current_node"] == "gather_data"
        assert result["material"] == "ABS"

    # ─── Node 2: Calculate Costs ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_calculate_costs(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"post_processes": ["support_removal", "sanding"]}, "run-1"
        )
        state["volume_cm3"] = 200.0
        state["material"] = "PLA"
        state["print_technology"] = "FDM"

        result = await agent._node_calculate_costs(state)
        assert result["current_node"] == "calculate_costs"
        assert result["material_cost_cents"] > 0
        assert result["time_cost_cents"] > 0
        assert result["total_cents"] > 0
        assert result["estimated_print_hours"] > 0
        assert result["markup_percent"] > 0

    @pytest.mark.asyncio
    async def test_node_calculate_costs_volume_discount(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"post_processes": ["support_removal"]}, "run-1"
        )
        state["volume_cm3"] = 600.0  # Above volume discount threshold
        state["material"] = "PLA"
        state["print_technology"] = "FDM"

        result = await agent._node_calculate_costs(state)
        assert result["markup_percent"] < 40.0  # Should have volume discount

    @pytest.mark.asyncio
    async def test_node_calculate_costs_min_order(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"post_processes": []}, "run-1"
        )
        state["volume_cm3"] = 1.0  # Very small volume
        state["material"] = "PLA"
        state["print_technology"] = "FDM"

        result = await agent._node_calculate_costs(state)
        assert result["total_cents"] >= 25_00  # min order enforced

    # ─── Node 3: Generate Quote ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_quote_with_llm(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "# Professional Quote\n\n**Total: $193.73**\n\nValid for 30 days."
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=300)
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["material"] = "NYLON_SLS"
        state["print_technology"] = "SLS"
        state["volume_cm3"] = 456.2
        state["material_cost_cents"] = 6843
        state["time_cost_cents"] = 4560
        state["estimated_print_hours"] = 18.2
        state["post_process_total_cents"] = 1500
        state["shipping_cost_cents"] = 999
        state["subtotal_cents"] = 14902
        state["markup_percent"] = 30.0
        state["markup_cents"] = 4471
        state["total_cents"] = 19373
        state["customer_name"] = "Bob"
        state["customer_email"] = "bob@example.com"
        state["print_job_id"] = "pj_001"
        state["quote_valid_days"] = 30

        result = await agent._node_generate_quote(state)
        assert result["current_node"] == "generate_quote"
        assert "Quote" in result["quote_document"]

    @pytest.mark.asyncio
    async def test_node_generate_quote_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["material"] = "PLA"
        state["print_technology"] = "FDM"
        state["volume_cm3"] = 100.0
        state["total_cents"] = 5000
        state["quote_valid_days"] = 30

        result = await agent._node_generate_quote(state)
        assert result["current_node"] == "generate_quote"
        assert "Print Quote" in result["quote_document"]

    # ─── Node 4: Human Review ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"total_cents": 19373, "customer_name": "Bob"}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_empty(self):
        agent = self._make_agent()
        state = {"total_cents": 0, "customer_name": ""}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ─── Node 5: Send Quote ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_send_quote_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "q_001"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["print_job_id"] = "pj_001"
        state["customer_id"] = "cust_1"
        state["customer_name"] = "Bob"
        state["customer_email"] = "bob@example.com"
        state["material"] = "NYLON_SLS"
        state["print_technology"] = "SLS"
        state["volume_cm3"] = 456.2
        state["total_cents"] = 19373
        state["quote_document"] = "# Quote"

        result = await agent._node_send_quote(state)
        assert result["current_node"] == "send_quote"
        assert result["quote_id"] == "q_001"
        assert result["quote_saved"] is True
        assert result["quote_sent"] is True
        assert result["sent_at"] != ""

    @pytest.mark.asyncio
    async def test_node_send_quote_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("insert fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["print_job_id"] = "pj_001"
        state["total_cents"] = 5000
        state["quote_document"] = "# Quote"

        result = await agent._node_send_quote(state)
        assert result["quote_saved"] is False
        assert result["quote_sent"] is True  # Attempted to send even if save failed

    # ─── Node 6: Report ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "material": "NYLON_SLS",
            "print_technology": "SLS",
            "volume_cm3": 456.2,
            "estimated_print_hours": 18.2,
            "material_cost_cents": 6843,
            "time_cost_cents": 4560,
            "post_process_total_cents": 1500,
            "shipping_cost_cents": 999,
            "markup_percent": 30.0,
            "total_cents": 19373,
            "quote_saved": True,
            "quote_sent": True,
            "quote_id": "q_001",
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Quote Generation Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_minimal(self):
        agent = self._make_agent()
        state = {
            "material": "PLA",
            "print_technology": "FDM",
            "volume_cm3": 100.0,
            "estimated_print_hours": 6.7,
            "material_cost_cents": 400,
            "time_cost_cents": 1000,
            "post_process_total_cents": 0,
            "shipping_cost_cents": 899,
            "markup_percent": 40.0,
            "total_cents": 3219,
            "quote_saved": False,
            "quote_sent": False,
            "quote_id": "",
        }
        result = await agent._node_report(state)
        assert "Quote Generation Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.quote_engine_agent import QuoteEngineAgent
        assert QuoteEngineAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.quote_engine_agent import QuoteEngineAgent
        assert QuoteEngineAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.quote_engine_agent import QuoteEngineAgent
        assert QuoteEngineAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"gather_data", "calculate_costs", "generate_quote", "human_review", "send_quote", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")
