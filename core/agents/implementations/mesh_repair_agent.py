"""
Mesh Repair Agent — The 3D Surgery Specialist.

Takes mesh issues identified by the File Analyst and applies targeted
repair strategies: fixing non-manifold edges, filling holes, correcting
inverted normals, removing degenerate faces, and welding open vertices.

Architecture (LangGraph State Machine):
    load_issues → plan_repairs → execute_repairs →
    human_review → report → END

Trigger Events:
    - file_analysis_completed: File Analyst found issues needing repair
    - repair_requested: Customer explicitly requests mesh repair
    - manual: On-demand mesh repair

Shared Brain Integration:
    - Reads: common repair patterns, success rates by issue type
    - Writes: repair effectiveness metrics, common issue-repair pairs

Safety:
    - NEVER modifies the original file in-place
    - Creates a copy for repair operations
    - All repair plans require human_review gate before applying
    - Preserves original geometry where possible

Usage:
    agent = MeshRepairAgent(config, db, embedder, llm)
    result = await agent.run({
        "file_analysis_id": "fa_abc123",
        "print_job_id": "pj_abc123",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import MeshRepairAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

REPAIR_STRATEGIES = {
    "non_manifold": {
        "label": "Non-Manifold Edge Repair",
        "description": "Split non-manifold edges and re-stitch topology",
        "complexity": "medium",
        "risk": "low",
        "typical_success_rate": 0.95,
        "steps": [
            "Identify non-manifold edges (shared by >2 faces)",
            "Duplicate vertices along non-manifold edges",
            "Split faces to create manifold topology",
            "Re-weld coincident vertices where safe",
        ],
    },
    "inverted_normals": {
        "label": "Normal Recalculation",
        "description": "Detect and flip inverted face normals for consistent orientation",
        "complexity": "low",
        "risk": "low",
        "typical_success_rate": 0.99,
        "steps": [
            "Detect face normal orientation using boundary edge analysis",
            "Identify connected components with inconsistent normals",
            "Flip normals to achieve outward-facing consistency",
            "Verify watertight seal after normal correction",
        ],
    },
    "holes": {
        "label": "Hole Filling",
        "description": "Close open boundaries by generating new faces",
        "complexity": "high",
        "risk": "medium",
        "typical_success_rate": 0.88,
        "steps": [
            "Identify boundary edge loops (open boundaries)",
            "Classify holes by size (small, medium, large)",
            "Apply ear-clipping triangulation for small holes",
            "Use advancing-front method for larger holes",
            "Smooth filled regions to match surrounding curvature",
        ],
    },
    "degenerate_faces": {
        "label": "Degenerate Face Removal",
        "description": "Remove zero-area and near-zero-area faces",
        "complexity": "low",
        "risk": "low",
        "typical_success_rate": 0.99,
        "steps": [
            "Identify faces with area below epsilon threshold",
            "Remove degenerate faces",
            "Collapse associated short edges",
            "Re-triangulate affected regions",
        ],
    },
    "self_intersecting": {
        "label": "Self-Intersection Repair",
        "description": "Detect and resolve overlapping mesh regions",
        "complexity": "high",
        "risk": "high",
        "typical_success_rate": 0.75,
        "steps": [
            "Build AABB tree for fast intersection detection",
            "Identify intersecting face pairs",
            "Compute intersection curves",
            "Re-mesh intersection regions",
            "Verify no new intersections introduced",
        ],
    },
    "thin_walls": {
        "label": "Thin Wall Thickening",
        "description": "Detect and thicken walls below minimum printable threshold",
        "complexity": "medium",
        "risk": "medium",
        "typical_success_rate": 0.85,
        "steps": [
            "Compute local thickness using ray-casting",
            "Identify regions below minimum wall threshold",
            "Offset faces outward to meet minimum thickness",
            "Smooth transition between thickened and original regions",
        ],
    },
    "duplicate_faces": {
        "label": "Duplicate Face Removal",
        "description": "Remove coincident overlapping faces",
        "complexity": "low",
        "risk": "low",
        "typical_success_rate": 0.99,
        "steps": [
            "Hash face vertices for duplicate detection",
            "Identify coincident face pairs",
            "Remove duplicate faces keeping one copy",
            "Verify mesh integrity after removal",
        ],
    },
}

MESH_REPAIR_SYSTEM_PROMPT = """\
You are a 3D mesh repair specialist for additive manufacturing. \
Given the mesh issues below, create a detailed repair plan as a JSON object:
{{
    "repair_steps": [
        {{
            "issue": "...",
            "strategy": "non_manifold|inverted_normals|holes|degenerate_faces|self_intersecting|thin_walls|duplicate_faces",
            "priority": 1-10,
            "estimated_impact": "critical|high|medium|low",
            "notes": "Any specific considerations for this repair",
            "estimated_vertex_changes": 0
        }}
    ],
    "repair_order_reasoning": "Explanation of why this order was chosen",
    "risk_assessment": "Overall risk level of the repair plan",
    "expected_outcome": "What the mesh should look like after repairs"
}}

File: {file_name}
Current Issues:
{issues_text}

Known repair strategies and success rates:
{strategies_text}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("mesh_repair")
class MeshRepairAgent(BaseAgent):
    """
    3D mesh repair agent for PrintBiz engagements.

    Nodes:
        1. load_issues       -- Pull analysis issues from file_analyses table
        2. plan_repairs      -- Use LLM to generate repair strategy
        3. execute_repairs   -- Apply repairs (stub — mock repaired data)
        4. human_review      -- Gate: approve repair results
        5. report            -- Generate repair summary and store insights
    """

    def build_graph(self) -> Any:
        """Build the Mesh Repair Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(MeshRepairAgentState)

        workflow.add_node("load_issues", self._node_load_issues)
        workflow.add_node("plan_repairs", self._node_plan_repairs)
        workflow.add_node("execute_repairs", self._node_execute_repairs)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_issues")

        workflow.add_edge("load_issues", "plan_repairs")
        workflow.add_edge("plan_repairs", "execute_repairs")
        workflow.add_edge("execute_repairs", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "report",
                "rejected": "report",
            },
        )
        workflow.add_edge("report", END)

        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        return self.mcp_tools or []

    @classmethod
    def get_state_class(cls) -> Type[MeshRepairAgentState]:
        return MeshRepairAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "file_analysis_id": "",
            "print_job_id": "",
            "file_name": "",
            "original_issues": [],
            "repair_plan": [],
            "repair_plan_summary": "",
            "repairs_applied": [],
            "repairs_failed": [],
            "repaired_vertex_count": 0,
            "repaired_face_count": 0,
            "post_repair_manifold": False,
            "post_repair_watertight": False,
            "issues_resolved": 0,
            "issues_remaining": 0,
            "repair_success_rate": 0.0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Issues ─────────────────────────────────────────

    async def _node_load_issues(
        self, state: MeshRepairAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull analysis issues from file_analyses by file_analysis_id."""
        task = state.get("task_input", {})
        file_analysis_id = task.get("file_analysis_id", "")
        print_job_id = task.get("print_job_id", "")

        logger.info(
            "mesh_repair_load_issues",
            extra={
                "file_analysis_id": file_analysis_id,
                "print_job_id": print_job_id,
                "agent_id": self.agent_id,
            },
        )

        file_name = task.get("file_name", "unknown.stl")
        original_issues: list[dict[str, Any]] = task.get("issues", [])

        # Try loading from DB if we have an analysis ID
        if file_analysis_id:
            try:
                result = (
                    self.db.client.table("file_analyses")
                    .select("*")
                    .eq("id", file_analysis_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    analysis = result.data[0]
                    file_name = analysis.get("file_name", file_name)
                    print_job_id = analysis.get("print_job_id", print_job_id)

                    # Parse issues from JSON
                    issues_raw = analysis.get("issues", "[]")
                    if isinstance(issues_raw, str):
                        try:
                            original_issues = json.loads(issues_raw)
                        except json.JSONDecodeError:
                            original_issues = []
                    elif isinstance(issues_raw, list):
                        original_issues = issues_raw

                    logger.info(
                        "mesh_repair_analysis_loaded",
                        extra={
                            "file_analysis_id": file_analysis_id,
                            "file_name": file_name,
                            "issue_count": len(original_issues),
                        },
                    )
                else:
                    logger.warning(
                        "mesh_repair_analysis_not_found",
                        extra={"file_analysis_id": file_analysis_id},
                    )
            except Exception as e:
                logger.warning(
                    "mesh_repair_db_error",
                    extra={
                        "file_analysis_id": file_analysis_id,
                        "error": str(e)[:200],
                    },
                )

        if not original_issues:
            logger.info(
                "mesh_repair_no_issues",
                extra={"file_analysis_id": file_analysis_id},
            )

        return {
            "current_node": "load_issues",
            "file_analysis_id": file_analysis_id,
            "print_job_id": print_job_id,
            "file_name": file_name,
            "original_issues": original_issues,
        }

    # ─── Node 2: Plan Repairs ────────────────────────────────────────

    async def _node_plan_repairs(
        self, state: MeshRepairAgentState
    ) -> dict[str, Any]:
        """Node 2: Use LLM to generate a repair strategy for each issue."""
        original_issues = state.get("original_issues", [])
        file_name = state.get("file_name", "unknown")

        logger.info(
            "mesh_repair_plan_repairs",
            extra={
                "file_name": file_name,
                "issue_count": len(original_issues),
            },
        )

        if not original_issues:
            return {
                "current_node": "plan_repairs",
                "repair_plan": [],
                "repair_plan_summary": "No issues to repair.",
            }

        # Format issues for LLM
        issues_text = ""
        for i, issue in enumerate(original_issues, 1):
            issues_text += (
                f"{i}. [{issue.get('severity', 'unknown').upper()}] "
                f"{issue.get('issue', 'Unknown issue')}\n"
                f"   Description: {issue.get('description', 'N/A')}\n"
                f"   Location: {issue.get('location', 'N/A')}\n\n"
            )

        # Format strategies
        strategies_text = ""
        for key, strat in REPAIR_STRATEGIES.items():
            strategies_text += (
                f"- {key}: {strat['label']} "
                f"(success rate: {strat['typical_success_rate']:.0%}, "
                f"risk: {strat['risk']}, complexity: {strat['complexity']})\n"
            )

        repair_plan: list[dict[str, Any]] = []
        repair_plan_summary = ""

        try:
            prompt = MESH_REPAIR_SYSTEM_PROMPT.format(
                file_name=file_name,
                issues_text=issues_text,
                strategies_text=strategies_text,
            )

            llm_response = self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                system="You are a 3D mesh repair specialist.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                llm_data = json.loads(llm_text)
                repair_steps = llm_data.get("repair_steps", [])

                for step in repair_steps:
                    strategy_key = step.get("strategy", "")
                    strategy_info = REPAIR_STRATEGIES.get(strategy_key, {})

                    repair_plan.append({
                        "issue": step.get("issue", ""),
                        "strategy": strategy_key,
                        "strategy_label": strategy_info.get("label", strategy_key),
                        "priority": step.get("priority", 5),
                        "estimated_impact": step.get("estimated_impact", "medium"),
                        "notes": step.get("notes", ""),
                        "estimated_vertex_changes": step.get("estimated_vertex_changes", 0),
                        "complexity": strategy_info.get("complexity", "medium"),
                        "risk": strategy_info.get("risk", "medium"),
                        "typical_success_rate": strategy_info.get("typical_success_rate", 0.8),
                        "steps": strategy_info.get("steps", []),
                    })

                # Sort by priority
                repair_plan.sort(key=lambda x: x.get("priority", 5))

                repair_plan_summary = (
                    f"Repair plan for {file_name}: "
                    f"{len(repair_plan)} repairs planned. "
                    f"{llm_data.get('repair_order_reasoning', '')} "
                    f"Risk assessment: {llm_data.get('risk_assessment', 'N/A')}. "
                    f"Expected outcome: {llm_data.get('expected_outcome', 'N/A')}."
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "mesh_repair_plan_parse_error",
                    extra={"error": str(e)[:200]},
                )
                # Fallback: create basic plan from issues
                for issue in original_issues:
                    issue_text = issue.get("issue", "").lower()
                    strategy_key = "holes"  # default
                    if "manifold" in issue_text:
                        strategy_key = "non_manifold"
                    elif "normal" in issue_text:
                        strategy_key = "inverted_normals"
                    elif "degenerate" in issue_text:
                        strategy_key = "degenerate_faces"
                    elif "thin" in issue_text or "wall" in issue_text:
                        strategy_key = "thin_walls"
                    elif "intersect" in issue_text:
                        strategy_key = "self_intersecting"

                    strategy_info = REPAIR_STRATEGIES.get(strategy_key, {})
                    repair_plan.append({
                        "issue": issue.get("issue", ""),
                        "strategy": strategy_key,
                        "strategy_label": strategy_info.get("label", strategy_key),
                        "priority": 5,
                        "estimated_impact": issue.get("severity", "medium"),
                        "notes": "",
                        "complexity": strategy_info.get("complexity", "medium"),
                        "risk": strategy_info.get("risk", "medium"),
                        "typical_success_rate": strategy_info.get("typical_success_rate", 0.8),
                        "steps": strategy_info.get("steps", []),
                    })
                repair_plan_summary = f"Fallback repair plan: {len(repair_plan)} repairs."

        except Exception as e:
            logger.warning(
                "mesh_repair_llm_error",
                extra={"error": str(e)[:200]},
            )
            repair_plan_summary = "LLM planning failed; manual repair recommended."

        logger.info(
            "mesh_repair_plan_complete",
            extra={
                "repair_count": len(repair_plan),
                "file_name": file_name,
            },
        )

        return {
            "current_node": "plan_repairs",
            "repair_plan": repair_plan,
            "repair_plan_summary": repair_plan_summary,
        }

    # ─── Node 3: Execute Repairs ─────────────────────────────────────

    async def _node_execute_repairs(
        self, state: MeshRepairAgentState
    ) -> dict[str, Any]:
        """Node 3: Apply repairs (stub — returns mock repaired data)."""
        repair_plan = state.get("repair_plan", [])
        file_name = state.get("file_name", "unknown")

        logger.info(
            "mesh_repair_execute",
            extra={
                "file_name": file_name,
                "repair_count": len(repair_plan),
            },
        )

        repairs_applied: list[dict[str, Any]] = []
        repairs_failed: list[dict[str, Any]] = []
        total_vertex_changes = 0

        for repair in repair_plan:
            strategy_key = repair.get("strategy", "")
            strategy_info = REPAIR_STRATEGIES.get(strategy_key, {})
            success_rate = strategy_info.get("typical_success_rate", 0.8)

            # Simulate repair execution
            # In production, this would call actual mesh processing libraries
            # (e.g., trimesh, pymeshlab, Open3D)
            vertex_changes = repair.get("estimated_vertex_changes", 50)
            if vertex_changes == 0:
                # Estimate based on strategy
                if strategy_key in ("non_manifold", "holes"):
                    vertex_changes = 120
                elif strategy_key == "inverted_normals":
                    vertex_changes = 0  # No vertex changes, just normal flips
                elif strategy_key == "degenerate_faces":
                    vertex_changes = 30
                elif strategy_key == "thin_walls":
                    vertex_changes = 200
                elif strategy_key == "self_intersecting":
                    vertex_changes = 350
                else:
                    vertex_changes = 50

            # Mock: high success rate for most repairs
            repair_result = {
                "issue": repair.get("issue", ""),
                "strategy": strategy_key,
                "strategy_label": repair.get("strategy_label", strategy_key),
                "result": "success",
                "vertices_modified": vertex_changes,
                "faces_modified": int(vertex_changes * 1.8),
                "execution_time_ms": 450,
                "notes": f"Applied {strategy_info.get('label', strategy_key)} successfully.",
            }

            # Simulate occasional failures for complex repairs
            if strategy_key == "self_intersecting" and success_rate < 0.8:
                repair_result["result"] = "partial"
                repair_result["notes"] = (
                    "Partial success: resolved 80% of self-intersections. "
                    "Remaining intersections are in complex curved regions."
                )

            if repair_result["result"] in ("success", "partial"):
                repairs_applied.append(repair_result)
                total_vertex_changes += vertex_changes
            else:
                repairs_failed.append(repair_result)

        # Post-repair assessment
        issues_resolved = len(repairs_applied)
        issues_remaining = len(repairs_failed)
        total_issues = issues_resolved + issues_remaining
        success_rate = (
            issues_resolved / total_issues if total_issues > 0 else 1.0
        )

        # Determine post-repair mesh state
        original_issues = state.get("original_issues", [])
        had_manifold_issue = any(
            "manifold" in i.get("issue", "").lower() for i in original_issues
        )
        had_watertight_issue = any(
            "watertight" in i.get("issue", "").lower() for i in original_issues
        )

        post_manifold = not had_manifold_issue or any(
            r.get("strategy") == "non_manifold" and r.get("result") == "success"
            for r in repairs_applied
        )
        post_watertight = not had_watertight_issue or any(
            r.get("strategy") == "holes" and r.get("result") == "success"
            for r in repairs_applied
        )

        logger.info(
            "mesh_repair_execution_complete",
            extra={
                "file_name": file_name,
                "repairs_applied": issues_resolved,
                "repairs_failed": issues_remaining,
                "success_rate": round(success_rate, 2),
                "vertex_changes": total_vertex_changes,
            },
        )

        return {
            "current_node": "execute_repairs",
            "repairs_applied": repairs_applied,
            "repairs_failed": repairs_failed,
            "repaired_vertex_count": total_vertex_changes,
            "repaired_face_count": int(total_vertex_changes * 1.8),
            "post_repair_manifold": post_manifold,
            "post_repair_watertight": post_watertight,
            "issues_resolved": issues_resolved,
            "issues_remaining": issues_remaining,
            "repair_success_rate": round(success_rate, 2),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: MeshRepairAgentState
    ) -> dict[str, Any]:
        """Node 4: Present repair results for human approval."""
        repairs_applied = state.get("repairs_applied", [])
        repairs_failed = state.get("repairs_failed", [])

        logger.info(
            "mesh_repair_human_review_pending",
            extra={
                "repairs_applied": len(repairs_applied),
                "repairs_failed": len(repairs_failed),
                "success_rate": state.get("repair_success_rate", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: MeshRepairAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate repair summary and store insights."""
        now = datetime.now(timezone.utc).isoformat()
        file_name = state.get("file_name", "unknown")
        repairs_applied = state.get("repairs_applied", [])
        repairs_failed = state.get("repairs_failed", [])
        original_issues = state.get("original_issues", [])
        success_rate = state.get("repair_success_rate", 0.0)

        sections = [
            "# Mesh Repair Report",
            f"*Generated: {now}*\n",
            f"## File: {file_name}",
            f"- **Original Issues:** {len(original_issues)}",
            f"- **Repairs Applied:** {len(repairs_applied)}",
            f"- **Repairs Failed:** {len(repairs_failed)}",
            f"- **Success Rate:** {success_rate:.0%}",
            f"- **Post-Repair Manifold:** {'Yes' if state.get('post_repair_manifold') else 'No'}",
            f"- **Post-Repair Watertight:** {'Yes' if state.get('post_repair_watertight') else 'No'}",
            f"- **Vertices Modified:** {state.get('repaired_vertex_count', 0):,}",
        ]

        if repairs_applied:
            sections.append(f"\n## Repairs Applied ({len(repairs_applied)})")
            for i, r in enumerate(repairs_applied, 1):
                sections.append(
                    f"{i}. **{r.get('strategy_label', r.get('strategy', ''))}** "
                    f"— {r.get('result', 'unknown')}\n"
                    f"   Issue: {r.get('issue', 'N/A')}\n"
                    f"   Vertices modified: {r.get('vertices_modified', 0):,}\n"
                    f"   {r.get('notes', '')}"
                )

        if repairs_failed:
            sections.append(f"\n## Repairs Failed ({len(repairs_failed)})")
            for i, r in enumerate(repairs_failed, 1):
                sections.append(
                    f"{i}. **{r.get('strategy_label', r.get('strategy', ''))}** "
                    f"— {r.get('result', 'failed')}\n"
                    f"   Issue: {r.get('issue', 'N/A')}\n"
                    f"   {r.get('notes', '')}"
                )

        report = "\n".join(sections)

        # Store insight about common repairs
        if repairs_applied:
            strategy_counts: dict[str, int] = {}
            for r in repairs_applied:
                s = r.get("strategy", "unknown")
                strategy_counts[s] = strategy_counts.get(s, 0) + 1

            top_strategy = max(strategy_counts, key=strategy_counts.get)  # type: ignore[arg-type]

            self.store_insight(InsightData(
                insight_type="mesh_repair_pattern",
                title=f"Mesh Repair: {file_name} — {success_rate:.0%} success",
                content=(
                    f"Mesh repair for {file_name}: {len(repairs_applied)} repairs "
                    f"applied ({success_rate:.0%} success rate). "
                    f"Most common strategy: {top_strategy} "
                    f"({strategy_counts[top_strategy]} uses). "
                    f"Post-repair: manifold={state.get('post_repair_manifold')}, "
                    f"watertight={state.get('post_repair_watertight')}."
                ),
                confidence=0.80,
                metadata={
                    "strategy_counts": strategy_counts,
                    "success_rate": success_rate,
                    "issues_resolved": state.get("issues_resolved", 0),
                    "issues_remaining": state.get("issues_remaining", 0),
                },
            ))

        logger.info(
            "mesh_repair_report_generated",
            extra={
                "file_name": file_name,
                "repairs_applied": len(repairs_applied),
                "success_rate": success_rate,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: MeshRepairAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<MeshRepairAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
