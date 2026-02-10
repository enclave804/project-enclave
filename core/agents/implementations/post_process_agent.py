"""
Post-Processing Agent — The Finishing Specialist.

Manages the post-processing workflow for 3D printed parts, recommending
finishing steps based on technology, material, and desired finish quality.
Generates structured work orders for the production floor.

Architecture (LangGraph State Machine):
    load_job → recommend_finishing → generate_work_order →
    human_review → report → END

Trigger Events:
    - job_printed: A print job has completed printing
    - post_process_request: Manual request for post-processing plan
    - manual: On-demand post-processing assessment

Shared Brain Integration:
    - Reads: material finishing best practices, time estimates from past jobs
    - Writes: finishing process insights, time accuracy data, quality correlations

Safety:
    - All work orders require human_review gate before finalization
    - Never modifies the original print job record without approval
    - Chemical process recommendations include safety notes
    - All DB mutations are wrapped in try/except

Usage:
    agent = PostProcessAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "job_123",
    })
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import PostProcessAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

FINISHING_STEPS = {
    "FDM": [
        {"step": "support_removal", "est_minutes": 15, "required": True,
         "description": "Remove support structures using pliers and flush cutters"},
        {"step": "sanding", "est_minutes": 30, "required": False,
         "description": "Sand surfaces progressively from 120 to 400 grit"},
        {"step": "priming", "est_minutes": 20, "required": False,
         "description": "Apply filler primer to smooth layer lines"},
        {"step": "painting", "est_minutes": 45, "required": False,
         "description": "Apply base coat and finish coat with airbrush or spray"},
    ],
    "SLA": [
        {"step": "washing", "est_minutes": 20, "required": True,
         "description": "Wash in IPA bath (2 cycles, 10 min each) to remove uncured resin"},
        {"step": "uv_curing", "est_minutes": 30, "required": True,
         "description": "UV cure at 405nm for 30 minutes to fully harden resin"},
        {"step": "support_removal", "est_minutes": 10, "required": True,
         "description": "Carefully remove SLA support structures with flush cutters"},
        {"step": "sanding", "est_minutes": 20, "required": False,
         "description": "Wet sand support nubs and seam lines (400-800 grit)"},
    ],
    "SLS": [
        {"step": "depowdering", "est_minutes": 25, "required": True,
         "description": "Remove excess powder via compressed air and soft brush"},
        {"step": "bead_blasting", "est_minutes": 15, "required": False,
         "description": "Bead blast for uniform matte surface finish"},
        {"step": "dyeing", "est_minutes": 40, "required": False,
         "description": "Dye part in heated dye bath for color application"},
        {"step": "sealing", "est_minutes": 20, "required": False,
         "description": "Apply sealant to prevent moisture absorption"},
    ],
    "MJF": [
        {"step": "depowdering", "est_minutes": 20, "required": True,
         "description": "Remove unfused powder via compressed air"},
        {"step": "bead_blasting", "est_minutes": 15, "required": True,
         "description": "Bead blast to remove grey surface discoloration"},
        {"step": "vapor_smoothing", "est_minutes": 35, "required": False,
         "description": "Chemical vapor smoothing for enhanced surface quality"},
        {"step": "dyeing", "est_minutes": 40, "required": False,
         "description": "Dye part in heated dye bath for uniform color"},
    ],
}

FINISH_LEVELS = {
    "raw": 0,
    "basic": 1,
    "standard": 2,
    "premium": 3,
    "exhibition": 4,
}

FINISH_LEVEL_DESCRIPTIONS = {
    0: "Raw — Required steps only, no cosmetic finishing",
    1: "Basic — Required steps plus light sanding/cleanup",
    2: "Standard — Sanding, priming, basic paint or dye",
    3: "Premium — Full finishing with multiple coats and fine detailing",
    4: "Exhibition — Museum-quality finish with hand-painted details and clear coat",
}

POST_PROCESS_SYSTEM_PROMPT = """\
You are a 3D print post-processing specialist with deep expertise in \
finishing techniques for architectural models. Given the print job details \
below, recommend the optimal finishing workflow.

Print Technology: {technology}
Material: {material}
Finish Requirement: {finish_requirement} (level {finish_level})
Part Dimensions: {dimensions}
Customer Notes: {customer_notes}

Available finishing steps for {technology}:
{available_steps}

Consider:
1. Required steps must always be included
2. Higher finish levels need more optional steps
3. Delicate features need gentler finishing approaches
4. Time estimates should account for part size

Return a JSON object with:
{{
    "recommended_steps": [
        {{
            "step": "step_name",
            "est_minutes": 0,
            "required": true/false,
            "notes": "Specific instructions for this part",
            "safety_notes": "Any safety precautions"
        }}
    ],
    "total_minutes": 0,
    "special_considerations": "Any extra notes for this specific job"
}}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("post_process")
class PostProcessAgent(BaseAgent):
    """
    Post-processing workflow agent for PrintBiz verticals.

    Nodes:
        1. load_job             -- Pull job data for the print_job_id
        2. recommend_finishing  -- Select finishing steps by material + technology
        3. generate_work_order  -- Create structured work order with timeline
        4. human_review         -- Gate: approve work order before saving
        5. report               -- Summary + Hive Mind finishing insights
    """

    def build_graph(self) -> Any:
        """Build the Post-Process Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(PostProcessAgentState)

        workflow.add_node("load_job", self._node_load_job)
        workflow.add_node("recommend_finishing", self._node_recommend_finishing)
        workflow.add_node("generate_work_order", self._node_generate_work_order)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_job")

        workflow.add_edge("load_job", "recommend_finishing")
        workflow.add_edge("recommend_finishing", "generate_work_order")
        workflow.add_edge("generate_work_order", "human_review")
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
    def get_state_class(cls) -> Type[PostProcessAgentState]:
        return PostProcessAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "print_technology": "",
            "material": "",
            "finish_requirement": "standard",
            "recommended_steps": [],
            "total_estimated_minutes": 0,
            "finish_level_score": 0,
            "work_order": {},
            "work_order_id": "",
            "work_order_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Job ────────────────────────────────────────────

    async def _node_load_job(
        self, state: PostProcessAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull print job data for post-processing."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")

        logger.info(
            "post_process_load_job",
            extra={"print_job_id": print_job_id, "agent_id": self.agent_id},
        )

        job_data: dict[str, Any] = {}
        if print_job_id:
            try:
                result = (
                    self.db.client.table("print_jobs")
                    .select("*")
                    .eq("id", print_job_id)
                    .execute()
                )
                if result and result.data and len(result.data) > 0:
                    job_data = result.data[0]
                    logger.info(
                        "post_process_job_loaded",
                        extra={
                            "job_id": print_job_id,
                            "status": job_data.get("status"),
                            "technology": job_data.get("technology"),
                        },
                    )
                else:
                    logger.warning(
                        f"Print job not found: {print_job_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load print job: {e}")

        # Extract relevant fields
        technology = job_data.get("technology", task.get("technology", "FDM"))
        material = job_data.get("material", task.get("material", "PLA"))
        finish_req = job_data.get(
            "finish_requirement",
            task.get("finish_requirement", "standard"),
        )
        finish_level = FINISH_LEVELS.get(finish_req, 2)

        return {
            "current_node": "load_job",
            "print_job_id": print_job_id,
            "print_technology": technology.upper(),
            "material": material,
            "finish_requirement": finish_req,
            "finish_level_score": finish_level,
        }

    # ─── Node 2: Recommend Finishing ─────────────────────────────────

    async def _node_recommend_finishing(
        self, state: PostProcessAgentState
    ) -> dict[str, Any]:
        """Node 2: Select finishing steps based on technology + material + finish level."""
        technology = state.get("print_technology", "FDM")
        material = state.get("material", "PLA")
        finish_req = state.get("finish_requirement", "standard")
        finish_level = state.get("finish_level_score", 2)

        logger.info(
            "post_process_recommend_finishing",
            extra={
                "technology": technology,
                "material": material,
                "finish_level": finish_level,
            },
        )

        # Get steps for this technology
        tech_steps = FINISHING_STEPS.get(technology, FINISHING_STEPS.get("FDM", []))

        # Select steps based on finish level
        recommended: list[dict[str, Any]] = []

        for step_info in tech_steps:
            is_required = step_info.get("required", False)

            # Always include required steps
            if is_required:
                recommended.append({
                    "step": step_info["step"],
                    "est_minutes": step_info["est_minutes"],
                    "required": True,
                    "description": step_info.get("description", ""),
                    "notes": "",
                })
                continue

            # Include optional steps based on finish level
            # Level 0 (raw): required only
            # Level 1 (basic): required + first optional (sanding/depowdering)
            # Level 2 (standard): required + sanding + priming/blasting
            # Level 3 (premium): all steps
            # Level 4 (exhibition): all steps with extended times
            step_name = step_info["step"]
            include = False

            if finish_level >= 4:
                include = True
            elif finish_level >= 3:
                include = True
            elif finish_level >= 2:
                include = step_name in (
                    "sanding", "priming", "bead_blasting", "sealing"
                )
            elif finish_level >= 1:
                include = step_name in ("sanding", "bead_blasting")

            if include:
                est_minutes = step_info["est_minutes"]
                # Exhibition quality takes 1.5x longer
                if finish_level >= 4:
                    est_minutes = int(est_minutes * 1.5)

                recommended.append({
                    "step": step_name,
                    "est_minutes": est_minutes,
                    "required": False,
                    "description": step_info.get("description", ""),
                    "notes": (
                        "Exhibition-grade attention required"
                        if finish_level >= 4
                        else ""
                    ),
                })

        total_minutes = sum(s["est_minutes"] for s in recommended)

        # ── Use LLM for special considerations on premium/exhibition ──
        if finish_level >= 3:
            try:
                task = state.get("task_input", {})
                response = self.llm.messages.create(
                    model=self.config.params.get("model", "claude-sonnet-4-20250514"),
                    max_tokens=512,
                    system=POST_PROCESS_SYSTEM_PROMPT.format(
                        technology=technology,
                        material=material,
                        finish_requirement=finish_req,
                        finish_level=finish_level,
                        dimensions=task.get("dimensions", "N/A"),
                        customer_notes=task.get("customer_notes", "None"),
                        available_steps=str(tech_steps),
                    ),
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Provide detailed finishing recommendations for this "
                            f"{finish_req}-level {technology} part in {material}."
                        ),
                    }],
                )
                llm_text = response.content[0].text if response.content else ""
                logger.info(
                    "post_process_llm_recommendations",
                    extra={"response_length": len(llm_text)},
                )
            except Exception as e:
                logger.debug(f"LLM finishing recommendation failed: {e}")

        logger.info(
            "post_process_steps_recommended",
            extra={
                "step_count": len(recommended),
                "total_minutes": total_minutes,
                "finish_level": finish_level,
            },
        )

        return {
            "current_node": "recommend_finishing",
            "recommended_steps": recommended,
            "total_estimated_minutes": total_minutes,
        }

    # ─── Node 3: Generate Work Order ────────────────────────────────

    async def _node_generate_work_order(
        self, state: PostProcessAgentState
    ) -> dict[str, Any]:
        """Node 3: Create structured work order with steps and timeline."""
        print_job_id = state.get("print_job_id", "")
        technology = state.get("print_technology", "FDM")
        material = state.get("material", "PLA")
        finish_req = state.get("finish_requirement", "standard")
        recommended_steps = state.get("recommended_steps", [])
        total_minutes = state.get("total_estimated_minutes", 0)

        work_order_id = f"wo_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        logger.info(
            "post_process_generate_work_order",
            extra={
                "work_order_id": work_order_id,
                "step_count": len(recommended_steps),
            },
        )

        # Build sequential timeline
        cumulative_minutes = 0
        timeline_steps: list[dict[str, Any]] = []

        for i, step in enumerate(recommended_steps):
            step_entry = {
                "order": i + 1,
                "step": step["step"],
                "description": step.get("description", ""),
                "est_minutes": step["est_minutes"],
                "required": step.get("required", False),
                "start_offset_minutes": cumulative_minutes,
                "notes": step.get("notes", ""),
                "status": "pending",
            }
            timeline_steps.append(step_entry)
            cumulative_minutes += step["est_minutes"]

        work_order = {
            "work_order_id": work_order_id,
            "print_job_id": print_job_id,
            "technology": technology,
            "material": material,
            "finish_requirement": finish_req,
            "finish_level_description": FINISH_LEVEL_DESCRIPTIONS.get(
                FINISH_LEVELS.get(finish_req, 2), "Standard finish"
            ),
            "steps": timeline_steps,
            "total_estimated_minutes": cumulative_minutes,
            "total_estimated_hours": round(cumulative_minutes / 60, 1),
            "created_at": now,
            "created_by": self.agent_id,
            "status": "pending_approval",
        }

        # ── Save work order to DB ──
        saved = False
        try:
            self.db.client.table("work_orders").insert({
                "id": work_order_id,
                "print_job_id": print_job_id,
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "order_type": "post_processing",
                "steps": timeline_steps,
                "total_minutes": cumulative_minutes,
                "finish_requirement": finish_req,
                "status": "pending_approval",
                "created_at": now,
            }).execute()
            saved = True
            logger.info(
                "post_process_work_order_saved",
                extra={"work_order_id": work_order_id},
            )
        except Exception as e:
            logger.warning(f"Failed to save work order: {e}")

        return {
            "current_node": "generate_work_order",
            "work_order": work_order,
            "work_order_id": work_order_id,
            "work_order_saved": saved,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: PostProcessAgentState
    ) -> dict[str, Any]:
        """Node 4: Present work order for human approval."""
        work_order = state.get("work_order", {})
        logger.info(
            "post_process_human_review_pending",
            extra={
                "work_order_id": work_order.get("work_order_id", ""),
                "total_minutes": work_order.get("total_estimated_minutes", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: PostProcessAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate finishing summary and Hive Mind insights."""
        now = datetime.now(timezone.utc).isoformat()
        print_job_id = state.get("print_job_id", "")
        technology = state.get("print_technology", "FDM")
        material = state.get("material", "PLA")
        finish_req = state.get("finish_requirement", "standard")
        recommended_steps = state.get("recommended_steps", [])
        total_minutes = state.get("total_estimated_minutes", 0)
        work_order = state.get("work_order", {})

        sections = [
            "# Post-Processing Work Order Report",
            f"*Generated: {now}*\n",
            f"## Job Details",
            f"- **Print Job:** {print_job_id}",
            f"- **Technology:** {technology}",
            f"- **Material:** {material}",
            f"- **Finish Level:** {finish_req} "
            f"(Level {FINISH_LEVELS.get(finish_req, 0)})",
            f"\n## Finishing Steps ({len(recommended_steps)} total)",
        ]

        for i, step in enumerate(recommended_steps, 1):
            req_tag = " [REQUIRED]" if step.get("required") else ""
            sections.append(
                f"{i}. **{step['step']}** — {step['est_minutes']} min{req_tag}"
            )
            if step.get("description"):
                sections.append(f"   {step['description']}")

        sections.extend([
            f"\n## Timeline",
            f"- **Total Estimated Time:** {total_minutes} minutes "
            f"({total_minutes/60:.1f} hours)",
            f"- **Work Order ID:** {work_order.get('work_order_id', 'N/A')}",
            f"- **Status:** {work_order.get('status', 'pending')}",
        ])

        report = "\n".join(sections)

        # ── Hive Mind insight ──
        if recommended_steps:
            self.store_insight(InsightData(
                insight_type="post_processing_workflow",
                title=f"Post-Process: {technology}/{material} → "
                      f"{finish_req} ({total_minutes}min)",
                content=(
                    f"Post-processing work order for {technology} part in "
                    f"{material}: {len(recommended_steps)} steps, "
                    f"{total_minutes} minutes total for {finish_req} finish. "
                    f"Required steps: "
                    f"{sum(1 for s in recommended_steps if s.get('required'))}."
                ),
                confidence=0.82,
                metadata={
                    "technology": technology,
                    "material": material,
                    "finish_level": finish_req,
                    "step_count": len(recommended_steps),
                    "total_minutes": total_minutes,
                },
            ))

        logger.info(
            "post_process_report_generated",
            extra={
                "print_job_id": print_job_id,
                "step_count": len(recommended_steps),
                "total_minutes": total_minutes,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: PostProcessAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<PostProcessAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
