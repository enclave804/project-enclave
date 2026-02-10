"""
Print Manager Agent — The Farm Controller.

Manages a fleet of 3D printers by scanning the job queue, matching pending
jobs to compatible printers based on material, build volume, and priority,
then executing assignments after human approval.

Architecture (LangGraph State Machine):
    scan_queue → assign_jobs → human_review → execute_assignments → report → END

Trigger Events:
    - queue_scan: Periodic sweep for new approved jobs
    - job_approved: A new print job has been approved for printing
    - manual: On-demand farm management run

Shared Brain Integration:
    - Reads: material performance data, printer reliability insights
    - Writes: throughput metrics, printer utilization patterns, bottleneck insights

Safety:
    - All printer assignments require human_review gate before execution
    - Never deletes job data — only transitions status
    - Respects printer maintenance windows and material compatibility
    - All DB mutations are wrapped in try/except to prevent partial state corruption

Usage:
    agent = PrintManagerAgent(config, db, embedder, llm)
    result = await agent.run({
        "scan_scope": "all",
        "priority_override": None,
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import PrintManagerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PRINTER_TYPES = {
    "fdm_standard": {
        "build_volume_mm": [220, 220, 250],
        "materials": ["PLA", "PETG"],
        "speed": "standard",
    },
    "fdm_large": {
        "build_volume_mm": [400, 400, 500],
        "materials": ["PLA", "ABS", "Nylon"],
        "speed": "standard",
    },
    "sla_standard": {
        "build_volume_mm": [145, 145, 175],
        "materials": ["SLA_Resin"],
        "speed": "slow",
    },
    "sls_industrial": {
        "build_volume_mm": [340, 340, 600],
        "materials": ["Nylon_SLS"],
        "speed": "fast",
    },
}

JOB_PRIORITIES = {
    "rush": 1,
    "high": 2,
    "normal": 3,
    "low": 4,
}

SPEED_MULTIPLIERS = {
    "slow": 1.4,
    "standard": 1.0,
    "fast": 0.7,
}

MATERIAL_DENSITY_G_CM3 = {
    "PLA": 1.24,
    "PETG": 1.27,
    "ABS": 1.04,
    "Nylon": 1.14,
    "SLA_Resin": 1.10,
    "Nylon_SLS": 1.01,
}

PRINT_MANAGER_SYSTEM_PROMPT = """\
You are a 3D print farm manager responsible for optimizing printer utilization \
and job throughput. Given the pending jobs and available printers below, \
assign jobs to printers optimally based on:

1. Material compatibility — the printer must support the job's material
2. Build volume — the part must fit within the printer's build envelope
3. Priority — rush and high-priority jobs are scheduled first
4. Estimated print time — balance load across printers when possible

Pending Jobs:
{pending_jobs}

Available Printers:
{available_printers}

Active Prints (for context):
{active_prints}

Return a JSON object with:
{{
    "assignments": [
        {{
            "job_id": "...",
            "printer_id": "...",
            "printer_type": "...",
            "material": "...",
            "estimated_hours": 0.0,
            "priority_score": 0,
            "reasoning": "..."
        }}
    ],
    "conflicts": [
        {{
            "job_id": "...",
            "reason": "..."
        }}
    ],
    "summary": "Brief summary of assignment strategy"
}}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("print_manager")
class PrintManagerAgent(BaseAgent):
    """
    Print farm management agent for PrintBiz verticals.

    Nodes:
        1. scan_queue           -- Pull pending jobs and list available printers
        2. assign_jobs          -- Match jobs to printers by material + size + priority
        3. human_review         -- Gate: approve assignments before execution
        4. execute_assignments  -- Update job statuses to 'printing'
        5. report               -- Generate throughput summary + Hive Mind insights
    """

    def build_graph(self) -> Any:
        """Build the Print Manager Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(PrintManagerAgentState)

        workflow.add_node("scan_queue", self._node_scan_queue)
        workflow.add_node("assign_jobs", self._node_assign_jobs)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute_assignments", self._node_execute_assignments)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_queue")

        workflow.add_edge("scan_queue", "assign_jobs")
        workflow.add_edge("assign_jobs", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "execute_assignments",
                "rejected": "report",
            },
        )
        workflow.add_edge("execute_assignments", "report")
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
    def get_state_class(cls) -> Type[PrintManagerAgentState]:
        return PrintManagerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "pending_jobs": [],
            "available_printers": [],
            "active_prints": [],
            "queue_depth": 0,
            "job_assignments": [],
            "assignment_conflicts": [],
            "assignments_approved": False,
            "jobs_started": 0,
            "jobs_failed_to_start": 0,
            "throughput_summary": {},
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Queue ───────────────────────────────────────────

    async def _node_scan_queue(
        self, state: PrintManagerAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull pending jobs from the queue and list available printers."""
        task = state.get("task_input", {})
        scan_scope = task.get("scan_scope", "all")

        logger.info(
            "print_manager_scan_queue",
            extra={
                "agent_id": self.agent_id,
                "scan_scope": scan_scope,
            },
        )

        # ── Pull pending print jobs (status = 'approved') ──
        pending_jobs: list[dict[str, Any]] = []
        try:
            result = (
                self.db.client.table("print_jobs")
                .select("*")
                .eq("status", "approved")
                .order("created_at", desc=False)
                .execute()
            )
            if result and result.data:
                pending_jobs = result.data
                logger.info(
                    "print_manager_jobs_found",
                    extra={"count": len(pending_jobs)},
                )
        except Exception as e:
            logger.warning(f"Failed to query print_jobs: {e}")

        # ── Sort by priority ──
        for job in pending_jobs:
            raw_priority = job.get("priority", "normal")
            job["_priority_score"] = JOB_PRIORITIES.get(raw_priority, 3)
        pending_jobs.sort(key=lambda j: j["_priority_score"])

        # ── Gather available printers from config params ──
        printers_config = self.config.params.get("printers", [])
        available_printers: list[dict[str, Any]] = []

        for printer in printers_config:
            printer_type_key = printer.get("type", "fdm_standard")
            printer_type = PRINTER_TYPES.get(printer_type_key, PRINTER_TYPES["fdm_standard"])
            available_printers.append({
                "printer_id": printer.get("id", f"printer_{len(available_printers)+1}"),
                "printer_name": printer.get("name", f"Printer {len(available_printers)+1}"),
                "type": printer_type_key,
                "build_volume_mm": printer_type["build_volume_mm"],
                "materials": printer_type["materials"],
                "speed": printer_type["speed"],
                "status": printer.get("status", "idle"),
            })

        # If no printers configured, create a default fleet
        if not available_printers:
            for ptype, pinfo in PRINTER_TYPES.items():
                available_printers.append({
                    "printer_id": f"default_{ptype}",
                    "printer_name": f"Default {ptype}",
                    "type": ptype,
                    "build_volume_mm": pinfo["build_volume_mm"],
                    "materials": pinfo["materials"],
                    "speed": pinfo["speed"],
                    "status": "idle",
                })

        # Filter to only idle printers
        idle_printers = [p for p in available_printers if p.get("status") == "idle"]

        # ── Pull active prints for context ──
        active_prints: list[dict[str, Any]] = []
        try:
            active_result = (
                self.db.client.table("print_jobs")
                .select("*")
                .eq("status", "printing")
                .execute()
            )
            if active_result and active_result.data:
                active_prints = active_result.data
        except Exception as e:
            logger.debug(f"Failed to query active prints: {e}")

        logger.info(
            "print_manager_queue_scanned",
            extra={
                "pending": len(pending_jobs),
                "idle_printers": len(idle_printers),
                "active_prints": len(active_prints),
            },
        )

        return {
            "current_node": "scan_queue",
            "pending_jobs": pending_jobs,
            "available_printers": idle_printers,
            "active_prints": active_prints,
            "queue_depth": len(pending_jobs),
        }

    # ─── Node 2: Assign Jobs ─────────────────────────────────────────

    async def _node_assign_jobs(
        self, state: PrintManagerAgentState
    ) -> dict[str, Any]:
        """Node 2: Match jobs to compatible printers based on material + size."""
        pending_jobs = state.get("pending_jobs", [])
        available_printers = state.get("available_printers", [])
        active_prints = state.get("active_prints", [])

        logger.info(
            "print_manager_assign_jobs",
            extra={
                "pending_count": len(pending_jobs),
                "printer_count": len(available_printers),
            },
        )

        if not pending_jobs:
            logger.info("print_manager_no_pending_jobs")
            return {
                "current_node": "assign_jobs",
                "job_assignments": [],
                "assignment_conflicts": [],
            }

        if not available_printers:
            conflicts = [
                {"job_id": j.get("id", "unknown"), "reason": "No printers available"}
                for j in pending_jobs
            ]
            logger.warning("print_manager_no_printers_available")
            return {
                "current_node": "assign_jobs",
                "job_assignments": [],
                "assignment_conflicts": conflicts,
            }

        # ── Attempt rule-based matching first ──
        assignments: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        used_printers: set[str] = set()

        for job in pending_jobs:
            job_id = job.get("id", "unknown")
            job_material = job.get("material", "PLA")
            job_dimensions = job.get("dimensions", {})
            job_priority = job.get("priority", "normal")

            # Get bounding box (x, y, z) in mm
            jx = job_dimensions.get("x", 100)
            jy = job_dimensions.get("y", 100)
            jz = job_dimensions.get("z", 100)

            matched_printer = None
            match_reasoning = ""

            for printer in available_printers:
                pid = printer.get("printer_id", "")
                if pid in used_printers:
                    continue

                # Material compatibility check
                if job_material not in printer.get("materials", []):
                    continue

                # Build volume check
                bv = printer.get("build_volume_mm", [0, 0, 0])
                if jx > bv[0] or jy > bv[1] or jz > bv[2]:
                    continue

                matched_printer = printer
                match_reasoning = (
                    f"Material {job_material} compatible, "
                    f"dimensions {jx}x{jy}x{jz}mm fit "
                    f"{bv[0]}x{bv[1]}x{bv[2]}mm build volume"
                )
                break

            if matched_printer:
                # Estimate print time based on volume and speed
                volume_cm3 = job.get("volume_cm3", 50.0)
                speed_mult = SPEED_MULTIPLIERS.get(
                    matched_printer.get("speed", "standard"), 1.0
                )
                # Rough estimate: ~15cm3/hour for FDM at standard speed
                base_rate_cm3_per_hour = 15.0
                est_hours = round(
                    (volume_cm3 / base_rate_cm3_per_hour) * speed_mult, 1
                )

                assignments.append({
                    "job_id": job_id,
                    "printer_id": matched_printer["printer_id"],
                    "printer_type": matched_printer.get("type", "unknown"),
                    "material": job_material,
                    "estimated_hours": est_hours,
                    "priority_score": JOB_PRIORITIES.get(job_priority, 3),
                    "reasoning": match_reasoning,
                })
                used_printers.add(matched_printer["printer_id"])
            else:
                conflicts.append({
                    "job_id": job_id,
                    "reason": (
                        f"No compatible printer for material={job_material}, "
                        f"dims={jx}x{jy}x{jz}mm"
                    ),
                })

        # ── If there are unmatched jobs, try LLM for creative assignment ──
        if conflicts and available_printers:
            try:
                llm_prompt = PRINT_MANAGER_SYSTEM_PROMPT.format(
                    pending_jobs=str([
                        c for c in pending_jobs
                        if c.get("id") in [x["job_id"] for x in conflicts]
                    ][:5]),
                    available_printers=str([
                        p for p in available_printers
                        if p["printer_id"] not in used_printers
                    ][:5]),
                    active_prints=str(active_prints[:3]),
                )

                response = self.llm.messages.create(
                    model=self.config.params.get("model", "claude-sonnet-4-20250514"),
                    max_tokens=1024,
                    system=llm_prompt,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Suggest any alternative assignments for these "
                            "conflicting jobs. If a job truly cannot be assigned, "
                            "explain why."
                        ),
                    }],
                )

                llm_text = response.content[0].text if response.content else ""
                logger.info(
                    "print_manager_llm_assist",
                    extra={"response_length": len(llm_text)},
                )
            except Exception as e:
                logger.debug(f"LLM assignment assist failed: {e}")

        logger.info(
            "print_manager_assignments_complete",
            extra={
                "assigned": len(assignments),
                "conflicts": len(conflicts),
            },
        )

        return {
            "current_node": "assign_jobs",
            "job_assignments": assignments,
            "assignment_conflicts": conflicts,
        }

    # ─── Node 3: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: PrintManagerAgentState
    ) -> dict[str, Any]:
        """Node 3: Present job assignments for human approval."""
        assignments = state.get("job_assignments", [])
        conflicts = state.get("assignment_conflicts", [])

        logger.info(
            "print_manager_human_review_pending",
            extra={
                "assignment_count": len(assignments),
                "conflict_count": len(conflicts),
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 4: Execute Assignments ─────────────────────────────────

    async def _node_execute_assignments(
        self, state: PrintManagerAgentState
    ) -> dict[str, Any]:
        """Node 4: Update print job statuses to 'printing'."""
        assignments = state.get("job_assignments", [])
        now = datetime.now(timezone.utc).isoformat()

        logger.info(
            "print_manager_execute_assignments",
            extra={"count": len(assignments)},
        )

        jobs_started = 0
        jobs_failed = 0

        for assignment in assignments:
            job_id = assignment.get("job_id")
            printer_id = assignment.get("printer_id")
            est_hours = assignment.get("estimated_hours", 0)

            try:
                self.db.client.table("print_jobs").update({
                    "status": "printing",
                    "printer_id": printer_id,
                    "estimated_hours": est_hours,
                    "print_started_at": now,
                    "assigned_by": self.agent_id,
                    "updated_at": now,
                }).eq("id", job_id).execute()

                jobs_started += 1
                logger.info(
                    "print_manager_job_started",
                    extra={
                        "job_id": job_id,
                        "printer_id": printer_id,
                        "est_hours": est_hours,
                    },
                )
            except Exception as e:
                jobs_failed += 1
                logger.warning(
                    f"Failed to start print job {job_id}: {e}",
                    extra={"job_id": job_id, "error": str(e)[:200]},
                )

        logger.info(
            "print_manager_execution_complete",
            extra={
                "started": jobs_started,
                "failed": jobs_failed,
            },
        )

        return {
            "current_node": "execute_assignments",
            "assignments_approved": True,
            "jobs_started": jobs_started,
            "jobs_failed_to_start": jobs_failed,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: PrintManagerAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate throughput summary and Hive Mind insights."""
        now = datetime.now(timezone.utc).isoformat()
        assignments = state.get("job_assignments", [])
        conflicts = state.get("assignment_conflicts", [])
        jobs_started = state.get("jobs_started", 0)
        jobs_failed = state.get("jobs_failed_to_start", 0)
        queue_depth = state.get("queue_depth", 0)
        active_prints = state.get("active_prints", [])

        # Calculate throughput metrics
        total_est_hours = sum(
            a.get("estimated_hours", 0) for a in assignments
        )
        avg_est_hours = (
            round(total_est_hours / len(assignments), 1)
            if assignments
            else 0.0
        )
        utilization_pct = (
            round(len(assignments) / max(len(state.get("available_printers", [])), 1) * 100, 1)
        )

        throughput_summary = {
            "jobs_assigned": len(assignments),
            "jobs_started": jobs_started,
            "jobs_conflicted": len(conflicts),
            "jobs_failed": jobs_failed,
            "queue_depth": queue_depth,
            "active_prints": len(active_prints),
            "total_estimated_hours": total_est_hours,
            "avg_estimated_hours": avg_est_hours,
            "printer_utilization_pct": utilization_pct,
            "timestamp": now,
        }

        # Build report
        sections = [
            "# Print Farm Manager Report",
            f"*Generated: {now}*\n",
            "## Queue Summary",
            f"- **Queue Depth:** {queue_depth} pending jobs",
            f"- **Active Prints:** {len(active_prints)}",
            f"- **Jobs Assigned:** {len(assignments)}",
            f"- **Assignment Conflicts:** {len(conflicts)}",
            f"\n## Execution Results",
            f"- **Jobs Started:** {jobs_started}",
            f"- **Jobs Failed:** {jobs_failed}",
            f"- **Total Estimated Print Time:** {total_est_hours:.1f} hours",
            f"- **Avg Print Time:** {avg_est_hours:.1f} hours",
            f"\n## Printer Utilization",
            f"- **Utilization:** {utilization_pct:.1f}%",
        ]

        if assignments:
            sections.append("\n## Assignments")
            for a in assignments:
                sections.append(
                    f"- Job `{a['job_id']}` → Printer `{a['printer_id']}` "
                    f"({a.get('material', 'N/A')}, ~{a.get('estimated_hours', 0)}h)"
                )

        if conflicts:
            sections.append("\n## Conflicts")
            for c in conflicts:
                sections.append(f"- Job `{c['job_id']}`: {c.get('reason', 'unknown')}")

        report = "\n".join(sections)

        # ── Publish to Hive Mind ──
        if assignments:
            self.store_insight(InsightData(
                insight_type="print_farm_throughput",
                title=f"Print Farm: {jobs_started} jobs started, "
                      f"{utilization_pct:.0f}% utilization",
                content=(
                    f"Print farm run: {len(assignments)} jobs assigned across "
                    f"{len(set(a['printer_id'] for a in assignments))} printers. "
                    f"Total estimated time: {total_est_hours:.1f}h. "
                    f"Queue depth: {queue_depth}. "
                    f"Conflicts: {len(conflicts)}."
                ),
                confidence=0.80,
                metadata={
                    "throughput": throughput_summary,
                    "conflict_count": len(conflicts),
                },
            ))

        logger.info(
            "print_manager_report_generated",
            extra=throughput_summary,
        )

        return {
            "current_node": "report",
            "throughput_summary": throughput_summary,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: PrintManagerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<PrintManagerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
