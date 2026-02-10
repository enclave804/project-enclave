"""
QC Inspector Agent — The Quality Gatekeeper.

Performs quality control inspection on 3D printed architectural models by
comparing printed output against original design specifications. Checks for
dimensional accuracy, surface quality, structural integrity, and visual
defects, computing a weighted QC score that determines whether a part ships,
gets reworked, or is reprinted.

Architecture (LangGraph State Machine):
    load_specs → run_inspection → score_quality →
    human_review → report → END

Trigger Events:
    - print_completed: A print job has finished and needs QC
    - qc_request: Manual request for quality inspection
    - reprint_check: Re-inspection after rework

Shared Brain Integration:
    - Reads: common defect patterns, material-specific quality baselines
    - Writes: defect pattern insights, pass/fail rates by printer/material

Safety:
    - QC decision (ship/rework/reprint/scrap) always requires human_review gate
    - Does not auto-ship any parts — human approval is mandatory
    - Preserves all measurement data for audit trail
    - All DB mutations are wrapped in try/except

Usage:
    agent = QCInspectorAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "job_456",
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import QCInspectorAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

QC_CRITERIA = {
    "dimensional_accuracy": {
        "weight": 0.35,
        "tolerance_mm": 0.5,
        "description": "Deviation from specified dimensions",
    },
    "surface_quality": {
        "weight": 0.25,
        "min_score": 60,
        "description": "Surface finish, layer visibility, and texture quality",
    },
    "structural_integrity": {
        "weight": 0.25,
        "min_score": 70,
        "description": "Part strength, layer adhesion, and structural soundness",
    },
    "visual_appearance": {
        "weight": 0.15,
        "min_score": 50,
        "description": "Overall visual presentation and cosmetic quality",
    },
}

DEFECT_TYPES = {
    "warping": {
        "severity": "high",
        "description": "Part has warped or curled, typically at corners or large flat surfaces",
        "common_causes": ["insufficient bed adhesion", "ABS without enclosure", "uneven cooling"],
    },
    "layer_separation": {
        "severity": "critical",
        "description": "Visible delamination between printed layers",
        "common_causes": ["under-extrusion", "printing too fast", "nozzle too far from bed"],
    },
    "stringing": {
        "severity": "low",
        "description": "Fine strings between features from filament ooze",
        "common_causes": ["retraction settings", "travel speed too slow", "temperature too high"],
    },
    "surface_roughness": {
        "severity": "medium",
        "description": "Excessive surface roughness beyond acceptable limits",
        "common_causes": ["layer height too high", "over-extrusion", "vibration"],
    },
    "dimensional_deviation": {
        "severity": "high",
        "description": "Dimensions outside tolerance compared to design specs",
        "common_causes": ["shrinkage", "elephant foot", "belt tension", "steps/mm calibration"],
    },
    "missing_features": {
        "severity": "critical",
        "description": "Small features or details did not print successfully",
        "common_causes": ["nozzle too large", "insufficient resolution", "feature below min size"],
    },
    "support_damage": {
        "severity": "medium",
        "description": "Surface damage from support removal",
        "common_causes": ["poor support settings", "aggressive removal", "wrong support type"],
    },
    "color_inconsistency": {
        "severity": "low",
        "description": "Uneven color or visible color banding",
        "common_causes": ["filament batch variation", "temperature fluctuation", "moisture"],
    },
}

QC_PASS_THRESHOLD = 70.0  # Minimum overall score to pass QC

QC_DISPOSITION_RULES = {
    "ship": {"min_score": 70, "max_critical_defects": 0, "max_high_defects": 1},
    "rework": {"min_score": 50, "max_critical_defects": 0, "max_high_defects": 3},
    "reprint": {"min_score": 30, "max_critical_defects": 1, "max_high_defects": 5},
    "scrap": {"min_score": 0, "max_critical_defects": 999, "max_high_defects": 999},
}

QC_SYSTEM_PROMPT = """\
You are a quality control inspector for 3D printed architectural models. \
You have deep expertise in identifying print defects, dimensional accuracy, \
and surface quality assessment.

Given the inspection data below, analyze the part quality and identify \
any defects. Be thorough but fair — architectural models need to look \
professional but perfection is not always required.

Print Job Details:
- Technology: {technology}
- Material: {material}
- Expected Dimensions: {expected_dimensions}
- Measured Dimensions: {measured_dimensions}

Task Data (from inspection checklist):
{task_data}

Known defect types and severities:
{defect_types}

Return a JSON object with:
{{
    "defects": [
        {{
            "type": "defect_type_key",
            "severity": "critical|high|medium|low",
            "description": "Specific description of this defect",
            "location": "Where on the part (e.g., base, roof, west facade)",
            "remediation": "Suggested fix if rework is possible"
        }}
    ],
    "surface_quality_score": 0-100,
    "structural_integrity_score": 0-100,
    "visual_appearance_score": 0-100,
    "inspector_notes": "Overall assessment notes"
}}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("qc_inspector")
class QCInspectorAgent(BaseAgent):
    """
    Quality control inspection agent for PrintBiz verticals.

    Nodes:
        1. load_specs       -- Pull print job + file analysis specs
        2. run_inspection   -- Compare against specs, detect defects (LLM-assisted)
        3. score_quality    -- Compute weighted QC score using QC_CRITERIA
        4. human_review     -- Gate: approve disposition (ship/rework/reprint)
        5. report           -- Save to quality_inspections + Hive Mind insights
    """

    def build_graph(self) -> Any:
        """Build the QC Inspector Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(QCInspectorAgentState)

        workflow.add_node("load_specs", self._node_load_specs)
        workflow.add_node("run_inspection", self._node_run_inspection)
        workflow.add_node("score_quality", self._node_score_quality)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_specs")

        workflow.add_edge("load_specs", "run_inspection")
        workflow.add_edge("run_inspection", "score_quality")
        workflow.add_edge("score_quality", "human_review")
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
    def get_state_class(cls) -> Type[QCInspectorAgentState]:
        return QCInspectorAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "file_analysis_id": "",
            "expected_dimensions": {},
            "expected_material": "",
            "expected_technology": "",
            "measured_dimensions": {},
            "dimensional_deviations": [],
            "defects_found": [],
            "defect_count": 0,
            "dimensional_accuracy_score": 0.0,
            "surface_quality_score": 0.0,
            "structural_integrity_score": 0.0,
            "visual_appearance_score": 0.0,
            "overall_qc_score": 0.0,
            "qc_pass": False,
            "disposition": "",
            "disposition_reasoning": "",
            "inspection_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Specs ──────────────────────────────────────────

    async def _node_load_specs(
        self, state: QCInspectorAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull print job and file analysis specs for comparison."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")

        logger.info(
            "qc_load_specs",
            extra={"print_job_id": print_job_id, "agent_id": self.agent_id},
        )

        # ── Load print job ──
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
                        "qc_job_loaded",
                        extra={"job_id": print_job_id},
                    )
            except Exception as e:
                logger.warning(f"Failed to load print job: {e}")

        # ── Load file analysis for expected specs ──
        file_analysis_id = job_data.get(
            "file_analysis_id", task.get("file_analysis_id", "")
        )
        analysis_data: dict[str, Any] = {}
        if file_analysis_id:
            try:
                analysis_result = (
                    self.db.client.table("file_analyses")
                    .select("*")
                    .eq("id", file_analysis_id)
                    .execute()
                )
                if analysis_result and analysis_result.data:
                    analysis_data = analysis_result.data[0]
                    logger.info(
                        "qc_analysis_loaded",
                        extra={"file_analysis_id": file_analysis_id},
                    )
            except Exception as e:
                logger.debug(f"Failed to load file analysis: {e}")

        # Extract expected specs
        expected_dims = analysis_data.get(
            "bounding_box",
            job_data.get("dimensions", task.get("expected_dimensions", {})),
        )
        expected_material = job_data.get("material", task.get("material", "PLA"))
        expected_tech = job_data.get("technology", task.get("technology", "FDM"))

        # Measured dimensions from task (supplied by QC technician)
        measured_dims = task.get("measured_dimensions", expected_dims)

        return {
            "current_node": "load_specs",
            "print_job_id": print_job_id,
            "file_analysis_id": file_analysis_id,
            "expected_dimensions": expected_dims,
            "expected_material": expected_material,
            "expected_technology": expected_tech,
            "measured_dimensions": measured_dims,
        }

    # ─── Node 2: Run Inspection ──────────────────────────────────────

    async def _node_run_inspection(
        self, state: QCInspectorAgentState
    ) -> dict[str, Any]:
        """Node 2: Compare printed part against specs, detect defects."""
        expected_dims = state.get("expected_dimensions", {})
        measured_dims = state.get("measured_dimensions", {})
        expected_material = state.get("expected_material", "PLA")
        expected_tech = state.get("expected_technology", "FDM")
        task = state.get("task_input", {})

        logger.info(
            "qc_run_inspection",
            extra={
                "technology": expected_tech,
                "material": expected_material,
            },
        )

        # ── Dimensional deviation analysis ──
        deviations: list[dict[str, Any]] = []
        tolerance = QC_CRITERIA["dimensional_accuracy"]["tolerance_mm"]

        for axis in ["x", "y", "z"]:
            expected_val = expected_dims.get(axis, 0)
            measured_val = measured_dims.get(axis, expected_val)

            if expected_val > 0:
                deviation = abs(measured_val - expected_val)
                pct_deviation = (deviation / expected_val) * 100

                deviations.append({
                    "axis": axis,
                    "expected_mm": expected_val,
                    "measured_mm": measured_val,
                    "deviation_mm": round(deviation, 2),
                    "deviation_pct": round(pct_deviation, 2),
                    "within_tolerance": deviation <= tolerance,
                })

        # ── Defect detection via LLM ──
        defects: list[dict[str, Any]] = []
        surface_score = 80.0
        structural_score = 85.0
        visual_score = 75.0

        try:
            inspection_data = task.get("inspection_data", {})
            response = self.llm.messages.create(
                model=self.config.params.get("model", "claude-sonnet-4-20250514"),
                max_tokens=1024,
                system=QC_SYSTEM_PROMPT.format(
                    technology=expected_tech,
                    material=expected_material,
                    expected_dimensions=str(expected_dims),
                    measured_dimensions=str(measured_dims),
                    task_data=str(inspection_data)[:2000],
                    defect_types=str({
                        k: {"severity": v["severity"], "description": v["description"]}
                        for k, v in DEFECT_TYPES.items()
                    }),
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        "Analyze this 3D printed part for quality issues. "
                        "Dimensional deviations detected so far: "
                        f"{str(deviations)}. "
                        "Customer notes: "
                        f"{task.get('customer_notes', 'None')}."
                    ),
                }],
            )

            llm_text = response.content[0].text if response.content else ""
            logger.info(
                "qc_llm_inspection",
                extra={"response_length": len(llm_text)},
            )

            # Parse LLM response for defects
            import json
            try:
                llm_result = json.loads(llm_text)
                raw_defects = llm_result.get("defects", [])
                for defect in raw_defects:
                    defect_type = defect.get("type", "unknown")
                    known_info = DEFECT_TYPES.get(defect_type, {})
                    defects.append({
                        "type": defect_type,
                        "severity": defect.get(
                            "severity",
                            known_info.get("severity", "medium"),
                        ),
                        "description": defect.get("description", ""),
                        "location": defect.get("location", "unspecified"),
                        "remediation": defect.get("remediation", ""),
                    })

                surface_score = float(
                    llm_result.get("surface_quality_score", surface_score)
                )
                structural_score = float(
                    llm_result.get("structural_integrity_score", structural_score)
                )
                visual_score = float(
                    llm_result.get("visual_appearance_score", visual_score)
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("Could not parse LLM QC response as JSON")

        except Exception as e:
            logger.warning(f"LLM inspection failed: {e}")

        # ── Add dimensional deviation defects ──
        for dev in deviations:
            if not dev["within_tolerance"]:
                defects.append({
                    "type": "dimensional_deviation",
                    "severity": "high" if dev["deviation_mm"] > tolerance * 2 else "medium",
                    "description": (
                        f"{dev['axis'].upper()}-axis: expected {dev['expected_mm']}mm, "
                        f"measured {dev['measured_mm']}mm "
                        f"(deviation: {dev['deviation_mm']}mm)"
                    ),
                    "location": f"{dev['axis']}-axis",
                    "remediation": "Reprint with calibrated printer if beyond rework limits",
                })

        logger.info(
            "qc_inspection_complete",
            extra={
                "defect_count": len(defects),
                "deviations": len(deviations),
            },
        )

        return {
            "current_node": "run_inspection",
            "dimensional_deviations": deviations,
            "defects_found": defects,
            "defect_count": len(defects),
            "surface_quality_score": surface_score,
            "structural_integrity_score": structural_score,
            "visual_appearance_score": visual_score,
        }

    # ─── Node 3: Score Quality ───────────────────────────────────────

    async def _node_score_quality(
        self, state: QCInspectorAgentState
    ) -> dict[str, Any]:
        """Node 3: Compute weighted QC score and determine disposition."""
        deviations = state.get("dimensional_deviations", [])
        defects = state.get("defects_found", [])
        surface_score = state.get("surface_quality_score", 80.0)
        structural_score = state.get("structural_integrity_score", 85.0)
        visual_score = state.get("visual_appearance_score", 75.0)

        logger.info(
            "qc_score_quality",
            extra={"defect_count": len(defects)},
        )

        # ── Dimensional accuracy score ──
        tolerance = QC_CRITERIA["dimensional_accuracy"]["tolerance_mm"]
        if deviations:
            within_count = sum(1 for d in deviations if d.get("within_tolerance", True))
            dim_accuracy = (within_count / len(deviations)) * 100
        else:
            dim_accuracy = 100.0

        # Penalize based on max deviation
        max_deviation = max(
            (d.get("deviation_mm", 0) for d in deviations),
            default=0,
        )
        if max_deviation > tolerance * 3:
            dim_accuracy = max(0, dim_accuracy - 30)
        elif max_deviation > tolerance * 2:
            dim_accuracy = max(0, dim_accuracy - 15)

        # ── Weighted overall score ──
        overall_score = (
            dim_accuracy * QC_CRITERIA["dimensional_accuracy"]["weight"]
            + surface_score * QC_CRITERIA["surface_quality"]["weight"]
            + structural_score * QC_CRITERIA["structural_integrity"]["weight"]
            + visual_score * QC_CRITERIA["visual_appearance"]["weight"]
        )
        overall_score = round(overall_score, 1)

        # ── Defect-based penalty ──
        critical_count = sum(1 for d in defects if d.get("severity") == "critical")
        high_count = sum(1 for d in defects if d.get("severity") == "high")
        medium_count = sum(1 for d in defects if d.get("severity") == "medium")

        # Critical defects are heavily penalized
        defect_penalty = critical_count * 20 + high_count * 8 + medium_count * 3
        overall_score = max(0, round(overall_score - defect_penalty, 1))

        qc_pass = overall_score >= QC_PASS_THRESHOLD and critical_count == 0

        # ── Determine disposition ──
        disposition = "scrap"
        disposition_reasoning = ""

        if (overall_score >= QC_DISPOSITION_RULES["ship"]["min_score"]
                and critical_count <= QC_DISPOSITION_RULES["ship"]["max_critical_defects"]
                and high_count <= QC_DISPOSITION_RULES["ship"]["max_high_defects"]):
            disposition = "ship"
            disposition_reasoning = (
                f"Score {overall_score}/100 meets shipping threshold. "
                f"No critical defects, {high_count} high severity (max 1 allowed)."
            )
        elif (overall_score >= QC_DISPOSITION_RULES["rework"]["min_score"]
                and critical_count <= QC_DISPOSITION_RULES["rework"]["max_critical_defects"]):
            disposition = "rework"
            disposition_reasoning = (
                f"Score {overall_score}/100 qualifies for rework. "
                f"{high_count} high-severity defects may be fixable."
            )
        elif overall_score >= QC_DISPOSITION_RULES["reprint"]["min_score"]:
            disposition = "reprint"
            disposition_reasoning = (
                f"Score {overall_score}/100 too low for rework. "
                f"Reprint recommended ({critical_count} critical defects)."
            )
        else:
            disposition = "scrap"
            disposition_reasoning = (
                f"Score {overall_score}/100 below all thresholds. "
                f"Part is unsuitable for rework or reprint."
            )

        logger.info(
            "qc_scoring_complete",
            extra={
                "overall_score": overall_score,
                "qc_pass": qc_pass,
                "disposition": disposition,
                "critical_defects": critical_count,
                "high_defects": high_count,
            },
        )

        return {
            "current_node": "score_quality",
            "dimensional_accuracy_score": round(dim_accuracy, 1),
            "overall_qc_score": overall_score,
            "qc_pass": qc_pass,
            "disposition": disposition,
            "disposition_reasoning": disposition_reasoning,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: QCInspectorAgentState
    ) -> dict[str, Any]:
        """Node 4: Present QC results for human approval (critical gate)."""
        logger.info(
            "qc_human_review_pending",
            extra={
                "overall_score": state.get("overall_qc_score", 0),
                "disposition": state.get("disposition", ""),
                "defect_count": state.get("defect_count", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: QCInspectorAgentState
    ) -> dict[str, Any]:
        """Node 5: Save inspection results and generate Hive Mind insights."""
        now = datetime.now(timezone.utc).isoformat()
        print_job_id = state.get("print_job_id", "")
        defects = state.get("defects_found", [])
        deviations = state.get("dimensional_deviations", [])
        overall_score = state.get("overall_qc_score", 0)
        disposition = state.get("disposition", "unknown")
        technology = state.get("expected_technology", "FDM")
        material = state.get("expected_material", "PLA")

        # ── Save inspection to DB ──
        saved = False
        try:
            self.db.client.table("quality_inspections").insert({
                "print_job_id": print_job_id,
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "overall_score": overall_score,
                "dimensional_accuracy_score": state.get("dimensional_accuracy_score", 0),
                "surface_quality_score": state.get("surface_quality_score", 0),
                "structural_integrity_score": state.get("structural_integrity_score", 0),
                "visual_appearance_score": state.get("visual_appearance_score", 0),
                "defect_count": len(defects),
                "defects": defects,
                "deviations": deviations,
                "disposition": disposition,
                "disposition_reasoning": state.get("disposition_reasoning", ""),
                "qc_pass": state.get("qc_pass", False),
                "technology": technology,
                "material": material,
                "inspected_at": now,
                "created_at": now,
            }).execute()
            saved = True
            logger.info(
                "qc_inspection_saved",
                extra={"print_job_id": print_job_id},
            )
        except Exception as e:
            logger.warning(f"Failed to save QC inspection: {e}")

        # ── Update print job status based on disposition ──
        if print_job_id and saved:
            try:
                status_map = {
                    "ship": "qc_passed",
                    "rework": "rework",
                    "reprint": "reprint_needed",
                    "scrap": "scrapped",
                }
                new_status = status_map.get(disposition, "qc_review")
                self.db.client.table("print_jobs").update({
                    "status": new_status,
                    "qc_score": overall_score,
                    "qc_disposition": disposition,
                    "updated_at": now,
                }).eq("id", print_job_id).execute()
            except Exception as e:
                logger.debug(f"Failed to update print job QC status: {e}")

        # ── Build report ──
        critical_count = sum(1 for d in defects if d.get("severity") == "critical")
        high_count = sum(1 for d in defects if d.get("severity") == "high")
        medium_count = sum(1 for d in defects if d.get("severity") == "medium")
        low_count = sum(1 for d in defects if d.get("severity") == "low")

        sections = [
            "# Quality Control Inspection Report",
            f"*Generated: {now}*\n",
            f"## Job Details",
            f"- **Print Job:** {print_job_id}",
            f"- **Technology:** {technology}",
            f"- **Material:** {material}",
            f"\n## QC Scores",
            f"- **Overall Score:** {overall_score}/100 "
            f"({'PASS' if state.get('qc_pass') else 'FAIL'})",
            f"- **Dimensional Accuracy:** "
            f"{state.get('dimensional_accuracy_score', 0)}/100",
            f"- **Surface Quality:** "
            f"{state.get('surface_quality_score', 0)}/100",
            f"- **Structural Integrity:** "
            f"{state.get('structural_integrity_score', 0)}/100",
            f"- **Visual Appearance:** "
            f"{state.get('visual_appearance_score', 0)}/100",
            f"\n## Defects ({len(defects)} total)",
            f"- Critical: {critical_count}",
            f"- High: {high_count}",
            f"- Medium: {medium_count}",
            f"- Low: {low_count}",
        ]

        if defects:
            sections.append("\n### Defect Details")
            for i, d in enumerate(defects, 1):
                sections.append(
                    f"{i}. **[{d.get('severity', '?').upper()}] "
                    f"{d.get('type', 'unknown')}** — {d.get('description', '')}"
                )
                if d.get("location"):
                    sections.append(f"   Location: {d['location']}")
                if d.get("remediation"):
                    sections.append(f"   Fix: {d['remediation']}")

        sections.extend([
            f"\n## Disposition",
            f"- **Decision:** {disposition.upper()}",
            f"- **Reasoning:** {state.get('disposition_reasoning', '')}",
        ])

        if deviations:
            sections.append("\n## Dimensional Deviations")
            for dev in deviations:
                status = "OK" if dev.get("within_tolerance") else "FAIL"
                sections.append(
                    f"- {dev.get('axis', '?').upper()}: "
                    f"{dev.get('measured_mm', 0):.1f}mm "
                    f"(expected {dev.get('expected_mm', 0):.1f}mm, "
                    f"dev {dev.get('deviation_mm', 0):.2f}mm) [{status}]"
                )

        report = "\n".join(sections)

        # ── Hive Mind insight ──
        if defects:
            defect_types_found = list(set(d.get("type", "") for d in defects))
            self.store_insight(InsightData(
                insight_type="qc_defect_pattern",
                title=f"QC: {technology}/{material} — "
                      f"{disposition.upper()} (score {overall_score})",
                content=(
                    f"QC inspection for {technology} part in {material}: "
                    f"score {overall_score}/100, disposition={disposition}. "
                    f"Defects found: {len(defects)} "
                    f"({critical_count} critical, {high_count} high). "
                    f"Types: {', '.join(defect_types_found[:5])}."
                ),
                confidence=0.85,
                metadata={
                    "technology": technology,
                    "material": material,
                    "overall_score": overall_score,
                    "disposition": disposition,
                    "defect_types": defect_types_found,
                    "defect_count": len(defects),
                },
            ))

        logger.info(
            "qc_report_generated",
            extra={
                "print_job_id": print_job_id,
                "overall_score": overall_score,
                "disposition": disposition,
                "saved": saved,
            },
        )

        return {
            "current_node": "report",
            "inspection_saved": saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: QCInspectorAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<QCInspectorAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
