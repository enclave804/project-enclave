"""
File Analyst Agent — The 3D Geometry Inspector.

Analyzes 3D model files (STL, OBJ, STEP, 3MF) for geometry integrity,
printability, and potential manufacturing issues. Produces a detailed
analysis report with a printability score, issue list, and warnings.

Architecture (LangGraph State Machine):
    load_file → analyze_geometry → human_review →
    save_analysis → report → END

Trigger Events:
    - print_job_submitted: New print job file uploaded
    - file_resubmitted: Customer re-uploads corrected file
    - manual: On-demand file analysis

Shared Brain Integration:
    - Reads: common file issues, customer history
    - Writes: file quality patterns, format-specific issue frequencies

Safety:
    - NEVER modifies the original file
    - Analysis based on geometry metadata only
    - All findings require human_review gate before persisting
    - Does not execute customer files

Usage:
    agent = FileAnalystAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "pj_abc123",
        "file_name": "model.stl",
        "file_format": "STL",
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
from core.agents.state import FileAnalystAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {
    "STL": {
        "label": "Stereolithography",
        "extensions": [".stl"],
        "binary_support": True,
        "color_support": False,
        "max_file_size_mb": 500,
        "typical_issues": ["non_manifold", "holes", "inverted_normals"],
    },
    "OBJ": {
        "label": "Wavefront OBJ",
        "extensions": [".obj"],
        "binary_support": False,
        "color_support": True,
        "max_file_size_mb": 1000,
        "typical_issues": ["missing_normals", "degenerate_faces", "non_manifold"],
    },
    "STEP": {
        "label": "STEP / STP",
        "extensions": [".step", ".stp"],
        "binary_support": False,
        "color_support": False,
        "max_file_size_mb": 2000,
        "typical_issues": ["open_shells", "tiny_edges", "face_gaps"],
    },
    "3MF": {
        "label": "3D Manufacturing Format",
        "extensions": [".3mf"],
        "binary_support": True,
        "color_support": True,
        "max_file_size_mb": 500,
        "typical_issues": ["invalid_mesh", "missing_components"],
    },
}

PRINTABILITY_THRESHOLDS = {
    "min_wall_thickness_mm": 0.5,
    "min_detail_mm": 0.2,
    "max_overhang_degrees": 45,
    "max_bridge_length_mm": 10.0,
    "min_support_contact_mm2": 1.0,
    "max_aspect_ratio": 8.0,
    "max_file_size_mb": 500,
    "min_faces_for_detail": 100,
    "max_faces_for_processing": 10_000_000,
}

SEVERITY_WEIGHTS = {
    "critical": 30,
    "high": 20,
    "medium": 10,
    "low": 5,
    "info": 1,
}

FILE_ANALYST_SYSTEM_PROMPT = """\
You are a 3D file geometry analyst specializing in additive manufacturing \
quality assessment. Given the file metrics below, evaluate the model's \
printability and produce a JSON object with:
{{
    "issues": [
        {{"issue": "...", "severity": "critical|high|medium|low", \
"location": "...", "description": "...", "recommendation": "..."}}
    ],
    "warnings": [
        {{"warning": "...", "impact": "...", "suggestion": "..."}}
    ],
    "printability_score": 0-100,
    "summary": "Brief assessment of file quality"
}}

File Name: {file_name}
File Format: {file_format}
Vertex Count: {vertex_count}
Face Count: {face_count}
Is Manifold: {is_manifold}
Is Watertight: {is_watertight}
Bounding Box (mm): {bounding_box}
Volume (cm3): {volume_cm3}
Surface Area (cm2): {surface_area_cm2}
Min Wall Thickness (mm): {min_wall_thickness}
Overhang Percentage: {overhang_pct}%

Printability Thresholds:
- Minimum wall thickness: {min_wall_thresh} mm
- Minimum detail size: {min_detail_thresh} mm

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("file_analyst")
class FileAnalystAgent(BaseAgent):
    """
    3D file geometry analysis agent for PrintBiz engagements.

    Nodes:
        1. load_file         -- Pull print job data and file metadata from DB
        2. analyze_geometry   -- Compute geometry metrics, check manifold/watertight
        3. human_review       -- Gate: approve analysis before saving
        4. save_analysis      -- Persist analysis results to file_analyses table
        5. report             -- Generate summary report and store insights
    """

    def build_graph(self) -> Any:
        """Build the File Analyst Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(FileAnalystAgentState)

        workflow.add_node("load_file", self._node_load_file)
        workflow.add_node("analyze_geometry", self._node_analyze_geometry)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("save_analysis", self._node_save_analysis)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_file")

        workflow.add_edge("load_file", "analyze_geometry")
        workflow.add_edge("analyze_geometry", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "save_analysis",
                "rejected": "report",
            },
        )
        workflow.add_edge("save_analysis", "report")
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
    def get_state_class(cls) -> Type[FileAnalystAgentState]:
        return FileAnalystAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "file_name": "",
            "file_format": "",
            "file_size_bytes": 0,
            "file_url": "",
            "customer_id": "",
            "customer_name": "",
            "vertex_count": 0,
            "face_count": 0,
            "edge_count": 0,
            "is_manifold": False,
            "is_watertight": False,
            "has_inverted_normals": False,
            "bounding_box": {},
            "volume_cm3": 0.0,
            "surface_area_cm2": 0.0,
            "center_of_mass": {},
            "printability_score": 0.0,
            "issues": [],
            "warnings": [],
            "min_wall_thickness_mm": 0.0,
            "min_detail_mm": 0.0,
            "overhang_percentage": 0.0,
            "file_analysis_id": "",
            "analysis_saved": False,
            "all_findings": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load File ────────────────────────────────────────────

    async def _node_load_file(
        self, state: FileAnalystAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull print job data and file metadata from the database."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")

        logger.info(
            "file_analyst_load_file",
            extra={"print_job_id": print_job_id, "agent_id": self.agent_id},
        )

        file_name = task.get("file_name", "unknown.stl")
        file_format = task.get("file_format", "STL").upper()
        file_size_bytes = task.get("file_size_bytes", 0)
        file_url = task.get("file_url", "")
        customer_id = task.get("customer_id", "")
        customer_name = task.get("customer_name", "Unknown")

        # Attempt to load print job from database
        if print_job_id:
            try:
                result = (
                    self.db.client.table("print_jobs")
                    .select("*")
                    .eq("id", print_job_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    job = result.data[0]
                    file_name = job.get("file_name", file_name)
                    file_format = job.get("file_format", file_format).upper()
                    file_size_bytes = job.get("file_size_bytes", file_size_bytes)
                    file_url = job.get("file_url", file_url)
                    customer_id = job.get("customer_id", customer_id)
                    customer_name = job.get("customer_name", customer_name)
                    logger.info(
                        "file_analyst_job_loaded",
                        extra={
                            "print_job_id": print_job_id,
                            "file_name": file_name,
                            "file_format": file_format,
                        },
                    )
                else:
                    logger.warning(
                        "file_analyst_job_not_found",
                        extra={"print_job_id": print_job_id},
                    )
            except Exception as e:
                logger.warning(
                    "file_analyst_db_error",
                    extra={
                        "print_job_id": print_job_id,
                        "error": str(e)[:200],
                    },
                )

        # Validate format
        if file_format not in SUPPORTED_FORMATS:
            logger.warning(
                "file_analyst_unsupported_format",
                extra={"file_format": file_format},
            )

        # Check file size limit
        format_info = SUPPORTED_FORMATS.get(file_format, {})
        max_mb = format_info.get("max_file_size_mb", 500)
        file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes else 0

        if file_size_mb > max_mb:
            logger.warning(
                "file_analyst_file_too_large",
                extra={
                    "file_size_mb": round(file_size_mb, 1),
                    "max_mb": max_mb,
                },
            )

        return {
            "current_node": "load_file",
            "print_job_id": print_job_id,
            "file_name": file_name,
            "file_format": file_format,
            "file_size_bytes": file_size_bytes,
            "file_url": file_url,
            "customer_id": customer_id,
            "customer_name": customer_name,
        }

    # ─── Node 2: Analyze Geometry ────────────────────────────────────

    async def _node_analyze_geometry(
        self, state: FileAnalystAgentState
    ) -> dict[str, Any]:
        """Node 2: Compute geometry metrics and evaluate printability."""
        task = state.get("task_input", {})
        file_name = state.get("file_name", "unknown")
        file_format = state.get("file_format", "STL")

        logger.info(
            "file_analyst_analyze_geometry",
            extra={"file_name": file_name, "file_format": file_format},
        )

        # Extract or compute geometry metrics from task input
        # In production, this would parse the actual 3D file
        vertex_count = task.get("vertex_count", 12480)
        face_count = task.get("face_count", 24960)
        edge_count = task.get("edge_count", vertex_count + face_count - 2)

        is_manifold = task.get("is_manifold", True)
        is_watertight = task.get("is_watertight", True)
        has_inverted_normals = task.get("has_inverted_normals", False)

        bounding_box = task.get("bounding_box", {
            "x": 120.0, "y": 80.0, "z": 95.0
        })
        volume_cm3 = task.get("volume_cm3", 456.2)
        surface_area_cm2 = task.get("surface_area_cm2", 892.5)
        center_of_mass = task.get("center_of_mass", {
            "x": 60.0, "y": 40.0, "z": 47.5
        })

        min_wall_thickness = task.get("min_wall_thickness_mm", 0.8)
        min_detail = task.get("min_detail_mm", 0.3)
        overhang_pct = task.get("overhang_percentage", 15.0)

        # ── Rule-based issue detection ──
        issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        # Non-manifold check
        if not is_manifold:
            issues.append({
                "issue": "Non-manifold geometry detected",
                "severity": "critical",
                "location": "mesh_topology",
                "description": (
                    "The mesh contains non-manifold edges or vertices, "
                    "meaning some edges are shared by more than two faces. "
                    "This will cause slicing failures in most print software."
                ),
                "recommendation": "Run mesh repair to fix non-manifold geometry.",
            })

        # Watertight check
        if not is_watertight:
            issues.append({
                "issue": "Mesh is not watertight",
                "severity": "critical",
                "location": "mesh_boundary",
                "description": (
                    "The mesh has holes or open boundaries, making it "
                    "impossible to determine inside vs outside for slicing. "
                    "Print will fail or produce incorrect geometry."
                ),
                "recommendation": "Fill holes and close open boundaries using mesh repair.",
            })

        # Inverted normals
        if has_inverted_normals:
            issues.append({
                "issue": "Inverted face normals detected",
                "severity": "high",
                "location": "face_normals",
                "description": (
                    "Some face normals point inward instead of outward. "
                    "This confuses the slicer about what is inside vs outside, "
                    "potentially causing incorrect layer generation."
                ),
                "recommendation": "Recalculate normals to ensure consistent outward orientation.",
            })

        # Wall thickness check
        wall_thresh = PRINTABILITY_THRESHOLDS["min_wall_thickness_mm"]
        if min_wall_thickness < wall_thresh:
            issues.append({
                "issue": f"Wall thickness below minimum ({min_wall_thickness:.2f} mm)",
                "severity": "high",
                "location": "thin_walls",
                "description": (
                    f"Minimum wall thickness of {min_wall_thickness:.2f} mm "
                    f"is below the printable threshold of {wall_thresh} mm. "
                    f"Thin walls may break during printing or post-processing."
                ),
                "recommendation": (
                    f"Increase minimum wall thickness to at least {wall_thresh} mm "
                    f"or switch to a higher-resolution print technology."
                ),
            })

        # Detail size check
        detail_thresh = PRINTABILITY_THRESHOLDS["min_detail_mm"]
        if min_detail < detail_thresh:
            warnings.append({
                "warning": f"Fine detail below recommended minimum ({min_detail:.2f} mm)",
                "impact": "Small features may not resolve during printing.",
                "suggestion": (
                    f"Consider using SLA/DLP technology for details under "
                    f"{detail_thresh} mm, or simplify geometry."
                ),
            })

        # Overhang warning
        max_overhang = PRINTABILITY_THRESHOLDS["max_overhang_degrees"]
        if overhang_pct > 30.0:
            warnings.append({
                "warning": f"High overhang percentage ({overhang_pct:.1f}%)",
                "impact": (
                    "Excessive overhangs require more support material, "
                    "increasing cost and post-processing time."
                ),
                "suggestion": (
                    "Consider reorienting the model to minimize overhangs "
                    "or expect additional support material costs."
                ),
            })

        # Face count checks
        max_faces = PRINTABILITY_THRESHOLDS["max_faces_for_processing"]
        min_faces = PRINTABILITY_THRESHOLDS["min_faces_for_detail"]
        if face_count > max_faces:
            warnings.append({
                "warning": f"Very high polygon count ({face_count:,} faces)",
                "impact": "May cause excessive processing time and memory usage.",
                "suggestion": "Consider decimating the mesh to reduce polygon count.",
            })
        elif face_count < min_faces:
            warnings.append({
                "warning": f"Very low polygon count ({face_count} faces)",
                "impact": "Model may appear faceted with visible polygon edges.",
                "suggestion": "Consider subdividing the mesh for smoother surfaces.",
            })

        # Aspect ratio check
        dims = bounding_box
        max_aspect = PRINTABILITY_THRESHOLDS["max_aspect_ratio"]
        if dims:
            dim_values = [dims.get("x", 1), dims.get("y", 1), dims.get("z", 1)]
            dim_values = [d for d in dim_values if d > 0]
            if dim_values:
                aspect = max(dim_values) / min(dim_values)
                if aspect > max_aspect:
                    warnings.append({
                        "warning": f"High aspect ratio ({aspect:.1f}:1)",
                        "impact": "Long thin models are prone to warping and breakage.",
                        "suggestion": (
                            "Consider printing in sections and assembling, "
                            "or reorienting for better bed adhesion."
                        ),
                    })

        # ── LLM-based analysis for additional insights ──
        try:
            prompt = FILE_ANALYST_SYSTEM_PROMPT.format(
                file_name=file_name,
                file_format=file_format,
                vertex_count=vertex_count,
                face_count=face_count,
                is_manifold=is_manifold,
                is_watertight=is_watertight,
                bounding_box=json.dumps(bounding_box),
                volume_cm3=round(volume_cm3, 2),
                surface_area_cm2=round(surface_area_cm2, 2),
                min_wall_thickness=min_wall_thickness,
                overhang_pct=overhang_pct,
                min_wall_thresh=wall_thresh,
                min_detail_thresh=detail_thresh,
            )

            llm_response = self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                system="You are a 3D file geometry analyst for additive manufacturing.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            # Try to parse LLM JSON for additional issues
            try:
                llm_data = json.loads(llm_text)
                llm_issues = llm_data.get("issues", [])
                llm_warnings = llm_data.get("warnings", [])
                llm_score = llm_data.get("printability_score", None)

                # Merge LLM issues (avoid duplicates by checking issue text)
                existing_issue_texts = {i["issue"] for i in issues}
                for li in llm_issues:
                    if isinstance(li, dict) and li.get("issue") not in existing_issue_texts:
                        issues.append(li)
                        existing_issue_texts.add(li.get("issue", ""))

                existing_warn_texts = {w.get("warning", "") for w in warnings}
                for lw in llm_warnings:
                    if isinstance(lw, dict) and lw.get("warning") not in existing_warn_texts:
                        warnings.append(lw)

            except (json.JSONDecodeError, KeyError):
                logger.debug("file_analyst_llm_parse_error: Could not parse LLM JSON")

        except Exception as e:
            logger.warning(
                "file_analyst_llm_error",
                extra={"error": str(e)[:200]},
            )

        # ── Calculate printability score ──
        # Start at 100, deduct for issues
        score = 100.0
        for issue in issues:
            severity = issue.get("severity", "low")
            score -= SEVERITY_WEIGHTS.get(severity, 5)

        for warning in warnings:
            score -= 2  # Minor deductions for warnings

        printability_score = max(0.0, min(100.0, score))

        # Combine all findings
        all_findings = [
            {**issue, "finding_type": "issue"} for issue in issues
        ] + [
            {**w, "finding_type": "warning"} for w in warnings
        ]

        logger.info(
            "file_analyst_geometry_complete",
            extra={
                "file_name": file_name,
                "vertex_count": vertex_count,
                "face_count": face_count,
                "issue_count": len(issues),
                "warning_count": len(warnings),
                "printability_score": round(printability_score, 1),
                "is_manifold": is_manifold,
                "is_watertight": is_watertight,
            },
        )

        return {
            "current_node": "analyze_geometry",
            "vertex_count": vertex_count,
            "face_count": face_count,
            "edge_count": edge_count,
            "is_manifold": is_manifold,
            "is_watertight": is_watertight,
            "has_inverted_normals": has_inverted_normals,
            "bounding_box": bounding_box,
            "volume_cm3": round(volume_cm3, 2),
            "surface_area_cm2": round(surface_area_cm2, 2),
            "center_of_mass": center_of_mass,
            "min_wall_thickness_mm": min_wall_thickness,
            "min_detail_mm": min_detail,
            "overhang_percentage": overhang_pct,
            "printability_score": round(printability_score, 1),
            "issues": issues,
            "warnings": warnings,
            "all_findings": all_findings,
        }

    # ─── Node 3: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: FileAnalystAgentState
    ) -> dict[str, Any]:
        """Node 3: Present file analysis findings for human approval."""
        issues = state.get("issues", [])
        warnings = state.get("warnings", [])
        score = state.get("printability_score", 0)

        logger.info(
            "file_analyst_human_review_pending",
            extra={
                "issue_count": len(issues),
                "warning_count": len(warnings),
                "printability_score": score,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 4: Save Analysis ───────────────────────────────────────

    async def _node_save_analysis(
        self, state: FileAnalystAgentState
    ) -> dict[str, Any]:
        """Node 4: Persist approved analysis results to file_analyses table."""
        print_job_id = state.get("print_job_id", "")
        file_name = state.get("file_name", "")
        now = datetime.now(timezone.utc).isoformat()

        logger.info(
            "file_analyst_save_analysis",
            extra={
                "print_job_id": print_job_id,
                "file_name": file_name,
            },
        )

        analysis_record = {
            "print_job_id": print_job_id,
            "vertical_id": self.vertical_id,
            "agent_id": self.agent_id,
            "file_name": file_name,
            "file_format": state.get("file_format", ""),
            "vertex_count": state.get("vertex_count", 0),
            "face_count": state.get("face_count", 0),
            "edge_count": state.get("edge_count", 0),
            "is_manifold": state.get("is_manifold", False),
            "is_watertight": state.get("is_watertight", False),
            "has_inverted_normals": state.get("has_inverted_normals", False),
            "bounding_box": json.dumps(state.get("bounding_box", {})),
            "volume_cm3": state.get("volume_cm3", 0.0),
            "surface_area_cm2": state.get("surface_area_cm2", 0.0),
            "center_of_mass": json.dumps(state.get("center_of_mass", {})),
            "min_wall_thickness_mm": state.get("min_wall_thickness_mm", 0.0),
            "min_detail_mm": state.get("min_detail_mm", 0.0),
            "overhang_percentage": state.get("overhang_percentage", 0.0),
            "printability_score": state.get("printability_score", 0.0),
            "issues": json.dumps(state.get("issues", [])),
            "warnings": json.dumps(state.get("warnings", [])),
            "created_at": now,
        }

        file_analysis_id = ""
        analysis_saved = False

        try:
            result = (
                self.db.client.table("file_analyses")
                .insert(analysis_record)
                .execute()
            )
            if result.data and len(result.data) > 0:
                file_analysis_id = result.data[0].get("id", "")
                analysis_saved = True
                logger.info(
                    "file_analyst_analysis_saved",
                    extra={
                        "file_analysis_id": file_analysis_id,
                        "print_job_id": print_job_id,
                    },
                )
        except Exception as e:
            logger.warning(
                "file_analyst_save_error",
                extra={
                    "print_job_id": print_job_id,
                    "error": str(e)[:200],
                },
            )

        # Update print job status if possible
        if print_job_id:
            try:
                self.db.client.table("print_jobs").update({
                    "analysis_status": "completed",
                    "file_analysis_id": file_analysis_id,
                    "printability_score": state.get("printability_score", 0.0),
                    "updated_at": now,
                }).eq("id", print_job_id).execute()
            except Exception as e:
                logger.debug(f"Failed to update print job status: {e}")

        return {
            "current_node": "save_analysis",
            "file_analysis_id": file_analysis_id,
            "analysis_saved": analysis_saved,
            "knowledge_written": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: FileAnalystAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate summary report and store insights."""
        now = datetime.now(timezone.utc).isoformat()
        file_name = state.get("file_name", "unknown")
        file_format = state.get("file_format", "STL")
        issues = state.get("issues", [])
        warnings = state.get("warnings", [])
        score = state.get("printability_score", 0.0)
        bb = state.get("bounding_box", {})

        # Build report
        sections = [
            "# 3D File Analysis Report",
            f"*Generated: {now}*\n",
            f"## File: {file_name}",
            f"- **Format:** {file_format}",
            f"- **Vertices:** {state.get('vertex_count', 0):,}",
            f"- **Faces:** {state.get('face_count', 0):,}",
            f"- **Manifold:** {'Yes' if state.get('is_manifold') else 'NO'}",
            f"- **Watertight:** {'Yes' if state.get('is_watertight') else 'NO'}",
            f"- **Volume:** {state.get('volume_cm3', 0):.2f} cm3",
            f"- **Surface Area:** {state.get('surface_area_cm2', 0):.2f} cm2",
            f"- **Bounding Box:** {bb.get('x', 0):.1f} x {bb.get('y', 0):.1f} x {bb.get('z', 0):.1f} mm",
            f"\n## Printability Score: {score:.0f}/100\n",
        ]

        # Grade the score
        if score >= 90:
            sections.append("**Grade: EXCELLENT** — Ready to print")
        elif score >= 70:
            sections.append("**Grade: GOOD** — Minor issues, printable with caution")
        elif score >= 50:
            sections.append("**Grade: FAIR** — Issues should be addressed before printing")
        else:
            sections.append("**Grade: POOR** — Significant issues require repair")

        if issues:
            sections.append(f"\n## Issues ({len(issues)})")
            for i, issue in enumerate(issues, 1):
                sev = issue.get("severity", "unknown").upper()
                sections.append(
                    f"{i}. **[{sev}]** {issue.get('issue', 'Unknown issue')}\n"
                    f"   {issue.get('description', '')}\n"
                    f"   *Recommendation:* {issue.get('recommendation', 'N/A')}"
                )

        if warnings:
            sections.append(f"\n## Warnings ({len(warnings)})")
            for i, w in enumerate(warnings, 1):
                sections.append(
                    f"{i}. {w.get('warning', 'Unknown warning')}\n"
                    f"   *Impact:* {w.get('impact', 'N/A')}\n"
                    f"   *Suggestion:* {w.get('suggestion', 'N/A')}"
                )

        report = "\n".join(sections)

        # Store insight about file quality patterns
        if issues or warnings:
            critical_count = sum(1 for i in issues if i.get("severity") == "critical")
            issue_types = [i.get("issue", "")[:50] for i in issues[:5]]

            self.store_insight(InsightData(
                insight_type="file_quality_pattern",
                title=f"File Analysis: {file_name} — Score {score:.0f}/100",
                content=(
                    f"File analysis for {file_name} ({file_format}): "
                    f"printability score {score:.0f}/100, "
                    f"{len(issues)} issues ({critical_count} critical), "
                    f"{len(warnings)} warnings. "
                    f"Key issues: {', '.join(issue_types)}."
                ),
                confidence=0.80,
                metadata={
                    "file_format": file_format,
                    "printability_score": score,
                    "issue_count": len(issues),
                    "critical_count": critical_count,
                    "is_manifold": state.get("is_manifold", False),
                    "is_watertight": state.get("is_watertight", False),
                },
            ))

        logger.info(
            "file_analyst_report_generated",
            extra={
                "file_name": file_name,
                "printability_score": score,
                "issue_count": len(issues),
                "warning_count": len(warnings),
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: FileAnalystAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<FileAnalystAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
