"""
CAD Advisor Agent — The Design-for-Manufacturing Consultant.

Advises architects and designers on optimizing their 3D models for
successful printing. Checks designs against printability rules, identifies
potential issues (thin walls, overhangs, unsupported features), and
generates actionable advisory reports with architecture-specific tips.

Architecture (LangGraph State Machine):
    load_consultation → analyze_design → generate_advisory →
    human_review → report → END

Trigger Events:
    - design_submitted: A new design file has been uploaded for review
    - consultation_request: Manual request for design advisory
    - pre_quote_check: Automated printability check before quoting

Shared Brain Integration:
    - Reads: common design issues by building type, successful modifications
    - Writes: design pattern insights, printability correlations, fix effectiveness

Safety:
    - NEVER modifies the original design files — advisory only
    - All recommendations require human_review gate before delivery
    - Uses claude-sonnet-4-5-20250514 for design analysis (creative/critical task)
    - All DB mutations are wrapped in try/except

Usage:
    agent = CADAdvisorAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "job_789",
        "file_analysis_id": "fa_123",
        "consultation_notes": "Client wants fine facade detail at 1:100",
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import CADAdvisorAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PRINTABILITY_RULES = {
    "min_wall_thickness_mm": {
        "value": 0.8,
        "severity": "high",
        "description": "Walls thinner than 0.8mm may not print or be fragile",
        "fix": "Increase wall thickness to at least 0.8mm (1.2mm recommended for FDM)",
    },
    "min_feature_size_mm": {
        "value": 0.4,
        "severity": "medium",
        "description": "Features smaller than 0.4mm may not resolve in the print",
        "fix": "Scale up or simplify small features for chosen technology",
    },
    "max_overhang_angle": {
        "value": 45,
        "severity": "medium",
        "description": "Overhangs beyond 45 degrees require support structures",
        "fix": "Redesign overhangs to 45 degrees or less, or plan for supports",
    },
    "max_bridge_length_mm": {
        "value": 10,
        "severity": "medium",
        "description": "Unsupported bridges longer than 10mm may sag or fail",
        "fix": "Add intermediate supports or redesign bridging geometry",
    },
    "min_clearance_mm": {
        "value": 0.3,
        "severity": "low",
        "description": "Clearances under 0.3mm may fuse together during printing",
        "fix": "Increase gaps between moving or separate parts to 0.3mm minimum",
    },
}

ARCHITECTURE_TIPS = {
    "detachable_features": {
        "tip": "Design delicate features as separate pieces for better printing",
        "applies_to": ["spires", "antennas", "railings", "canopies"],
        "benefit": "Reduces breakage risk and allows optimal orientation per piece",
    },
    "hollow_sections": {
        "tip": "Use hollow sections to reduce material cost and weight",
        "applies_to": ["large_volumes", "solid_walls", "foundations"],
        "benefit": "30-60% material savings without visual impact for display models",
    },
    "base_stability": {
        "tip": "Ensure flat base surface for stable printing and display",
        "applies_to": ["all_models", "terrain", "landscape"],
        "benefit": "Better bed adhesion during printing and stable display presentation",
    },
    "scale_features": {
        "tip": "At 1:100 scale, features under 1mm may not print clearly",
        "applies_to": ["windows", "mullions", "door_handles", "textures"],
        "benefit": "Prevents unresolvable features and reduces post-processing",
    },
    "drainage_holes": {
        "tip": "Add drainage holes for SLA/resin models to prevent trapped resin",
        "applies_to": ["enclosed_volumes", "hollow_sections", "domes"],
        "benefit": "Prevents uncured resin from being trapped inside the model",
    },
    "layer_orientation": {
        "tip": "Orient the model so critical facades face away from the build plate",
        "applies_to": ["facade_details", "signage", "surface_textures"],
        "benefit": "Best surface quality on visible faces, supports on hidden sides",
    },
}

TECHNOLOGY_CONSTRAINTS = {
    "FDM": {
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "layer_height_range_mm": [0.1, 0.3],
        "max_overhang_deg": 45,
        "strengths": ["cost-effective", "large_build_volume", "strong_parts"],
        "weaknesses": ["visible_layers", "limited_detail", "support_marks"],
    },
    "SLA": {
        "min_wall_mm": 0.5,
        "min_feature_mm": 0.15,
        "layer_height_range_mm": [0.025, 0.1],
        "max_overhang_deg": 30,
        "strengths": ["high_detail", "smooth_surfaces", "fine_features"],
        "weaknesses": ["small_build_volume", "brittle", "uv_sensitive"],
    },
    "SLS": {
        "min_wall_mm": 0.7,
        "min_feature_mm": 0.3,
        "layer_height_range_mm": [0.06, 0.12],
        "max_overhang_deg": 90,
        "strengths": ["no_supports_needed", "functional_parts", "batch_production"],
        "weaknesses": ["grainy_surface", "limited_materials", "expensive"],
    },
    "MJF": {
        "min_wall_mm": 0.5,
        "min_feature_mm": 0.2,
        "layer_height_range_mm": [0.08, 0.08],
        "max_overhang_deg": 90,
        "strengths": ["consistent_quality", "fast", "good_detail"],
        "weaknesses": ["limited_materials", "grey_surface", "cost"],
    },
}

CAD_ADVISOR_SYSTEM_PROMPT = """\
You are a design-for-manufacturing advisor specializing in 3D printing \
for architectural models. Your expertise spans FDM, SLA, SLS, and MJF \
technologies. Help architects optimize their designs for successful \
3D printing.

Design File Information:
- File Name: {file_name}
- Target Technology: {technology}
- Target Scale: {scale}
- Technology Constraints: {tech_constraints}

File Analysis Data:
{analysis_data}

Consultation Notes from Client:
{consultation_notes}

Printability Rules:
{printability_rules}

Architecture-Specific Tips:
{architecture_tips}

Analyze this design and return a JSON object with:
{{
    "printability_issues": [
        {{
            "rule": "rule_name",
            "violation": "Specific description of the issue",
            "severity": "critical|high|medium|low",
            "location": "Where in the design (e.g., north facade, roof detail)",
            "fix": "Specific actionable recommendation"
        }}
    ],
    "design_warnings": [
        {{
            "category": "warning_category",
            "description": "What to watch out for",
            "impact": "What could go wrong",
            "mitigation": "How to address it"
        }}
    ],
    "printability_score": 0-100,
    "architecture_tips": [
        "tip_key_1",
        "tip_key_2"
    ],
    "suggestions": [
        {{
            "category": "geometry|orientation|scaling|material|assembly",
            "suggestion": "Specific recommendation",
            "impact": "Expected improvement",
            "priority": "high|medium|low"
        }}
    ],
    "executive_summary": "2-3 sentence overview for the architect"
}}

Be constructive and specific. Architects appreciate precise guidance with \
clear rationale. Reference specific features and dimensions where possible.

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("cad_advisor")
class CADAdvisorAgent(BaseAgent):
    """
    Design-for-manufacturing advisory agent for PrintBiz verticals.

    Nodes:
        1. load_consultation  -- Pull design files + consultation context
        2. analyze_design     -- Check against printability rules + identify issues
        3. generate_advisory  -- Create advisory report with suggestions (LLM)
        4. human_review       -- Gate: approve advisory before delivery
        5. report             -- Summary + Hive Mind design pattern insights
    """

    def build_graph(self) -> Any:
        """Build the CAD Advisor Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(CADAdvisorAgentState)

        workflow.add_node("load_consultation", self._node_load_consultation)
        workflow.add_node("analyze_design", self._node_analyze_design)
        workflow.add_node("generate_advisory", self._node_generate_advisory)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_consultation")

        workflow.add_edge("load_consultation", "analyze_design")
        workflow.add_edge("analyze_design", "generate_advisory")
        workflow.add_edge("generate_advisory", "human_review")
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
    def get_state_class(cls) -> Type[CADAdvisorAgentState]:
        return CADAdvisorAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "file_analysis_id": "",
            "design_file_name": "",
            "consultation_notes": "",
            "target_technology": "FDM",
            "target_scale": "1:100",
            "printability_issues": [],
            "design_warnings": [],
            "printability_score": 0.0,
            "advisory_report": "",
            "suggestions": [],
            "architecture_tips_applied": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Consultation ───────────────────────────────────

    async def _node_load_consultation(
        self, state: CADAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull design file analysis and consultation context."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")
        file_analysis_id = task.get("file_analysis_id", "")
        consultation_notes = task.get("consultation_notes", "")

        logger.info(
            "cad_advisor_load_consultation",
            extra={
                "agent_id": self.agent_id,
                "print_job_id": print_job_id,
                "file_analysis_id": file_analysis_id,
            },
        )

        # ── Load file analysis ──
        analysis_data: dict[str, Any] = {}
        design_file_name = task.get("file_name", "")

        if file_analysis_id:
            try:
                result = (
                    self.db.client.table("file_analyses")
                    .select("*")
                    .eq("id", file_analysis_id)
                    .execute()
                )
                if result and result.data and len(result.data) > 0:
                    analysis_data = result.data[0]
                    design_file_name = analysis_data.get(
                        "file_name", design_file_name
                    )
                    logger.info(
                        "cad_advisor_analysis_loaded",
                        extra={"file_analysis_id": file_analysis_id},
                    )
            except Exception as e:
                logger.warning(f"Failed to load file analysis: {e}")

        # ── Load print job for technology context ──
        job_data: dict[str, Any] = {}
        if print_job_id:
            try:
                job_result = (
                    self.db.client.table("print_jobs")
                    .select("*")
                    .eq("id", print_job_id)
                    .execute()
                )
                if job_result and job_result.data:
                    job_data = job_result.data[0]
            except Exception as e:
                logger.debug(f"Failed to load print job: {e}")

        target_tech = job_data.get(
            "technology", task.get("technology", "FDM")
        ).upper()
        target_scale = job_data.get(
            "scale", task.get("scale", "1:100")
        )

        return {
            "current_node": "load_consultation",
            "print_job_id": print_job_id,
            "file_analysis_id": file_analysis_id,
            "design_file_name": design_file_name,
            "consultation_notes": consultation_notes,
            "target_technology": target_tech,
            "target_scale": target_scale,
        }

    # ─── Node 2: Analyze Design ──────────────────────────────────────

    async def _node_analyze_design(
        self, state: CADAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 2: Check design against printability rules and identify issues."""
        file_analysis_id = state.get("file_analysis_id", "")
        target_tech = state.get("target_technology", "FDM")
        target_scale = state.get("target_scale", "1:100")
        design_file_name = state.get("design_file_name", "")
        consultation_notes = state.get("consultation_notes", "")

        logger.info(
            "cad_advisor_analyze_design",
            extra={
                "technology": target_tech,
                "scale": target_scale,
                "file": design_file_name,
            },
        )

        # ── Load file analysis data for rule checking ──
        analysis_data: dict[str, Any] = {}
        if file_analysis_id:
            try:
                result = (
                    self.db.client.table("file_analyses")
                    .select("*")
                    .eq("id", file_analysis_id)
                    .execute()
                )
                if result and result.data:
                    analysis_data = result.data[0]
            except Exception as e:
                logger.debug(f"Failed to reload analysis: {e}")

        # ── Rule-based checks ──
        issues: list[dict[str, Any]] = []
        tech_constraints = TECHNOLOGY_CONSTRAINTS.get(
            target_tech, TECHNOLOGY_CONSTRAINTS["FDM"]
        )

        # Wall thickness check
        min_wall = analysis_data.get("min_wall_thickness_mm", 0)
        tech_min_wall = tech_constraints.get("min_wall_mm", 0.8)
        if min_wall > 0 and min_wall < tech_min_wall:
            issues.append({
                "rule": "min_wall_thickness_mm",
                "violation": (
                    f"Minimum wall thickness {min_wall}mm is below "
                    f"{target_tech} minimum of {tech_min_wall}mm"
                ),
                "severity": "high",
                "location": "Thin wall sections",
                "fix": PRINTABILITY_RULES["min_wall_thickness_mm"]["fix"],
            })

        # Feature size check
        min_detail = analysis_data.get("min_detail_mm", 0)
        tech_min_feature = tech_constraints.get("min_feature_mm", 0.4)
        if min_detail > 0 and min_detail < tech_min_feature:
            issues.append({
                "rule": "min_feature_size_mm",
                "violation": (
                    f"Minimum feature size {min_detail}mm is below "
                    f"{target_tech} resolution of {tech_min_feature}mm"
                ),
                "severity": "medium",
                "location": "Fine detail areas",
                "fix": PRINTABILITY_RULES["min_feature_size_mm"]["fix"],
            })

        # Overhang check
        overhang_pct = analysis_data.get("overhang_percentage", 0)
        max_overhang = tech_constraints.get("max_overhang_deg", 45)
        if overhang_pct > 30:  # >30% of surfaces are overhangs
            issues.append({
                "rule": "max_overhang_angle",
                "violation": (
                    f"{overhang_pct}% of surfaces exceed {max_overhang} degree "
                    f"overhang limit for {target_tech}"
                ),
                "severity": "medium",
                "location": "Overhang regions",
                "fix": PRINTABILITY_RULES["max_overhang_angle"]["fix"],
            })

        # Scale-specific checks
        if target_scale:
            try:
                scale_parts = target_scale.replace("1:", "").strip()
                scale_factor = float(scale_parts) if scale_parts else 100
                if scale_factor >= 100 and tech_min_feature >= 0.3:
                    issues.append({
                        "rule": "scale_features",
                        "violation": (
                            f"At {target_scale} scale with {target_tech}, "
                            f"features smaller than {tech_min_feature * scale_factor:.0f}mm "
                            f"in real life will not print"
                        ),
                        "severity": "medium",
                        "location": "Small architectural details",
                        "fix": "Exaggerate small features or use higher-resolution technology",
                    })
            except (ValueError, TypeError):
                pass

        # Manifold check
        if analysis_data.get("is_manifold") is False:
            issues.append({
                "rule": "mesh_integrity",
                "violation": "Model is not manifold — has holes or non-closed surfaces",
                "severity": "critical",
                "location": "Mesh topology",
                "fix": "Run mesh repair to close holes and fix normals before printing",
            })

        # Watertight check
        if analysis_data.get("is_watertight") is False:
            issues.append({
                "rule": "watertight_mesh",
                "violation": "Model is not watertight — slicers may produce incorrect output",
                "severity": "high",
                "location": "Mesh topology",
                "fix": "Repair mesh to ensure all surfaces are properly closed",
            })

        # ── Determine applicable architecture tips ──
        tips_applied: list[str] = []
        bounding_box = analysis_data.get("bounding_box", {})
        volume = analysis_data.get("volume_cm3", 0)

        # Large volumes benefit from hollow sections
        if volume > 500:
            tips_applied.append("hollow_sections")

        # All models benefit from stable base
        tips_applied.append("base_stability")

        # Scale models need feature awareness
        if target_scale and "1:" in target_scale:
            tips_applied.append("scale_features")

        # Check for SLA/resin drainage needs
        if target_tech in ("SLA",):
            tips_applied.append("drainage_holes")

        # Orientation advice for detailed facades
        if consultation_notes and any(
            word in consultation_notes.lower()
            for word in ["facade", "detail", "texture", "surface"]
        ):
            tips_applied.append("layer_orientation")
            tips_applied.append("detachable_features")

        # Calculate preliminary printability score
        severity_penalties = {"critical": 25, "high": 15, "medium": 8, "low": 3}
        penalty = sum(
            severity_penalties.get(i.get("severity", "low"), 3)
            for i in issues
        )
        printability_score = max(0.0, min(100.0, 100.0 - penalty))

        logger.info(
            "cad_advisor_analysis_complete",
            extra={
                "issues": len(issues),
                "printability_score": printability_score,
                "tips": len(tips_applied),
            },
        )

        return {
            "current_node": "analyze_design",
            "printability_issues": issues,
            "design_warnings": [],  # Will be populated by LLM in next node
            "printability_score": printability_score,
            "architecture_tips_applied": tips_applied,
        }

    # ─── Node 3: Generate Advisory ───────────────────────────────────

    async def _node_generate_advisory(
        self, state: CADAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 3: Create detailed advisory report with suggestions (LLM)."""
        target_tech = state.get("target_technology", "FDM")
        target_scale = state.get("target_scale", "1:100")
        design_file_name = state.get("design_file_name", "design.stl")
        consultation_notes = state.get("consultation_notes", "")
        issues = state.get("printability_issues", [])
        tips_applied = state.get("architecture_tips_applied", [])
        file_analysis_id = state.get("file_analysis_id", "")

        logger.info(
            "cad_advisor_generate_advisory",
            extra={
                "technology": target_tech,
                "issue_count": len(issues),
            },
        )

        # ── Load analysis data for LLM context ──
        analysis_data: dict[str, Any] = {}
        if file_analysis_id:
            try:
                result = (
                    self.db.client.table("file_analyses")
                    .select("*")
                    .eq("id", file_analysis_id)
                    .execute()
                )
                if result and result.data:
                    analysis_data = result.data[0]
            except Exception as e:
                logger.debug(f"Failed to load analysis for advisory: {e}")

        tech_constraints = TECHNOLOGY_CONSTRAINTS.get(
            target_tech, TECHNOLOGY_CONSTRAINTS["FDM"]
        )

        # Format architecture tips for prompt
        formatted_tips = {
            k: ARCHITECTURE_TIPS[k]["tip"]
            for k in tips_applied
            if k in ARCHITECTURE_TIPS
        }

        # ── LLM advisory generation (uses Sonnet for creative/critical task) ──
        advisory_model = self.config.params.get(
            "advisory_model", "claude-sonnet-4-5-20250514"
        )

        suggestions: list[dict[str, Any]] = []
        design_warnings: list[dict[str, Any]] = []
        advisory_text = ""

        try:
            response = self.llm.messages.create(
                model=advisory_model,
                max_tokens=2048,
                system=CAD_ADVISOR_SYSTEM_PROMPT.format(
                    file_name=design_file_name,
                    technology=target_tech,
                    scale=target_scale,
                    tech_constraints=str(tech_constraints),
                    analysis_data=str({
                        k: v for k, v in analysis_data.items()
                        if k not in ("embedding", "raw_data")
                    })[:2000],
                    consultation_notes=consultation_notes or "No specific notes provided",
                    printability_rules=str({
                        k: {"value": v["value"], "description": v["description"]}
                        for k, v in PRINTABILITY_RULES.items()
                    }),
                    architecture_tips=str(formatted_tips),
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Analyze {design_file_name} for {target_tech} printing "
                        f"at {target_scale} scale. Known issues from rule checking: "
                        f"{str(issues)[:1500]}. "
                        f"Provide comprehensive design advisory."
                    ),
                }],
            )

            llm_text = response.content[0].text if response.content else ""
            logger.info(
                "cad_advisor_llm_response",
                extra={"response_length": len(llm_text)},
            )

            # Parse LLM response
            import json
            try:
                llm_result = json.loads(llm_text)

                # Extract suggestions
                raw_suggestions = llm_result.get("suggestions", [])
                for s in raw_suggestions:
                    suggestions.append({
                        "category": s.get("category", "general"),
                        "suggestion": s.get("suggestion", ""),
                        "impact": s.get("impact", ""),
                        "priority": s.get("priority", "medium"),
                    })

                # Extract warnings
                raw_warnings = llm_result.get("design_warnings", [])
                for w in raw_warnings:
                    design_warnings.append({
                        "category": w.get("category", "general"),
                        "description": w.get("description", ""),
                        "impact": w.get("impact", ""),
                        "mitigation": w.get("mitigation", ""),
                    })

                # Extract additional issues from LLM
                llm_issues = llm_result.get("printability_issues", [])
                for li in llm_issues:
                    # Only add if not already found by rule engine
                    if not any(
                        i.get("rule") == li.get("rule") for i in issues
                    ):
                        issues.append(li)

                # Use LLM printability score if it seems reasonable
                llm_score = llm_result.get("printability_score", 0)
                if 0 < llm_score <= 100:
                    # Average with rule-based score
                    rule_score = state.get("printability_score", 50)
                    combined_score = round((rule_score + llm_score) / 2, 1)
                else:
                    combined_score = state.get("printability_score", 50)

                # Additional tips from LLM
                llm_tips = llm_result.get("architecture_tips", [])
                tips_applied = list(set(tips_applied + llm_tips))

                executive_summary = llm_result.get("executive_summary", "")

            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("Could not parse LLM advisory response as JSON")
                combined_score = state.get("printability_score", 50)
                executive_summary = ""

        except Exception as e:
            logger.warning(f"LLM advisory generation failed: {e}")
            combined_score = state.get("printability_score", 50)
            executive_summary = ""

        # ── Build advisory report ──
        report_lines = [
            f"# Design Advisory Report: {design_file_name}",
            f"\n**Technology:** {target_tech} | **Scale:** {target_scale} "
            f"| **Printability Score:** {combined_score}/100\n",
        ]

        if executive_summary:
            report_lines.append(f"## Executive Summary\n{executive_summary}\n")

        if issues:
            report_lines.append(f"## Printability Issues ({len(issues)})")
            for i, issue in enumerate(issues, 1):
                sev = issue.get("severity", "medium").upper()
                report_lines.append(
                    f"\n### {i}. [{sev}] {issue.get('rule', 'Issue')}"
                )
                report_lines.append(f"- **Problem:** {issue.get('violation', '')}")
                report_lines.append(f"- **Location:** {issue.get('location', 'N/A')}")
                report_lines.append(f"- **Fix:** {issue.get('fix', 'See recommendations')}")

        if suggestions:
            report_lines.append(f"\n## Design Suggestions ({len(suggestions)})")
            for s in suggestions:
                priority = s.get("priority", "medium").upper()
                report_lines.append(
                    f"- **[{priority}] {s.get('category', '').title()}:** "
                    f"{s.get('suggestion', '')}"
                )
                if s.get("impact"):
                    report_lines.append(f"  Impact: {s['impact']}")

        if tips_applied:
            report_lines.append("\n## Architecture Tips")
            for tip_key in tips_applied:
                tip_info = ARCHITECTURE_TIPS.get(tip_key, {})
                if tip_info:
                    report_lines.append(f"- **{tip_key}:** {tip_info.get('tip', '')}")

        if design_warnings:
            report_lines.append(f"\n## Warnings ({len(design_warnings)})")
            for w in design_warnings:
                report_lines.append(
                    f"- **{w.get('category', 'General')}:** {w.get('description', '')}"
                )

        advisory_text = "\n".join(report_lines)

        logger.info(
            "cad_advisor_advisory_generated",
            extra={
                "issues": len(issues),
                "suggestions": len(suggestions),
                "tips": len(tips_applied),
                "score": combined_score,
            },
        )

        return {
            "current_node": "generate_advisory",
            "printability_issues": issues,
            "design_warnings": design_warnings,
            "printability_score": combined_score,
            "advisory_report": advisory_text,
            "suggestions": suggestions,
            "architecture_tips_applied": tips_applied,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: CADAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 4: Present advisory for human approval before delivery."""
        logger.info(
            "cad_advisor_human_review_pending",
            extra={
                "printability_score": state.get("printability_score", 0),
                "issue_count": len(state.get("printability_issues", [])),
                "suggestion_count": len(state.get("suggestions", [])),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: CADAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate summary and Hive Mind design pattern insights."""
        now = datetime.now(timezone.utc).isoformat()
        design_file_name = state.get("design_file_name", "")
        target_tech = state.get("target_technology", "FDM")
        target_scale = state.get("target_scale", "1:100")
        issues = state.get("printability_issues", [])
        suggestions = state.get("suggestions", [])
        score = state.get("printability_score", 0)
        advisory_report = state.get("advisory_report", "")
        tips_applied = state.get("architecture_tips_applied", [])

        # ── Save advisory to DB ──
        try:
            self.db.client.table("design_advisories").insert({
                "print_job_id": state.get("print_job_id", ""),
                "file_analysis_id": state.get("file_analysis_id", ""),
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "technology": target_tech,
                "scale": target_scale,
                "printability_score": score,
                "issue_count": len(issues),
                "issues": issues,
                "suggestions": suggestions,
                "tips_applied": tips_applied,
                "advisory_report": advisory_report[:5000],
                "created_at": now,
            }).execute()
            logger.info("cad_advisor_advisory_saved")
        except Exception as e:
            logger.debug(f"Failed to save advisory: {e}")

        # ── Build concise report ──
        severity_counts = {}
        for issue in issues:
            sev = issue.get("severity", "medium")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        sections = [
            "# CAD Advisory Summary",
            f"*Generated: {now}*\n",
            f"## Design: {design_file_name}",
            f"- **Technology:** {target_tech}",
            f"- **Scale:** {target_scale}",
            f"- **Printability Score:** {score}/100",
            f"\n## Issues Found: {len(issues)}",
        ]

        for sev, count in sorted(severity_counts.items()):
            sections.append(f"- {sev.title()}: {count}")

        sections.append(f"\n## Suggestions: {len(suggestions)}")
        sections.append(f"## Architecture Tips Applied: {len(tips_applied)}")

        report = "\n".join(sections)

        # ── Hive Mind insight ──
        if issues or suggestions:
            issue_types = list(set(i.get("rule", "") for i in issues if i.get("rule")))
            self.store_insight(InsightData(
                insight_type="design_advisory_pattern",
                title=f"CAD Advisory: {target_tech}/{target_scale} — "
                      f"score {score}/100",
                content=(
                    f"Design advisory for {design_file_name} "
                    f"({target_tech}, {target_scale}): "
                    f"printability score {score}/100, "
                    f"{len(issues)} issues found, "
                    f"{len(suggestions)} suggestions provided. "
                    f"Common issues: {', '.join(issue_types[:5])}."
                ),
                confidence=0.80,
                metadata={
                    "technology": target_tech,
                    "scale": target_scale,
                    "score": score,
                    "issue_types": issue_types,
                    "tips": tips_applied,
                },
            ))

        logger.info(
            "cad_advisor_report_generated",
            extra={
                "score": score,
                "issues": len(issues),
                "suggestions": len(suggestions),
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: CADAdvisorAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<CADAdvisorAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
