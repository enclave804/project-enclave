"""
Scale Optimizer Agent — The Architectural Scale Specialist.

Analyzes 3D models for architectural scale optimization, ensuring that
detail features survive at the target print scale and recommending
adjustments for optimal print quality across FDM, SLA, SLS, and MJF
technologies.

Architecture (LangGraph State Machine):
    load_dimensions → analyze_scale → recommend_adjustments →
    human_review → report → END

Trigger Events:
    - file_analysis_completed: Geometry data available for scale check
    - scale_requested: Customer requests scale optimization
    - manual: On-demand scale analysis

Shared Brain Integration:
    - Reads: technology capabilities, common scale issues
    - Writes: scale optimization patterns, detail loss metrics

Safety:
    - NEVER modifies the original model geometry
    - Recommendations only — human decides final scale
    - All recommendations require human_review gate
    - Preserves all original dimensions for reference

Usage:
    agent = ScaleOptimizerAgent(config, db, embedder, llm)
    result = await agent.run({
        "file_analysis_id": "fa_abc123",
        "target_scale": "1:100",
        "print_technology": "SLA",
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
from core.agents.state import ScaleOptimizerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

SCALE_PRESETS = {
    "1:10": 0.10,
    "1:20": 0.05,
    "1:25": 0.04,
    "1:50": 0.02,
    "1:75": 0.01333,
    "1:100": 0.01,
    "1:150": 0.00667,
    "1:200": 0.005,
    "1:250": 0.004,
    "1:500": 0.002,
    "1:1000": 0.001,
}

DETAIL_THRESHOLDS = {
    "FDM": {
        "min_feature_mm": 0.8,
        "min_wall_mm": 0.8,
        "layer_height_mm": 0.2,
        "nozzle_diameter_mm": 0.4,
        "xy_resolution_mm": 0.4,
        "label": "Fused Deposition Modeling",
        "best_for": "Large models, low cost, functional prototypes",
    },
    "SLA": {
        "min_feature_mm": 0.2,
        "min_wall_mm": 0.3,
        "layer_height_mm": 0.05,
        "xy_resolution_mm": 0.05,
        "label": "Stereolithography",
        "best_for": "High detail, smooth surfaces, architectural models",
    },
    "SLS": {
        "min_feature_mm": 0.5,
        "min_wall_mm": 0.7,
        "layer_height_mm": 0.1,
        "xy_resolution_mm": 0.1,
        "label": "Selective Laser Sintering",
        "best_for": "Durable parts, complex geometry, no supports needed",
    },
    "MJF": {
        "min_feature_mm": 0.5,
        "min_wall_mm": 0.5,
        "layer_height_mm": 0.08,
        "xy_resolution_mm": 0.08,
        "label": "Multi Jet Fusion",
        "best_for": "Production parts, good detail, fast turnaround",
    },
    "DLP": {
        "min_feature_mm": 0.15,
        "min_wall_mm": 0.25,
        "layer_height_mm": 0.025,
        "xy_resolution_mm": 0.035,
        "label": "Digital Light Processing",
        "best_for": "Ultra-high detail, jewelry, dental, miniatures",
    },
}

BUILD_PLATES = {
    "FDM": {"x_mm": 250, "y_mm": 250, "z_mm": 300},
    "SLA": {"x_mm": 145, "y_mm": 145, "z_mm": 185},
    "SLS": {"x_mm": 340, "y_mm": 340, "z_mm": 600},
    "MJF": {"x_mm": 380, "y_mm": 284, "z_mm": 380},
    "DLP": {"x_mm": 120, "y_mm": 68, "z_mm": 150},
}

SCALE_OPTIMIZER_SYSTEM_PROMPT = """\
You are an architectural scale optimization specialist for 3D printing. \
Given the model dimensions and target scale below, analyze whether the \
scaled model will retain sufficient detail for the chosen print technology.

Produce a JSON object with:
{{
    "scale_assessment": "excellent|good|fair|poor",
    "detail_analysis": {{
        "features_at_risk": [
            {{"feature": "...", "original_mm": 0.0, "scaled_mm": 0.0, \
"printable": true, "recommendation": "..."}}
        ],
        "detail_loss_percentage": 0.0,
        "critical_features_lost": 0
    }},
    "recommendations": [
        {{"adjustment": "...", "reason": "...", "impact": "...", "priority": 1-5}}
    ],
    "optimal_scale_factor": 0.0,
    "optimal_technology": "FDM|SLA|SLS|MJF|DLP",
    "summary": "Brief assessment"
}}

Model: {file_name}
Original Dimensions: {original_dims} mm
Target Scale: {target_scale} (factor: {scale_factor})
Scaled Dimensions: {scaled_dims} mm
Print Technology: {print_tech}
Tech Min Feature: {min_feature} mm
Min Feature in Model: {model_min_feature} mm
Min Feature at Scale: {scaled_min_feature} mm

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("scale_optimizer")
class ScaleOptimizerAgent(BaseAgent):
    """
    Architectural scale optimization agent for PrintBiz engagements.

    Nodes:
        1. load_dimensions        -- Pull bounding box from file_analyses
        2. analyze_scale          -- Check detail vs min feature at target scale
        3. recommend_adjustments  -- Suggest scale factor, warn about detail loss
        4. human_review           -- Gate: approve recommendations
        5. report                 -- Generate summary report
    """

    def build_graph(self) -> Any:
        """Build the Scale Optimizer Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ScaleOptimizerAgentState)

        workflow.add_node("load_dimensions", self._node_load_dimensions)
        workflow.add_node("analyze_scale", self._node_analyze_scale)
        workflow.add_node("recommend_adjustments", self._node_recommend_adjustments)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_dimensions")

        workflow.add_edge("load_dimensions", "analyze_scale")
        workflow.add_edge("analyze_scale", "recommend_adjustments")
        workflow.add_edge("recommend_adjustments", "human_review")
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
    def get_state_class(cls) -> Type[ScaleOptimizerAgentState]:
        return ScaleOptimizerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "file_analysis_id": "",
            "print_job_id": "",
            "file_name": "",
            "original_dimensions": {},
            "target_scale": "1:100",
            "print_technology": "FDM",
            "target_scale_factor": 0.01,
            "scaled_dimensions": {},
            "min_feature_size_mm": 0.0,
            "tech_min_feature_mm": 0.0,
            "detail_loss_percentage": 0.0,
            "features_at_risk": [],
            "recommended_scale_factor": 0.0,
            "recommended_technology": "",
            "scale_adjustments": [],
            "fit_on_build_plate": True,
            "build_plate_utilization": 0.0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Dimensions ─────────────────────────────────────

    async def _node_load_dimensions(
        self, state: ScaleOptimizerAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull bounding box from file_analyses or task input."""
        task = state.get("task_input", {})
        file_analysis_id = task.get("file_analysis_id", "")
        print_job_id = task.get("print_job_id", "")
        target_scale = task.get("target_scale", "1:100")
        print_technology = task.get("print_technology", "FDM").upper()

        logger.info(
            "scale_optimizer_load_dimensions",
            extra={
                "file_analysis_id": file_analysis_id,
                "target_scale": target_scale,
                "print_technology": print_technology,
            },
        )

        file_name = task.get("file_name", "unknown.stl")
        original_dimensions = task.get("original_dimensions", {})
        min_feature_size_mm = task.get("min_feature_size_mm", 1.0)

        # Try loading from DB
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

                    # Parse bounding box
                    bb_raw = analysis.get("bounding_box", "{}")
                    if isinstance(bb_raw, str):
                        try:
                            original_dimensions = json.loads(bb_raw)
                        except json.JSONDecodeError:
                            original_dimensions = {}
                    elif isinstance(bb_raw, dict):
                        original_dimensions = bb_raw

                    min_feature_size_mm = analysis.get(
                        "min_detail_mm", min_feature_size_mm
                    )

                    logger.info(
                        "scale_optimizer_analysis_loaded",
                        extra={
                            "file_analysis_id": file_analysis_id,
                            "file_name": file_name,
                            "dimensions": original_dimensions,
                        },
                    )
            except Exception as e:
                logger.warning(
                    "scale_optimizer_db_error",
                    extra={"error": str(e)[:200]},
                )

        if not original_dimensions:
            original_dimensions = {"x": 200.0, "y": 150.0, "z": 120.0}
            logger.info("scale_optimizer_using_default_dimensions")

        # Resolve scale factor
        scale_factor = SCALE_PRESETS.get(target_scale)
        if scale_factor is None:
            # Try parsing custom scale like "1:75"
            try:
                parts = target_scale.split(":")
                if len(parts) == 2:
                    scale_factor = 1.0 / float(parts[1])
                else:
                    scale_factor = float(target_scale)
            except (ValueError, ZeroDivisionError):
                scale_factor = 0.01  # Default to 1:100
                logger.warning(
                    "scale_optimizer_invalid_scale",
                    extra={"target_scale": target_scale, "using_default": "1:100"},
                )

        return {
            "current_node": "load_dimensions",
            "file_analysis_id": file_analysis_id,
            "print_job_id": print_job_id,
            "file_name": file_name,
            "original_dimensions": original_dimensions,
            "target_scale": target_scale,
            "target_scale_factor": scale_factor,
            "print_technology": print_technology,
            "min_feature_size_mm": min_feature_size_mm,
        }

    # ─── Node 2: Analyze Scale ───────────────────────────────────────

    async def _node_analyze_scale(
        self, state: ScaleOptimizerAgentState
    ) -> dict[str, Any]:
        """Node 2: Check detail vs min feature size at target scale."""
        original_dims = state.get("original_dimensions", {})
        scale_factor = state.get("target_scale_factor", 0.01)
        print_tech = state.get("print_technology", "FDM")
        min_feature = state.get("min_feature_size_mm", 1.0)

        logger.info(
            "scale_optimizer_analyze",
            extra={
                "scale_factor": scale_factor,
                "print_technology": print_tech,
                "original_dims": original_dims,
            },
        )

        # Calculate scaled dimensions
        scaled_dims = {
            axis: round(dim * scale_factor, 3)
            for axis, dim in original_dims.items()
        }

        # Get technology thresholds
        tech_info = DETAIL_THRESHOLDS.get(print_tech, DETAIL_THRESHOLDS["FDM"])
        tech_min_feature = tech_info.get("min_feature_mm", 0.8)
        tech_min_wall = tech_info.get("min_wall_mm", 0.8)

        # Calculate scaled minimum feature
        scaled_min_feature = min_feature * scale_factor

        # Determine detail loss
        features_at_risk: list[dict[str, Any]] = []

        # Check if smallest features survive at scale
        if scaled_min_feature < tech_min_feature:
            features_at_risk.append({
                "feature": "Fine details",
                "original_mm": round(min_feature, 3),
                "scaled_mm": round(scaled_min_feature, 3),
                "printable": False,
                "recommendation": (
                    f"Details of {min_feature:.2f}mm scale to "
                    f"{scaled_min_feature:.3f}mm, below {print_tech} minimum "
                    f"of {tech_min_feature}mm. Consider SLA/DLP for finer detail."
                ),
            })

        # Check common architectural features at various sizes
        architectural_features = [
            {"feature": "Window frames", "typical_mm": 5.0},
            {"feature": "Door handles", "typical_mm": 2.0},
            {"feature": "Railings", "typical_mm": 3.0},
            {"feature": "Roof tiles/shingles", "typical_mm": 1.5},
            {"feature": "Facade ornaments", "typical_mm": 4.0},
            {"feature": "Column fluting", "typical_mm": 2.5},
            {"feature": "Staircase treads", "typical_mm": 8.0},
            {"feature": "Balcony details", "typical_mm": 3.0},
        ]

        for feat in architectural_features:
            scaled_size = feat["typical_mm"] * scale_factor
            printable = scaled_size >= tech_min_feature
            if not printable:
                features_at_risk.append({
                    "feature": feat["feature"],
                    "original_mm": feat["typical_mm"],
                    "scaled_mm": round(scaled_size, 3),
                    "printable": False,
                    "recommendation": (
                        f"{feat['feature']} ({feat['typical_mm']}mm) scale to "
                        f"{scaled_size:.3f}mm — below printable threshold. "
                        f"Will be lost or merged in final print."
                    ),
                })

        # Calculate detail loss percentage
        total_features = len(architectural_features) + 1  # +1 for fine details
        lost_features = len(features_at_risk)
        detail_loss_pct = round(
            (lost_features / total_features * 100) if total_features > 0 else 0, 1
        )

        # Check build plate fit
        build_plate = BUILD_PLATES.get(print_tech, BUILD_PLATES["FDM"])
        fits = all(
            scaled_dims.get(axis, 0) <= build_plate.get(f"{axis}_mm", 999)
            for axis in ["x", "y", "z"]
        )

        # Calculate build plate utilization
        model_volume = 1.0
        plate_volume = 1.0
        for axis in ["x", "y", "z"]:
            model_volume *= scaled_dims.get(axis, 0)
            plate_volume *= build_plate.get(f"{axis}_mm", 1)

        utilization = round(
            (model_volume / plate_volume * 100) if plate_volume > 0 else 0, 1
        )

        logger.info(
            "scale_optimizer_analysis_complete",
            extra={
                "scaled_dims": scaled_dims,
                "features_at_risk": len(features_at_risk),
                "detail_loss_pct": detail_loss_pct,
                "fits_build_plate": fits,
                "utilization": utilization,
            },
        )

        return {
            "current_node": "analyze_scale",
            "scaled_dimensions": scaled_dims,
            "tech_min_feature_mm": tech_min_feature,
            "detail_loss_percentage": detail_loss_pct,
            "features_at_risk": features_at_risk,
            "fit_on_build_plate": fits,
            "build_plate_utilization": utilization,
        }

    # ─── Node 3: Recommend Adjustments ───────────────────────────────

    async def _node_recommend_adjustments(
        self, state: ScaleOptimizerAgentState
    ) -> dict[str, Any]:
        """Node 3: Suggest optimal scale factor and technology."""
        file_name = state.get("file_name", "unknown")
        original_dims = state.get("original_dimensions", {})
        target_scale = state.get("target_scale", "1:100")
        scale_factor = state.get("target_scale_factor", 0.01)
        print_tech = state.get("print_technology", "FDM")
        features_at_risk = state.get("features_at_risk", [])
        detail_loss_pct = state.get("detail_loss_percentage", 0.0)
        scaled_dims = state.get("scaled_dimensions", {})
        min_feature = state.get("min_feature_size_mm", 1.0)

        logger.info(
            "scale_optimizer_recommend",
            extra={
                "file_name": file_name,
                "features_at_risk": len(features_at_risk),
                "detail_loss_pct": detail_loss_pct,
            },
        )

        adjustments: list[dict[str, Any]] = []
        recommended_scale = scale_factor
        recommended_tech = print_tech

        # If significant detail loss, suggest alternatives
        if detail_loss_pct > 30:
            # Find best technology for this scale
            best_tech = print_tech
            best_loss = detail_loss_pct

            for tech_name, tech_info in DETAIL_THRESHOLDS.items():
                tech_min = tech_info["min_feature_mm"]
                scaled_min = min_feature * scale_factor
                if scaled_min >= tech_min:
                    # This tech can handle the detail
                    if tech_name != print_tech:
                        adjustments.append({
                            "adjustment": f"Switch to {tech_name} ({tech_info['label']})",
                            "reason": (
                                f"{tech_name} can resolve features down to "
                                f"{tech_min}mm vs {DETAIL_THRESHOLDS[print_tech]['min_feature_mm']}mm "
                                f"for {print_tech}."
                            ),
                            "impact": f"Preserves detail at {target_scale} scale",
                            "priority": 1,
                        })
                        best_tech = tech_name
                        break

            recommended_tech = best_tech

        # If model doesn't fit build plate
        if not state.get("fit_on_build_plate", True):
            # Calculate max scale that fits
            build_plate = BUILD_PLATES.get(print_tech, BUILD_PLATES["FDM"])
            max_scales = []
            for axis in ["x", "y", "z"]:
                orig = original_dims.get(axis, 1)
                plate = build_plate.get(f"{axis}_mm", 999)
                if orig > 0:
                    max_scales.append(plate / orig)

            if max_scales:
                max_scale = min(max_scales)
                adjustments.append({
                    "adjustment": f"Reduce scale to fit build plate (max factor: {max_scale:.4f})",
                    "reason": (
                        f"Model at {target_scale} ({scaled_dims}) exceeds "
                        f"{print_tech} build plate "
                        f"({build_plate['x_mm']}x{build_plate['y_mm']}x{build_plate['z_mm']}mm)."
                    ),
                    "impact": "Model must be scaled down or split into sections",
                    "priority": 1,
                })
                recommended_scale = min(scale_factor, max_scale * 0.95)

        # Suggest larger scale for better detail
        if detail_loss_pct > 15:
            tech_info = DETAIL_THRESHOLDS.get(recommended_tech, DETAIL_THRESHOLDS["FDM"])
            tech_min = tech_info["min_feature_mm"]
            if min_feature > 0:
                optimal_factor = tech_min / min_feature
                if optimal_factor > scale_factor:
                    # Find nearest standard scale
                    nearest_scale = target_scale
                    nearest_diff = float("inf")
                    for preset_name, preset_factor in SCALE_PRESETS.items():
                        if preset_factor >= optimal_factor:
                            diff = abs(preset_factor - optimal_factor)
                            if diff < nearest_diff:
                                nearest_diff = diff
                                nearest_scale = preset_name
                                recommended_scale = preset_factor

                    adjustments.append({
                        "adjustment": f"Increase scale to {nearest_scale} for full detail preservation",
                        "reason": (
                            f"At {target_scale}, {detail_loss_pct:.0f}% of features are "
                            f"below the printable threshold. Scale {nearest_scale} "
                            f"preserves all major architectural details."
                        ),
                        "impact": f"Larger model but no detail loss",
                        "priority": 2,
                    })

        # General recommendations
        if features_at_risk:
            adjustments.append({
                "adjustment": "Consider selective detail enhancement",
                "reason": (
                    f"{len(features_at_risk)} features are at risk of being "
                    f"lost at {target_scale} scale. Key features like window frames "
                    f"and railings can be slightly exaggerated for visibility."
                ),
                "impact": "Better visual representation at the cost of strict accuracy",
                "priority": 3,
            })

        # Use LLM for more nuanced recommendations
        try:
            tech_info = DETAIL_THRESHOLDS.get(print_tech, DETAIL_THRESHOLDS["FDM"])
            scaled_min_feature = min_feature * scale_factor

            prompt = SCALE_OPTIMIZER_SYSTEM_PROMPT.format(
                file_name=file_name,
                original_dims=json.dumps(original_dims),
                target_scale=target_scale,
                scale_factor=scale_factor,
                scaled_dims=json.dumps(scaled_dims),
                print_tech=print_tech,
                min_feature=tech_info["min_feature_mm"],
                model_min_feature=min_feature,
                scaled_min_feature=round(scaled_min_feature, 4),
            )

            llm_response = self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                system="You are an architectural scale optimization specialist.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                llm_data = json.loads(llm_text)

                # Merge any additional LLM recommendations
                llm_recs = llm_data.get("recommendations", [])
                existing_adjustments = {a["adjustment"][:30] for a in adjustments}
                for rec in llm_recs:
                    if isinstance(rec, dict):
                        adj_text = rec.get("adjustment", "")
                        if adj_text[:30] not in existing_adjustments:
                            adjustments.append(rec)

                # Use LLM optimal values if provided
                llm_optimal_factor = llm_data.get("optimal_scale_factor")
                if llm_optimal_factor and isinstance(llm_optimal_factor, (int, float)):
                    if llm_optimal_factor > 0:
                        recommended_scale = llm_optimal_factor

                llm_optimal_tech = llm_data.get("optimal_technology")
                if llm_optimal_tech and llm_optimal_tech in DETAIL_THRESHOLDS:
                    recommended_tech = llm_optimal_tech

            except (json.JSONDecodeError, KeyError):
                logger.debug("scale_optimizer_llm_parse_error")

        except Exception as e:
            logger.warning(
                "scale_optimizer_llm_error",
                extra={"error": str(e)[:200]},
            )

        # Sort adjustments by priority
        adjustments.sort(key=lambda x: x.get("priority", 5))

        logger.info(
            "scale_optimizer_recommendations_complete",
            extra={
                "adjustment_count": len(adjustments),
                "recommended_scale": recommended_scale,
                "recommended_tech": recommended_tech,
            },
        )

        return {
            "current_node": "recommend_adjustments",
            "recommended_scale_factor": recommended_scale,
            "recommended_technology": recommended_tech,
            "scale_adjustments": adjustments,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ScaleOptimizerAgentState
    ) -> dict[str, Any]:
        """Node 4: Present scale recommendations for human approval."""
        adjustments = state.get("scale_adjustments", [])
        features_at_risk = state.get("features_at_risk", [])

        logger.info(
            "scale_optimizer_human_review_pending",
            extra={
                "adjustment_count": len(adjustments),
                "features_at_risk": len(features_at_risk),
                "detail_loss_pct": state.get("detail_loss_percentage", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: ScaleOptimizerAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate scale optimization summary report."""
        now = datetime.now(timezone.utc).isoformat()
        file_name = state.get("file_name", "unknown")
        target_scale = state.get("target_scale", "1:100")
        print_tech = state.get("print_technology", "FDM")
        original_dims = state.get("original_dimensions", {})
        scaled_dims = state.get("scaled_dimensions", {})
        features_at_risk = state.get("features_at_risk", [])
        adjustments = state.get("scale_adjustments", [])
        detail_loss = state.get("detail_loss_percentage", 0.0)

        sections = [
            "# Scale Optimization Report",
            f"*Generated: {now}*\n",
            f"## Model: {file_name}",
            f"- **Target Scale:** {target_scale}",
            f"- **Scale Factor:** {state.get('target_scale_factor', 0):.4f}",
            f"- **Print Technology:** {print_tech}",
            f"- **Original Dimensions:** "
            f"{original_dims.get('x', 0):.1f} x "
            f"{original_dims.get('y', 0):.1f} x "
            f"{original_dims.get('z', 0):.1f} mm",
            f"- **Scaled Dimensions:** "
            f"{scaled_dims.get('x', 0):.1f} x "
            f"{scaled_dims.get('y', 0):.1f} x "
            f"{scaled_dims.get('z', 0):.1f} mm",
            f"- **Fits Build Plate:** {'Yes' if state.get('fit_on_build_plate') else 'NO'}",
            f"- **Build Plate Utilization:** {state.get('build_plate_utilization', 0):.1f}%",
            f"\n## Detail Analysis",
            f"- **Detail Loss:** {detail_loss:.1f}%",
            f"- **Features at Risk:** {len(features_at_risk)}",
        ]

        # Grade the scale
        if detail_loss < 5:
            sections.append("- **Grade: EXCELLENT** — All details preserved")
        elif detail_loss < 15:
            sections.append("- **Grade: GOOD** — Minor detail loss, acceptable")
        elif detail_loss < 30:
            sections.append("- **Grade: FAIR** — Notable detail loss, consider adjustments")
        else:
            sections.append("- **Grade: POOR** — Significant detail loss, adjustments needed")

        if features_at_risk:
            sections.append(f"\n## Features at Risk ({len(features_at_risk)})")
            for feat in features_at_risk:
                status = "PRINTABLE" if feat.get("printable") else "LOST"
                sections.append(
                    f"- **{feat.get('feature', 'Unknown')}** [{status}]: "
                    f"{feat.get('original_mm', 0):.1f}mm -> "
                    f"{feat.get('scaled_mm', 0):.3f}mm"
                )

        if adjustments:
            sections.append(f"\n## Recommendations ({len(adjustments)})")
            for i, adj in enumerate(adjustments, 1):
                sections.append(
                    f"{i}. **{adj.get('adjustment', '')}**\n"
                    f"   Reason: {adj.get('reason', 'N/A')}\n"
                    f"   Impact: {adj.get('impact', 'N/A')}"
                )

        rec_tech = state.get("recommended_technology", print_tech)
        rec_scale = state.get("recommended_scale_factor", 0)
        if rec_tech != print_tech or rec_scale != state.get("target_scale_factor", 0):
            sections.append("\n## Optimal Configuration")
            sections.append(f"- **Recommended Technology:** {rec_tech}")
            sections.append(f"- **Recommended Scale Factor:** {rec_scale:.4f}")

        report = "\n".join(sections)

        logger.info(
            "scale_optimizer_report_generated",
            extra={
                "file_name": file_name,
                "detail_loss_pct": detail_loss,
                "adjustment_count": len(adjustments),
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ScaleOptimizerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ScaleOptimizerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
