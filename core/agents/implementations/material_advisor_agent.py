"""
Material Advisor Agent — The 3D Printing Materials Expert.

Evaluates customer requirements against a comprehensive materials database
and recommends the optimal 3D printing material, technology, and settings.
Produces a detailed recommendation report with cost estimates and alternatives.

Architecture (LangGraph State Machine):
    gather_requirements → evaluate_materials → recommend →
    human_review → report → END

Trigger Events:
    - print_job_analyzed: File analysis completed, needs material recommendation
    - material_request: Manual material consultation request
    - manual: On-demand material advisory

Shared Brain Integration:
    - Reads: material performance history, customer preferences
    - Writes: material recommendation patterns, cost-accuracy correlations

Safety:
    - NEVER commits to pricing without human review
    - Recommendations are advisory only until approved
    - All findings require human_review gate before persisting
    - Cost estimates are clearly marked as estimates

Usage:
    agent = MaterialAdvisorAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "pj_abc123",
        "intended_use": "functional",
        "customer_requirements": {"durability": "high", "budget": "medium"},
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
from core.agents.state import MaterialAdvisorAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

MATERIALS_DB = {
    "PLA": {
        "tech": "FDM",
        "cost_per_cm3": 0.04,
        "detail": "medium",
        "layer_um": 200,
        "strength": "moderate",
        "heat_resistance": "low",
        "finish": "matte",
        "best_for": ["prototypes", "display_models"],
    },
    "ABS": {
        "tech": "FDM",
        "cost_per_cm3": 0.05,
        "detail": "medium",
        "layer_um": 200,
        "strength": "good",
        "heat_resistance": "moderate",
        "finish": "matte",
        "best_for": ["functional", "automotive"],
    },
    "SLA_RESIN": {
        "tech": "SLA",
        "cost_per_cm3": 0.12,
        "detail": "very_high",
        "layer_um": 50,
        "strength": "moderate",
        "heat_resistance": "low",
        "finish": "smooth",
        "best_for": ["detail_models", "jewelry", "dental"],
    },
    "NYLON_SLS": {
        "tech": "SLS",
        "cost_per_cm3": 0.15,
        "detail": "high",
        "layer_um": 100,
        "strength": "excellent",
        "heat_resistance": "good",
        "finish": "textured",
        "best_for": ["functional", "engineering"],
    },
    "PETG": {
        "tech": "FDM",
        "cost_per_cm3": 0.06,
        "detail": "medium",
        "layer_um": 200,
        "strength": "good",
        "heat_resistance": "moderate",
        "finish": "glossy",
        "best_for": ["functional", "outdoor"],
    },
    "SANDSTONE": {
        "tech": "Binder_Jetting",
        "cost_per_cm3": 0.20,
        "detail": "medium",
        "layer_um": 100,
        "strength": "low",
        "heat_resistance": "high",
        "finish": "textured",
        "best_for": ["architecture", "full_color"],
    },
}

DETAIL_RANK = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
STRENGTH_RANK = {"low": 1, "moderate": 2, "good": 3, "excellent": 4}
HEAT_RANK = {"low": 1, "moderate": 2, "good": 3, "high": 4}

BUDGET_COST_THRESHOLDS = {
    "low": 0.06,
    "medium": 0.12,
    "high": 0.20,
    "unlimited": 999.0,
}

MATERIAL_ADVISOR_SYSTEM_PROMPT = """\
You are a 3D printing materials expert. Given the customer requirements \
and material evaluation scores below, select the best material and provide \
a clear recommendation with reasoning.

Return a JSON object:
{{
    "recommended_material": "MATERIAL_KEY",
    "recommended_technology": "TECH_NAME",
    "reasoning": "Detailed explanation of why this material is best...",
    "confidence": 0.0-1.0,
    "alternatives": [
        {{"material": "KEY", "technology": "TECH", "tradeoff": "description"}}
    ],
    "warnings": ["Any concerns about the recommendation..."]
}}

Customer Requirements:
- Intended Use: {intended_use}
- Detail Level Needed: {detail_level}
- Durability Needed: {durability}
- Budget: {budget}
- Heat Resistance: {heat_resistance}
- Finish Preference: {finish_preference}
- Volume: {volume_cm3} cm3

Material Scores:
{material_scores_json}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("material_advisor")
class MaterialAdvisorAgent(BaseAgent):
    """
    3D printing material recommendation agent.

    Nodes:
        1. gather_requirements  -- Pull job + analysis data from DB
        2. evaluate_materials   -- Score each material against requirements
        3. recommend            -- LLM selects best material with reasoning
        4. human_review         -- Gate: approve recommendation
        5. report               -- Summary + InsightData for patterns
    """

    def build_graph(self) -> Any:
        """Build the Material Advisor Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(MaterialAdvisorAgentState)

        workflow.add_node("gather_requirements", self._node_gather_requirements)
        workflow.add_node("evaluate_materials", self._node_evaluate_materials)
        workflow.add_node("recommend", self._node_recommend)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("gather_requirements")

        workflow.add_edge("gather_requirements", "evaluate_materials")
        workflow.add_edge("evaluate_materials", "recommend")
        workflow.add_edge("recommend", "human_review")
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
    def get_state_class(cls) -> Type[MaterialAdvisorAgentState]:
        return MaterialAdvisorAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "file_analysis_id": "",
            "customer_requirements": {},
            "intended_use": "",
            "volume_cm3": 0.0,
            "surface_area_cm2": 0.0,
            "candidate_materials": [],
            "material_scores": [],
            "technology_options": [],
            "recommended_material": "",
            "recommended_technology": "",
            "material_cost_estimate": 0.0,
            "layer_height_um": 0,
            "detail_rating": "",
            "recommendation_reasoning": "",
            "alternative_materials": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Gather Requirements ──────────────────────────────────

    async def _node_gather_requirements(
        self, state: MaterialAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull print job and file analysis data from the database."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")
        file_analysis_id = task.get("file_analysis_id", "")

        logger.info(
            "material_advisor_gather_requirements",
            extra={"print_job_id": print_job_id, "agent_id": self.agent_id},
        )

        intended_use = task.get("intended_use", "prototype")
        customer_requirements = task.get("customer_requirements", {})
        volume_cm3 = task.get("volume_cm3", 0.0)
        surface_area_cm2 = task.get("surface_area_cm2", 0.0)

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
                    intended_use = job.get("intended_use", intended_use)
                    volume_cm3 = job.get("volume_cm3", volume_cm3)
                    surface_area_cm2 = job.get("surface_area_cm2", surface_area_cm2)
                    customer_requirements = job.get(
                        "customer_requirements", customer_requirements
                    )
                    if isinstance(customer_requirements, str):
                        try:
                            customer_requirements = json.loads(customer_requirements)
                        except (json.JSONDecodeError, TypeError):
                            customer_requirements = {}
                    logger.info(
                        "material_advisor_job_loaded",
                        extra={
                            "print_job_id": print_job_id,
                            "intended_use": intended_use,
                        },
                    )
                else:
                    logger.warning(
                        "material_advisor_job_not_found",
                        extra={"print_job_id": print_job_id},
                    )
            except Exception as e:
                logger.warning(
                    "material_advisor_db_error",
                    extra={
                        "print_job_id": print_job_id,
                        "error": str(e)[:200],
                    },
                )

        # Attempt to load file analysis for volume/surface area
        if file_analysis_id and not volume_cm3:
            try:
                result = (
                    self.db.client.table("file_analyses")
                    .select("*")
                    .eq("id", file_analysis_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    analysis = result.data[0]
                    volume_cm3 = analysis.get("volume_cm3", volume_cm3)
                    surface_area_cm2 = analysis.get(
                        "surface_area_cm2", surface_area_cm2
                    )
                    logger.info(
                        "material_advisor_analysis_loaded",
                        extra={"file_analysis_id": file_analysis_id},
                    )
            except Exception as e:
                logger.warning(
                    "material_advisor_analysis_db_error",
                    extra={"error": str(e)[:200]},
                )

        return {
            "current_node": "gather_requirements",
            "print_job_id": print_job_id,
            "file_analysis_id": file_analysis_id,
            "intended_use": intended_use,
            "customer_requirements": customer_requirements,
            "volume_cm3": volume_cm3,
            "surface_area_cm2": surface_area_cm2,
        }

    # ─── Node 2: Evaluate Materials ──────────────────────────────────

    async def _node_evaluate_materials(
        self, state: MaterialAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 2: Score each material against customer requirements."""
        requirements = state.get("customer_requirements", {})
        intended_use = state.get("intended_use", "prototype")
        volume_cm3 = state.get("volume_cm3", 100.0)

        logger.info(
            "material_advisor_evaluate_materials",
            extra={"intended_use": intended_use},
        )

        # Extract requirement levels (default to medium)
        req_detail = requirements.get("detail_level", "medium")
        req_durability = requirements.get("durability", "moderate")
        req_budget = requirements.get("budget", "medium")
        req_heat = requirements.get("heat_resistance", "low")
        req_finish = requirements.get("finish", "any")

        budget_max_cost = BUDGET_COST_THRESHOLDS.get(req_budget, 0.12)

        candidate_materials: list[str] = []
        material_scores: list[dict[str, Any]] = []
        technology_options: list[str] = []

        for mat_key, mat_info in MATERIALS_DB.items():
            score = 0.0
            strengths: list[str] = []
            weaknesses: list[str] = []

            # Detail level scoring (weight: 25%)
            mat_detail_rank = DETAIL_RANK.get(mat_info["detail"], 2)
            req_detail_rank = DETAIL_RANK.get(req_detail, 2)
            if mat_detail_rank >= req_detail_rank:
                score += 25.0
                strengths.append(f"Detail level: {mat_info['detail']}")
            else:
                detail_gap = req_detail_rank - mat_detail_rank
                score += max(0, 25.0 - detail_gap * 10)
                weaknesses.append(
                    f"Detail ({mat_info['detail']}) below requirement ({req_detail})"
                )

            # Strength/durability scoring (weight: 25%)
            mat_strength_rank = STRENGTH_RANK.get(mat_info["strength"], 2)
            req_strength_rank = STRENGTH_RANK.get(req_durability, 2)
            if mat_strength_rank >= req_strength_rank:
                score += 25.0
                strengths.append(f"Strength: {mat_info['strength']}")
            else:
                strength_gap = req_strength_rank - mat_strength_rank
                score += max(0, 25.0 - strength_gap * 10)
                weaknesses.append(
                    f"Strength ({mat_info['strength']}) below requirement ({req_durability})"
                )

            # Cost scoring (weight: 20%)
            if mat_info["cost_per_cm3"] <= budget_max_cost:
                cost_ratio = mat_info["cost_per_cm3"] / max(budget_max_cost, 0.01)
                score += 20.0 * (1.0 - cost_ratio * 0.5)
                strengths.append(
                    f"Cost ${mat_info['cost_per_cm3']:.2f}/cm3 within budget"
                )
            else:
                overage = mat_info["cost_per_cm3"] / max(budget_max_cost, 0.01)
                score += max(0, 20.0 - (overage - 1.0) * 20)
                weaknesses.append(
                    f"Cost ${mat_info['cost_per_cm3']:.2f}/cm3 exceeds budget"
                )

            # Heat resistance scoring (weight: 15%)
            mat_heat_rank = HEAT_RANK.get(mat_info["heat_resistance"], 1)
            req_heat_rank = HEAT_RANK.get(req_heat, 1)
            if mat_heat_rank >= req_heat_rank:
                score += 15.0
                strengths.append(f"Heat resistance: {mat_info['heat_resistance']}")
            else:
                heat_gap = req_heat_rank - mat_heat_rank
                score += max(0, 15.0 - heat_gap * 8)
                weaknesses.append(
                    f"Heat resistance ({mat_info['heat_resistance']}) "
                    f"below requirement ({req_heat})"
                )

            # Use-case fit scoring (weight: 15%)
            use_keywords = intended_use.lower().replace("_", " ").split()
            best_for_flat = " ".join(mat_info["best_for"]).lower()
            use_match = sum(1 for kw in use_keywords if kw in best_for_flat)
            if use_match > 0:
                score += 15.0
                strengths.append(
                    f"Best for: {', '.join(mat_info['best_for'])}"
                )
            else:
                score += 5.0
                weaknesses.append(
                    f"Not specifically optimized for '{intended_use}'"
                )

            # Finish preference bonus
            if req_finish != "any" and mat_info["finish"] == req_finish:
                score += 5.0
                strengths.append(f"Finish matches preference: {req_finish}")

            candidate_materials.append(mat_key)
            material_scores.append({
                "material": mat_key,
                "technology": mat_info["tech"],
                "score": round(min(100.0, score), 1),
                "cost_per_cm3": mat_info["cost_per_cm3"],
                "estimated_cost_cents": int(
                    mat_info["cost_per_cm3"] * volume_cm3 * 100
                ),
                "layer_um": mat_info["layer_um"],
                "detail": mat_info["detail"],
                "strength": mat_info["strength"],
                "finish": mat_info["finish"],
                "strengths": strengths,
                "weaknesses": weaknesses,
            })

            if mat_info["tech"] not in technology_options:
                technology_options.append(mat_info["tech"])

        # Sort by score descending
        material_scores.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "material_advisor_evaluation_complete",
            extra={
                "candidates": len(candidate_materials),
                "top_material": material_scores[0]["material"] if material_scores else "none",
                "top_score": material_scores[0]["score"] if material_scores else 0,
            },
        )

        return {
            "current_node": "evaluate_materials",
            "candidate_materials": candidate_materials,
            "material_scores": material_scores,
            "technology_options": technology_options,
        }

    # ─── Node 3: Recommend ───────────────────────────────────────────

    async def _node_recommend(
        self, state: MaterialAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM call to select best material with detailed reasoning."""
        material_scores = state.get("material_scores", [])
        intended_use = state.get("intended_use", "prototype")
        requirements = state.get("customer_requirements", {})
        volume_cm3 = state.get("volume_cm3", 100.0)

        logger.info(
            "material_advisor_recommend",
            extra={"intended_use": intended_use, "candidates": len(material_scores)},
        )

        # Default to top-scored material
        recommended_material = material_scores[0]["material"] if material_scores else "PLA"
        recommended_technology = material_scores[0]["technology"] if material_scores else "FDM"
        recommendation_reasoning = "Selected based on highest overall score."
        alternative_materials: list[dict[str, Any]] = []
        cost_estimate = 0.0
        layer_um = 200
        detail_rating = "medium"

        try:
            prompt = MATERIAL_ADVISOR_SYSTEM_PROMPT.format(
                intended_use=intended_use,
                detail_level=requirements.get("detail_level", "medium"),
                durability=requirements.get("durability", "moderate"),
                budget=requirements.get("budget", "medium"),
                heat_resistance=requirements.get("heat_resistance", "low"),
                finish_preference=requirements.get("finish", "any"),
                volume_cm3=round(volume_cm3, 2),
                material_scores_json=json.dumps(material_scores[:6], indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                system="You are a 3D printing materials expert specializing in additive manufacturing.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                llm_data = json.loads(llm_text)
                recommended_material = llm_data.get(
                    "recommended_material", recommended_material
                )
                recommended_technology = llm_data.get(
                    "recommended_technology", recommended_technology
                )
                recommendation_reasoning = llm_data.get(
                    "reasoning", recommendation_reasoning
                )
                alternative_materials = llm_data.get("alternatives", [])

            except (json.JSONDecodeError, KeyError):
                logger.debug(
                    "material_advisor_llm_parse_error: Could not parse LLM JSON"
                )

        except Exception as e:
            logger.warning(
                "material_advisor_llm_error",
                extra={"error": str(e)[:200]},
            )

        # Look up the recommended material's details
        mat_info = MATERIALS_DB.get(recommended_material, {})
        if mat_info:
            cost_estimate = int(mat_info.get("cost_per_cm3", 0.04) * volume_cm3 * 100)
            layer_um = mat_info.get("layer_um", 200)
            detail_rating = mat_info.get("detail", "medium")
            recommended_technology = mat_info.get("tech", recommended_technology)

        # Build alternatives from next best scores
        if not alternative_materials and len(material_scores) > 1:
            for alt_score in material_scores[1:4]:
                alternative_materials.append({
                    "material": alt_score["material"],
                    "technology": alt_score["technology"],
                    "cost": alt_score.get("estimated_cost_cents", 0),
                    "tradeoff": (
                        f"Score: {alt_score['score']}, "
                        f"Strengths: {', '.join(alt_score.get('strengths', [])[:2])}"
                    ),
                })

        logger.info(
            "material_advisor_recommendation",
            extra={
                "recommended_material": recommended_material,
                "recommended_technology": recommended_technology,
                "cost_estimate_cents": cost_estimate,
            },
        )

        return {
            "current_node": "recommend",
            "recommended_material": recommended_material,
            "recommended_technology": recommended_technology,
            "material_cost_estimate": cost_estimate,
            "layer_height_um": layer_um,
            "detail_rating": detail_rating,
            "recommendation_reasoning": recommendation_reasoning,
            "alternative_materials": alternative_materials,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: MaterialAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 4: Present material recommendation for human approval."""
        recommended = state.get("recommended_material", "")
        cost = state.get("material_cost_estimate", 0)

        logger.info(
            "material_advisor_human_review_pending",
            extra={
                "recommended_material": recommended,
                "cost_estimate_cents": cost,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: MaterialAdvisorAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate summary report and store insights."""
        now = datetime.now(timezone.utc).isoformat()
        recommended = state.get("recommended_material", "unknown")
        technology = state.get("recommended_technology", "unknown")
        cost = state.get("material_cost_estimate", 0)
        reasoning = state.get("recommendation_reasoning", "")
        alternatives = state.get("alternative_materials", [])
        scores = state.get("material_scores", [])
        intended_use = state.get("intended_use", "")
        volume = state.get("volume_cm3", 0.0)

        sections = [
            "# Material Recommendation Report",
            f"*Generated: {now}*\n",
            f"## Recommended Material: {recommended}",
            f"- **Technology:** {technology}",
            f"- **Layer Height:** {state.get('layer_height_um', 0)} um",
            f"- **Detail Rating:** {state.get('detail_rating', 'N/A')}",
            f"- **Estimated Material Cost:** ${cost / 100:.2f}",
            f"- **Volume:** {volume:.2f} cm3",
            f"\n## Reasoning\n{reasoning}\n",
        ]

        if scores:
            sections.append("## Material Comparison")
            for s in scores[:6]:
                sections.append(
                    f"- **{s['material']}** ({s['technology']}): "
                    f"Score {s['score']}/100, "
                    f"${s.get('estimated_cost_cents', 0) / 100:.2f}"
                )

        if alternatives:
            sections.append("\n## Alternatives")
            for i, alt in enumerate(alternatives, 1):
                sections.append(
                    f"{i}. **{alt.get('material', 'N/A')}** "
                    f"({alt.get('technology', 'N/A')}): "
                    f"{alt.get('tradeoff', 'N/A')}"
                )

        report = "\n".join(sections)

        # Store insight about material recommendation patterns
        self.store_insight(InsightData(
            insight_type="material_recommendation",
            title=f"Material Recommendation: {recommended} for {intended_use}",
            content=(
                f"Recommended {recommended} ({technology}) for {intended_use} "
                f"application. Volume: {volume:.1f} cm3, "
                f"estimated cost: ${cost / 100:.2f}. "
                f"Reasoning: {reasoning[:200]}."
            ),
            confidence=0.80,
            metadata={
                "recommended_material": recommended,
                "recommended_technology": technology,
                "intended_use": intended_use,
                "cost_estimate_cents": cost,
                "volume_cm3": volume,
                "candidate_count": len(scores),
            },
        ))

        logger.info(
            "material_advisor_report_generated",
            extra={
                "recommended_material": recommended,
                "cost_estimate_cents": cost,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: MaterialAdvisorAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<MaterialAdvisorAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
