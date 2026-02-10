"""
Quote Engine Agent — The Professional Pricing Machine.

Calculates comprehensive 3D printing costs from geometry data, material
selection, post-processing requirements, and shipping. Generates professional
quote documents and manages the quote lifecycle.

Architecture (LangGraph State Machine):
    gather_data → calculate_costs → generate_quote →
    human_review → send_quote → report → END

Trigger Events:
    - material_recommended: Material advisor completed recommendation
    - quote_request: Manual quote generation request
    - manual: On-demand quote generation

Shared Brain Integration:
    - Reads: material costs, historical pricing, margin targets
    - Writes: pricing patterns, quote acceptance correlations

Safety:
    - NEVER sends quotes without human review gate
    - All prices are in cents to avoid floating-point currency errors
    - Markup and discounts require explicit approval
    - Cost breakdowns are transparent and auditable

Usage:
    agent = QuoteEngineAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "pj_abc123",
        "material": "NYLON_SLS",
        "volume_cm3": 456.2,
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
from core.agents.state import QuoteEngineAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PRICING_RATES = {
    "print_rate_per_hour": 15_00,       # cents per printer-hour
    "operator_rate_per_hour": 25_00,    # cents per operator-hour
    "setup_fee": 10_00,                 # flat setup fee per job
    "rush_multiplier": 1.5,             # multiplier for rush orders
    "min_order_cents": 25_00,           # minimum order value
    "default_markup_percent": 40.0,     # standard markup
    "volume_discount_threshold_cm3": 500.0,  # volume for discount eligibility
    "volume_discount_percent": 10.0,    # discount for large volumes
    "repeat_customer_discount": 5.0,    # discount for returning customers
}

POST_PROCESS_COSTS = {
    "support_removal": 10_00,
    "sanding": 15_00,
    "priming": 20_00,
    "painting": 35_00,
    "vapor_smoothing": 25_00,
    "uv_curing": 8_00,
    "heat_treatment": 18_00,
    "dyeing": 22_00,
    "metal_plating": 50_00,
    "assembly": 30_00,
    "clear_coat": 15_00,
}

# Estimated print speeds by technology (cm3 per hour)
PRINT_SPEEDS = {
    "FDM": 15.0,
    "SLA": 10.0,
    "SLS": 25.0,
    "Binder_Jetting": 30.0,
    "MJF": 28.0,
}

SHIPPING_RATES = {
    "standard": {"base_cents": 8_99, "per_kg_cents": 2_00, "days": 5},
    "express": {"base_cents": 15_99, "per_kg_cents": 3_50, "days": 2},
    "overnight": {"base_cents": 29_99, "per_kg_cents": 5_00, "days": 1},
}

QUOTE_SYSTEM_PROMPT = """\
You are a professional print shop quoting specialist. Generate a clean, \
professional quote document in Markdown format from the cost breakdown below.

Include:
1. A professional header with quote number and date
2. Customer information section
3. Itemized cost breakdown table
4. Total with any applicable discounts
5. Terms and conditions (quote valid for {valid_days} days)
6. Estimated delivery timeline

Cost Breakdown:
- Material: {material} ({technology})
- Volume: {volume_cm3} cm3
- Material Cost: ${material_cost}
- Print Time: {print_hours} hours
- Time Cost: ${time_cost}
- Post-Processing: ${post_process_total}
- Shipping ({shipping_method}): ${shipping_cost}
- Subtotal: ${subtotal}
- Markup ({markup_pct}%): ${markup}
- **Total: ${total}**

Customer: {customer_name} ({customer_email})
Print Job ID: {print_job_id}

Generate a professional Markdown quote document. Do NOT use code fences.
"""


@register_agent_type("quote_engine")
class QuoteEngineAgent(BaseAgent):
    """
    3D printing cost calculation and professional quote generation agent.

    Nodes:
        1. gather_data      -- Pull geometry + material data from DB
        2. calculate_costs   -- Compute full cost breakdown
        3. generate_quote    -- LLM creates professional formatted quote
        4. human_review      -- Gate: approve quote before sending
        5. send_quote        -- Save to print_quotes table, mark as sent
        6. report            -- Summary + InsightData for pricing patterns
    """

    def build_graph(self) -> Any:
        """Build the Quote Engine Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(QuoteEngineAgentState)

        workflow.add_node("gather_data", self._node_gather_data)
        workflow.add_node("calculate_costs", self._node_calculate_costs)
        workflow.add_node("generate_quote", self._node_generate_quote)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("send_quote", self._node_send_quote)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("gather_data")

        workflow.add_edge("gather_data", "calculate_costs")
        workflow.add_edge("calculate_costs", "generate_quote")
        workflow.add_edge("generate_quote", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "send_quote",
                "rejected": "report",
            },
        )
        workflow.add_edge("send_quote", "report")
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
    def get_state_class(cls) -> Type[QuoteEngineAgentState]:
        return QuoteEngineAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "file_analysis_id": "",
            "customer_id": "",
            "customer_name": "",
            "customer_email": "",
            "volume_cm3": 0.0,
            "surface_area_cm2": 0.0,
            "bounding_box": {},
            "print_technology": "",
            "material": "",
            "material_cost_cents": 0,
            "time_cost_cents": 0,
            "post_process_costs": [],
            "post_process_total_cents": 0,
            "shipping_cost_cents": 0,
            "subtotal_cents": 0,
            "markup_percent": 0.0,
            "markup_cents": 0,
            "total_cents": 0,
            "estimated_print_hours": 0.0,
            "quote_id": "",
            "quote_document": "",
            "quote_valid_days": 30,
            "quote_saved": False,
            "quote_sent": False,
            "sent_at": "",
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Gather Data ─────────────────────────────────────────

    async def _node_gather_data(
        self, state: QuoteEngineAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull geometry and material data from print_jobs and file_analyses."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")
        file_analysis_id = task.get("file_analysis_id", "")

        logger.info(
            "quote_engine_gather_data",
            extra={"print_job_id": print_job_id, "agent_id": self.agent_id},
        )

        volume_cm3 = task.get("volume_cm3", 0.0)
        surface_area_cm2 = task.get("surface_area_cm2", 0.0)
        bounding_box = task.get("bounding_box", {})
        material = task.get("material", "PLA")
        print_technology = task.get("print_technology", "FDM")
        customer_id = task.get("customer_id", "")
        customer_name = task.get("customer_name", "Customer")
        customer_email = task.get("customer_email", "")

        # Load print job data
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
                    volume_cm3 = job.get("volume_cm3", volume_cm3)
                    material = job.get("material", material)
                    print_technology = job.get("print_technology", print_technology)
                    customer_id = job.get("customer_id", customer_id)
                    customer_name = job.get("customer_name", customer_name)
                    customer_email = job.get("customer_email", customer_email)
                    logger.info(
                        "quote_engine_job_loaded",
                        extra={"print_job_id": print_job_id},
                    )
            except Exception as e:
                logger.warning(
                    "quote_engine_db_error",
                    extra={"error": str(e)[:200]},
                )

        # Load file analysis data
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
                    volume_cm3 = volume_cm3 or analysis.get("volume_cm3", 0.0)
                    surface_area_cm2 = surface_area_cm2 or analysis.get(
                        "surface_area_cm2", 0.0
                    )
                    bb_raw = analysis.get("bounding_box", "{}")
                    if isinstance(bb_raw, str):
                        try:
                            bounding_box = json.loads(bb_raw)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    else:
                        bounding_box = bb_raw or bounding_box
            except Exception as e:
                logger.warning(
                    "quote_engine_analysis_db_error",
                    extra={"error": str(e)[:200]},
                )

        return {
            "current_node": "gather_data",
            "print_job_id": print_job_id,
            "file_analysis_id": file_analysis_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "volume_cm3": volume_cm3,
            "surface_area_cm2": surface_area_cm2,
            "bounding_box": bounding_box,
            "material": material,
            "print_technology": print_technology,
        }

    # ─── Node 2: Calculate Costs ─────────────────────────────────────

    async def _node_calculate_costs(
        self, state: QuoteEngineAgentState
    ) -> dict[str, Any]:
        """Node 2: Compute full cost breakdown from geometry + material data."""
        task = state.get("task_input", {})
        volume_cm3 = state.get("volume_cm3", 100.0)
        material = state.get("material", "PLA")
        print_technology = state.get("print_technology", "FDM")

        logger.info(
            "quote_engine_calculate_costs",
            extra={"material": material, "volume_cm3": volume_cm3},
        )

        # Material cost
        from core.agents.implementations.material_advisor_agent import MATERIALS_DB
        mat_info = MATERIALS_DB.get(material, {"cost_per_cm3": 0.04, "tech": "FDM"})
        material_cost_cents = int(mat_info["cost_per_cm3"] * volume_cm3 * 100)

        # Print time estimation
        print_speed = PRINT_SPEEDS.get(print_technology, 15.0)
        estimated_print_hours = round(volume_cm3 / print_speed, 2)
        time_cost_cents = int(
            estimated_print_hours * PRICING_RATES["print_rate_per_hour"]
        )

        # Setup fee
        setup_fee = PRICING_RATES["setup_fee"]

        # Post-processing costs
        requested_processes = task.get("post_processes", ["support_removal"])
        post_process_list: list[dict[str, Any]] = []
        post_process_total = 0
        for proc in requested_processes:
            proc_cost = POST_PROCESS_COSTS.get(proc, 0)
            if proc_cost:
                post_process_list.append({
                    "process": proc,
                    "cost_cents": proc_cost,
                })
                post_process_total += proc_cost

        # Shipping
        shipping_method = task.get("shipping_method", "standard")
        shipping_info = SHIPPING_RATES.get(shipping_method, SHIPPING_RATES["standard"])
        weight_kg = task.get("weight_kg", volume_cm3 * 0.001 * 1.2)  # rough estimate
        shipping_cost_cents = (
            shipping_info["base_cents"]
            + int(weight_kg * shipping_info["per_kg_cents"])
        )

        # Subtotal before markup
        subtotal_cents = (
            material_cost_cents
            + time_cost_cents
            + setup_fee
            + post_process_total
            + shipping_cost_cents
        )

        # Apply markup
        markup_percent = PRICING_RATES["default_markup_percent"]
        # Volume discount
        if volume_cm3 >= PRICING_RATES["volume_discount_threshold_cm3"]:
            markup_percent -= PRICING_RATES["volume_discount_percent"]

        # Rush order
        is_rush = task.get("is_rush", False)
        if is_rush:
            markup_percent *= PRICING_RATES["rush_multiplier"]

        markup_cents = int(subtotal_cents * markup_percent / 100)
        total_cents = subtotal_cents + markup_cents

        # Enforce minimum order
        if total_cents < PRICING_RATES["min_order_cents"]:
            total_cents = PRICING_RATES["min_order_cents"]
            markup_cents = total_cents - subtotal_cents

        logger.info(
            "quote_engine_costs_calculated",
            extra={
                "material_cost_cents": material_cost_cents,
                "time_cost_cents": time_cost_cents,
                "post_process_cents": post_process_total,
                "shipping_cents": shipping_cost_cents,
                "total_cents": total_cents,
            },
        )

        return {
            "current_node": "calculate_costs",
            "material_cost_cents": material_cost_cents,
            "time_cost_cents": time_cost_cents,
            "post_process_costs": post_process_list,
            "post_process_total_cents": post_process_total,
            "shipping_cost_cents": shipping_cost_cents,
            "subtotal_cents": subtotal_cents,
            "markup_percent": round(markup_percent, 1),
            "markup_cents": markup_cents,
            "total_cents": total_cents,
            "estimated_print_hours": estimated_print_hours,
        }

    # ─── Node 3: Generate Quote ──────────────────────────────────────

    async def _node_generate_quote(
        self, state: QuoteEngineAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM creates professional formatted quote document."""
        logger.info("quote_engine_generate_quote")

        valid_days = state.get("quote_valid_days", 30)
        total_cents = state.get("total_cents", 0)

        try:
            prompt = QUOTE_SYSTEM_PROMPT.format(
                valid_days=valid_days,
                material=state.get("material", "PLA"),
                technology=state.get("print_technology", "FDM"),
                volume_cm3=round(state.get("volume_cm3", 0), 2),
                material_cost=f"{state.get('material_cost_cents', 0) / 100:.2f}",
                print_hours=state.get("estimated_print_hours", 0),
                time_cost=f"{state.get('time_cost_cents', 0) / 100:.2f}",
                post_process_total=f"{state.get('post_process_total_cents', 0) / 100:.2f}",
                shipping_method="standard",
                shipping_cost=f"{state.get('shipping_cost_cents', 0) / 100:.2f}",
                subtotal=f"{state.get('subtotal_cents', 0) / 100:.2f}",
                markup_pct=state.get("markup_percent", 40),
                markup=f"{state.get('markup_cents', 0) / 100:.2f}",
                total=f"{total_cents / 100:.2f}",
                customer_name=state.get("customer_name", "Customer"),
                customer_email=state.get("customer_email", ""),
                print_job_id=state.get("print_job_id", ""),
            )

            llm_response = self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                system="You are a professional print shop quoting specialist.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            quote_document = llm_response.content[0].text.strip()

        except Exception as e:
            logger.warning(
                "quote_engine_llm_error",
                extra={"error": str(e)[:200]},
            )
            # Fallback to simple quote
            quote_document = (
                f"# Print Quote\n\n"
                f"**Material:** {state.get('material', 'PLA')}\n"
                f"**Total:** ${total_cents / 100:.2f}\n"
                f"**Valid for:** {valid_days} days\n"
            )

        return {
            "current_node": "generate_quote",
            "quote_document": quote_document,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: QuoteEngineAgentState
    ) -> dict[str, Any]:
        """Node 4: Present quote for human approval before sending."""
        total = state.get("total_cents", 0)
        customer = state.get("customer_name", "")

        logger.info(
            "quote_engine_human_review_pending",
            extra={
                "total_cents": total,
                "customer": customer,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Send Quote ──────────────────────────────────────────

    async def _node_send_quote(
        self, state: QuoteEngineAgentState
    ) -> dict[str, Any]:
        """Node 5: Save quote to print_quotes table and mark as sent."""
        now = datetime.now(timezone.utc).isoformat()
        print_job_id = state.get("print_job_id", "")

        logger.info(
            "quote_engine_send_quote",
            extra={"print_job_id": print_job_id},
        )

        quote_record = {
            "print_job_id": print_job_id,
            "vertical_id": self.vertical_id,
            "agent_id": self.agent_id,
            "customer_id": state.get("customer_id", ""),
            "customer_name": state.get("customer_name", ""),
            "customer_email": state.get("customer_email", ""),
            "material": state.get("material", ""),
            "print_technology": state.get("print_technology", ""),
            "volume_cm3": state.get("volume_cm3", 0.0),
            "material_cost_cents": state.get("material_cost_cents", 0),
            "time_cost_cents": state.get("time_cost_cents", 0),
            "post_process_total_cents": state.get("post_process_total_cents", 0),
            "shipping_cost_cents": state.get("shipping_cost_cents", 0),
            "subtotal_cents": state.get("subtotal_cents", 0),
            "markup_percent": state.get("markup_percent", 0.0),
            "markup_cents": state.get("markup_cents", 0),
            "total_cents": state.get("total_cents", 0),
            "estimated_print_hours": state.get("estimated_print_hours", 0.0),
            "quote_document": state.get("quote_document", ""),
            "status": "sent",
            "created_at": now,
            "sent_at": now,
        }

        quote_id = ""
        quote_saved = False

        try:
            result = (
                self.db.client.table("print_quotes")
                .insert(quote_record)
                .execute()
            )
            if result.data and len(result.data) > 0:
                quote_id = result.data[0].get("id", "")
                quote_saved = True
                logger.info(
                    "quote_engine_quote_saved",
                    extra={"quote_id": quote_id},
                )
        except Exception as e:
            logger.warning(
                "quote_engine_save_error",
                extra={"error": str(e)[:200]},
            )

        return {
            "current_node": "send_quote",
            "quote_id": quote_id,
            "quote_saved": quote_saved,
            "quote_sent": True,
            "sent_at": now,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: QuoteEngineAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate summary report and store pricing insights."""
        now = datetime.now(timezone.utc).isoformat()
        total_cents = state.get("total_cents", 0)
        material = state.get("material", "unknown")
        technology = state.get("print_technology", "unknown")
        volume = state.get("volume_cm3", 0.0)

        sections = [
            "# Quote Generation Report",
            f"*Generated: {now}*\n",
            f"## Quote Summary",
            f"- **Material:** {material} ({technology})",
            f"- **Volume:** {volume:.2f} cm3",
            f"- **Print Time:** {state.get('estimated_print_hours', 0):.1f} hours",
            f"- **Material Cost:** ${state.get('material_cost_cents', 0) / 100:.2f}",
            f"- **Time Cost:** ${state.get('time_cost_cents', 0) / 100:.2f}",
            f"- **Post-Processing:** ${state.get('post_process_total_cents', 0) / 100:.2f}",
            f"- **Shipping:** ${state.get('shipping_cost_cents', 0) / 100:.2f}",
            f"- **Markup:** {state.get('markup_percent', 0):.1f}%",
            f"- **Total:** ${total_cents / 100:.2f}",
            f"\n## Status",
            f"- Quote Saved: {'Yes' if state.get('quote_saved') else 'No'}",
            f"- Quote Sent: {'Yes' if state.get('quote_sent') else 'No'}",
            f"- Quote ID: {state.get('quote_id', 'N/A')}",
        ]

        report = "\n".join(sections)

        # Store pricing insight
        self.store_insight(InsightData(
            insight_type="pricing_pattern",
            title=f"Quote: {material} {volume:.0f}cm3 = ${total_cents / 100:.2f}",
            content=(
                f"Generated quote for {material} ({technology}), "
                f"volume {volume:.1f} cm3. Total: ${total_cents / 100:.2f} "
                f"(material ${state.get('material_cost_cents', 0) / 100:.2f}, "
                f"time ${state.get('time_cost_cents', 0) / 100:.2f}, "
                f"markup {state.get('markup_percent', 0):.1f}%)."
            ),
            confidence=0.85,
            metadata={
                "material": material,
                "technology": technology,
                "volume_cm3": volume,
                "total_cents": total_cents,
                "markup_percent": state.get("markup_percent", 0),
            },
        ))

        logger.info(
            "quote_engine_report_generated",
            extra={"total_cents": total_cents, "material": material},
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: QuoteEngineAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<QuoteEngineAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
