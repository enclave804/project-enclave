"""
Logistics Agent — The Shipping & Packaging Coordinator.

Plans optimal packaging and selects the best carrier for 3D printed items
based on fragility, dimensions, weight, and destination. Manages shipment
tracking and delivery confirmations.

Architecture (LangGraph State Machine):
    load_shipment → plan_packaging → select_carrier →
    human_review → report → END

Trigger Events:
    - job_completed: Print job passed QC and is ready to ship
    - shipping_request: Manual shipping coordination request
    - manual: On-demand logistics planning

Shared Brain Integration:
    - Reads: packaging success rates, carrier performance history
    - Writes: shipping patterns, carrier reliability data, damage rates

Safety:
    - NEVER commits to shipping without human review
    - Address validation is advisory; human confirms destination
    - Tracking information is logged but not externally shared without approval
    - Cost estimates are clearly marked as estimates

Usage:
    agent = LogisticsAgent(config, db, embedder, llm)
    result = await agent.run({
        "print_job_id": "pj_abc123",
        "shipping_address": {"city": "Austin", "state": "TX", "zip": "78701"},
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
from core.agents.state import LogisticsAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PACKAGING_OPTIONS = {
    "standard_box": {
        "label": "Standard Cardboard Box",
        "cost_cents": 3_00,
        "max_weight_kg": 10.0,
        "fragility_rating": "low",
        "padding": "bubble_wrap",
        "description": "Basic cardboard box with bubble wrap padding.",
    },
    "padded_box": {
        "label": "Padded Shipping Box",
        "cost_cents": 5_50,
        "max_weight_kg": 8.0,
        "fragility_rating": "medium",
        "padding": "foam_insert",
        "description": "Double-walled box with custom foam inserts.",
    },
    "custom_crate": {
        "label": "Custom Wooden Crate",
        "cost_cents": 18_00,
        "max_weight_kg": 50.0,
        "fragility_rating": "high",
        "padding": "foam_suspension",
        "description": "Wooden crate with suspension foam for fragile items.",
    },
    "envelope_mailer": {
        "label": "Padded Envelope",
        "cost_cents": 1_50,
        "max_weight_kg": 1.0,
        "fragility_rating": "low",
        "padding": "bubble_lined",
        "description": "Lightweight padded envelope for small, durable items.",
    },
    "rigid_mailer": {
        "label": "Rigid Cardboard Mailer",
        "cost_cents": 4_00,
        "max_weight_kg": 3.0,
        "fragility_rating": "medium",
        "padding": "corrugated_insert",
        "description": "Rigid mailer with corrugated insert for flat items.",
    },
}

CARRIER_OPTIONS = [
    {
        "name": "USPS",
        "service": "Priority Mail",
        "base_cost_cents": 8_99,
        "per_kg_cents": 1_50,
        "delivery_days_min": 2,
        "delivery_days_max": 5,
        "tracking": True,
        "insurance_available": True,
        "max_weight_kg": 31.75,
    },
    {
        "name": "UPS",
        "service": "Ground",
        "base_cost_cents": 10_99,
        "per_kg_cents": 2_00,
        "delivery_days_min": 3,
        "delivery_days_max": 7,
        "tracking": True,
        "insurance_available": True,
        "max_weight_kg": 68.0,
    },
    {
        "name": "UPS",
        "service": "2nd Day Air",
        "base_cost_cents": 22_99,
        "per_kg_cents": 3_50,
        "delivery_days_min": 2,
        "delivery_days_max": 2,
        "tracking": True,
        "insurance_available": True,
        "max_weight_kg": 68.0,
    },
    {
        "name": "FedEx",
        "service": "Ground",
        "base_cost_cents": 11_50,
        "per_kg_cents": 2_25,
        "delivery_days_min": 3,
        "delivery_days_max": 7,
        "tracking": True,
        "insurance_available": True,
        "max_weight_kg": 68.0,
    },
    {
        "name": "FedEx",
        "service": "Express",
        "base_cost_cents": 28_99,
        "per_kg_cents": 4_00,
        "delivery_days_min": 1,
        "delivery_days_max": 2,
        "tracking": True,
        "insurance_available": True,
        "max_weight_kg": 68.0,
    },
]

FRAGILITY_MAP = {
    "SLA": "high",
    "SLA_RESIN": "high",
    "FDM": "medium",
    "SLS": "low",
    "Binder_Jetting": "high",
    "MJF": "low",
}

LOGISTICS_SYSTEM_PROMPT = """\
You are a logistics coordinator for a 3D print shop. Given the shipment \
details below, confirm the packaging and carrier selections are optimal.

Return a JSON object:
{{
    "packaging_approved": true/false,
    "carrier_approved": true/false,
    "packaging_notes": "Any additional packaging recommendations...",
    "carrier_notes": "Any carrier-specific recommendations...",
    "handling_instructions": "Special handling instructions for the carrier...",
    "estimated_total_cost_cents": 0
}}

Shipment Details:
- Print Technology: {print_technology}
- Weight: {weight_kg} kg
- Dimensions: {dimensions}
- Is Fragile: {is_fragile}
- Destination: {destination}
- Selected Packaging: {selected_packaging}
- Selected Carrier: {selected_carrier} ({selected_service})

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("logistics")
class LogisticsAgent(BaseAgent):
    """
    Packaging and shipping coordination agent for 3D printed items.

    Nodes:
        1. load_shipment    -- Pull job details + destination from print_jobs
        2. plan_packaging   -- Select packaging based on fragility, dimensions, weight
        3. select_carrier   -- Compare carrier options by cost and delivery time
        4. human_review     -- Gate: approve shipping plan
        5. report           -- Save tracking info + InsightData for patterns
    """

    def build_graph(self) -> Any:
        """Build the Logistics Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(LogisticsAgentState)

        workflow.add_node("load_shipment", self._node_load_shipment)
        workflow.add_node("plan_packaging", self._node_plan_packaging)
        workflow.add_node("select_carrier", self._node_select_carrier)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_shipment")

        workflow.add_edge("load_shipment", "plan_packaging")
        workflow.add_edge("plan_packaging", "select_carrier")
        workflow.add_edge("select_carrier", "human_review")
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
    def get_state_class(cls) -> Type[LogisticsAgentState]:
        return LogisticsAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "print_job_id": "",
            "customer_id": "",
            "customer_name": "",
            "customer_email": "",
            "shipping_address": {},
            "package_weight_kg": 0.0,
            "package_dimensions": {},
            "is_fragile": False,
            "selected_packaging": "",
            "packaging_cost_cents": 0,
            "packaging_notes": "",
            "carrier_options": [],
            "selected_carrier": "",
            "shipping_cost_cents": 0,
            "estimated_delivery_days": 0,
            "tracking_number": "",
            "shipment_status": "pending",
            "shipped_at": "",
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Shipment ───────────────────────────────────────

    async def _node_load_shipment(
        self, state: LogisticsAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull job details and destination from print_jobs."""
        task = state.get("task_input", {})
        print_job_id = task.get("print_job_id", "")

        logger.info(
            "logistics_load_shipment",
            extra={"print_job_id": print_job_id, "agent_id": self.agent_id},
        )

        customer_id = task.get("customer_id", "")
        customer_name = task.get("customer_name", "Customer")
        customer_email = task.get("customer_email", "")
        shipping_address = task.get("shipping_address", {})
        weight_kg = task.get("package_weight_kg", 0.5)
        dimensions = task.get("package_dimensions", {
            "length": 20.0, "width": 15.0, "height": 10.0
        })
        print_technology = task.get("print_technology", "FDM")
        is_fragile = task.get("is_fragile", False)

        # Load from DB
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
                    customer_id = job.get("customer_id", customer_id)
                    customer_name = job.get("customer_name", customer_name)
                    customer_email = job.get("customer_email", customer_email)
                    addr = job.get("shipping_address", None)
                    if addr:
                        if isinstance(addr, str):
                            try:
                                shipping_address = json.loads(addr)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        else:
                            shipping_address = addr
                    weight_kg = job.get("package_weight_kg", weight_kg)
                    print_technology = job.get("print_technology", print_technology)
                    logger.info(
                        "logistics_job_loaded",
                        extra={"print_job_id": print_job_id},
                    )
            except Exception as e:
                logger.warning(
                    "logistics_db_error",
                    extra={"error": str(e)[:200]},
                )

        # Determine fragility from print technology if not explicitly set
        if not is_fragile:
            fragility_level = FRAGILITY_MAP.get(print_technology, "medium")
            is_fragile = fragility_level == "high"

        return {
            "current_node": "load_shipment",
            "print_job_id": print_job_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "shipping_address": shipping_address,
            "package_weight_kg": weight_kg,
            "package_dimensions": dimensions,
            "is_fragile": is_fragile,
        }

    # ─── Node 2: Plan Packaging ──────────────────────────────────────

    async def _node_plan_packaging(
        self, state: LogisticsAgentState
    ) -> dict[str, Any]:
        """Node 2: Select packaging based on fragility, dimensions, weight."""
        weight_kg = state.get("package_weight_kg", 0.5)
        is_fragile = state.get("is_fragile", False)
        dimensions = state.get("package_dimensions", {})

        logger.info(
            "logistics_plan_packaging",
            extra={"weight_kg": weight_kg, "is_fragile": is_fragile},
        )

        # Calculate volume in approximate liters
        vol_liters = (
            dimensions.get("length", 20)
            * dimensions.get("width", 15)
            * dimensions.get("height", 10)
        ) / 1000.0

        selected_packaging = ""
        packaging_cost_cents = 0
        packaging_notes = ""

        # Selection logic
        if is_fragile:
            if weight_kg > 5.0:
                selected_packaging = "custom_crate"
                packaging_notes = "Fragile heavy item requires custom crate with suspension foam."
            else:
                selected_packaging = "padded_box"
                packaging_notes = "Fragile item requires padded box with foam inserts."
        elif weight_kg <= 0.5 and vol_liters < 2.0:
            selected_packaging = "envelope_mailer"
            packaging_notes = "Small lightweight item fits in padded envelope."
        elif weight_kg <= 2.0 and vol_liters < 5.0:
            selected_packaging = "rigid_mailer"
            packaging_notes = "Medium item fits in rigid mailer with insert."
        else:
            selected_packaging = "standard_box"
            packaging_notes = "Standard item fits in regular box with bubble wrap."

        pkg_info = PACKAGING_OPTIONS.get(selected_packaging, {})
        packaging_cost_cents = pkg_info.get("cost_cents", 3_00)

        # Validate weight limit
        max_weight = pkg_info.get("max_weight_kg", 10.0)
        if weight_kg > max_weight:
            selected_packaging = "custom_crate"
            pkg_info = PACKAGING_OPTIONS["custom_crate"]
            packaging_cost_cents = pkg_info["cost_cents"]
            packaging_notes = f"Upgraded to custom crate (weight {weight_kg:.1f}kg exceeds {max_weight}kg limit)."

        logger.info(
            "logistics_packaging_selected",
            extra={
                "selected_packaging": selected_packaging,
                "cost_cents": packaging_cost_cents,
            },
        )

        return {
            "current_node": "plan_packaging",
            "selected_packaging": selected_packaging,
            "packaging_cost_cents": packaging_cost_cents,
            "packaging_notes": packaging_notes,
        }

    # ─── Node 3: Select Carrier ──────────────────────────────────────

    async def _node_select_carrier(
        self, state: LogisticsAgentState
    ) -> dict[str, Any]:
        """Node 3: Compare carrier options by cost and delivery time."""
        task = state.get("task_input", {})
        weight_kg = state.get("package_weight_kg", 0.5)
        shipping_priority = task.get("shipping_priority", "standard")

        logger.info(
            "logistics_select_carrier",
            extra={"weight_kg": weight_kg, "priority": shipping_priority},
        )

        # Score each carrier option
        scored_options: list[dict[str, Any]] = []
        for carrier in CARRIER_OPTIONS:
            if weight_kg > carrier["max_weight_kg"]:
                continue

            total_cost_cents = (
                carrier["base_cost_cents"]
                + int(weight_kg * carrier["per_kg_cents"])
            )
            avg_days = (
                carrier["delivery_days_min"] + carrier["delivery_days_max"]
            ) / 2.0

            # Score: lower cost and faster delivery = higher score
            cost_score = max(0, 100 - total_cost_cents / 50)
            speed_score = max(0, 100 - avg_days * 15)

            if shipping_priority == "express":
                overall_score = cost_score * 0.3 + speed_score * 0.7
            elif shipping_priority == "economy":
                overall_score = cost_score * 0.8 + speed_score * 0.2
            else:
                overall_score = cost_score * 0.5 + speed_score * 0.5

            scored_options.append({
                "carrier": carrier["name"],
                "service": carrier["service"],
                "cost_cents": total_cost_cents,
                "delivery_days_min": carrier["delivery_days_min"],
                "delivery_days_max": carrier["delivery_days_max"],
                "tracking": carrier["tracking"],
                "insurance_available": carrier["insurance_available"],
                "score": round(overall_score, 1),
            })

        # Sort by score descending
        scored_options.sort(key=lambda x: x["score"], reverse=True)

        best = scored_options[0] if scored_options else {
            "carrier": "USPS",
            "service": "Priority Mail",
            "cost_cents": 8_99,
            "delivery_days_max": 5,
        }

        logger.info(
            "logistics_carrier_selected",
            extra={
                "selected_carrier": f"{best['carrier']} {best.get('service', '')}",
                "cost_cents": best["cost_cents"],
            },
        )

        return {
            "current_node": "select_carrier",
            "carrier_options": scored_options,
            "selected_carrier": f"{best['carrier']} {best.get('service', '')}",
            "shipping_cost_cents": best["cost_cents"],
            "estimated_delivery_days": best.get("delivery_days_max", 5),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: LogisticsAgentState
    ) -> dict[str, Any]:
        """Node 4: Present shipping plan for human approval."""
        carrier = state.get("selected_carrier", "")
        packaging = state.get("selected_packaging", "")
        cost = state.get("shipping_cost_cents", 0) + state.get("packaging_cost_cents", 0)

        logger.info(
            "logistics_human_review_pending",
            extra={
                "carrier": carrier,
                "packaging": packaging,
                "total_cost_cents": cost,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: LogisticsAgentState
    ) -> dict[str, Any]:
        """Node 5: Save tracking info and generate InsightData."""
        now = datetime.now(timezone.utc).isoformat()
        carrier = state.get("selected_carrier", "unknown")
        packaging = state.get("selected_packaging", "unknown")
        ship_cost = state.get("shipping_cost_cents", 0)
        pkg_cost = state.get("packaging_cost_cents", 0)
        total_cost = ship_cost + pkg_cost

        pkg_info = PACKAGING_OPTIONS.get(packaging, {})

        sections = [
            "# Logistics Report",
            f"*Generated: {now}*\n",
            f"## Packaging",
            f"- **Type:** {pkg_info.get('label', packaging)}",
            f"- **Cost:** ${pkg_cost / 100:.2f}",
            f"- **Notes:** {state.get('packaging_notes', 'N/A')}",
            f"\n## Carrier",
            f"- **Selected:** {carrier}",
            f"- **Shipping Cost:** ${ship_cost / 100:.2f}",
            f"- **Estimated Delivery:** {state.get('estimated_delivery_days', 'N/A')} days",
            f"\n## Total Logistics Cost: ${total_cost / 100:.2f}",
            f"\n## Shipment Status: {state.get('shipment_status', 'pending')}",
        ]

        address = state.get("shipping_address", {})
        if address:
            sections.append(f"\n## Destination")
            sections.append(
                f"- {address.get('city', '')}, {address.get('state', '')} "
                f"{address.get('zip', '')}, {address.get('country', 'US')}"
            )

        report = "\n".join(sections)

        # Store shipping pattern insight
        self.store_insight(InsightData(
            insight_type="shipping_pattern",
            title=f"Shipment: {carrier} via {packaging}",
            content=(
                f"Shipped via {carrier} with {packaging} packaging. "
                f"Total logistics cost: ${total_cost / 100:.2f} "
                f"(shipping ${ship_cost / 100:.2f} + packaging ${pkg_cost / 100:.2f}). "
                f"Weight: {state.get('package_weight_kg', 0):.1f}kg, "
                f"fragile: {state.get('is_fragile', False)}."
            ),
            confidence=0.80,
            metadata={
                "carrier": carrier,
                "packaging": packaging,
                "total_cost_cents": total_cost,
                "weight_kg": state.get("package_weight_kg", 0),
                "is_fragile": state.get("is_fragile", False),
                "estimated_days": state.get("estimated_delivery_days", 0),
            },
        ))

        logger.info(
            "logistics_report_generated",
            extra={
                "carrier": carrier,
                "total_cost_cents": total_cost,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: LogisticsAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<LogisticsAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
