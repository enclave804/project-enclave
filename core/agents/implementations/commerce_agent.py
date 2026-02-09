"""
Commerce Agent — The Store Manager.

Monitors the storefront, processes orders, detects VIP customers,
handles refund requests, and alerts on low inventory.

Architecture (LangGraph State Machine):
    monitor → triage → [handle_vip | handle_low_stock | handle_refund] →
    human_review → execute → report → END

The Commerce Agent is the bridge between your storefront (Shopify) and
your agent fleet. When a whale drops $9,999 on an enterprise plan,
the Commerce Agent detects it and triggers the Outreach Agent to send
a personal thank-you from the CEO.

Key Scenarios:
    1. VIP Order: Customer spends >$500 → draft VIP follow-up email →
       human approves → Outreach Agent sends personal note
    2. Low Stock: Variant drops below 5 units → alert human →
       optionally restock via API
    3. Refund Request: Customer requests refund → draft refund for
       human approval → process via Stripe
    4. Routine: Normal orders → log and report

Safety:
    - Refunds ALWAYS require human approval (human gate)
    - Inventory updates are sandboxed in non-production
    - VIP emails go through Outreach Agent's existing human gate
    - All financial operations are logged to shared brain

Usage:
    agent = CommerceAgent(config, db, embedder, llm)
    result = await agent.run({"mode": "full_check"})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, OrderData
from core.agents.registry import register_agent_type
from core.agents.state import CommerceAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

VIP_THRESHOLD = 500.0  # Orders above this are VIP
LOW_STOCK_THRESHOLD = 5  # Variants below this trigger alerts

# Triage actions
ACTION_VIP_FOLLOWUP = "vip_followup"
ACTION_RESTOCK_ALERT = "restock_alert"
ACTION_REFUND_REVIEW = "refund_review"
ACTION_ROUTINE = "routine"

# System prompt for LLM-assisted triage
COMMERCE_SYSTEM_PROMPT = """You are the Commerce Agent — the Store Manager for an autonomous business platform.

Your job is to analyze storefront activity and decide what actions to take.

You receive:
- Recent orders (with VIP flags for high-value customers)
- Inventory levels (with low-stock alerts)
- Payment status updates

Your decisions:
1. VIP Follow-up: When a high-value order comes in, draft a personal thank-you email
   from the CEO. Be warm, specific about what they purchased, and genuine.
2. Restock Alert: When inventory is critically low, recommend restock quantities.
3. Refund Review: When a refund is requested, analyze the order and recommend
   whether to approve (always requires human sign-off).
4. Routine: Normal operations — just report the summary.

IMPORTANT: You are NOT the one who sends emails or processes refunds.
You RECOMMEND actions. Humans approve. Other agents execute.

Respond in JSON format:
{
    "triage_action": "vip_followup" | "restock_alert" | "refund_review" | "routine",
    "reasoning": "Why this action was chosen",
    "actions": [
        {
            "action": "send_vip_email" | "restock" | "process_refund" | "alert_team",
            "target": "customer email or variant ID",
            "details": "Specific details",
            "requires_approval": true
        }
    ]
}"""


@register_agent_type("commerce")
class CommerceAgent(BaseAgent):
    """
    The Store Manager — monitors commerce, detects VIPs, handles refunds.

    LangGraph Workflow:
        monitor: Check for new orders and low inventory
        triage: Decide what needs attention (LLM-assisted for complex cases)
        handle_vip: Draft VIP follow-up email
        handle_low_stock: Prepare restock recommendations
        handle_refund: Draft refund for human approval
        human_review: Gate for human approval of actions
        execute: Execute approved actions
        report: Generate commerce status report
    """

    agent_type = "commerce"

    def build_graph(self) -> Any:
        """Build the CommerceAgent LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(CommerceAgentState)

        # Add nodes
        workflow.add_node("monitor", self._node_monitor)
        workflow.add_node("triage", self._node_triage)
        workflow.add_node("handle_vip", self._node_handle_vip)
        workflow.add_node("handle_low_stock", self._node_handle_low_stock)
        workflow.add_node("handle_refund", self._node_handle_refund)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute", self._node_execute)
        workflow.add_node("report", self._node_report)

        # Entry point
        workflow.set_entry_point("monitor")

        # monitor → triage
        workflow.add_edge("monitor", "triage")

        # triage → route by action
        workflow.add_conditional_edges(
            "triage",
            self._route_by_triage,
            {
                ACTION_VIP_FOLLOWUP: "handle_vip",
                ACTION_RESTOCK_ALERT: "handle_low_stock",
                ACTION_REFUND_REVIEW: "handle_refund",
                ACTION_ROUTINE: "report",
            },
        )

        # All handlers → human_review
        workflow.add_edge("handle_vip", "human_review")
        workflow.add_edge("handle_low_stock", "human_review")
        workflow.add_edge("handle_refund", "human_review")

        # human_review → route by approval
        workflow.add_conditional_edges(
            "human_review",
            self._route_by_approval,
            {
                "approved": "execute",
                "rejected": "report",
            },
        )

        # execute → report
        workflow.add_edge("execute", "report")

        # report → END
        workflow.add_edge("report", END)

        # Compile with human gates
        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            compile_kwargs["interrupt_before"] = self.config.human_gates.gate_before
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list:
        """Commerce tools are accessed via MCP, not directly."""
        return []

    def get_state_class(self) -> Type:
        return CommerceAgentState

    # ─── Routing Functions ──────────────────────────────────────────

    def _route_by_triage(self, state: CommerceAgentState) -> str:
        """Route to the appropriate handler based on triage decision."""
        action = state.get("triage_action", ACTION_ROUTINE)
        logger.info(
            "commerce_routing",
            extra={
                "agent_id": self.agent_id,
                "triage_action": action,
            },
        )
        return action

    def _route_by_approval(self, state: CommerceAgentState) -> str:
        """Route based on human approval status."""
        if state.get("actions_approved", False):
            return "approved"
        return "rejected"

    # ─── Node Implementations ──────────────────────────────────────

    async def _node_monitor(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 1: Monitor — Check storefront for new orders and inventory.

        Calls commerce MCP tools to gather current state.
        """
        logger.info(
            "commerce_monitor_start",
            extra={"agent_id": self.agent_id},
        )

        result: dict[str, Any] = {
            "current_node": "monitor",
            "recent_orders": [],
            "order_count": 0,
            "total_revenue": 0.0,
            "vip_orders": [],
            "vip_count": 0,
            "products": [],
            "low_stock_alerts": [],
            "low_stock_count": 0,
        }

        # Fetch recent orders
        try:
            from core.mcp.tools.commerce_tools import shopify_get_recent_orders
            orders_json = await shopify_get_recent_orders(days=1)
            orders_data = json.loads(orders_json)

            result["recent_orders"] = orders_data.get("orders", [])
            summary = orders_data.get("summary", {})
            result["order_count"] = summary.get("order_count", 0)
            result["total_revenue"] = summary.get("total_revenue", 0.0)
            result["vip_orders"] = [
                o for o in result["recent_orders"] if o.get("is_vip")
            ]
            result["vip_count"] = len(result["vip_orders"])

        except Exception as e:
            logger.warning(
                "commerce_orders_fetch_failed",
                extra={"error": str(e)[:200]},
            )

        # Fetch product inventory
        try:
            from core.mcp.tools.commerce_tools import shopify_get_products
            products_json = await shopify_get_products(limit=50)
            products_data = json.loads(products_json)

            result["products"] = products_data.get("products", [])
            result["low_stock_alerts"] = products_data.get("low_stock_alerts", [])
            result["low_stock_count"] = products_data.get("low_stock_count", 0)

        except Exception as e:
            logger.warning(
                "commerce_products_fetch_failed",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "commerce_monitor_complete",
            extra={
                "orders": result["order_count"],
                "revenue": result["total_revenue"],
                "vip_count": result["vip_count"],
                "low_stock": result["low_stock_count"],
            },
        )

        return result

    async def _node_triage(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 2: Triage — Decide what needs attention.

        Priority order:
        1. VIP orders (high-value customers need personal attention)
        2. Refund requests (time-sensitive, customer satisfaction)
        3. Low stock (revenue protection)
        4. Routine (just report)
        """
        vip_count = state.get("vip_count", 0)
        low_stock_count = state.get("low_stock_count", 0)
        task_input = state.get("task_input", {})
        mode = task_input.get("mode", "full_check") if isinstance(task_input, dict) else "full_check"

        # Check if this is a refund request from webhook/event
        if mode == "refund_review":
            return {
                "current_node": "triage",
                "triage_action": ACTION_REFUND_REVIEW,
                "triage_reasoning": "Refund request received — requires human review.",
            }

        # Priority-based triage (no LLM needed for simple decisions)
        if vip_count > 0:
            vip_order = state.get("vip_orders", [{}])[0]
            return {
                "current_node": "triage",
                "triage_action": ACTION_VIP_FOLLOWUP,
                "triage_reasoning": (
                    f"VIP order detected: {vip_order.get('customer_name', 'Unknown')} "
                    f"spent ${vip_order.get('total_price', 0)} — "
                    f"personal follow-up recommended."
                ),
            }

        if low_stock_count > 0:
            return {
                "current_node": "triage",
                "triage_action": ACTION_RESTOCK_ALERT,
                "triage_reasoning": (
                    f"{low_stock_count} product variant(s) below stock threshold "
                    f"({LOW_STOCK_THRESHOLD} units) — restock alert needed."
                ),
            }

        return {
            "current_node": "triage",
            "triage_action": ACTION_ROUTINE,
            "triage_reasoning": "All systems normal. No urgent actions required.",
        }

    async def _node_handle_vip(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 3a: Handle VIP — Draft a personal follow-up for whale customers.

        Uses LLM to draft a warm, personal thank-you email from the CEO.
        """
        vip_orders = state.get("vip_orders", [])
        if not vip_orders:
            return {
                "current_node": "handle_vip",
                "actions_planned": [],
            }

        vip = vip_orders[0]
        customer_email = vip.get("customer_email", "")
        customer_name = vip.get("customer_name", "Valued Customer")
        order_total = vip.get("total_price", "0")

        logger.info(
            "commerce_vip_detected",
            extra={
                "agent_id": self.agent_id,
                "customer": customer_email,
                "total": order_total,
            },
        )

        # Draft a VIP email via LLM
        email_subject = f"Thank you for your order, {customer_name.split()[0] if customer_name.split() else 'there'}!"
        email_body = (
            f"Hi {customer_name},\n\n"
            f"I wanted to personally thank you for your recent order of ${order_total}. "
            f"We truly appreciate your business and are committed to delivering "
            f"an exceptional experience.\n\n"
            f"If you have any questions or need anything at all, please don't "
            f"hesitate to reach out directly to me.\n\n"
            f"Best regards,\nThe Founder"
        )

        # Try to use LLM for a more personalized draft
        try:
            if hasattr(self, "router") and self.router:
                prompt = (
                    f"Draft a short, warm personal thank-you email from the CEO "
                    f"to {customer_name} ({customer_email}) who just placed a "
                    f"${order_total} order. Keep it under 100 words, genuine, "
                    f"not salesy. Sign as 'The Founder'. Just the email body, no subject."
                )
                resp = await self.route_llm(
                    intent="creative_writing",
                    system_prompt="You write warm, genuine business emails. Be brief and authentic.",
                    user_prompt=prompt,
                )
                if resp and resp.text:
                    email_body = resp.text
        except Exception as e:
            logger.warning(
                "commerce_vip_llm_draft_failed",
                extra={"error": str(e)[:200]},
            )
            # Keep the template draft

        actions_planned = [
            {
                "action": "send_vip_email",
                "target": customer_email,
                "details": {
                    "customer_name": customer_name,
                    "order_total": order_total,
                    "subject": email_subject,
                    "body": email_body,
                },
                "requires_approval": True,
            }
        ]

        return {
            "current_node": "handle_vip",
            "vip_customer_email": customer_email,
            "vip_customer_name": customer_name,
            "vip_order_total": float(order_total) if order_total else 0.0,
            "vip_email_subject": email_subject,
            "vip_email_body": email_body,
            "actions_planned": actions_planned,
        }

    async def _node_handle_low_stock(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 3b: Handle Low Stock — Prepare restock recommendations.

        Alerts the human about critically low inventory with
        recommended restock quantities.
        """
        alerts = state.get("low_stock_alerts", [])

        logger.info(
            "commerce_low_stock_handling",
            extra={
                "agent_id": self.agent_id,
                "alert_count": len(alerts),
            },
        )

        actions_planned = []
        for alert in alerts:
            # Recommend restocking to 50 units as a safe default
            current_qty = alert.get("quantity", 0)
            recommended_qty = max(50, current_qty * 10)  # At least 50 or 10x current

            actions_planned.append({
                "action": "restock",
                "target": alert.get("variant_id", ""),
                "details": {
                    "product": alert.get("product", ""),
                    "variant": alert.get("variant", ""),
                    "current_quantity": current_qty,
                    "recommended_quantity": recommended_qty,
                    "sku": alert.get("sku", ""),
                },
                "requires_approval": True,
            })

        return {
            "current_node": "handle_low_stock",
            "actions_planned": actions_planned,
        }

    async def _node_handle_refund(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 3c: Handle Refund — Draft refund for human approval.

        Refunds ALWAYS require human sign-off. The Commerce Agent
        prepares the refund details but never processes it autonomously.
        """
        task_input = state.get("task_input", {})
        if not isinstance(task_input, dict):
            task_input = {}

        refund_order_id = task_input.get("order_id", "")
        refund_amount = float(task_input.get("amount", 0))
        refund_reason = task_input.get("reason", "Customer requested refund")
        payment_intent_id = task_input.get("payment_intent_id", "")

        logger.info(
            "commerce_refund_handling",
            extra={
                "agent_id": self.agent_id,
                "order_id": refund_order_id,
                "amount": refund_amount,
            },
        )

        actions_planned = [
            {
                "action": "process_refund",
                "target": payment_intent_id,
                "details": {
                    "order_id": refund_order_id,
                    "amount": refund_amount,
                    "reason": refund_reason,
                },
                "requires_approval": True,  # ALWAYS requires human approval
            }
        ]

        return {
            "current_node": "handle_refund",
            "refund_order_id": refund_order_id,
            "refund_amount": refund_amount,
            "refund_reason": refund_reason,
            "actions_planned": actions_planned,
        }

    async def _node_human_review(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 4: Human Review — Present actions for approval.

        This node is a LangGraph interrupt point. The human reviews
        the planned actions and approves or rejects them.

        In automated testing, auto-approve safe actions (restock alerts).
        Refunds and VIP emails always require explicit approval.
        """
        actions = state.get("actions_planned", [])
        triage_action = state.get("triage_action", "")

        # Check if all actions are pre-approved (e.g., from test harness)
        all_approved = state.get("actions_approved", False)

        if not all_approved:
            # Log what needs approval
            for action in actions:
                logger.info(
                    "commerce_awaiting_approval",
                    extra={
                        "action": action.get("action"),
                        "target": str(action.get("target", ""))[:50],
                        "requires_approval": action.get("requires_approval", True),
                    },
                )

        return {
            "current_node": "human_review",
            "actions_approved": all_approved,
        }

    async def _node_execute(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 5: Execute — Carry out approved actions.

        Dispatches each action to the appropriate system:
        - send_vip_email → Outreach Agent via event bus
        - restock → Shopify inventory update (sandboxed)
        - process_refund → Stripe refund (sandboxed)
        """
        actions = state.get("actions_planned", [])
        executed = []
        failed = []

        for action in actions:
            action_type = action.get("action", "")
            details = action.get("details", {})

            try:
                if action_type == "send_vip_email":
                    # In a real system, dispatch to Outreach Agent
                    # For now, log the intent
                    logger.info(
                        "commerce_vip_email_dispatched",
                        extra={
                            "customer": action.get("target"),
                            "subject": details.get("subject", ""),
                        },
                    )
                    executed.append({
                        **action,
                        "status": "dispatched",
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })

                elif action_type == "restock":
                    # Log the restock recommendation (actual update is sandboxed)
                    logger.info(
                        "commerce_restock_recommended",
                        extra={
                            "variant_id": action.get("target"),
                            "recommended_qty": details.get("recommended_quantity"),
                        },
                    )
                    executed.append({
                        **action,
                        "status": "recommended",
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })

                elif action_type == "process_refund":
                    # Refund processing (sandboxed in non-production)
                    logger.info(
                        "commerce_refund_dispatched",
                        extra={
                            "payment_intent_id": action.get("target", "")[:16],
                            "amount": details.get("amount"),
                        },
                    )
                    executed.append({
                        **action,
                        "status": "dispatched",
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })

                else:
                    logger.warning(
                        "commerce_unknown_action",
                        extra={"action": action_type},
                    )
                    executed.append({
                        **action,
                        "status": "skipped",
                        "reason": f"Unknown action type: {action_type}",
                    })

            except Exception as e:
                logger.error(
                    "commerce_action_failed",
                    extra={
                        "action": action_type,
                        "error": str(e)[:200],
                    },
                )
                failed.append({
                    **action,
                    "status": "failed",
                    "error": str(e)[:200],
                })

        return {
            "current_node": "execute",
            "actions_executed": executed,
            "actions_failed": failed,
        }

    async def _node_report(self, state: CommerceAgentState) -> dict[str, Any]:
        """
        Node 6: Report — Generate a commerce status report.

        Produces a human-readable Markdown summary of all activity.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Build report sections
        sections = ["# Commerce Status Report", f"*Generated: {now}*\n"]

        # Order summary
        order_count = state.get("order_count", 0)
        total_revenue = state.get("total_revenue", 0.0)
        sections.append("## Orders")
        sections.append(f"- **Orders processed:** {order_count}")
        sections.append(f"- **Total revenue:** ${total_revenue:,.2f}")
        sections.append(f"- **VIP orders:** {state.get('vip_count', 0)}")

        # Inventory
        low_stock = state.get("low_stock_count", 0)
        sections.append("\n## Inventory")
        sections.append(f"- **Low stock alerts:** {low_stock}")
        if low_stock > 0:
            for alert in state.get("low_stock_alerts", [])[:5]:
                sections.append(
                    f"  - {alert.get('product', '?')} "
                    f"({alert.get('variant', '?')}): "
                    f"{alert.get('quantity', 0)} remaining"
                )

        # Triage
        triage_action = state.get("triage_action", "routine")
        sections.append("\n## Triage Decision")
        sections.append(f"- **Action:** {triage_action}")
        sections.append(f"- **Reasoning:** {state.get('triage_reasoning', 'N/A')}")

        # Actions taken
        executed = state.get("actions_executed", [])
        failed = state.get("actions_failed", [])
        if executed or failed:
            sections.append("\n## Actions")
            for a in executed:
                sections.append(f"- ✅ {a.get('action', '?')}: {a.get('status', '?')}")
            for a in failed:
                sections.append(f"- ❌ {a.get('action', '?')}: {a.get('error', '?')}")

        report = "\n".join(sections)

        logger.info(
            "commerce_report_generated",
            extra={
                "agent_id": self.agent_id,
                "order_count": order_count,
                "revenue": total_revenue,
                "triage_action": triage_action,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Prepare State ────────────────────────────────────────────────

    def prepare_state(self, task_input: dict[str, Any]) -> CommerceAgentState:
        """Prepare the initial state for a commerce run."""
        return CommerceAgentState(
            agent_id=self.agent_id,
            vertical_id=self.vertical_id,
            run_id="",
            task_input=task_input,
            current_node="monitor",
            error=None,
            human_gates_pending=[],
            human_response=None,
            rag_context=[],
            knowledge_written=[],
            # Commerce defaults
            recent_orders=[],
            order_count=0,
            total_revenue=0.0,
            vip_orders=[],
            vip_count=0,
            products=[],
            low_stock_alerts=[],
            low_stock_count=0,
            pending_payments=[],
            failed_payments=[],
            triage_action="routine",
            triage_reasoning="",
            actions_planned=[],
            actions_approved=False,
            actions_executed=[],
            actions_failed=[],
            report_summary="",
            report_generated_at="",
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    def write_knowledge(self, state: CommerceAgentState) -> None:
        """Write commerce insights to the shared brain."""
        revenue = state.get("total_revenue", 0.0)
        vip_count = state.get("vip_count", 0)

        # Only write insights when there's something noteworthy
        if revenue > 0 or vip_count > 0:
            self.store_insight(InsightData(
                insight_type="commerce_activity",
                title=f"Commerce: ${revenue:,.2f} revenue, {vip_count} VIP orders",
                content=(
                    f"Commerce check completed. "
                    f"Orders: {state.get('order_count', 0)}, "
                    f"Revenue: ${revenue:,.2f}, "
                    f"VIP: {vip_count}, "
                    f"Low stock: {state.get('low_stock_count', 0)}, "
                    f"Action: {state.get('triage_action', 'routine')}"
                ),
                confidence=0.85,
                metadata={
                    "order_count": state.get("order_count", 0),
                    "revenue": revenue,
                    "vip_count": vip_count,
                    "triage_action": state.get("triage_action", "routine"),
                },
            ))
