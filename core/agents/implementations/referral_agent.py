"""
Referral Agent — The Network Builder.

Identifies happy clients with high NPS scores, generates personalized
referral outreach messages, and tracks the referral pipeline from
initial ask through conversion. Works across all verticals.

Architecture (LangGraph State Machine):
    identify_candidates → generate_referral_asks → human_review →
    execute_referrals → report → END

Trigger Events:
    - scheduled: Monthly referral campaign sweep
    - nps_response: New NPS survey response with score >= 9
    - manual: On-demand referral campaign

Shared Brain Integration:
    - Reads: client satisfaction data, engagement history, project outcomes
    - Writes: referral success patterns, optimal ask timing insights

Safety:
    - NEVER contacts clients without human approval
    - Referral asks require human review before sending
    - Commission tracking is advisory; finance team validates
    - Client privacy preserved — no PII in logs or insights

Usage:
    agent = ReferralAgent(config, db, embedder, llm)
    result = await agent.run({
        "min_nps": 9,
        "campaign_name": "Q1 Referral Drive",
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
from core.agents.state import ReferralAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

MIN_NPS_FOR_REFERRAL = 9

DEFAULT_COMMISSION_PERCENT = 10.0

REFERRAL_STATUSES = [
    "submitted",
    "contacted",
    "qualified",
    "converted",
    "lost",
    "expired",
]

REFERRAL_ASK_PROMPT = """\
You are a B2B referral specialist. Your job is to craft personalized, \
warm referral outreach messages for happy clients.

Client Information:
{client_json}

Company Context:
- Vertical: {vertical_id}
- Services provided: {services}
- Average relationship: {relationship_months} months

Guidelines:
- Be warm and genuine, not pushy or transactional
- Reference the specific value they received
- Make the ask easy and low-friction
- Mention the referral incentive naturally
- Keep the message concise (under 200 words)
- Commission: {commission_percent}% for successful referrals

For each client, return a JSON array:
[
    {{
        "client_id": "id",
        "client_name": "name",
        "email": "email",
        "ask_message": "The personalized referral outreach message",
        "reasoning": "Why this client is a good referral candidate",
        "ask_type": "email|call|meeting",
        "confidence": 0.0-1.0
    }}
]

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("referral")
class ReferralAgent(BaseAgent):
    """
    Referral pipeline management agent.

    Nodes:
        1. identify_candidates     -- Find happy clients (NPS 9+, multiple projects)
        2. generate_referral_asks  -- LLM drafts personalized referral outreach
        3. human_review            -- Gate: approve referral messages
        4. execute_referrals       -- Save to referrals table, track pipeline
        5. report                  -- Summary + InsightData on referral patterns
    """

    def build_graph(self) -> Any:
        """Build the Referral Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ReferralAgentState)

        workflow.add_node("identify_candidates", self._node_identify_candidates)
        workflow.add_node("generate_referral_asks", self._node_generate_referral_asks)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute_referrals", self._node_execute_referrals)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("identify_candidates")

        workflow.add_edge("identify_candidates", "generate_referral_asks")
        workflow.add_edge("generate_referral_asks", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "execute_referrals",
                "rejected": "report",
            },
        )
        workflow.add_edge("execute_referrals", "report")
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
    def get_state_class(cls) -> Type[ReferralAgentState]:
        return ReferralAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "happy_clients": [],
            "referral_candidates": [],
            "referral_requests": [],
            "requests_approved": False,
            "active_referrals": [],
            "total_referrals": 0,
            "converted_referrals": 0,
            "total_commission_cents": 0,
            "requests_sent": 0,
            "referrals_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Identify Candidates ─────────────────────────────────

    async def _node_identify_candidates(
        self, state: ReferralAgentState
    ) -> dict[str, Any]:
        """Node 1: Find happy clients with NPS 9+ and strong engagement."""
        task = state.get("task_input", {})
        min_nps = task.get("min_nps", MIN_NPS_FOR_REFERRAL)

        logger.info(
            "referral_identify_candidates",
            extra={
                "agent_id": self.agent_id,
                "min_nps": min_nps,
            },
        )

        happy_clients: list[dict[str, Any]] = []
        referral_candidates: list[dict[str, Any]] = []

        # Query feedback_responses for high NPS scores
        try:
            result = (
                self.db.client.table("feedback_responses")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .gte("nps_score", min_nps)
                .order("nps_score", desc=True)
                .limit(100)
                .execute()
            )

            nps_map: dict[str, dict[str, Any]] = {}
            if result.data:
                for response in result.data:
                    client_id = response.get("client_id", "")
                    if client_id and client_id not in nps_map:
                        nps_map[client_id] = {
                            "client_id": client_id,
                            "nps_score": response.get("nps_score", 0),
                            "feedback_comment": response.get("comment", ""),
                            "response_date": response.get("created_at", ""),
                        }

            logger.info(
                "referral_nps_clients_found",
                extra={"count": len(nps_map)},
            )

            # Enrich with company data
            if nps_map:
                client_ids = list(nps_map.keys())
                try:
                    companies_result = (
                        self.db.client.table("companies")
                        .select("*")
                        .eq("vertical_id", self.vertical_id)
                        .in_("id", client_ids[:50])
                        .execute()
                    )

                    if companies_result.data:
                        for company in companies_result.data:
                            cid = company.get("id", "")
                            if cid in nps_map:
                                nps_map[cid]["company_name"] = company.get("name", "")
                                nps_map[cid]["company_domain"] = company.get("domain", "")
                                nps_map[cid]["contact_email"] = company.get("primary_email", "")
                                nps_map[cid]["contact_name"] = company.get("primary_contact", "")
                                nps_map[cid]["industry"] = company.get("industry", "")
                except Exception as e:
                    logger.warning(
                        "referral_company_enrichment_error",
                        extra={"error": str(e)[:200]},
                    )

                happy_clients = list(nps_map.values())

        except Exception as e:
            logger.warning(
                "referral_nps_query_error",
                extra={"error": str(e)[:200]},
            )

        # Add any task-provided candidates
        task_candidates = task.get("candidates", [])
        if task_candidates:
            happy_clients.extend(task_candidates)

        # Filter out clients who already have active referrals
        try:
            active_result = (
                self.db.client.table("referrals")
                .select("referrer_client_id")
                .eq("vertical_id", self.vertical_id)
                .in_("status", ["submitted", "contacted", "qualified"])
                .execute()
            )
            if active_result.data:
                active_ids = {r.get("referrer_client_id") for r in active_result.data}
                before_count = len(happy_clients)
                referral_candidates = [
                    c for c in happy_clients
                    if c.get("client_id") not in active_ids
                ]
                filtered_out = before_count - len(referral_candidates)
                if filtered_out > 0:
                    logger.info(
                        "referral_filtered_active_referrals",
                        extra={"filtered_count": filtered_out},
                    )
            else:
                referral_candidates = happy_clients
        except Exception as e:
            logger.debug(f"referral_active_check_error: {e}")
            referral_candidates = happy_clients

        logger.info(
            "referral_candidates_identified",
            extra={
                "happy_clients": len(happy_clients),
                "referral_candidates": len(referral_candidates),
            },
        )

        return {
            "current_node": "identify_candidates",
            "happy_clients": happy_clients,
            "referral_candidates": referral_candidates,
        }

    # ─── Node 2: Generate Referral Asks ──────────────────────────────

    async def _node_generate_referral_asks(
        self, state: ReferralAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM drafts personalized referral outreach messages."""
        candidates = state.get("referral_candidates", [])
        commission = self.config.params.get(
            "commission_percent", DEFAULT_COMMISSION_PERCENT
        )

        logger.info(
            "referral_generate_asks",
            extra={"candidate_count": len(candidates)},
        )

        referral_requests: list[dict[str, Any]] = []

        if not candidates:
            logger.info("referral_no_candidates_for_asks")
            return {
                "current_node": "generate_referral_asks",
                "referral_requests": [],
            }

        # Process in batches of 10 to keep LLM context manageable
        batch_size = 10
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            try:
                # Calculate approximate relationship duration for context
                for client in batch:
                    response_date = client.get("response_date", "")
                    if response_date:
                        try:
                            response_dt = datetime.fromisoformat(
                                response_date.replace("Z", "+00:00")
                            )
                            months = max(
                                1,
                                (datetime.now(timezone.utc) - response_dt).days // 30,
                            )
                            client["relationship_months"] = months
                        except (ValueError, TypeError):
                            client["relationship_months"] = 6
                    else:
                        client["relationship_months"] = 6

                services = self.config.params.get(
                    "services_description", "professional services"
                )
                avg_months = sum(
                    c.get("relationship_months", 6) for c in batch
                ) // max(len(batch), 1)

                prompt = REFERRAL_ASK_PROMPT.format(
                    client_json=json.dumps(batch, indent=2, default=str),
                    vertical_id=self.vertical_id,
                    services=services,
                    relationship_months=avg_months,
                    commission_percent=commission,
                )

                llm_response = self.llm.messages.create(
                    model="claude-haiku-4-5-20250514",
                    system="You are a B2B referral outreach specialist.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )

                llm_text = llm_response.content[0].text.strip()

                try:
                    asks_data = json.loads(llm_text)
                    if isinstance(asks_data, list):
                        referral_requests.extend(asks_data)
                except (json.JSONDecodeError, KeyError):
                    logger.debug(
                        "referral_asks_parse_error",
                        extra={"batch_start": batch_start},
                    )
                    # Fallback: create basic asks from candidates
                    for client in batch:
                        referral_requests.append({
                            "client_id": client.get("client_id", ""),
                            "client_name": client.get("company_name", "Valued Client"),
                            "email": client.get("contact_email", ""),
                            "ask_message": (
                                f"Hi {client.get('contact_name', 'there')}, "
                                f"we have truly enjoyed working with "
                                f"{client.get('company_name', 'your team')}. "
                                f"If you know anyone who could benefit from our "
                                f"services, we offer a {commission}% referral commission."
                            ),
                            "reasoning": f"NPS score: {client.get('nps_score', 'N/A')}",
                            "ask_type": "email",
                            "confidence": 0.6,
                        })

            except Exception as e:
                logger.warning(
                    "referral_asks_llm_error",
                    extra={
                        "error": str(e)[:200],
                        "batch_start": batch_start,
                    },
                )

        logger.info(
            "referral_asks_generated",
            extra={"ask_count": len(referral_requests)},
        )

        return {
            "current_node": "generate_referral_asks",
            "referral_requests": referral_requests,
        }

    # ─── Node 3: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ReferralAgentState
    ) -> dict[str, Any]:
        """Node 3: Present referral asks for human approval."""
        requests = state.get("referral_requests", [])
        candidates = state.get("referral_candidates", [])

        logger.info(
            "referral_human_review_pending",
            extra={
                "request_count": len(requests),
                "candidate_count": len(candidates),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 4: Execute Referrals ───────────────────────────────────

    async def _node_execute_referrals(
        self, state: ReferralAgentState
    ) -> dict[str, Any]:
        """Node 4: Save referral records to DB and track pipeline."""
        requests = state.get("referral_requests", [])
        now = datetime.now(timezone.utc).isoformat()

        logger.info(
            "referral_execute",
            extra={"request_count": len(requests)},
        )

        active_referrals: list[dict[str, Any]] = []
        requests_sent = 0
        referrals_saved = False

        for req in requests:
            try:
                referral_record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "referrer_client_id": req.get("client_id", ""),
                    "referrer_name": req.get("client_name", ""),
                    "referrer_email": req.get("email", ""),
                    "ask_message": req.get("ask_message", ""),
                    "ask_type": req.get("ask_type", "email"),
                    "commission_percent": self.config.params.get(
                        "commission_percent", DEFAULT_COMMISSION_PERCENT
                    ),
                    "status": "submitted",
                    "confidence": req.get("confidence", 0.5),
                    "created_at": now,
                }

                result = (
                    self.db.client.table("referrals")
                    .insert(referral_record)
                    .execute()
                )

                if result.data and len(result.data) > 0:
                    created = result.data[0]
                    active_referrals.append(created)
                    requests_sent += 1
                    referrals_saved = True
                    logger.debug(
                        "referral_record_created",
                        extra={
                            "referrer": req.get("client_name", ""),
                            "referral_id": created.get("id", ""),
                        },
                    )

            except Exception as e:
                logger.warning(
                    "referral_save_error",
                    extra={
                        "client_id": req.get("client_id", ""),
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "referral_execution_complete",
            extra={
                "referrals_created": len(active_referrals),
                "requests_sent": requests_sent,
            },
        )

        return {
            "current_node": "execute_referrals",
            "active_referrals": active_referrals,
            "total_referrals": len(active_referrals),
            "requests_sent": requests_sent,
            "requests_approved": True,
            "referrals_saved": referrals_saved,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: ReferralAgentState
    ) -> dict[str, Any]:
        """Node 5: Summary + InsightData on referral success patterns."""
        now = datetime.now(timezone.utc).isoformat()
        happy_clients = state.get("happy_clients", [])
        candidates = state.get("referral_candidates", [])
        requests = state.get("referral_requests", [])
        active = state.get("active_referrals", [])
        requests_sent = state.get("requests_sent", 0)
        approved = state.get("human_approval_status") == "approved"

        # Build report
        sections = [
            "# Referral Campaign Report",
            f"*Generated: {now}*\n",
            f"## Summary",
            f"- **Happy Clients (NPS 9+):** {len(happy_clients)}",
            f"- **Eligible Candidates:** {len(candidates)}",
            f"- **Referral Asks Drafted:** {len(requests)}",
            f"- **Referrals Created:** {len(active)}",
            f"- **Requests Sent:** {requests_sent}",
            f"- **Status:** {'Approved & Executed' if approved else 'Pending/Rejected'}",
        ]

        if happy_clients:
            sections.append("\n## Top Happy Clients")
            for i, c in enumerate(happy_clients[:10], 1):
                sections.append(
                    f"{i}. **{c.get('company_name', 'Unknown')}** "
                    f"(NPS: {c.get('nps_score', 'N/A')}) — "
                    f"{c.get('contact_name', 'N/A')}"
                )

        if requests:
            sections.append("\n## Referral Asks")
            for i, r in enumerate(requests[:5], 1):
                sections.append(
                    f"{i}. **{r.get('client_name', 'Unknown')}**: "
                    f"{r.get('reasoning', 'N/A')[:80]}"
                )

        report = "\n".join(sections)

        # Store insight on referral patterns
        if happy_clients:
            avg_nps = sum(
                c.get("nps_score", 0) for c in happy_clients
            ) / max(len(happy_clients), 1)

            self.store_insight(InsightData(
                insight_type="referral_pattern",
                title=f"Referral Campaign: {len(candidates)} candidates, {requests_sent} sent",
                content=(
                    f"Identified {len(happy_clients)} happy clients with "
                    f"avg NPS {avg_nps:.1f}. Generated {len(requests)} "
                    f"personalized referral asks. {requests_sent} referrals "
                    f"created in pipeline."
                ),
                confidence=0.75,
                metadata={
                    "happy_client_count": len(happy_clients),
                    "avg_nps": round(avg_nps, 1),
                    "asks_generated": len(requests),
                    "referrals_sent": requests_sent,
                },
            ))

        logger.info(
            "referral_report_generated",
            extra={
                "happy_clients": len(happy_clients),
                "asks": len(requests),
                "sent": requests_sent,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ReferralAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ReferralAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
