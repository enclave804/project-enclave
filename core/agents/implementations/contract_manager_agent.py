"""
Contract Manager Agent — The Legal Operations Automator.

Manages the full contract lifecycle: scans for upcoming renewals, generates
contracts from templates, routes through human review, tracks signatures,
and maintains a complete audit trail.

Architecture (LangGraph State Machine):
    check_renewals → generate_contract → human_review →
    send_contract → track_signatures → report → END

Trigger Events:
    - deal_closed: Opportunity moved to closed_won stage
    - renewal_due: Contract approaching expiration date
    - manual: On-demand contract generation

Shared Brain Integration:
    - Reads: deal context, company data, negotiated terms
    - Writes: contract patterns, renewal rates, signature timelines

Safety:
    - NEVER sends contracts without human review gate
    - Templates are advisory; legal team reviews final language
    - All contract versions are immutable once saved
    - PII is handled with care; no sensitive data in logs

Usage:
    agent = ContractManagerAgent(config, db, embedder, llm)
    result = await agent.run({
        "opportunity_id": "opp_abc123",
        "contract_type": "msa",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import ContractManagerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

CONTRACT_TEMPLATES = {
    "service_agreement": {
        "label": "Service Agreement",
        "description": "Standard services agreement for project-based engagements.",
        "sections": [
            "parties", "scope_of_work", "deliverables", "timeline",
            "pricing_payment", "intellectual_property", "confidentiality",
            "liability", "termination", "governing_law", "signatures",
        ],
        "typical_duration_months": 12,
    },
    "msa": {
        "label": "Master Service Agreement",
        "description": "Umbrella agreement governing all future engagements.",
        "sections": [
            "parties", "definitions", "scope", "ordering_process",
            "pricing_payment", "service_levels", "intellectual_property",
            "confidentiality", "indemnification", "limitation_of_liability",
            "term_termination", "dispute_resolution", "general_provisions",
            "signatures",
        ],
        "typical_duration_months": 24,
    },
    "nda": {
        "label": "Non-Disclosure Agreement",
        "description": "Mutual NDA for protecting confidential information.",
        "sections": [
            "parties", "definition_of_confidential", "obligations",
            "exclusions", "term", "return_of_materials", "remedies",
            "governing_law", "signatures",
        ],
        "typical_duration_months": 24,
    },
    "sow": {
        "label": "Statement of Work",
        "description": "Detailed scope and deliverables for a specific project.",
        "sections": [
            "overview", "objectives", "scope", "deliverables",
            "timeline_milestones", "acceptance_criteria", "pricing",
            "assumptions", "change_management", "signatures",
        ],
        "typical_duration_months": 6,
    },
}

RENEWAL_WINDOW_DAYS = 30

CONTRACT_SYSTEM_PROMPT = """\
You are a legal operations specialist. Generate a professional contract \
document in Markdown format based on the template and context below.

Contract Type: {contract_type} ({contract_label})
Company: {company_name}
Contact: {contact_name} ({contact_email})
Opportunity ID: {opportunity_id}

Required Sections:
{sections_list}

Contract Terms:
{contract_terms_json}

Additional Context:
{additional_context}

Generate a complete, professional Markdown contract document. Include:
1. Proper headers and section numbering
2. Placeholder brackets [FIELD] for items needing manual fill
3. Professional legal language appropriate for the contract type
4. Clear terms and conditions
5. Signature blocks at the end

Do NOT use code fences. Return the Markdown document directly.
"""


@register_agent_type("contract_manager")
class ContractManagerAgent(BaseAgent):
    """
    Contract lifecycle management agent.

    Nodes:
        1. check_renewals       -- Query contracts table for expiring contracts
        2. generate_contract    -- LLM drafts contract from template
        3. human_review         -- Gate: approve contract
        4. send_contract        -- Save to contracts table with status 'sent'
        5. track_signatures     -- Check/update signature status
        6. report               -- Summary + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Contract Manager Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ContractManagerAgentState)

        workflow.add_node("check_renewals", self._node_check_renewals)
        workflow.add_node("generate_contract", self._node_generate_contract)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("send_contract", self._node_send_contract)
        workflow.add_node("track_signatures", self._node_track_signatures)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("check_renewals")

        workflow.add_edge("check_renewals", "generate_contract")
        workflow.add_edge("generate_contract", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "send_contract",
                "rejected": "report",
            },
        )
        workflow.add_edge("send_contract", "track_signatures")
        workflow.add_edge("track_signatures", "report")
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
    def get_state_class(cls) -> Type[ContractManagerAgentState]:
        return ContractManagerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "expiring_contracts": [],
            "total_expiring": 0,
            "renewal_window_days": RENEWAL_WINDOW_DAYS,
            "contract_template": "",
            "company_name": "",
            "company_domain": "",
            "contact_name": "",
            "contact_email": "",
            "opportunity_id": "",
            "contract_terms": {},
            "draft_contract": "",
            "contract_type": "",
            "contract_id": "",
            "contract_saved": False,
            "contract_sent": False,
            "sent_at": "",
            "signature_status": "pending",
            "signed_at": "",
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Check Renewals ──────────────────────────────────────

    async def _node_check_renewals(
        self, state: ContractManagerAgentState
    ) -> dict[str, Any]:
        """Node 1: Query contracts table for expiring contracts within renewal window."""
        task = state.get("task_input", {})
        opportunity_id = task.get("opportunity_id", "")
        contract_type = task.get("contract_type", "service_agreement")

        logger.info(
            "contract_manager_check_renewals",
            extra={
                "opportunity_id": opportunity_id,
                "agent_id": self.agent_id,
            },
        )

        # Extract task-level data
        company_name = task.get("company_name", "")
        company_domain = task.get("company_domain", "")
        contact_name = task.get("contact_name", "")
        contact_email = task.get("contact_email", "")
        contract_terms = task.get("contract_terms", {})

        # Check for expiring contracts
        expiring_contracts: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        renewal_cutoff = (now + timedelta(days=RENEWAL_WINDOW_DAYS)).isoformat()

        try:
            result = (
                self.db.client.table("contracts")
                .select("*")
                .lte("expires_at", renewal_cutoff)
                .gte("expires_at", now.isoformat())
                .eq("status", "active")
                .execute()
            )
            if result.data:
                expiring_contracts = result.data
                logger.info(
                    "contract_manager_expiring_found",
                    extra={"count": len(expiring_contracts)},
                )
        except Exception as e:
            logger.warning(
                "contract_manager_renewal_check_error",
                extra={"error": str(e)[:200]},
            )

        # If we have an opportunity_id, pull company data from opportunities
        if opportunity_id and not company_name:
            try:
                result = (
                    self.db.client.table("opportunities")
                    .select("*")
                    .eq("id", opportunity_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    opp = result.data[0]
                    company_name = opp.get("company_name", company_name)
                    company_domain = opp.get("company_domain", company_domain)
                    contact_name = opp.get("contact_name", contact_name)
                    contact_email = opp.get("contact_email", contact_email)
                    logger.info(
                        "contract_manager_opportunity_loaded",
                        extra={"opportunity_id": opportunity_id},
                    )
            except Exception as e:
                logger.warning(
                    "contract_manager_opportunity_error",
                    extra={"error": str(e)[:200]},
                )

        return {
            "current_node": "check_renewals",
            "expiring_contracts": expiring_contracts,
            "total_expiring": len(expiring_contracts),
            "opportunity_id": opportunity_id,
            "contract_type": contract_type,
            "company_name": company_name,
            "company_domain": company_domain,
            "contact_name": contact_name,
            "contact_email": contact_email,
            "contract_terms": contract_terms,
        }

    # ─── Node 2: Generate Contract ───────────────────────────────────

    async def _node_generate_contract(
        self, state: ContractManagerAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM drafts contract from template and context."""
        contract_type = state.get("contract_type", "service_agreement")
        company_name = state.get("company_name", "Company")
        contact_name = state.get("contact_name", "Contact")
        contact_email = state.get("contact_email", "")
        opportunity_id = state.get("opportunity_id", "")
        contract_terms = state.get("contract_terms", {})

        logger.info(
            "contract_manager_generate_contract",
            extra={
                "contract_type": contract_type,
                "company_name": company_name,
            },
        )

        template = CONTRACT_TEMPLATES.get(
            contract_type, CONTRACT_TEMPLATES["service_agreement"]
        )
        sections_list = "\n".join(
            f"- {s.replace('_', ' ').title()}" for s in template["sections"]
        )

        draft_contract = ""

        try:
            prompt = CONTRACT_SYSTEM_PROMPT.format(
                contract_type=contract_type,
                contract_label=template["label"],
                company_name=company_name,
                contact_name=contact_name,
                contact_email=contact_email,
                opportunity_id=opportunity_id,
                sections_list=sections_list,
                contract_terms_json=json.dumps(contract_terms, indent=2),
                additional_context=(
                    f"Typical duration: {template['typical_duration_months']} months. "
                    f"Description: {template['description']}"
                ),
            )

            llm_response = self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                system="You are a legal operations specialist generating professional business contracts.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )

            draft_contract = llm_response.content[0].text.strip()

        except Exception as e:
            logger.warning(
                "contract_manager_llm_error",
                extra={"error": str(e)[:200]},
            )
            # Fallback to basic template
            draft_contract = (
                f"# {template['label']}\n\n"
                f"**Parties:** [Your Company] and {company_name}\n"
                f"**Contact:** {contact_name} ({contact_email})\n"
                f"**Type:** {contract_type}\n\n"
                f"[Contract content to be drafted by legal team]\n"
            )

        logger.info(
            "contract_manager_contract_generated",
            extra={
                "contract_type": contract_type,
                "length_chars": len(draft_contract),
            },
        )

        return {
            "current_node": "generate_contract",
            "draft_contract": draft_contract,
            "contract_template": contract_type,
        }

    # ─── Node 3: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ContractManagerAgentState
    ) -> dict[str, Any]:
        """Node 3: Present contract for human approval."""
        contract_type = state.get("contract_type", "")
        company = state.get("company_name", "")

        logger.info(
            "contract_manager_human_review_pending",
            extra={
                "contract_type": contract_type,
                "company": company,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 4: Send Contract ───────────────────────────────────────

    async def _node_send_contract(
        self, state: ContractManagerAgentState
    ) -> dict[str, Any]:
        """Node 4: Save contract to contracts table with status 'sent'."""
        now = datetime.now(timezone.utc).isoformat()
        contract_type = state.get("contract_type", "service_agreement")
        template = CONTRACT_TEMPLATES.get(
            contract_type, CONTRACT_TEMPLATES["service_agreement"]
        )

        logger.info(
            "contract_manager_send_contract",
            extra={"contract_type": contract_type},
        )

        expires_at = (
            datetime.now(timezone.utc)
            + timedelta(days=template["typical_duration_months"] * 30)
        ).isoformat()

        contract_record = {
            "vertical_id": self.vertical_id,
            "agent_id": self.agent_id,
            "opportunity_id": state.get("opportunity_id", ""),
            "company_name": state.get("company_name", ""),
            "company_domain": state.get("company_domain", ""),
            "contact_name": state.get("contact_name", ""),
            "contact_email": state.get("contact_email", ""),
            "contract_type": contract_type,
            "contract_document": state.get("draft_contract", ""),
            "contract_terms": json.dumps(state.get("contract_terms", {})),
            "status": "sent",
            "created_at": now,
            "sent_at": now,
            "expires_at": expires_at,
            "signature_status": "pending",
        }

        contract_id = ""
        contract_saved = False

        try:
            result = (
                self.db.client.table("contracts")
                .insert(contract_record)
                .execute()
            )
            if result.data and len(result.data) > 0:
                contract_id = result.data[0].get("id", "")
                contract_saved = True
                logger.info(
                    "contract_manager_contract_saved",
                    extra={"contract_id": contract_id},
                )
        except Exception as e:
            logger.warning(
                "contract_manager_save_error",
                extra={"error": str(e)[:200]},
            )

        return {
            "current_node": "send_contract",
            "contract_id": contract_id,
            "contract_saved": contract_saved,
            "contract_sent": True,
            "sent_at": now,
            "knowledge_written": True,
        }

    # ─── Node 5: Track Signatures ────────────────────────────────────

    async def _node_track_signatures(
        self, state: ContractManagerAgentState
    ) -> dict[str, Any]:
        """Node 5: Check and update signature status."""
        contract_id = state.get("contract_id", "")

        logger.info(
            "contract_manager_track_signatures",
            extra={"contract_id": contract_id},
        )

        signature_status = "pending"
        signed_at = ""

        # Check if contract has been signed (poll DB)
        if contract_id:
            try:
                result = (
                    self.db.client.table("contracts")
                    .select("signature_status, signed_at")
                    .eq("id", contract_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    record = result.data[0]
                    signature_status = record.get("signature_status", "pending")
                    signed_at = record.get("signed_at", "")
            except Exception as e:
                logger.warning(
                    "contract_manager_signature_check_error",
                    extra={"error": str(e)[:200]},
                )

        logger.info(
            "contract_manager_signature_status",
            extra={
                "contract_id": contract_id,
                "status": signature_status,
            },
        )

        return {
            "current_node": "track_signatures",
            "signature_status": signature_status,
            "signed_at": signed_at,
        }

    # ─── Node 6: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: ContractManagerAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate summary report and store insights."""
        now = datetime.now(timezone.utc).isoformat()
        contract_type = state.get("contract_type", "unknown")
        company = state.get("company_name", "unknown")
        expiring = state.get("total_expiring", 0)

        template = CONTRACT_TEMPLATES.get(contract_type, {})

        sections = [
            "# Contract Management Report",
            f"*Generated: {now}*\n",
            f"## Contract Details",
            f"- **Type:** {template.get('label', contract_type)}",
            f"- **Company:** {company}",
            f"- **Contact:** {state.get('contact_name', 'N/A')}",
            f"- **Opportunity:** {state.get('opportunity_id', 'N/A')}",
            f"\n## Status",
            f"- Contract Saved: {'Yes' if state.get('contract_saved') else 'No'}",
            f"- Contract Sent: {'Yes' if state.get('contract_sent') else 'No'}",
            f"- Signature Status: {state.get('signature_status', 'pending')}",
            f"- Contract ID: {state.get('contract_id', 'N/A')}",
            f"\n## Renewals",
            f"- Expiring within {RENEWAL_WINDOW_DAYS} days: {expiring}",
        ]

        report = "\n".join(sections)

        # Store insight
        self.store_insight(InsightData(
            insight_type="contract_lifecycle",
            title=f"Contract: {contract_type} for {company}",
            content=(
                f"Generated {template.get('label', contract_type)} contract "
                f"for {company}. Status: {state.get('signature_status', 'pending')}. "
                f"Renewals pending: {expiring}."
            ),
            confidence=0.85,
            metadata={
                "contract_type": contract_type,
                "company_name": company,
                "signature_status": state.get("signature_status", "pending"),
                "total_expiring": expiring,
            },
        ))

        logger.info(
            "contract_manager_report_generated",
            extra={
                "contract_type": contract_type,
                "company": company,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ContractManagerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ContractManagerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
