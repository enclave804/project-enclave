"""
Proposal Builder Agent — The Deal Closer.

Converts discovery call notes + company enrichment data into
professional proposals/SOWs that close deals. This is the revenue
multiplier — every meeting that gets a proposal is 5x more likely to close.

Architecture (LangGraph State Machine):
    gather_context → research_company → generate_proposal →
    human_review → finalize → deliver → END

Shared Brain: Reads outreach insights (what messaging resonates),
appointment data (what was discussed), and SEO authority topics
to build highly-personalized proposals.

RLHF Hook: If a human edits the proposal before sending,
the (draft, edit) pair is saved for future optimization.

Pricing Tiers (configurable via YAML params):
    starter:        Lightweight assessment, 2-week delivery
    professional:   Full assessment + remediation plan, 4-week delivery
    enterprise:     Multi-phase engagement, 8-12 week delivery
    custom:         LLM generates pricing from meeting context

Usage:
    agent = ProposalBuilderAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_name": "Acme Corp",
        "company_domain": "acme.com",
        "contact_name": "Jane Doe",
        "contact_email": "jane@acme.com",
        "contact_title": "CISO",
        "meeting_notes": "Discussed SOC 2 compliance gaps...",
        "proposal_type": "full_proposal",
        "pricing_tier": "professional",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, ProposalRequest, GeneratedProposal
from core.agents.registry import register_agent_type
from core.agents.state import ProposalBuilderAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PROPOSAL_TYPES = {"sow", "one_pager", "executive_summary", "full_proposal"}
PRICING_TIERS = {"starter", "professional", "enterprise", "custom"}

DEFAULT_PRICING = {
    "starter": {
        "amount": 5000.0,
        "timeline_weeks": 2,
        "deliverables": [
            "Security posture assessment report",
            "Top 10 vulnerability summary",
            "Executive risk briefing",
        ],
    },
    "professional": {
        "amount": 15000.0,
        "timeline_weeks": 4,
        "deliverables": [
            "Comprehensive security assessment",
            "Vulnerability scan & penetration test report",
            "Remediation roadmap with priorities",
            "Compliance gap analysis",
            "30-day follow-up review",
        ],
    },
    "enterprise": {
        "amount": 45000.0,
        "timeline_weeks": 10,
        "deliverables": [
            "Multi-phase security engagement",
            "External & internal penetration testing",
            "Cloud infrastructure security review",
            "Compliance framework alignment (SOC 2, ISO 27001)",
            "Incident response plan development",
            "Security awareness training program",
            "Quarterly review cadence (12 months)",
        ],
    },
    "custom": {
        "amount": 0.0,
        "timeline_weeks": 0,
        "deliverables": [],
    },
}


@register_agent_type("proposal_builder")
class ProposalBuilderAgent(BaseAgent):
    """
    AI-powered proposal builder that converts meetings into revenue.

    Nodes:
        1. gather_context — Pull meeting notes, enrichment, shared brain
        2. research_company — Deep company research via enrichment + RAG
        3. generate_proposal — LLM creates full proposal with sections
        4. human_review — Gate for human approval/editing
        5. finalize — Apply edits, capture RLHF, format final version
        6. deliver — Send proposal via email/link
    """

    def build_graph(self) -> Any:
        """
        Build the Proposal Builder's LangGraph state machine.

        Graph flow:
            gather_context → research_company → generate_proposal →
            human_review → route_after_review → finalize → deliver → END
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(ProposalBuilderAgentState)

        # Add nodes
        workflow.add_node("gather_context", self._node_gather_context)
        workflow.add_node("research_company", self._node_research_company)
        workflow.add_node("generate_proposal", self._node_generate_proposal)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("finalize", self._node_finalize)
        workflow.add_node("deliver", self._node_deliver)

        # Entry point
        workflow.set_entry_point("gather_context")

        # Linear flow with one conditional branch
        workflow.add_edge("gather_context", "research_company")
        workflow.add_edge("research_company", "generate_proposal")
        workflow.add_edge("generate_proposal", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "finalize",
                "rejected": END,
            },
        )
        workflow.add_edge("finalize", "deliver")
        workflow.add_edge("deliver", END)

        # Compile with human gate
        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        """Proposal builder uses enrichment and email MCP tools."""
        return []  # Tools accessed via MCP, not injected

    def get_state_class(self) -> Type[ProposalBuilderAgentState]:
        return ProposalBuilderAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state from proposal request task."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "company_name": task.get("company_name", ""),
            "company_domain": task.get("company_domain", ""),
            "contact_name": task.get("contact_name", ""),
            "contact_email": task.get("contact_email", ""),
            "contact_title": task.get("contact_title", ""),
            "meeting_notes": task.get("meeting_notes", ""),
            "meeting_date": task.get("meeting_date", ""),
            "deal_stage": task.get("deal_stage", "proposal"),
            "company_profile": {},
            "company_tech_stack": {},
            "company_pain_points": [],
            "competitive_landscape": [],
            "proposal_type": task.get("proposal_type", "full_proposal"),
            "proposal_outline": "",
            "proposal_sections": [],
            "draft_proposal": "",
            "pricing_tier": task.get("pricing_tier", "professional"),
            "pricing_amount": 0.0,
            "pricing_breakdown": [],
            "timeline_weeks": 0,
            "deliverables": [],
            "template_id": "",
            "template_variables": {},
            "proposal_approved": False,
            "human_edited_proposal": None,
            "revision_count": 0,
            "delivered": False,
            "delivery_method": task.get("delivery_method", "email"),
            "delivery_url": "",
            "delivered_at": "",
            "rlhf_captured": False,
        })
        return state

    # ─── Node 1: Gather Context ──────────────────────────────────────

    async def _node_gather_context(
        self, state: ProposalBuilderAgentState
    ) -> dict[str, Any]:
        """
        Gather all available context for the proposal.

        Pulls from:
        - Task input (meeting notes, company info)
        - Shared brain (outreach insights, appointment context)
        - Knowledge base (company-specific data)
        """
        company_name = state.get("company_name", "")
        company_domain = state.get("company_domain", "")

        logger.info(
            "proposal_gather_context",
            extra={
                "agent_id": self.agent_id,
                "company": company_name,
                "domain": company_domain,
            },
        )

        # Pull relevant insights from shared brain
        rag_context: list[dict[str, Any]] = []
        try:
            query = f"company {company_name} {company_domain} cybersecurity assessment"
            query_embedding = self.embedder.embed_query(query)
            chunks = self.db.search_knowledge(
                query_embedding=query_embedding,
                limit=10,
            )
            rag_context = chunks if isinstance(chunks, list) else []
        except Exception as e:
            logger.debug(f"Could not search knowledge base: {e}")

        # Pull outreach insights for this company
        outreach_insights: list[dict[str, Any]] = []
        try:
            query = f"outreach {company_name} winning pattern messaging"
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="winning_pattern",
                limit=5,
            )
            outreach_insights = insights if isinstance(insights, list) else []
        except Exception as e:
            logger.debug(f"Could not fetch outreach insights: {e}")

        # Merge insights into RAG context
        rag_context.extend(outreach_insights)

        return {
            "current_node": "gather_context",
            "rag_context": rag_context,
        }

    # ─── Node 2: Research Company ────────────────────────────────────

    async def _node_research_company(
        self, state: ProposalBuilderAgentState
    ) -> dict[str, Any]:
        """
        Deep company research to personalize the proposal.

        Uses Apollo enrichment and tech stack scanning data if available.
        Falls back to meeting notes + RAG context for personalization.
        """
        company_domain = state.get("company_domain", "")
        company_name = state.get("company_name", "")
        meeting_notes = state.get("meeting_notes", "")

        logger.info(
            "proposal_research_company",
            extra={
                "agent_id": self.agent_id,
                "company": company_name,
            },
        )

        company_profile: dict[str, Any] = {}
        tech_stack: dict[str, Any] = {}
        pain_points: list[str] = []

        # Try to enrich company via Apollo
        if company_domain:
            try:
                from core.mcp.tools.apollo_tools import enrich_company
                enrichment_json = await enrich_company(domain=company_domain)
                enrichment = json.loads(enrichment_json)
                company_profile = enrichment.get("company", {})
            except Exception as e:
                logger.debug(f"Apollo enrichment failed: {e}")

        # Extract pain points from meeting notes using LLM
        if meeting_notes:
            try:
                pain_prompt = (
                    "Extract the top 3-5 business pain points or challenges "
                    "mentioned in these meeting notes. Return ONLY a JSON array "
                    "of strings, no code fences.\n\n"
                    f"Meeting notes:\n{meeting_notes[:3000]}"
                )
                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=500,
                    temperature=0.2,
                    messages=[{"role": "user", "content": pain_prompt}],
                )
                text = response.content[0].text.strip() if response.content else "[]"
                # Clean JSON
                if "```" in text:
                    parts = text.split("```")
                    text = parts[1] if len(parts) > 1 else text
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    pain_points = [str(p) for p in parsed[:5]]
            except Exception as e:
                logger.debug(f"Pain point extraction failed: {e}")
                # Fallback: simple keyword extraction
                pain_points = ["Security assessment needed"]

        return {
            "current_node": "research_company",
            "company_profile": company_profile,
            "company_tech_stack": tech_stack,
            "company_pain_points": pain_points,
        }

    # ─── Node 3: Generate Proposal ───────────────────────────────────

    async def _node_generate_proposal(
        self, state: ProposalBuilderAgentState
    ) -> dict[str, Any]:
        """
        Generate a full proposal using LLM with all gathered context.

        Creates a professional, multi-section proposal in Markdown format
        with personalized content based on meeting notes, company research,
        and shared brain insights.
        """
        company_name = state.get("company_name", "Unknown Company")
        contact_name = state.get("contact_name", "")
        contact_title = state.get("contact_title", "")
        meeting_notes = state.get("meeting_notes", "")
        pain_points = state.get("company_pain_points", [])
        company_profile = state.get("company_profile", {})
        rag_context = state.get("rag_context", [])
        proposal_type = state.get("proposal_type", "full_proposal")
        pricing_tier = state.get("pricing_tier", "professional")

        logger.info(
            "proposal_generate_started",
            extra={
                "agent_id": self.agent_id,
                "company": company_name,
                "type": proposal_type,
                "tier": pricing_tier,
            },
        )

        # Get pricing details
        pricing_config = self.config.params.get("pricing", DEFAULT_PRICING)
        tier_info = pricing_config.get(pricing_tier, DEFAULT_PRICING["professional"])
        pricing_amount = tier_info.get("amount", 15000.0)
        timeline_weeks = tier_info.get("timeline_weeks", 4)
        deliverables = tier_info.get("deliverables", [])

        # Build context for LLM
        our_company = self.config.params.get("company_name", "Enclave Guard")
        our_tagline = self.config.params.get(
            "tagline", "Enterprise Cybersecurity Consulting"
        )

        pain_section = ""
        if pain_points:
            pain_section = "\nIDENTIFIED PAIN POINTS:\n"
            for p in pain_points:
                pain_section += f"  - {p}\n"

        profile_section = ""
        if company_profile:
            industry = company_profile.get("industry", "")
            employees = company_profile.get("estimated_num_employees", "")
            profile_section = (
                f"\nCOMPANY PROFILE:\n"
                f"  Industry: {industry}\n"
                f"  Size: {employees} employees\n"
            )

        rag_section = ""
        if rag_context:
            rag_section = "\nRELEVANT INSIGHTS (from our knowledge base):\n"
            for chunk in rag_context[:5]:
                text = chunk.get("content", chunk.get("text", ""))[:300]
                rag_section += f"  - {text}\n"

        deliverables_section = ""
        if deliverables:
            deliverables_section = "\nDELIVERABLES FOR THIS TIER:\n"
            for d in deliverables:
                deliverables_section += f"  - {d}\n"

        type_instructions = {
            "sow": "Generate a formal Statement of Work with scope, timeline, milestones, and payment terms.",
            "one_pager": "Generate a concise 1-page executive summary that sells the engagement.",
            "executive_summary": "Generate a 2-3 page executive summary with key findings and recommendations.",
            "full_proposal": (
                "Generate a comprehensive proposal with these sections: "
                "Executive Summary, Situation Analysis, Proposed Approach, "
                "Scope & Deliverables, Timeline & Milestones, Pricing, "
                "About Our Team, Next Steps."
            ),
        }

        generation_prompt = (
            f"Generate a professional {proposal_type.replace('_', ' ')} for {company_name}.\n\n"
            f"OUR COMPANY: {our_company} — {our_tagline}\n"
            f"CLIENT: {company_name}\n"
            f"CONTACT: {contact_name}{', ' + contact_title if contact_title else ''}\n"
            f"PRICING TIER: {pricing_tier} (${pricing_amount:,.0f})\n"
            f"TIMELINE: {timeline_weeks} weeks\n\n"
            f"MEETING NOTES:\n{meeting_notes[:3000] if meeting_notes else 'No meeting notes provided.'}\n"
            f"{pain_section}"
            f"{profile_section}"
            f"{rag_section}"
            f"{deliverables_section}\n"
            f"INSTRUCTIONS:\n"
            f"{type_instructions.get(proposal_type, type_instructions['full_proposal'])}\n\n"
            f"FORMAT: Write in clean Markdown. Use ## for section headers.\n"
            f"Be specific, professional, and persuasive. Reference their specific "
            f"pain points and show how our approach addresses each one.\n"
            f"Include the pricing naturally within the proposal.\n"
            f"End with clear next steps and a call-to-action.\n"
        )

        draft = ""
        sections: list[dict[str, Any]] = []

        try:
            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                messages=[{"role": "user", "content": generation_prompt}],
                system=self._get_system_prompt(),
            )
            draft = response.content[0].text.strip() if response.content else ""

            # Parse sections from markdown
            sections = self._parse_sections(draft)

        except Exception as e:
            logger.error(
                "proposal_generate_failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e)[:200],
                },
            )
            return {
                "current_node": "generate_proposal",
                "error": f"Proposal generation failed: {str(e)[:200]}",
                "draft_proposal": "",
            }

        # Build pricing breakdown
        pricing_breakdown = [
            {
                "item": d,
                "amount": round(pricing_amount / max(len(deliverables), 1), 2),
                "description": d,
            }
            for d in deliverables
        ]

        logger.info(
            "proposal_generate_completed",
            extra={
                "agent_id": self.agent_id,
                "company": company_name,
                "draft_length": len(draft),
                "sections": len(sections),
            },
        )

        return {
            "current_node": "generate_proposal",
            "draft_proposal": draft,
            "proposal_sections": sections,
            "pricing_amount": pricing_amount,
            "pricing_breakdown": pricing_breakdown,
            "timeline_weeks": timeline_weeks,
            "deliverables": deliverables,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ProposalBuilderAgentState
    ) -> dict[str, Any]:
        """
        Human review gate. LangGraph's interrupt_before pauses here.

        The human can:
        - Approve the proposal as-is
        - Edit the proposal (triggers RLHF capture)
        - Reject the proposal (skips delivery)
        """
        logger.info(
            "proposal_human_review_pending",
            extra={
                "agent_id": self.agent_id,
                "company": state.get("company_name", ""),
                "type": state.get("proposal_type", ""),
                "pricing": state.get("pricing_amount", 0),
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Finalize ────────────────────────────────────────────

    async def _node_finalize(
        self, state: ProposalBuilderAgentState
    ) -> dict[str, Any]:
        """
        Finalize the proposal after human review.

        Applies any human edits and captures RLHF data if the
        proposal was modified during review.
        """
        draft = state.get("draft_proposal", "")
        human_edited = state.get("human_edited_proposal")
        final_proposal = human_edited if human_edited else draft
        revision_count = state.get("revision_count", 0)

        logger.info(
            "proposal_finalize",
            extra={
                "agent_id": self.agent_id,
                "was_edited": human_edited is not None,
                "revision_count": revision_count,
            },
        )

        # ─── RLHF Capture ─────────────────────────────────────────
        rlhf_captured = False
        if human_edited and human_edited != draft:
            self.learn(
                task_input={
                    "company_name": state.get("company_name", ""),
                    "proposal_type": state.get("proposal_type", ""),
                    "pricing_tier": state.get("pricing_tier", ""),
                    "pain_points": state.get("company_pain_points", []),
                },
                model_output=draft,
                human_correction=human_edited,
                source="manual_review",
                metadata={
                    "agent_type": "proposal_builder",
                    "company": state.get("company_name", ""),
                    "proposal_type": state.get("proposal_type", ""),
                },
            )
            rlhf_captured = True

        return {
            "current_node": "finalize",
            "draft_proposal": final_proposal,
            "proposal_approved": True,
            "revision_count": revision_count + (1 if human_edited else 0),
            "rlhf_captured": rlhf_captured,
        }

    # ─── Node 6: Deliver ─────────────────────────────────────────────

    async def _node_deliver(
        self, state: ProposalBuilderAgentState
    ) -> dict[str, Any]:
        """
        Deliver the finalized proposal to the prospect.

        Sends via email with the proposal as the body. In production,
        this would also generate a PDF and host it as a trackable link.
        """
        contact_email = state.get("contact_email", "")
        contact_name = state.get("contact_name", "")
        company_name = state.get("company_name", "")
        proposal = state.get("draft_proposal", "")
        delivery_method = state.get("delivery_method", "email")

        logger.info(
            "proposal_deliver",
            extra={
                "agent_id": self.agent_id,
                "to": contact_email,
                "method": delivery_method,
                "company": company_name,
            },
        )

        delivered = False
        delivered_at = ""

        if delivery_method == "email" and contact_email:
            try:
                from core.mcp.tools.email_tools import send_email

                our_company = self.config.params.get("company_name", "Enclave Guard")
                subject = f"Proposal: {our_company} × {company_name}"

                # Convert markdown to simple HTML
                html_body = f"<div style='font-family: Arial, sans-serif;'>"
                for line in proposal.split("\n"):
                    if line.startswith("## "):
                        html_body += f"<h2>{line[3:]}</h2>"
                    elif line.startswith("# "):
                        html_body += f"<h1>{line[2:]}</h1>"
                    elif line.startswith("- "):
                        html_body += f"<li>{line[2:]}</li>"
                    elif line.strip():
                        html_body += f"<p>{line}</p>"
                html_body += "</div>"

                await send_email(
                    to_email=contact_email,
                    to_name=contact_name,
                    subject=subject,
                    body_html=html_body,
                    body_text=proposal,
                )
                delivered = True
                delivered_at = datetime.now(timezone.utc).isoformat()

            except Exception as e:
                logger.error(
                    "proposal_deliver_failed",
                    extra={"error": str(e)[:200]},
                )

        # Write insight to shared brain
        self.store_insight(InsightData(
            insight_type="proposal_sent",
            title=f"Proposal sent to {company_name}",
            content=(
                f"Proposal ({state.get('proposal_type', 'full_proposal')}) "
                f"delivered to {contact_name} at {company_name}. "
                f"Pricing: ${state.get('pricing_amount', 0):,.0f} "
                f"({state.get('pricing_tier', 'professional')} tier). "
                f"Pain points: {', '.join(state.get('company_pain_points', []))}"
            ),
            confidence=0.85,
            metadata={
                "company_name": company_name,
                "contact_email": contact_email,
                "pricing_tier": state.get("pricing_tier", ""),
                "pricing_amount": state.get("pricing_amount", 0),
                "proposal_type": state.get("proposal_type", ""),
            },
        ))

        return {
            "current_node": "deliver",
            "delivered": delivered,
            "delivered_at": delivered_at,
            "knowledge_written": True,
        }

    # ─── Routing Functions ────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ProposalBuilderAgentState) -> str:
        """Route after human review: approved → finalize, rejected → END."""
        status = state.get("human_approval_status", "approved")
        if status == "rejected":
            return "rejected"
        return "approved"

    # ─── Helpers ──────────────────────────────────────────────────────

    def _parse_sections(self, markdown: str) -> list[dict[str, Any]]:
        """Parse markdown into structured sections."""
        sections: list[dict[str, Any]] = []
        current_title = ""
        current_content: list[str] = []
        order = 0

        for line in markdown.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_title:
                    sections.append({
                        "title": current_title,
                        "content": "\n".join(current_content).strip(),
                        "order": order,
                    })
                    order += 1
                current_title = line[3:].strip()
                current_content = []
            elif line.startswith("# ") and not current_title:
                current_title = line[2:].strip()
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_title:
            sections.append({
                "title": current_title,
                "content": "\n".join(current_content).strip(),
                "order": order,
            })

        return sections

    def _get_system_prompt(self) -> str:
        """Load system prompt from file or use default."""
        if self.config.system_prompt_path:
            try:
                with open(self.config.system_prompt_path) as f:
                    return f.read()
            except FileNotFoundError:
                logger.warning(
                    f"System prompt not found: {self.config.system_prompt_path}"
                )

        return (
            "You are a senior proposal writer for an elite cybersecurity consulting firm. "
            "You create persuasive, professional proposals that convert discovery calls into "
            "signed engagements. Your proposals are data-driven, specifically address the "
            "prospect's pain points, and clearly articulate ROI. "
            "Write in a confident, consultative tone. Never use filler or fluff. "
            "Every sentence should serve the goal of closing the deal."
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        Write proposal insights to shared brain.
        Main knowledge writing happens in the deliver node.
        """
        pass  # Handled in _node_deliver

    def __repr__(self) -> str:
        return (
            f"<ProposalBuilderAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
