"""
Support Agent — The Customer Service Intelligence Layer.

Classifies inbound support tickets by category, priority, and sentiment.
Searches the knowledge base for relevant articles, drafts helpful responses,
and routes escalations to the appropriate team.

Architecture (LangGraph State Machine):
    classify_ticket → search_knowledge → draft_response →
    human_review → report → END

Trigger Events:
    - ticket_created: New support ticket submitted
    - ticket_escalated: Ticket escalated from automated system
    - manual: On-demand ticket handling

Shared Brain Integration:
    - Reads: common support patterns, FAQ articles, product documentation
    - Writes: ticket category patterns, response effectiveness data

Safety:
    - NEVER sends responses without human review gate
    - Sensitive customer data is not logged in detail
    - Escalation keywords trigger automatic priority elevation
    - Response tone adapts to customer sentiment

Usage:
    agent = SupportAgentImpl(config, db, embedder, llm)
    result = await agent.run({
        "ticket_id": "tk_abc123",
        "ticket_subject": "Can't access my dashboard",
        "ticket_body": "I've been locked out since yesterday...",
        "customer_email": "jane@acme.com",
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
from core.agents.state import SupportAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

CATEGORIES = [
    "billing",
    "technical",
    "feature_request",
    "bug",
    "account_access",
    "onboarding",
    "general",
]

PRIORITY_RULES = {
    "critical": {
        "keywords": ["down", "outage", "breach", "security", "data loss", "emergency"],
        "description": "System-wide or security-impacting issues requiring immediate attention.",
    },
    "high": {
        "keywords": ["broken", "not working", "error", "can't access", "locked out", "urgent"],
        "description": "Significant functionality issues affecting customer operations.",
    },
    "medium": {
        "keywords": ["slow", "intermittent", "sometimes", "issue", "problem", "help"],
        "description": "Non-critical issues that impact customer experience.",
    },
    "low": {
        "keywords": ["question", "how to", "wondering", "curious", "suggestion", "nice to have"],
        "description": "General inquiries, how-to questions, and feature suggestions.",
    },
}

ESCALATION_KEYWORDS = [
    "lawyer", "legal", "attorney", "sue", "lawsuit",
    "cancel", "refund", "chargeback", "fraud",
    "ceo", "manager", "supervisor", "escalate",
    "unacceptable", "terrible", "worst",
    "breach", "leak", "compromised", "hacked",
]

SENTIMENT_INDICATORS = {
    "angry": ["furious", "outraged", "disgusted", "unacceptable", "terrible", "worst", "horrible"],
    "negative": ["frustrated", "disappointed", "annoyed", "unhappy", "upset", "confused"],
    "neutral": ["wondering", "question", "how to", "looking for", "need help"],
    "positive": ["thanks", "great", "appreciate", "love", "excellent", "wonderful"],
}

SUPPORT_CLASSIFY_PROMPT = """\
You are a support ticket classifier. Analyze the ticket below and return \
a JSON object with classification results.

Ticket Subject: {subject}
Ticket Body:
{body}

Return a JSON object:
{{
    "category": "one of: {categories}",
    "priority": "critical|high|medium|low",
    "sentiment": "angry|negative|neutral|positive",
    "escalation_needed": true/false,
    "reasoning": "Brief explanation of classification...",
    "summary": "One sentence summary of the issue"
}}

Return ONLY the JSON object, no markdown code fences.
"""

SUPPORT_RESPONSE_PROMPT = """\
You are a customer support agent. Draft a helpful, empathetic response \
to the support ticket below. Use the knowledge base articles for context.

Ticket Subject: {subject}
Ticket Body:
{body}

Customer: {customer_name} ({customer_email})
Category: {category}
Priority: {priority}
Sentiment: {sentiment}

Relevant Knowledge Base Articles:
{knowledge_articles}

Guidelines:
- Match tone to sentiment (empathetic for angry/negative, helpful for neutral)
- Address the specific issue mentioned in the ticket
- Reference knowledge base articles where relevant
- Include actionable next steps
- Keep response professional and concise

Return the response text directly, no JSON wrapping needed.
"""


@register_agent_type("support_agent")
class SupportAgentImpl(BaseAgent):
    """
    Customer support ticket classification and response agent.

    Nodes:
        1. classify_ticket     -- LLM categorizes, prioritizes, detects sentiment
        2. search_knowledge    -- RAG search for relevant articles
        3. draft_response      -- LLM drafts response based on knowledge + context
        4. human_review        -- Gate
        5. report              -- Save to support_tickets table + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Support Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(SupportAgentState)

        workflow.add_node("classify_ticket", self._node_classify_ticket)
        workflow.add_node("search_knowledge", self._node_search_knowledge)
        workflow.add_node("draft_response", self._node_draft_response)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("classify_ticket")

        workflow.add_edge("classify_ticket", "search_knowledge")
        workflow.add_edge("search_knowledge", "draft_response")
        workflow.add_edge("draft_response", "human_review")
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
    def get_state_class(cls) -> Type[SupportAgentState]:
        return SupportAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "ticket_id": "",
            "ticket_subject": "",
            "ticket_body": "",
            "customer_email": "",
            "customer_name": "",
            "customer_id": "",
            "category": "",
            "priority": "",
            "sentiment": "",
            "escalation_needed": False,
            "classification_reasoning": "",
            "relevant_articles": [],
            "knowledge_sources_checked": 0,
            "draft_response": "",
            "response_tone": "",
            "suggested_actions": [],
            "response_sent": False,
            "ticket_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Classify Ticket ─────────────────────────────────────

    async def _node_classify_ticket(
        self, state: SupportAgentState
    ) -> dict[str, Any]:
        """Node 1: LLM categorizes, prioritizes, and detects sentiment."""
        task = state.get("task_input", {})
        ticket_id = task.get("ticket_id", "")
        subject = task.get("ticket_subject", "")
        body = task.get("ticket_body", "")
        customer_email = task.get("customer_email", "")
        customer_name = task.get("customer_name", "Customer")
        customer_id = task.get("customer_id", "")

        logger.info(
            "support_classify_ticket",
            extra={"ticket_id": ticket_id, "agent_id": self.agent_id},
        )

        # Rule-based pre-classification
        combined_text = f"{subject} {body}".lower()

        # Check for escalation keywords
        escalation_needed = any(kw in combined_text for kw in ESCALATION_KEYWORDS)

        # Rule-based priority detection
        priority = "medium"
        for prio_level in ["critical", "high", "medium", "low"]:
            keywords = PRIORITY_RULES[prio_level]["keywords"]
            if any(kw in combined_text for kw in keywords):
                priority = prio_level
                break

        # Rule-based sentiment detection
        sentiment = "neutral"
        for sent_level in ["angry", "negative", "positive", "neutral"]:
            indicators = SENTIMENT_INDICATORS.get(sent_level, [])
            if any(ind in combined_text for ind in indicators):
                sentiment = sent_level
                break

        category = "general"
        classification_reasoning = ""

        # LLM classification for more nuanced understanding
        try:
            prompt = SUPPORT_CLASSIFY_PROMPT.format(
                subject=subject,
                body=body[:2000],
                categories=", ".join(CATEGORIES),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a customer support ticket classifier.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                llm_data = json.loads(llm_text)
                category = llm_data.get("category", category)
                if category not in CATEGORIES:
                    category = "general"
                # Use LLM priority if rule-based didn't find critical/high
                if priority in ("medium", "low"):
                    priority = llm_data.get("priority", priority)
                sentiment = llm_data.get("sentiment", sentiment)
                escalation_needed = escalation_needed or llm_data.get(
                    "escalation_needed", False
                )
                classification_reasoning = llm_data.get("reasoning", "")

            except (json.JSONDecodeError, KeyError):
                logger.debug("support_classify_parse_error: Could not parse LLM JSON")

        except Exception as e:
            logger.warning(
                "support_classify_llm_error",
                extra={"error": str(e)[:200]},
            )

        # Force escalation to bump priority
        if escalation_needed and priority not in ("critical", "high"):
            priority = "high"

        logger.info(
            "support_ticket_classified",
            extra={
                "ticket_id": ticket_id,
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "escalation_needed": escalation_needed,
            },
        )

        return {
            "current_node": "classify_ticket",
            "ticket_id": ticket_id,
            "ticket_subject": subject,
            "ticket_body": body,
            "customer_email": customer_email,
            "customer_name": customer_name,
            "customer_id": customer_id,
            "category": category,
            "priority": priority,
            "sentiment": sentiment,
            "escalation_needed": escalation_needed,
            "classification_reasoning": classification_reasoning,
        }

    # ─── Node 2: Search Knowledge ────────────────────────────────────

    async def _node_search_knowledge(
        self, state: SupportAgentState
    ) -> dict[str, Any]:
        """Node 2: RAG search for relevant knowledge base articles."""
        subject = state.get("ticket_subject", "")
        body = state.get("ticket_body", "")
        category = state.get("category", "general")

        logger.info(
            "support_search_knowledge",
            extra={"category": category},
        )

        search_query = f"{subject} {category} {body[:200]}"
        relevant_articles: list[dict[str, Any]] = []
        knowledge_sources_checked = 0

        # Search knowledge base via DB
        try:
            result = (
                self.db.client.table("knowledge_base")
                .select("*")
                .ilike("title", f"%{category}%")
                .limit(5)
                .execute()
            )
            knowledge_sources_checked += 1
            if result.data:
                for article in result.data:
                    relevant_articles.append({
                        "title": article.get("title", ""),
                        "content": article.get("content", "")[:500],
                        "relevance_score": 0.8,
                        "source": "knowledge_base",
                    })
        except Exception as e:
            logger.debug(f"Knowledge base search error: {e}")

        # Also search insights from shared brain
        try:
            result = (
                self.db.client.table("agent_insights")
                .select("*")
                .ilike("content", f"%{category}%")
                .limit(3)
                .execute()
            )
            knowledge_sources_checked += 1
            if result.data:
                for insight in result.data:
                    relevant_articles.append({
                        "title": insight.get("title", ""),
                        "content": insight.get("content", "")[:300],
                        "relevance_score": 0.6,
                        "source": "agent_insights",
                    })
        except Exception as e:
            logger.debug(f"Insights search error: {e}")

        logger.info(
            "support_knowledge_searched",
            extra={
                "articles_found": len(relevant_articles),
                "sources_checked": knowledge_sources_checked,
            },
        )

        return {
            "current_node": "search_knowledge",
            "relevant_articles": relevant_articles,
            "knowledge_sources_checked": knowledge_sources_checked,
        }

    # ─── Node 3: Draft Response ──────────────────────────────────────

    async def _node_draft_response(
        self, state: SupportAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM drafts response based on knowledge and ticket context."""
        subject = state.get("ticket_subject", "")
        body = state.get("ticket_body", "")
        category = state.get("category", "general")
        priority = state.get("priority", "medium")
        sentiment = state.get("sentiment", "neutral")
        customer_name = state.get("customer_name", "Customer")
        customer_email = state.get("customer_email", "")
        articles = state.get("relevant_articles", [])

        logger.info(
            "support_draft_response",
            extra={"category": category, "sentiment": sentiment},
        )

        # Format knowledge articles
        articles_text = ""
        if articles:
            for i, article in enumerate(articles[:5], 1):
                articles_text += (
                    f"\n{i}. **{article.get('title', 'Article')}**: "
                    f"{article.get('content', 'No content')[:300]}\n"
                )
        else:
            articles_text = "No relevant articles found in knowledge base."

        # Determine response tone
        tone_map = {
            "angry": "empathetic",
            "negative": "empathetic",
            "neutral": "professional",
            "positive": "professional",
        }
        response_tone = tone_map.get(sentiment, "professional")

        draft_response = ""
        suggested_actions: list[str] = []

        try:
            prompt = SUPPORT_RESPONSE_PROMPT.format(
                subject=subject,
                body=body[:2000],
                customer_name=customer_name,
                customer_email=customer_email,
                category=category,
                priority=priority,
                sentiment=sentiment,
                knowledge_articles=articles_text,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system=(
                    f"You are an {response_tone} customer support agent. "
                    "Draft a helpful response to the support ticket."
                ),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )

            draft_response = llm_response.content[0].text.strip()

        except Exception as e:
            logger.warning(
                "support_draft_llm_error",
                extra={"error": str(e)[:200]},
            )
            draft_response = (
                f"Dear {customer_name},\n\n"
                f"Thank you for reaching out. We've received your support request "
                f"regarding '{subject}' and our team is looking into it.\n\n"
                f"We'll get back to you shortly with a resolution.\n\n"
                f"Best regards,\nSupport Team"
            )

        # Determine suggested actions
        if state.get("escalation_needed"):
            suggested_actions.append("Escalate to manager/supervisor")
        if priority == "critical":
            suggested_actions.append("Create incident ticket")
        if category == "billing":
            suggested_actions.append("Review billing records")
        if category == "bug":
            suggested_actions.append("Create engineering issue")

        logger.info(
            "support_response_drafted",
            extra={
                "response_length": len(draft_response),
                "tone": response_tone,
                "actions": len(suggested_actions),
            },
        )

        return {
            "current_node": "draft_response",
            "draft_response": draft_response,
            "response_tone": response_tone,
            "suggested_actions": suggested_actions,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: SupportAgentState
    ) -> dict[str, Any]:
        """Node 4: Present draft response for human approval."""
        priority = state.get("priority", "")
        category = state.get("category", "")
        escalation = state.get("escalation_needed", False)

        logger.info(
            "support_human_review_pending",
            extra={
                "priority": priority,
                "category": category,
                "escalation_needed": escalation,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: SupportAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to support_tickets table and generate insights."""
        now = datetime.now(timezone.utc).isoformat()
        ticket_id = state.get("ticket_id", "")
        category = state.get("category", "general")
        priority = state.get("priority", "medium")
        sentiment = state.get("sentiment", "neutral")

        # Save ticket record
        ticket_saved = False
        try:
            ticket_record = {
                "ticket_id": ticket_id,
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "ticket_subject": state.get("ticket_subject", ""),
                "customer_email": state.get("customer_email", ""),
                "customer_name": state.get("customer_name", ""),
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "escalation_needed": state.get("escalation_needed", False),
                "draft_response": state.get("draft_response", ""),
                "response_tone": state.get("response_tone", ""),
                "suggested_actions": json.dumps(state.get("suggested_actions", [])),
                "status": "responded" if state.get("human_approval_status") == "approved" else "pending",
                "created_at": now,
            }

            result = (
                self.db.client.table("support_tickets")
                .insert(ticket_record)
                .execute()
            )
            if result.data:
                ticket_saved = True
                logger.info(
                    "support_ticket_saved",
                    extra={"ticket_id": ticket_id},
                )
        except Exception as e:
            logger.warning(
                "support_ticket_save_error",
                extra={"error": str(e)[:200]},
            )

        # Build report
        sections = [
            "# Support Ticket Report",
            f"*Generated: {now}*\n",
            f"## Ticket: {state.get('ticket_subject', 'N/A')}",
            f"- **Ticket ID:** {ticket_id}",
            f"- **Category:** {category}",
            f"- **Priority:** {priority}",
            f"- **Sentiment:** {sentiment}",
            f"- **Escalation Needed:** {'Yes' if state.get('escalation_needed') else 'No'}",
            f"- **Response Tone:** {state.get('response_tone', 'N/A')}",
            f"\n## Knowledge Articles Used: {len(state.get('relevant_articles', []))}",
            f"## Suggested Actions: {', '.join(state.get('suggested_actions', ['None']))}",
            f"\n## Status",
            f"- Ticket Saved: {'Yes' if ticket_saved else 'No'}",
        ]

        report = "\n".join(sections)

        # Store insight
        self.store_insight(InsightData(
            insight_type="support_pattern",
            title=f"Support: {category}/{priority} — {state.get('ticket_subject', '')[:50]}",
            content=(
                f"Support ticket classified as {category}/{priority}, "
                f"sentiment: {sentiment}. "
                f"Escalation: {state.get('escalation_needed', False)}. "
                f"Knowledge articles used: {len(state.get('relevant_articles', []))}."
            ),
            confidence=0.75,
            metadata={
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "escalation_needed": state.get("escalation_needed", False),
                "articles_found": len(state.get("relevant_articles", [])),
            },
        ))

        logger.info(
            "support_report_generated",
            extra={
                "ticket_id": ticket_id,
                "category": category,
                "priority": priority,
            },
        )

        return {
            "current_node": "report",
            "ticket_saved": ticket_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: SupportAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<SupportAgentImpl agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
