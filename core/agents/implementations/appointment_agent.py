"""
Appointment Setter Agent — The Revenue Closer.

Converts email replies into booked discovery calls. This is the final
piece of the autonomous sales pipeline:

    Outreach Agent → generates interest
    SEO Agent → builds authority
    Appointment Setter → closes the meeting

Architecture (LangGraph State Machine):
    classify_intent → route_by_intent → [
        handle_interested     → draft_reply → human_review → send_reply
        handle_objection      → draft_reply → human_review → send_reply
        handle_question       → draft_reply → human_review → send_reply
        handle_ooo            → draft_reply → human_review → send_reply
        handle_not_interested → draft_reply → human_review → send_reply
    ]

Shared Brain: Reads winning rebuttals and messaging insights from
shared_insights (cross-agent knowledge). Writes back what objections
were encountered and how they were resolved.

RLHF Hook: If a human edits the drafted reply in human_review,
the (draft, edit) pair is saved for future optimization.

Usage:
    agent = AppointmentSetterAgent(config, db, embedder, llm)
    result = await agent.run({
        "inbound_email": {
            "from_email": "jane@acme.com",
            "subject": "Re: Security Assessment",
            "body": "This sounds interesting. Can we schedule a call?",
        },
        "contact_id": "...",
        "company_id": "...",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, ReplyClassification
from core.agents.registry import register_agent_type
from core.agents.state import AppointmentAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Intent constants ─────────────────────────────────────────────────
INTENT_INTERESTED = "interested"
INTENT_OBJECTION = "objection"
INTENT_QUESTION = "question"
INTENT_OOO = "ooo"
INTENT_WRONG_PERSON = "wrong_person"
INTENT_UNSUBSCRIBE = "unsubscribe"
INTENT_NOT_INTERESTED = "not_interested"

VALID_INTENTS = {
    INTENT_INTERESTED,
    INTENT_OBJECTION,
    INTENT_QUESTION,
    INTENT_OOO,
    INTENT_WRONG_PERSON,
    INTENT_UNSUBSCRIBE,
    INTENT_NOT_INTERESTED,
}

# Intents that should propose meeting times
BOOKING_INTENTS = {INTENT_INTERESTED, INTENT_QUESTION}

# Intents where we draft a reply
REPLY_INTENTS = {
    INTENT_INTERESTED,
    INTENT_OBJECTION,
    INTENT_QUESTION,
    INTENT_OOO,
    INTENT_WRONG_PERSON,
    INTENT_NOT_INTERESTED,
}


@register_agent_type("appointment_setter")
class AppointmentSetterAgent(BaseAgent):
    """
    AI-powered appointment setter that converts email replies into meetings.

    Nodes:
        1. classify_intent — LLM-based reply intent classification
        2. handle_interested — Propose meeting times (soft close)
        3. handle_objection — Pull rebuttals from shared brain, address concern
        4. handle_question — Answer from knowledge base
        5. handle_ooo — Schedule follow-up after return
        6. handle_not_interested — Polite close + archive
        7. draft_reply — Generate the response email
        8. human_review — Gate for human approval/editing
        9. send_reply — Send via email tool
    """

    def build_graph(self) -> Any:
        """
        Build the Appointment Setter's LangGraph state machine.

        Graph flow:
            classify_intent → route_by_intent → [handler] → draft_reply
            → human_review → route_after_review → send_reply / END
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AppointmentAgentState)

        # Add nodes
        workflow.add_node("classify_intent", self._node_classify_intent)
        workflow.add_node("handle_interested", self._node_handle_interested)
        workflow.add_node("handle_objection", self._node_handle_objection)
        workflow.add_node("handle_question", self._node_handle_question)
        workflow.add_node("handle_ooo", self._node_handle_ooo)
        workflow.add_node("handle_not_interested", self._node_handle_not_interested)
        workflow.add_node("draft_reply", self._node_draft_reply)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("send_reply", self._node_send_reply)

        # Entry point
        workflow.set_entry_point("classify_intent")

        # Intent routing: classify → appropriate handler
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                INTENT_INTERESTED: "handle_interested",
                INTENT_OBJECTION: "handle_objection",
                INTENT_QUESTION: "handle_question",
                INTENT_OOO: "handle_ooo",
                INTENT_WRONG_PERSON: "handle_not_interested",
                INTENT_UNSUBSCRIBE: END,
                INTENT_NOT_INTERESTED: "handle_not_interested",
            },
        )

        # All handlers flow to draft_reply
        for handler in [
            "handle_interested",
            "handle_objection",
            "handle_question",
            "handle_ooo",
            "handle_not_interested",
        ]:
            workflow.add_edge(handler, "draft_reply")

        # Draft → human review → route
        workflow.add_edge("draft_reply", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "send_reply",
                "rejected": END,
            },
        )
        workflow.add_edge("send_reply", END)

        # Compile with human gate interrupt
        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        """Appointment setter uses calendar and email MCP tools."""
        return []  # Tools accessed via MCP, not injected

    def get_state_class(self) -> Type[AppointmentAgentState]:
        return AppointmentAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state from inbound email task."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "inbound_email": task.get("inbound_email", {}),
            "contact_id": task.get("contact_id", ""),
            "company_id": task.get("company_id", ""),
            "conversation_history": task.get("conversation_history", []),
            "reply_intent": "",
            "reply_sentiment": "neutral",
            "objection_type": None,
            "rebuttal_context": [],
            "draft_reply_subject": "",
            "draft_reply_body": "",
            "proposed_times": [],
            "calendar_link": None,
            "meeting_booked": False,
            "meeting_datetime": None,
            "follow_up_sequence_step": task.get("follow_up_sequence_step", 0),
            "next_follow_up_at": None,
        })
        return state

    # ─── Node 1: Intent Classification ────────────────────────────────

    async def _node_classify_intent(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Classify the inbound email reply intent using LLM.

        Returns structured classification: intent, sentiment, confidence,
        objection_type (if applicable).
        """
        inbound = state.get("inbound_email", {})
        email_body = inbound.get("body", "")
        email_subject = inbound.get("subject", "")
        conversation = state.get("conversation_history", [])

        logger.info(
            "appointment_classify_started",
            extra={
                "agent_id": self.agent_id,
                "subject": email_subject[:80],
                "body_length": len(email_body),
            },
        )

        # Build classification prompt
        conversation_context = ""
        if conversation:
            for msg in conversation[-5:]:  # Last 5 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:500]
                conversation_context += f"\n[{role}]: {content}\n"

        classification_prompt = (
            "Classify this email reply. Return ONLY a JSON object with these fields:\n"
            '- "intent": one of [interested, objection, question, ooo, wrong_person, unsubscribe, not_interested]\n'
            '- "sentiment": one of [positive, negative, neutral]\n'
            '- "confidence": float 0.0-1.0\n'
            '- "objection_type": string or null (if intent is "objection", specify type: '
            'price, timing, already_have, need_approval, send_info, other)\n'
            '- "summary": brief 1-sentence summary of the reply\n\n'
            f"Previous conversation:\n{conversation_context or 'No previous messages.'}\n\n"
            f"Subject: {email_subject}\n"
            f"Reply:\n{email_body}\n\n"
            "Return ONLY the JSON object, no markdown code fences."
        )

        intent = INTENT_NOT_INTERESTED
        sentiment = "neutral"
        confidence = 0.0
        objection_type = None

        try:
            response = self.llm.messages.create(
                model=self.config.params.get(
                    "classification_model", self.config.model.model
                ),
                max_tokens=500,
                temperature=0.1,  # Low temp for precise classification
                messages=[{"role": "user", "content": classification_prompt}],
            )

            response_text = response.content[0].text.strip() if response.content else "{}"

            # Parse JSON (handle possible markdown fences)
            json_text = response_text
            if "```" in json_text:
                # Extract content between code fences
                parts = json_text.split("```")
                json_text = parts[1] if len(parts) > 1 else json_text
                if json_text.startswith("json"):
                    json_text = json_text[4:]
                json_text = json_text.strip()

            parsed = json.loads(json_text)

            intent = parsed.get("intent", INTENT_NOT_INTERESTED)
            if intent not in VALID_INTENTS:
                logger.warning(
                    f"Unknown intent '{intent}', defaulting to not_interested"
                )
                intent = INTENT_NOT_INTERESTED

            sentiment = parsed.get("sentiment", "neutral")
            confidence = float(parsed.get("confidence", 0.5))
            objection_type = parsed.get("objection_type")

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "appointment_classify_failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e)[:200],
                },
            )
            # Fallback: simple heuristic classification
            intent, sentiment = self._heuristic_classify(email_body)
            confidence = 0.3

        logger.info(
            "appointment_intent_classified",
            extra={
                "agent_id": self.agent_id,
                "intent": intent,
                "sentiment": sentiment,
                "confidence": confidence,
                "objection_type": objection_type,
            },
        )

        return {
            "current_node": "classify_intent",
            "reply_intent": intent,
            "reply_sentiment": sentiment,
            "objection_type": objection_type,
        }

    def _heuristic_classify(self, body: str) -> tuple[str, str]:
        """Fallback heuristic classification when LLM fails."""
        body_lower = body.lower()

        # OOO detection
        ooo_signals = ["out of office", "on vacation", "returning on", "auto-reply", "away from"]
        if any(s in body_lower for s in ooo_signals):
            return INTENT_OOO, "neutral"

        # Unsubscribe detection
        unsub_signals = ["unsubscribe", "remove me", "stop emailing", "opt out"]
        if any(s in body_lower for s in unsub_signals):
            return INTENT_UNSUBSCRIBE, "negative"

        # Not interested
        reject_signals = ["not interested", "no thanks", "no thank you", "not a fit", "pass on this"]
        if any(s in body_lower for s in reject_signals):
            return INTENT_NOT_INTERESTED, "negative"

        # Wrong person
        wrong_signals = ["wrong person", "not the right", "try reaching", "not my area"]
        if any(s in body_lower for s in wrong_signals):
            return INTENT_WRONG_PERSON, "neutral"

        # Interested
        interest_signals = ["interested", "sounds good", "let's chat", "schedule", "available", "set up a call"]
        if any(s in body_lower for s in interest_signals):
            return INTENT_INTERESTED, "positive"

        # Question
        if "?" in body:
            return INTENT_QUESTION, "neutral"

        return INTENT_NOT_INTERESTED, "neutral"

    # ─── Node 2: Handle Interested ────────────────────────────────────

    async def _node_handle_interested(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Handle interested reply: fetch calendar availability, propose times.

        The Soft Close pattern:
        1. Propose 3 specific times (in prospect's timezone if known)
        2. Include Calendly/Cal.com link as fallback
        """
        logger.info(
            "appointment_handle_interested",
            extra={"agent_id": self.agent_id, "intent": "interested"},
        )

        # Fetch available time slots
        proposed_times: list[str] = []
        calendar_link: Optional[str] = None

        try:
            from core.integrations.calendar_client import CalendarClient

            calendar = CalendarClient.from_env()
            max_times = self.config.params.get("max_proposed_times", 3)
            days_ahead = self.config.params.get("days_ahead_for_slots", 5)

            slots = await calendar.get_available_slots(
                days_ahead=days_ahead,
                max_slots=max_times,
            )
            proposed_times = [s["start"] for s in slots]
            calendar_link = calendar.get_booking_link()

        except Exception as e:
            logger.warning(
                "appointment_calendar_fetch_failed",
                extra={"error": str(e)[:200]},
            )
            calendar_link = self.config.params.get(
                "fallback_booking_link", "https://cal.enclaveguard.com"
            )

        return {
            "current_node": "handle_interested",
            "proposed_times": proposed_times,
            "calendar_link": calendar_link,
        }

    # ─── Node 3: Handle Objection ─────────────────────────────────────

    async def _node_handle_objection(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Handle objection reply: search shared brain for winning rebuttals.

        Reads shared_insights for objection_rebuttal entries that match
        the objection type. This is cross-agent intelligence — the outreach
        agent may have logged successful rebuttals that we can reuse.
        """
        objection_type = state.get("objection_type", "other")

        logger.info(
            "appointment_handle_objection",
            extra={
                "agent_id": self.agent_id,
                "objection_type": objection_type,
            },
        )

        rebuttal_context: list[dict[str, Any]] = []

        # Search shared brain for relevant rebuttals
        try:
            query = f"objection handling {objection_type} cybersecurity"
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="objection_rebuttal",
                limit=3,
            )
            rebuttal_context = insights if isinstance(insights, list) else []
        except Exception as e:
            logger.debug(f"Could not fetch rebuttal insights: {e}")

        # If no specific rebuttals, try general winning patterns
        if not rebuttal_context:
            try:
                query = f"winning response {objection_type}"
                insights = self.db.search_insights(
                    query_embedding=self.embedder.embed_query(query),
                    insight_type="winning_pattern",
                    limit=3,
                )
                rebuttal_context = insights if isinstance(insights, list) else []
            except Exception as e:
                logger.debug(f"Could not fetch winning pattern insights: {e}")

        return {
            "current_node": "handle_objection",
            "rebuttal_context": rebuttal_context,
        }

    # ─── Node 4: Handle Question ──────────────────────────────────────

    async def _node_handle_question(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Handle question reply: answer from knowledge base, include soft CTA.
        """
        inbound = state.get("inbound_email", {})
        question = inbound.get("body", "")

        logger.info(
            "appointment_handle_question",
            extra={"agent_id": self.agent_id},
        )

        # Search knowledge base for relevant context
        rag_context: list[dict[str, Any]] = []
        try:
            query_embedding = self.embedder.embed_query(question[:500])
            chunks = self.db.search_knowledge(
                query_embedding=query_embedding,
                limit=5,
            )
            rag_context = chunks if isinstance(chunks, list) else []
        except Exception as e:
            logger.debug(f"Could not search knowledge base: {e}")

        # Also try to propose times (soft CTA)
        proposed_times: list[str] = []
        calendar_link: Optional[str] = None
        try:
            from core.integrations.calendar_client import CalendarClient

            calendar = CalendarClient.from_env()
            slots = await calendar.get_available_slots(days_ahead=5, max_slots=2)
            proposed_times = [s["start"] for s in slots]
            calendar_link = calendar.get_booking_link()
        except Exception:
            pass

        return {
            "current_node": "handle_question",
            "rag_context": rag_context,
            "proposed_times": proposed_times,
            "calendar_link": calendar_link,
        }

    # ─── Node 5: Handle OOO ───────────────────────────────────────────

    async def _node_handle_ooo(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Handle out-of-office reply: parse return date, schedule follow-up.

        Adds a buffer (default 1 day) after their stated return date.
        """
        inbound = state.get("inbound_email", {})
        body = inbound.get("body", "")
        buffer_days = self.config.params.get("ooo_buffer_days", 1)

        logger.info(
            "appointment_handle_ooo",
            extra={"agent_id": self.agent_id},
        )

        # Try to extract return date using LLM
        next_follow_up = None
        try:
            parse_prompt = (
                "Extract the return date from this out-of-office message. "
                "Return ONLY an ISO date string (YYYY-MM-DD) or 'unknown' if no date found.\n\n"
                f"Message:\n{body[:1000]}"
            )

            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=50,
                temperature=0.0,
                messages=[{"role": "user", "content": parse_prompt}],
            )
            date_str = response.content[0].text.strip() if response.content else "unknown"

            if date_str != "unknown":
                return_date = datetime.fromisoformat(date_str)
                follow_up_date = return_date + timedelta(days=buffer_days)
                next_follow_up = follow_up_date.isoformat()

        except Exception as e:
            logger.debug(f"Could not parse OOO return date: {e}")

        # Default: follow up in 7 days if no return date found
        if not next_follow_up:
            next_follow_up = (
                datetime.now(timezone.utc) + timedelta(days=7)
            ).date().isoformat()

        return {
            "current_node": "handle_ooo",
            "next_follow_up_at": next_follow_up,
        }

    # ─── Node 6: Handle Not Interested / Wrong Person ─────────────────

    async def _node_handle_not_interested(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Handle not-interested or wrong-person replies.

        For wrong_person: ask for referral to the right contact.
        For not_interested: polite close, no chase.
        """
        intent = state.get("reply_intent", INTENT_NOT_INTERESTED)

        logger.info(
            "appointment_handle_not_interested",
            extra={
                "agent_id": self.agent_id,
                "intent": intent,
            },
        )

        return {
            "current_node": "handle_not_interested",
        }

    # ─── Node 7: Draft Reply ──────────────────────────────────────────

    async def _node_draft_reply(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Generate the response email using LLM with full context.

        Builds a context-rich prompt incorporating:
        - The prospect's reply and conversation history
        - Intent classification result
        - Available meeting times (for interested/question intents)
        - Rebuttal context (for objection intent)
        - Knowledge base results (for question intent)
        - OOO follow-up date (for ooo intent)
        """
        inbound = state.get("inbound_email", {})
        intent = state.get("reply_intent", "")
        objection_type = state.get("objection_type")
        proposed_times = state.get("proposed_times", [])
        calendar_link = state.get("calendar_link")
        rebuttal_context = state.get("rebuttal_context", [])
        rag_context = state.get("rag_context", [])
        next_follow_up = state.get("next_follow_up_at")
        conversation = state.get("conversation_history", [])

        from_email = inbound.get("from_email", "")
        subject = inbound.get("subject", "")
        body = inbound.get("body", "")
        company_name = self.config.params.get("company_name", "Enclave Guard")
        value_prop = self.config.params.get("value_proposition", "security posture review")

        logger.info(
            "appointment_draft_started",
            extra={
                "agent_id": self.agent_id,
                "intent": intent,
                "has_times": len(proposed_times) > 0,
            },
        )

        # Build context sections
        times_section = ""
        if proposed_times:
            times_section = "\nAVAILABLE MEETING TIMES:\n"
            for i, t in enumerate(proposed_times[:3], 1):
                times_section += f"  {i}. {t}\n"
            if calendar_link:
                times_section += f"\nSelf-service booking link: {calendar_link}\n"

        rebuttal_section = ""
        if rebuttal_context:
            rebuttal_section = "\nSHARED BRAIN — WINNING REBUTTALS:\n"
            for r in rebuttal_context[:3]:
                content = r.get("content", r.get("title", ""))[:300]
                rebuttal_section += f"  - {content}\n"

        knowledge_section = ""
        if rag_context:
            knowledge_section = "\nKNOWLEDGE BASE CONTEXT:\n"
            for chunk in rag_context[:3]:
                text = chunk.get("content", chunk.get("text", ""))[:300]
                knowledge_section += f"  - {text}\n"

        conversation_section = ""
        if conversation:
            conversation_section = "\nCONVERSATION HISTORY:\n"
            for msg in conversation[-5:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:300]
                conversation_section += f"  [{role}]: {content}\n"

        ooo_section = ""
        if next_follow_up:
            ooo_section = f"\nFOLLOW-UP SCHEDULED: {next_follow_up}\n"

        draft_prompt = (
            f"Draft a reply email for this conversation.\n\n"
            f"INTENT: {intent}\n"
            f"{'OBJECTION TYPE: ' + objection_type if objection_type else ''}\n"
            f"SENTIMENT: {state.get('reply_sentiment', 'neutral')}\n"
            f"OUR COMPANY: {company_name}\n"
            f"VALUE PROPOSITION: {value_prop}\n\n"
            f"THEIR REPLY:\n"
            f"From: {from_email}\n"
            f"Subject: {subject}\n"
            f"Body: {body}\n"
            f"{conversation_section}"
            f"{times_section}"
            f"{rebuttal_section}"
            f"{knowledge_section}"
            f"{ooo_section}\n"
            f"INSTRUCTIONS:\n"
            f"1. Write a professional, warm reply that addresses their specific message\n"
            f"2. Match their energy and tone\n"
            f"3. Include ONE clear call-to-action\n"
            f"{'4. Propose the available meeting times naturally' if proposed_times else ''}\n"
            f"{'4. Address their objection using the rebuttal context, then pivot to a meeting' if intent == INTENT_OBJECTION else ''}\n"
            f"{'4. Answer their question using the knowledge context, then include a soft meeting CTA' if intent == INTENT_QUESTION else ''}\n"
            f"{'4. Acknowledge their OOO, mention you will follow up after they return' if intent == INTENT_OOO else ''}\n"
            f"{'4. Thank them politely. For wrong_person, ask for a referral.' if intent in (INTENT_NOT_INTERESTED, INTENT_WRONG_PERSON) else ''}\n"
            f"5. Keep it concise (under 150 words)\n"
            f"6. Do NOT include a subject line — just the body text\n"
        )

        draft_subject = f"Re: {subject}" if subject else "Re: Follow-up"
        draft_body = ""

        try:
            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                messages=[{"role": "user", "content": draft_prompt}],
                system=self._get_system_prompt(),
            )
            draft_body = response.content[0].text.strip() if response.content else ""
        except Exception as e:
            logger.error(
                "appointment_draft_failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e)[:200],
                },
            )
            return {
                "current_node": "draft_reply",
                "error": f"Draft generation failed: {str(e)[:200]}",
                "draft_reply_subject": draft_subject,
                "draft_reply_body": "",
            }

        logger.info(
            "appointment_draft_completed",
            extra={
                "agent_id": self.agent_id,
                "intent": intent,
                "draft_length": len(draft_body),
            },
        )

        return {
            "current_node": "draft_reply",
            "draft_reply_subject": draft_subject,
            "draft_reply_body": draft_body,
        }

    # ─── Node 8: Human Review Gate ────────────────────────────────────

    async def _node_human_review(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Human review gate. LangGraph's interrupt_before pauses here.

        The human can:
        - Approve the draft as-is
        - Edit the draft (triggers RLHF capture)
        - Reject the draft (skips sending)
        """
        logger.info(
            "appointment_human_review_pending",
            extra={
                "agent_id": self.agent_id,
                "intent": state.get("reply_intent", ""),
                "subject": state.get("draft_reply_subject", "")[:80],
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 9: Send Reply ───────────────────────────────────────────

    async def _node_send_reply(
        self, state: AppointmentAgentState
    ) -> dict[str, Any]:
        """
        Send the approved reply email and capture RLHF data if edited.

        Also writes insights to the shared brain about what worked.
        """
        inbound = state.get("inbound_email", {})
        to_email = inbound.get("from_email", "")
        intent = state.get("reply_intent", "")

        # Use human-edited content if available
        human_edited = state.get("human_edited_content")
        draft_body = state.get("draft_reply_body", "")
        final_body = human_edited if human_edited else draft_body
        subject = state.get("draft_reply_subject", "")

        logger.info(
            "appointment_send_reply",
            extra={
                "agent_id": self.agent_id,
                "to_email": to_email,
                "intent": intent,
                "was_edited": human_edited is not None,
            },
        )

        # ─── RLHF Capture ─────────────────────────────────────────
        if human_edited and human_edited != draft_body:
            self.learn(
                task_input={
                    "intent": intent,
                    "objection_type": state.get("objection_type"),
                    "inbound_body": inbound.get("body", "")[:500],
                },
                model_output=draft_body,
                human_correction=human_edited,
                source="manual_review",
                metadata={
                    "agent_type": "appointment_setter",
                    "intent": intent,
                    "contact_id": state.get("contact_id", ""),
                },
            )

        # ─── Send Email (via sandboxed tool) ───────────────────────
        send_result = {}
        try:
            from core.mcp.tools.email_tools import send_email

            send_result = await send_email(
                to_email=to_email,
                to_name=inbound.get("from_name", ""),
                subject=subject,
                body_html=f"<p>{final_body.replace(chr(10), '</p><p>')}</p>",
                body_text=final_body,
            )
        except Exception as e:
            logger.error(
                "appointment_send_failed",
                extra={"error": str(e)[:200]},
            )
            return {
                "current_node": "send_reply",
                "error": f"Send failed: {str(e)[:200]}",
            }

        # ─── Write Insight to Shared Brain ─────────────────────────
        if intent in (INTENT_INTERESTED, INTENT_OBJECTION):
            insight_type = (
                "objection_rebuttal" if intent == INTENT_OBJECTION
                else "winning_pattern"
            )
            self.store_insight(InsightData(
                insight_type=insight_type,
                title=f"Appointment: {intent} → replied",
                content=(
                    f"Handled {intent} reply"
                    f"{' (objection: ' + str(state.get('objection_type')) + ')' if state.get('objection_type') else ''}. "
                    f"Response sent to {to_email}."
                ),
                confidence=0.75,
                metadata={
                    "intent": intent,
                    "objection_type": state.get("objection_type"),
                    "contact_id": state.get("contact_id", ""),
                },
            ))

        # Check if this was a booking flow
        meeting_booked = state.get("meeting_booked", False)

        return {
            "current_node": "send_reply",
            "meeting_booked": meeting_booked,
            "knowledge_written": True,
        }

    # ─── Routing Functions ────────────────────────────────────────────

    @staticmethod
    def _route_by_intent(state: AppointmentAgentState) -> str:
        """Route to the appropriate handler based on classified intent."""
        intent = state.get("reply_intent", INTENT_NOT_INTERESTED)
        if intent not in VALID_INTENTS:
            return INTENT_NOT_INTERESTED
        return intent

    @staticmethod
    def _route_after_review(state: AppointmentAgentState) -> str:
        """Route after human review: approved → send, rejected → END."""
        status = state.get("human_approval_status", "approved")
        if status == "rejected":
            return "rejected"
        return "approved"

    # ─── System Prompt ────────────────────────────────────────────────

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
            "You are a professional appointment setter for a cybersecurity consulting firm. "
            "Your goal is to convert email replies into booked discovery calls. "
            "Be warm, consultative, and never pushy. Mirror the prospect's tone. "
            "Always include a clear call-to-action with specific meeting times when appropriate."
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        Write appointment insights to shared brain.
        Main knowledge writing happens in send_reply node.
        """
        pass  # Handled in _node_send_reply

    def __repr__(self) -> str:
        return (
            f"<AppointmentSetterAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
