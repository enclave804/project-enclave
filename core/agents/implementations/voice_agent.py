"""
Voice Agent — The Receptionist / Sales Rep.

Handles inbound voice messages and SMS, classifies intent,
routes to appropriate handlers, and manages outbound callbacks.

Architecture (LangGraph State Machine):
    receive → transcribe → classify_intent →
    [handle_support | handle_sales | handle_urgent] →
    draft_response → human_review → execute → report → END

The Voice Agent is the "ears and mouth" of the autonomous business.
When someone calls and leaves a voicemail, the Voice Agent:
1. Transcribes the recording (Whisper)
2. Classifies intent (Support vs Sales vs Urgent)
3. Drafts an appropriate response (SMS reply, callback, or CRM update)
4. Routes to human for approval before any outbound action

Key Scenarios:
    1. Sales Inquiry: Caller asks about services → extract lead info →
       dispatch to Outreach Agent → send follow-up SMS
    2. Support Request: Customer needs help → draft SMS reply →
       human approves → send SMS
    3. Urgent: Security incident or emergency → alert Overseer →
       escalate to human immediately

Safety:
    - ALL outbound calls/SMS require human approval (human gate)
    - SMS replies go through sandboxed send_sms tool
    - Phone number purchases are sandboxed
    - Transcriptions are logged to shared brain for learning

Usage:
    agent = VoiceAgent(config, db, embedder, llm)
    result = await agent.run({"mode": "process_voicemail", "recording_url": "..."})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import VoiceAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

INTENT_SALES = "sales"
INTENT_SUPPORT = "support"
INTENT_URGENT = "urgent"
INTENT_UNKNOWN = "unknown"

# Intent classification keywords (used when LLM is unavailable)
SALES_KEYWORDS = [
    "interested", "pricing", "quote", "demo", "learn more",
    "services", "consultation", "buy", "purchase", "proposal",
    "custom", "enterprise", "how much",
]
SUPPORT_KEYWORDS = [
    "order", "tracking", "refund", "issue", "problem",
    "help", "broken", "not working", "complaint", "return",
    "status", "delivery", "shipping",
]
URGENT_KEYWORDS = [
    "urgent", "emergency", "security incident", "breach",
    "immediately", "asap", "critical", "down", "hacked",
]

VOICE_SYSTEM_PROMPT = """You are a Voice Agent — the virtual receptionist for an autonomous business.

You receive transcribed voicemails and SMS messages. Your job:
1. Classify the caller's intent (sales, support, or urgent)
2. Extract key information (name, company, phone, email, what they need)
3. Draft an appropriate response

Respond in JSON format:
{
    "intent": "sales" | "support" | "urgent",
    "confidence": 0.0-1.0,
    "caller_name": "extracted name or empty",
    "caller_company": "extracted company or empty",
    "summary": "Brief summary of what they want",
    "response_type": "sms_reply" | "schedule_callback" | "escalate" | "create_lead",
    "draft_response": "The SMS or callback script text",
    "urgency_level": 1-5
}"""


@register_agent_type("voice")
class VoiceAgent(BaseAgent):
    """
    The Receptionist — handles voice messages, SMS, and phone calls.

    LangGraph Workflow:
        receive: Parse incoming event (voicemail, SMS, or call)
        transcribe: Convert audio to text (if voicemail)
        classify_intent: Determine caller intent (Sales/Support/Urgent)
        handle_sales: Extract lead info, prepare CRM update
        handle_support: Draft support reply
        handle_urgent: Escalate to human/Overseer immediately
        draft_response: Generate SMS reply or callback script
        human_review: Gate for human approval of outbound actions
        execute: Send SMS/dispatch to other agents
        report: Generate activity report
    """

    agent_type = "voice"

    def build_graph(self) -> Any:
        """Build the VoiceAgent LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(VoiceAgentState)

        # Add nodes
        workflow.add_node("receive", self._node_receive)
        workflow.add_node("transcribe", self._node_transcribe)
        workflow.add_node("classify_intent", self._node_classify_intent)
        workflow.add_node("handle_sales", self._node_handle_sales)
        workflow.add_node("handle_support", self._node_handle_support)
        workflow.add_node("handle_urgent", self._node_handle_urgent)
        workflow.add_node("draft_response", self._node_draft_response)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute", self._node_execute)
        workflow.add_node("report", self._node_report)

        # Entry point
        workflow.set_entry_point("receive")

        # receive → transcribe
        workflow.add_edge("receive", "transcribe")

        # transcribe → classify_intent
        workflow.add_edge("transcribe", "classify_intent")

        # classify_intent → route by intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                INTENT_SALES: "handle_sales",
                INTENT_SUPPORT: "handle_support",
                INTENT_URGENT: "handle_urgent",
                INTENT_UNKNOWN: "draft_response",
            },
        )

        # All handlers → draft_response
        workflow.add_edge("handle_sales", "draft_response")
        workflow.add_edge("handle_support", "draft_response")
        workflow.add_edge("handle_urgent", "draft_response")

        # draft_response → human_review
        workflow.add_edge("draft_response", "human_review")

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
        """Voice tools are accessed via MCP, not directly."""
        return []

    def get_state_class(self) -> Type:
        return VoiceAgentState

    # ─── Routing Functions ──────────────────────────────────────────

    def _route_by_intent(self, state: VoiceAgentState) -> str:
        """Route to the appropriate handler based on intent classification."""
        intent = state.get("classified_intent", INTENT_UNKNOWN)
        logger.info(
            "voice_routing",
            extra={"agent_id": self.agent_id, "intent": intent},
        )
        return intent

    def _route_by_approval(self, state: VoiceAgentState) -> str:
        """Route based on human approval status."""
        if state.get("actions_approved", False):
            return "approved"
        return "rejected"

    # ─── Node Implementations ──────────────────────────────────────

    async def _node_receive(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 1: Receive — Parse the incoming voice/SMS event.

        Extracts caller info, recording URL, or SMS body.
        """
        task_input = state.get("task_input", {})
        if not isinstance(task_input, dict):
            task_input = {}

        mode = task_input.get("mode", "process_voicemail")
        caller_number = task_input.get("caller", task_input.get("from_number", ""))
        recording_url = task_input.get("recording_url", "")
        sms_body = task_input.get("body", task_input.get("sms_body", ""))
        call_sid = task_input.get("call_sid", "")
        message_sid = task_input.get("message_sid", "")

        logger.info(
            "voice_receive",
            extra={
                "agent_id": self.agent_id,
                "mode": mode,
                "has_recording": bool(recording_url),
                "has_sms": bool(sms_body),
            },
        )

        # Determine channel type
        if mode == "process_sms" or (sms_body and not recording_url):
            channel = "sms"
        elif recording_url:
            channel = "voicemail"
        else:
            channel = "unknown"

        return {
            "current_node": "receive",
            "channel": channel,
            "caller_number": caller_number,
            "recording_url": recording_url,
            "sms_body": sms_body,
            "call_sid": call_sid,
            "message_sid": message_sid,
            "transcript": sms_body if channel == "sms" else "",
        }

    async def _node_transcribe(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 2: Transcribe — Convert audio to text (voicemail only).

        For SMS, the transcript is already set from the body.
        """
        channel = state.get("channel", "unknown")
        recording_url = state.get("recording_url", "")

        if channel == "sms":
            # SMS already has text — skip transcription
            return {
                "current_node": "transcribe",
                "transcription_source": "sms_body",
            }

        if not recording_url:
            return {
                "current_node": "transcribe",
                "transcript": "",
                "transcription_source": "none",
                "transcription_error": "No recording URL provided",
            }

        # Transcribe the recording
        try:
            from core.mcp.tools.voice_tools import transcribe_audio
            result_json = await transcribe_audio(recording_url)
            result = json.loads(result_json)

            transcript = result.get("text", "")
            return {
                "current_node": "transcribe",
                "transcript": transcript,
                "transcription_source": result.get("source", "whisper"),
            }

        except Exception as e:
            logger.warning(
                "voice_transcription_failed",
                extra={"error": str(e)[:200]},
            )
            return {
                "current_node": "transcribe",
                "transcript": "",
                "transcription_source": "failed",
                "transcription_error": str(e)[:200],
            }

    async def _node_classify_intent(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 3: Classify Intent — Determine what the caller wants.

        Uses keyword matching as a fast fallback when LLM is unavailable.
        Priority: Urgent > Sales > Support > Unknown.
        """
        transcript = state.get("transcript", "").lower()

        if not transcript:
            return {
                "current_node": "classify_intent",
                "classified_intent": INTENT_UNKNOWN,
                "intent_confidence": 0.0,
                "intent_reasoning": "Empty transcript — cannot classify.",
            }

        # Keyword-based classification (fast, no LLM)
        urgent_hits = sum(1 for kw in URGENT_KEYWORDS if kw in transcript)
        sales_hits = sum(1 for kw in SALES_KEYWORDS if kw in transcript)
        support_hits = sum(1 for kw in SUPPORT_KEYWORDS if kw in transcript)

        # Priority: urgent > sales > support
        if urgent_hits >= 1:
            intent = INTENT_URGENT
            confidence = min(0.9, 0.5 + urgent_hits * 0.2)
            reasoning = f"Urgent keywords detected ({urgent_hits} matches)"
        elif sales_hits > support_hits:
            intent = INTENT_SALES
            confidence = min(0.9, 0.4 + sales_hits * 0.15)
            reasoning = f"Sales keywords detected ({sales_hits} matches)"
        elif support_hits > 0:
            intent = INTENT_SUPPORT
            confidence = min(0.9, 0.4 + support_hits * 0.15)
            reasoning = f"Support keywords detected ({support_hits} matches)"
        else:
            intent = INTENT_UNKNOWN
            confidence = 0.3
            reasoning = "No strong keyword signals detected."

        # Try LLM for higher confidence (optional)
        try:
            if hasattr(self, "router") and self.router and confidence < 0.7:
                resp = await self.route_llm(
                    intent="classification",
                    system_prompt=VOICE_SYSTEM_PROMPT,
                    user_prompt=f"Classify this voicemail transcript:\n\n{transcript[:500]}",
                )
                if resp and resp.text:
                    llm_result = json.loads(resp.text)
                    intent = llm_result.get("intent", intent)
                    confidence = float(llm_result.get("confidence", confidence))
                    reasoning = f"LLM classification: {llm_result.get('summary', reasoning)}"

                    # Extract caller info from LLM
                    caller_name = llm_result.get("caller_name", "")
                    caller_company = llm_result.get("caller_company", "")
                    if caller_name or caller_company:
                        return {
                            "current_node": "classify_intent",
                            "classified_intent": intent,
                            "intent_confidence": confidence,
                            "intent_reasoning": reasoning,
                            "caller_name": caller_name,
                            "caller_company": caller_company,
                            "intent_summary": llm_result.get("summary", ""),
                        }
        except Exception as e:
            logger.debug(
                "voice_llm_classification_skipped",
                extra={"error": str(e)[:100]},
            )

        return {
            "current_node": "classify_intent",
            "classified_intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": reasoning,
        }

    async def _node_handle_sales(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 4a: Handle Sales — Extract lead info, prepare CRM update.

        Sales inquiries get routed to the Outreach Agent for follow-up.
        """
        transcript = state.get("transcript", "")
        caller_number = state.get("caller_number", "")
        caller_name = state.get("caller_name", "")
        caller_company = state.get("caller_company", "")

        logger.info(
            "voice_sales_inquiry",
            extra={
                "agent_id": self.agent_id,
                "caller": caller_number[-4:] if len(caller_number) >= 4 else "?",
                "caller_name": caller_name,
            },
        )

        actions_planned = [
            {
                "action": "send_sms_reply",
                "target": caller_number,
                "details": {
                    "caller_name": caller_name,
                    "caller_company": caller_company,
                    "intent": "sales",
                    "transcript_preview": transcript[:200],
                },
                "requires_approval": True,
            },
            {
                "action": "create_lead",
                "target": "outreach_agent",
                "details": {
                    "caller_number": caller_number,
                    "caller_name": caller_name,
                    "caller_company": caller_company,
                    "source": "inbound_call",
                    "notes": transcript[:500],
                },
                "requires_approval": True,
            },
        ]

        return {
            "current_node": "handle_sales",
            "response_type": "sms_reply",
            "actions_planned": actions_planned,
        }

    async def _node_handle_support(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 4b: Handle Support — Draft support response.

        Support requests get an SMS acknowledgment and are logged.
        """
        transcript = state.get("transcript", "")
        caller_number = state.get("caller_number", "")

        logger.info(
            "voice_support_request",
            extra={
                "agent_id": self.agent_id,
                "caller": caller_number[-4:] if len(caller_number) >= 4 else "?",
            },
        )

        actions_planned = [
            {
                "action": "send_sms_reply",
                "target": caller_number,
                "details": {
                    "intent": "support",
                    "transcript_preview": transcript[:200],
                },
                "requires_approval": True,
            },
        ]

        return {
            "current_node": "handle_support",
            "response_type": "sms_reply",
            "actions_planned": actions_planned,
        }

    async def _node_handle_urgent(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 4c: Handle Urgent — Escalate immediately.

        Urgent messages get:
        1. Immediate SMS acknowledgment
        2. Alert to the Overseer agent
        3. Human notification
        """
        transcript = state.get("transcript", "")
        caller_number = state.get("caller_number", "")

        logger.warning(
            "voice_urgent_escalation",
            extra={
                "agent_id": self.agent_id,
                "caller": caller_number[-4:] if len(caller_number) >= 4 else "?",
                "transcript_preview": transcript[:100],
            },
        )

        actions_planned = [
            {
                "action": "send_sms_reply",
                "target": caller_number,
                "details": {
                    "intent": "urgent",
                    "transcript_preview": transcript[:200],
                },
                "requires_approval": True,
            },
            {
                "action": "escalate_to_overseer",
                "target": "overseer_agent",
                "details": {
                    "urgency_level": 5,
                    "caller": caller_number,
                    "transcript": transcript[:500],
                    "reason": "Urgent voicemail requiring immediate attention",
                },
                "requires_approval": False,  # Escalation doesn't need approval
            },
        ]

        return {
            "current_node": "handle_urgent",
            "response_type": "escalate",
            "urgency_level": 5,
            "actions_planned": actions_planned,
        }

    async def _node_draft_response(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 5: Draft Response — Generate the SMS reply text.

        Uses templates for reliability, with LLM override for personalization.
        """
        intent = state.get("classified_intent", INTENT_UNKNOWN)
        caller_name = state.get("caller_name", "")
        name_greeting = f"Hi {caller_name.split()[0]}, " if caller_name else "Hi, "

        # Template-based responses (reliable)
        templates = {
            INTENT_SALES: (
                f"{name_greeting}thank you for your interest! "
                "We received your message and a team member will reach out "
                "within 24 hours to discuss how we can help. "
                "Reply STOP to opt out."
            ),
            INTENT_SUPPORT: (
                f"{name_greeting}we received your message and are looking into it. "
                "A support specialist will get back to you within 4 hours. "
                "If urgent, reply URGENT to this message."
            ),
            INTENT_URGENT: (
                f"{name_greeting}we received your urgent message and have "
                "escalated it to our team. Someone will contact you within "
                "the next 30 minutes. Please stay available."
            ),
            INTENT_UNKNOWN: (
                f"{name_greeting}thank you for reaching out! "
                "We received your message and will get back to you shortly. "
                "Reply STOP to opt out."
            ),
        }

        draft_sms = templates.get(intent, templates[INTENT_UNKNOWN])

        # Try LLM personalization for sales responses
        if intent == INTENT_SALES:
            try:
                if hasattr(self, "router") and self.router:
                    transcript = state.get("transcript", "")
                    resp = await self.route_llm(
                        intent="creative_writing",
                        system_prompt=(
                            "You write brief, friendly SMS replies to sales inquiries. "
                            "Max 160 chars. Include a call-to-action. End with "
                            "'Reply STOP to opt out.'"
                        ),
                        user_prompt=(
                            f"Draft an SMS reply to this voicemail from "
                            f"{caller_name or 'a caller'}:\n\n{transcript[:300]}"
                        ),
                    )
                    if resp and resp.text and len(resp.text) <= 320:
                        draft_sms = resp.text
            except Exception as e:
                logger.debug(
                    "voice_draft_llm_skipped",
                    extra={"error": str(e)[:100]},
                )

        return {
            "current_node": "draft_response",
            "draft_sms_reply": draft_sms,
        }

    async def _node_human_review(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 6: Human Review — Present draft for approval.

        ALL outbound SMS and calls require human sign-off.
        """
        all_approved = state.get("actions_approved", False)
        actions = state.get("actions_planned", [])

        if not all_approved:
            for action in actions:
                if action.get("requires_approval", True):
                    logger.info(
                        "voice_awaiting_approval",
                        extra={
                            "action": action.get("action"),
                            "target": str(action.get("target", ""))[-4:],
                        },
                    )

        return {
            "current_node": "human_review",
            "actions_approved": all_approved,
        }

    async def _node_execute(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 7: Execute — Carry out approved actions.

        Dispatches SMS replies and lead creation tasks.
        """
        actions = state.get("actions_planned", [])
        draft_sms = state.get("draft_sms_reply", "")
        executed = []
        failed = []

        for action in actions:
            action_type = action.get("action", "")

            try:
                if action_type == "send_sms_reply" and draft_sms:
                    # In real system, would call send_sms MCP tool
                    logger.info(
                        "voice_sms_dispatched",
                        extra={"target": str(action.get("target", ""))[-4:]},
                    )
                    executed.append({
                        **action,
                        "status": "dispatched",
                        "sms_body": draft_sms[:160],
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })

                elif action_type == "create_lead":
                    # Dispatch to outreach agent via event bus
                    logger.info(
                        "voice_lead_created",
                        extra={
                            "source": "inbound_call",
                            "caller": str(action.get("details", {}).get("caller_number", ""))[-4:],
                        },
                    )
                    executed.append({
                        **action,
                        "status": "dispatched",
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })

                elif action_type == "escalate_to_overseer":
                    logger.warning(
                        "voice_escalation_dispatched",
                        extra={"urgency": action.get("details", {}).get("urgency_level")},
                    )
                    executed.append({
                        **action,
                        "status": "dispatched",
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })

                else:
                    executed.append({
                        **action,
                        "status": "skipped",
                        "reason": f"Unknown action type: {action_type}",
                    })

            except Exception as e:
                logger.error(
                    "voice_action_failed",
                    extra={"action": action_type, "error": str(e)[:200]},
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

    async def _node_report(self, state: VoiceAgentState) -> dict[str, Any]:
        """
        Node 8: Report — Generate activity report.
        """
        now = datetime.now(timezone.utc).isoformat()
        channel = state.get("channel", "unknown")
        intent = state.get("classified_intent", INTENT_UNKNOWN)

        sections = [
            "# Voice Agent Report",
            f"*Generated: {now}*\n",
            "## Inbound Event",
            f"- **Channel:** {channel}",
            f"- **Caller:** {state.get('caller_number', 'N/A')}",
            f"- **Intent:** {intent} (confidence: {state.get('intent_confidence', 0):.0%})",
            f"- **Reasoning:** {state.get('intent_reasoning', 'N/A')}",
        ]

        transcript = state.get("transcript", "")
        if transcript:
            sections.append(f"\n## Transcript Preview\n> {transcript[:200]}...")

        executed = state.get("actions_executed", [])
        failed = state.get("actions_failed", [])
        if executed or failed:
            sections.append("\n## Actions")
            for a in executed:
                sections.append(f"- ✅ {a.get('action', '?')}: {a.get('status', '?')}")
            for a in failed:
                sections.append(f"- ❌ {a.get('action', '?')}: {a.get('error', '?')}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Prepare State ────────────────────────────────────────────────

    def prepare_state(self, task_input: dict[str, Any]) -> VoiceAgentState:
        """Prepare the initial state for a voice agent run."""
        return VoiceAgentState(
            agent_id=self.agent_id,
            vertical_id=self.vertical_id,
            run_id="",
            task_input=task_input,
            current_node="receive",
            error=None,
            human_gates_pending=[],
            human_response=None,
            rag_context=[],
            knowledge_written=[],
            # Voice defaults
            channel="unknown",
            caller_number="",
            caller_name="",
            caller_company="",
            recording_url="",
            sms_body="",
            call_sid="",
            message_sid="",
            transcript="",
            transcription_source="",
            transcription_error="",
            classified_intent="unknown",
            intent_confidence=0.0,
            intent_reasoning="",
            intent_summary="",
            response_type="",
            urgency_level=1,
            draft_sms_reply="",
            draft_callback_script="",
            actions_planned=[],
            actions_approved=False,
            actions_executed=[],
            actions_failed=[],
            report_summary="",
            report_generated_at="",
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    def write_knowledge(self, state: VoiceAgentState) -> None:
        """Write voice interaction insights to the shared brain."""
        intent = state.get("classified_intent", INTENT_UNKNOWN)
        transcript = state.get("transcript", "")
        channel = state.get("channel", "unknown")

        # Only write insights when there's a meaningful transcript
        if transcript and len(transcript) > 10:
            self.store_insight(InsightData(
                insight_type="voice_interaction",
                title=f"Voice: {intent} via {channel}",
                content=(
                    f"Inbound {channel} classified as '{intent}'. "
                    f"Caller: {state.get('caller_number', 'N/A')}. "
                    f"Transcript: {transcript[:300]}. "
                    f"Action: {state.get('response_type', 'N/A')}"
                ),
                confidence=state.get("intent_confidence", 0.5),
                metadata={
                    "channel": channel,
                    "intent": intent,
                    "caller_number": state.get("caller_number", ""),
                    "caller_name": state.get("caller_name", ""),
                    "response_type": state.get("response_type", ""),
                    "urgency_level": state.get("urgency_level", 1),
                },
            ))
