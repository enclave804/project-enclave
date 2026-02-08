"""
Agent state schemas for the Sovereign Venture Engine.

Each agent has its own state TypedDict that extends BaseAgentState.
All states include common fields for tracking, logging, and cross-agent knowledge.
"""

from __future__ import annotations

from typing import Any, Optional, Literal

from typing_extensions import TypedDict


class BaseAgentState(TypedDict, total=False):
    """Common state fields shared by all agents."""

    # ---- Agent Identity ----
    agent_id: str
    vertical_id: str
    run_id: str

    # ---- Task ----
    task_input: dict[str, Any]
    task_type: str

    # ---- Control Flow ----
    current_node: str
    error: Optional[str]
    error_node: Optional[str]
    retry_count: int
    started_at: str
    completed_at: Optional[str]

    # ---- Human-in-the-loop ----
    requires_human_approval: bool
    human_approval_status: Optional[
        Literal["approved", "rejected", "edited"]
    ]
    human_feedback: Optional[str]

    # ---- Knowledge ----
    rag_context: list[dict[str, Any]]
    knowledge_written: bool


class OutreachAgentState(BaseAgentState, total=False):
    """State for the Outreach Agent (mirrors existing LeadState)."""

    # ---- Lead Identity ----
    lead_id: str
    company_id: str
    contact_id: str
    company_name: str
    company_domain: str
    contact_name: str
    contact_email: str
    contact_title: str
    contact_seniority: str

    # ---- Enrichment ----
    tech_stack: dict[str, Any]
    vulnerabilities: list[dict[str, Any]]
    company_industry: str
    company_size: int
    enrichment_sources: list[str]

    # ---- Duplicate Check ----
    is_duplicate: bool
    last_contacted_at: Optional[str]
    previous_outreach: list[dict[str, Any]]

    # ---- Qualification ----
    qualification_score: float
    qualified: bool
    matching_signals: list[str]
    matching_disqualifiers: list[str]
    disqualification_reason: Optional[str]

    # ---- Strategy ----
    selected_persona: str
    selected_approach: str
    rag_patterns: list[dict[str, Any]]
    vulnerability_context: list[dict[str, Any]]
    template_id: Optional[str]

    # ---- Draft ----
    draft_email_subject: str
    draft_email_body: str
    draft_reasoning: str

    # ---- Compliance ----
    compliance_passed: bool
    compliance_issues: list[str]
    is_suppressed: bool

    # ---- Human Review ----
    human_review_status: Optional[str]
    edited_subject: Optional[str]
    edited_body: Optional[str]
    review_attempts: int

    # ---- Sending ----
    email_sent: bool
    sending_provider_id: Optional[str]
    sent_at: Optional[str]

    # ---- Pipeline Control ----
    skip_reason: Optional[str]
    pipeline_run_id: str


class SEOContentAgentState(BaseAgentState, total=False):
    """State for the SEO Content Agent."""

    # ---- Research ----
    target_keywords: list[str]
    keyword_research_results: list[dict[str, Any]]
    competitor_content: list[dict[str, Any]]
    serp_analysis: dict[str, Any]

    # ---- Content ----
    content_type: str  # blog_post, landing_page, case_study
    content_outline: str
    draft_title: str
    draft_content: str
    seo_score: float
    meta_title: str
    meta_description: str

    # ---- Review ----
    content_approved: bool
    publish_url: Optional[str]


class AppointmentAgentState(BaseAgentState, total=False):
    """State for the Appointment Setter Agent."""

    # ---- Reply Processing ----
    inbound_email: dict[str, Any]
    reply_intent: str  # interested, objection, question, ooo, wrong_person, unsubscribe
    reply_sentiment: str
    conversation_history: list[dict[str, Any]]
    contact_id: str
    company_id: str

    # ---- Objection Handling ----
    objection_type: Optional[str]
    rebuttal_context: list[dict[str, Any]]

    # ---- Response ----
    draft_reply_subject: str
    draft_reply_body: str

    # ---- Scheduling ----
    proposed_times: list[str]
    calendar_link: Optional[str]
    meeting_booked: bool
    meeting_datetime: Optional[str]

    # ---- Follow-up ----
    follow_up_sequence_step: int
    next_follow_up_at: Optional[str]
