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
    """
    State for the SEO Content Agent.

    Flow: research_competitors → draft_content → human_review → finalize_and_learn
    """

    # ---- Topic & Keywords ----
    topic: str
    target_keywords: list[str]
    keyword_research_results: list[dict[str, Any]]

    # ---- Competitor Research (Visual Cortex) ----
    competitor_content: list[dict[str, Any]]
    competitor_analysis: list[dict[str, Any]]
    # Each entry: {url, title, summary, design_score, content_gaps, screenshot_path}
    serp_analysis: dict[str, Any]

    # ---- Shared Brain Context ----
    outreach_insights: list[dict[str, Any]]

    # ---- Content ----
    content_type: str  # blog_post, landing_page, case_study
    content_outline: str
    draft_title: str
    draft_content: str
    seo_score: float
    meta_title: str
    meta_description: str
    word_count: int

    # ---- Review ----
    content_approved: bool
    human_edited_content: Optional[str]
    publish_url: Optional[str]

    # ---- RLHF ----
    rlhf_captured: bool


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


class ArchitectAgentState(BaseAgentState, total=False):
    """
    State for the Architect Agent — the meta-agent that drives Genesis.

    Flow: gather_context → analyze_market → generate_blueprint →
          human_review_blueprint → generate_configs → validate_configs →
          request_credentials → launch

    The Architect doesn't do sales work — it *hires* the workers.
    """

    # ---- Interview ----
    conversation_id: str
    interview_phase: str  # InterviewPhase value
    business_context: dict[str, Any]  # Accumulating BusinessContext fields
    questions_asked: list[str]  # Question IDs already asked
    questions_remaining: int
    interview_complete: bool

    # ---- Blueprint ----
    blueprint: Optional[dict[str, Any]]  # Serialized BusinessBlueprint
    blueprint_approved: bool
    blueprint_feedback: Optional[str]  # Human feedback on rejected blueprint
    blueprint_version: int

    # ---- Config Generation ----
    generated_config_paths: list[str]
    generated_vertical_id: Optional[str]
    configs_validated: bool
    validation_errors: list[str]

    # ---- Credentials ----
    required_credentials: list[dict[str, Any]]  # [{name, env_var, required, instructions}]
    credentials_collected: bool

    # ---- Launch ----
    launch_status: Optional[str]  # pending, shadow_mode, live, failed
    launch_errors: list[str]
    launched_agent_ids: list[str]


class OverseerAgentState(BaseAgentState, total=False):
    """
    State for the Overseer Agent — the SRE meta-agent.

    Flow: collect_metrics → diagnose → plan_actions → human_review → execute_actions → report

    The Overseer doesn't do sales work — it *watches the workers*.
    """

    # ---- Health Check Results ----
    system_health: dict[str, Any]          # Full health report from get_system_health()
    health_status: str                      # "healthy", "degraded", "critical"

    # ---- Diagnostics ----
    error_logs: list[dict[str, Any]]        # Recent error/warning logs
    agent_error_rates: dict[str, Any]       # Per-agent failure analysis
    task_queue_status: dict[str, Any]       # Queue depth and zombie count
    cache_performance: dict[str, Any]       # LLM cache hit rates
    knowledge_stats: dict[str, Any]         # Shared brain utilization

    # ---- Issues Detected ----
    issues: list[dict[str, Any]]            # [{severity, component, message, recommended_action}]
    issue_count: int
    critical_count: int

    # ---- Diagnosis ----
    diagnosis: str                          # LLM-generated diagnosis summary
    root_causes: list[str]                  # Identified root causes
    recommended_actions: list[dict[str, Any]]  # [{action, target, priority, reasoning}]

    # ---- Actions ----
    actions_planned: list[dict[str, Any]]   # Actions proposed to human
    actions_approved: bool
    actions_executed: list[dict[str, Any]]  # Actions completed
    actions_failed: list[dict[str, Any]]    # Actions that failed

    # ---- Report ----
    report_summary: str                     # Human-readable status report
    report_generated_at: str


class CommerceAgentState(BaseAgentState, total=False):
    """
    State for the Commerce Agent — the Store Manager.

    Flow: monitor → triage → [handle_vip | handle_low_stock | handle_refund] →
          human_review → execute → report → END

    The Commerce Agent watches the storefront and reacts to orders,
    inventory, and payment events.
    """

    # ---- Order Data ----
    recent_orders: list[dict[str, Any]]     # Orders from last check
    order_count: int
    total_revenue: float
    vip_orders: list[dict[str, Any]]        # Orders above VIP threshold
    vip_count: int

    # ---- Inventory Data ----
    products: list[dict[str, Any]]          # Product catalog snapshot
    low_stock_alerts: list[dict[str, Any]]  # Variants below threshold
    low_stock_count: int

    # ---- Payment Data ----
    pending_payments: list[dict[str, Any]]
    failed_payments: list[dict[str, Any]]

    # ---- Triage Results ----
    triage_action: str                      # "vip_followup", "restock_alert", "refund_review", "routine"
    triage_reasoning: str

    # ---- Actions ----
    actions_planned: list[dict[str, Any]]   # [{action, target, details, requires_approval}]
    actions_approved: bool
    actions_executed: list[dict[str, Any]]
    actions_failed: list[dict[str, Any]]

    # ---- VIP Handling ----
    vip_customer_email: str
    vip_customer_name: str
    vip_order_total: float
    vip_followup_drafted: bool
    vip_email_subject: str
    vip_email_body: str

    # ---- Refund Handling ----
    refund_order_id: str
    refund_amount: float
    refund_reason: str
    refund_approved: bool
    refund_result: dict[str, Any]

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class VoiceAgentState(BaseAgentState, total=False):
    """
    State for the Voice Agent — the Receptionist / Sales Rep.

    Flow: receive → transcribe → classify_intent →
          [handle_sales | handle_support | handle_urgent] →
          draft_response → human_review → execute → report → END

    The Voice Agent handles inbound calls, voicemails, and SMS.
    """

    # ---- Channel Info ----
    channel: str                        # "voicemail", "sms", "unknown"
    caller_number: str
    caller_name: str
    caller_company: str
    recording_url: str
    sms_body: str
    call_sid: str
    message_sid: str

    # ---- Transcription ----
    transcript: str                     # Transcribed text (from Whisper or SMS body)
    transcription_source: str           # "whisper_api", "sms_body", "mock", "none"
    transcription_error: str

    # ---- Intent Classification ----
    classified_intent: str              # "sales", "support", "urgent", "unknown"
    intent_confidence: float
    intent_reasoning: str
    intent_summary: str

    # ---- Response ----
    response_type: str                  # "sms_reply", "schedule_callback", "escalate", "create_lead"
    urgency_level: int                  # 1-5, 5 = most urgent
    draft_sms_reply: str
    draft_callback_script: str

    # ---- Actions ----
    actions_planned: list[dict[str, Any]]
    actions_approved: bool
    actions_executed: list[dict[str, Any]]
    actions_failed: list[dict[str, Any]]

    # ---- Report ----
    report_summary: str
    report_generated_at: str
