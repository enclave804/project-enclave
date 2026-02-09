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


# ══════════════════════════════════════════════════════════════════════
# Phase 11: Growth Agents
# ══════════════════════════════════════════════════════════════════════


class ProposalBuilderAgentState(BaseAgentState, total=False):
    """
    State for the Proposal Builder Agent — The Deal Closer.

    Flow: gather_context → research_company → generate_proposal →
          human_review → finalize → deliver → END

    The Proposal Builder converts meeting notes + enrichment data into
    professional SOWs/proposals that close deals. It reads from the
    shared brain (outreach insights, appointment context, SEO authority)
    to create highly-personalized proposals.
    """

    # ---- Deal Context ----
    company_name: str
    company_domain: str
    contact_name: str
    contact_email: str
    contact_title: str
    meeting_notes: str                     # Notes from the discovery call
    meeting_date: str                      # ISO date of the meeting
    deal_stage: str                        # "discovery", "proposal", "negotiation", "closed"

    # ---- Company Research ----
    company_profile: dict[str, Any]        # Enriched company data
    company_tech_stack: dict[str, Any]     # Known tech stack from enrichment
    company_pain_points: list[str]         # Identified pain points
    competitive_landscape: list[dict[str, Any]]  # Competitor info

    # ---- Proposal Generation ----
    proposal_type: str                     # "sow", "one_pager", "executive_summary", "full_proposal"
    proposal_outline: str                  # AI-generated outline
    proposal_sections: list[dict[str, Any]]  # [{title, content, order}]
    draft_proposal: str                    # Full proposal markdown
    pricing_tier: str                      # "starter", "professional", "enterprise", "custom"
    pricing_amount: float                  # Proposed price
    pricing_breakdown: list[dict[str, Any]]  # [{item, amount, description}]
    timeline_weeks: int                    # Estimated delivery timeline
    deliverables: list[str]               # What the client gets

    # ---- Templates ----
    template_id: str                       # Which template was used
    template_variables: dict[str, Any]     # Variables injected into template

    # ---- Review ----
    proposal_approved: bool
    human_edited_proposal: Optional[str]
    revision_count: int

    # ---- Delivery ----
    delivered: bool
    delivery_method: str                   # "email", "pdf", "link"
    delivery_url: str                      # Link to hosted proposal
    delivered_at: str

    # ---- RLHF ----
    rlhf_captured: bool


class SocialMediaAgentState(BaseAgentState, total=False):
    """
    State for the Social Media Agent — The Brand Builder.

    Flow: analyze_trends → plan_content → generate_posts →
          human_review → schedule → report → END

    The Social Media Agent creates LinkedIn/X content calendars,
    generates posts from outreach insights, and tracks engagement.
    """

    # ---- Content Planning ----
    platform: str                           # "linkedin", "x", "both"
    content_calendar: list[dict[str, Any]]  # [{date, platform, topic, post_type}]
    trending_topics: list[str]
    audience_insights: dict[str, Any]

    # ---- Post Generation ----
    posts_generated: list[dict[str, Any]]  # [{platform, content, hashtags, media_suggestion}]
    post_count: int
    draft_posts: list[dict[str, Any]]      # Posts awaiting review

    # ---- Engagement Data ----
    engagement_metrics: dict[str, Any]     # {impressions, clicks, shares, comments}
    top_performing_posts: list[dict[str, Any]]
    audience_growth: dict[str, Any]

    # ---- Outreach Synergy ----
    outreach_insights: list[dict[str, Any]]  # What messaging resonates from outreach
    seo_keywords: list[str]                  # Keywords from SEO agent for alignment

    # ---- Review ----
    posts_approved: bool
    human_edits: list[dict[str, Any]]

    # ---- Scheduling ----
    scheduled_posts: list[dict[str, Any]]  # [{platform, scheduled_at, post_id}]
    posts_published: int

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class AdsStrategyAgentState(BaseAgentState, total=False):
    """
    State for the Ads Strategy Agent — The Growth Engine.

    Flow: analyze_performance → research_keywords → generate_campaigns →
          human_review → deploy → monitor → report → END

    The Ads Strategy Agent creates Google/Meta ad campaigns, generates
    copy from outreach data, optimizes spend based on conversion data,
    and A/B tests creative using shared brain insights.
    """

    # ---- Campaign Context ----
    platform: str                           # "google", "meta", "linkedin", "both"
    campaign_objective: str                 # "lead_gen", "brand_awareness", "conversions"
    budget_daily: float
    budget_total: float
    target_cpa: float                       # Target cost per acquisition

    # ---- Keyword Research ----
    seed_keywords: list[str]
    keyword_research: list[dict[str, Any]]  # [{keyword, volume, cpc, competition}]
    negative_keywords: list[str]
    selected_keywords: list[str]

    # ---- Ad Copy Generation ----
    ad_groups: list[dict[str, Any]]        # [{name, keywords, ads}]
    generated_ads: list[dict[str, Any]]    # [{headline, description, url, cta}]
    ad_variants: list[dict[str, Any]]      # A/B test variants

    # ---- Audience Targeting ----
    target_audience: dict[str, Any]        # Demographics, interests, behaviors
    lookalike_audiences: list[dict[str, Any]]
    retargeting_rules: list[dict[str, Any]]

    # ---- Performance Data ----
    campaign_performance: dict[str, Any]   # {impressions, clicks, ctr, conversions, cpa, roas}
    ad_performance: list[dict[str, Any]]   # Per-ad metrics
    optimization_suggestions: list[str]

    # ---- Review ----
    campaigns_approved: bool

    # ---- Deployment ----
    deployed_campaigns: list[dict[str, Any]]
    campaigns_active: int

    # ---- Report ----
    report_summary: str
    report_generated_at: str


# ──────────────────────────────────────────────────────────────────────
# Phase 12: Operations Agents
# ──────────────────────────────────────────────────────────────────────


class FinanceAgentState(BaseAgentState, total=False):
    """
    State for the Finance Agent — The CFO.

    Flow: scan_invoices → identify_overdue → draft_reminders →
          human_review → send_reminders → generate_pnl → report → END

    The Finance Agent automates the post-sale money flow: generates
    invoices from accepted proposals, chases overdue payments, and
    calculates monthly P&L from all revenue streams.
    """

    # ---- Invoice Context ----
    active_invoices: list[dict[str, Any]]      # All open invoices
    overdue_invoices: list[dict[str, Any]]     # Unpaid past due date
    paid_invoices: list[dict[str, Any]]        # Recently paid
    new_proposal_ids: list[str]                # Accepted proposals needing invoices

    # ---- Invoice Generation ----
    invoices_to_create: list[dict[str, Any]]   # Proposals → invoice data
    invoices_created: list[dict[str, Any]]     # Created Stripe invoices
    invoices_sent: int

    # ---- Payment Reminders ----
    reminder_drafts: list[dict[str, Any]]      # [{invoice_id, client, amount, tone, draft_text}]
    reminders_approved: bool
    reminders_sent: int

    # ---- Financial Metrics ----
    total_revenue: float                       # All-time or period
    accounts_receivable: float                 # Outstanding invoices
    monthly_recurring: float                   # MRR estimate
    service_revenue: float                     # From proposals/invoices
    commerce_revenue: float                    # From e-commerce
    total_costs: float                         # API costs, infrastructure
    net_profit: float
    pnl_data: dict[str, Any]                   # Full P&L breakdown

    # ---- Review ----
    finance_actions_approved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class CustomerSuccessAgentState(BaseAgentState, total=False):
    """
    State for the Customer Success Agent — The Account Manager.

    Flow: detect_signals → generate_outreach → schedule_followup →
          human_review → execute → report → END

    The CS Agent automates client onboarding, check-ins, and churn
    prevention. Triggered when proposals are accepted or on periodic
    schedule for retention monitoring.
    """

    # ---- Client Context ----
    active_clients: list[dict[str, Any]]       # All active clients
    new_clients: list[dict[str, Any]]          # Recently signed (needs onboarding)
    at_risk_clients: list[dict[str, Any]]      # No contact in 30+ days
    client_id: str                             # Current client being processed

    # ---- Onboarding ----
    onboarding_email_draft: str
    onboarding_checklist: list[str]
    kickoff_meeting_requested: bool
    kickoff_meeting_scheduled: bool
    welcome_packet_sent: bool

    # ---- Check-ins ----
    checkin_type: str                          # "30_day", "60_day", "quarterly", "health_check"
    checkin_drafts: list[dict[str, Any]]       # [{client, email_draft, tone}]
    last_contact_dates: dict[str, str]         # {client_id: ISO date}
    sentiment_scores: dict[str, float]         # {client_id: 0.0-1.0}

    # ---- Retention ----
    churn_risk_score: float                    # 0.0-1.0
    retention_actions: list[dict[str, Any]]    # Suggested actions

    # ---- Review ----
    cs_actions_approved: bool
    human_edits: list[dict[str, Any]]

    # ---- Execution ----
    emails_sent: int
    meetings_scheduled: int

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class AutopilotAgentState(BaseAgentState, total=False):
    """
    State for the Autopilot Agent — The Self-Driver.

    Flow: analyze_system → detect_issues → generate_strategy →
          human_review → execute_strategy → report → END

    The Autopilot is the "self-driving" layer that orchestrates
    autonomous healing, budget optimization, and strategic
    recommendations across the entire platform.
    """

    # ---- System Snapshot ----
    performance_snapshot: dict[str, Any]       # Cross-agent metrics
    budget_snapshot: dict[str, Any]            # Current spend/ROAS
    experiment_snapshot: list[dict[str, Any]]  # Active experiments

    # ---- Diagnosis ----
    detected_issues: list[dict[str, Any]]      # [{agent_id, issue_type, severity, details}]
    health_scores: dict[str, float]            # {agent_id: 0.0-1.0}
    optimization_opportunities: list[dict[str, Any]]

    # ---- Strategy ----
    healing_actions: list[dict[str, Any]]      # Config fixes to apply
    budget_actions: list[dict[str, Any]]       # Budget reallocations
    experiment_proposals: list[dict[str, Any]] # New experiments to launch
    strategy_summary: str                       # LLM-generated strategy text
    strategy_confidence: float                  # 0.0-1.0

    # ---- Actions ----
    actions_planned: list[dict[str, Any]]
    actions_approved: bool
    actions_executed: list[dict[str, Any]]
    actions_failed: list[dict[str, Any]]

    # ---- Session ----
    session_id: str
    session_type: str                           # full_analysis, healing, budget, strategy

    # ---- Report ----
    report_summary: str
    report_generated_at: str


# ══════════════════════════════════════════════════════════════════════
# Phase 16: Sales Pipeline Agents — The Money Machine
# ══════════════════════════════════════════════════════════════════════


class FollowUpAgentState(BaseAgentState, total=False):
    """
    State for the Follow-Up Agent — The Persistence Engine.

    Flow: load_sequences → generate_followups → human_review →
          send_followups → report → END

    The Follow-Up Agent manages multi-touch email sequences,
    ensuring no lead falls through the cracks after initial outreach.
    It reads active sequences, generates personalized follow-up emails,
    and tracks progression through the drip campaign.
    """

    # ---- Sequence Data ----
    active_sequences: list[dict[str, Any]]      # All active sequences
    due_sequences: list[dict[str, Any]]         # Sequences due for next touch
    total_sequences: int

    # ---- Follow-Up Drafts ----
    draft_followups: list[dict[str, Any]]       # [{sequence_id, step, subject, body, contact_email}]
    followups_approved: bool
    human_edits: list[dict[str, Any]]

    # ---- Execution Results ----
    followups_sent: int
    sequences_completed: int                     # Reached max_steps
    sequences_paused: int                        # Paused due to reply/issue
    reply_detected: bool                         # Any sequence got a reply

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class MeetingSchedulerAgentState(BaseAgentState, total=False):
    """
    State for the Meeting Scheduler Agent — The Calendar Manager.

    Flow: check_requests → propose_times → human_review →
          send_invites → report → END

    The Meeting Scheduler handles booking meetings with prospects,
    sending calendar invites, and tracking confirmations. It bridges
    the gap between a warm reply and an actual discovery call.
    """

    # ---- Meeting Requests ----
    pending_requests: list[dict[str, Any]]      # Inbound meeting requests
    total_requests: int

    # ---- Proposed Meetings ----
    proposed_meetings: list[dict[str, Any]]     # [{contact, times, type, duration}]
    meetings_approved: bool
    human_edits: list[dict[str, Any]]

    # ---- Execution Results ----
    invites_sent: int
    meetings_confirmed: int
    meetings_cancelled: int
    calendar_links: list[str]

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class SalesPipelineAgentState(BaseAgentState, total=False):
    """
    State for the Sales Pipeline Agent — The Deal Tracker.

    Flow: scan_pipeline → analyze_deals → recommend_actions →
          human_review → execute_actions → report → END

    The Sales Pipeline Agent monitors all opportunities through their
    lifecycle stages (prospect → qualified → proposal → negotiation →
    closed_won/closed_lost), identifies stalled or at-risk deals,
    and recommends actions to keep the pipeline flowing.
    """

    # ---- Pipeline Data ----
    opportunities: list[dict[str, Any]]         # All active opportunities
    stage_metrics: dict[str, Any]               # {stage: {count, value, avg_age}}
    total_pipeline_value: float                  # Total value in cents

    # ---- Analysis ----
    stalled_deals: list[dict[str, Any]]         # No activity in stale_days
    at_risk_deals: list[dict[str, Any]]         # Likely to be lost
    hot_deals: list[dict[str, Any]]             # High probability close

    # ---- Recommendations ----
    recommended_actions: list[dict[str, Any]]   # [{deal_id, action, reasoning, priority}]
    actions_approved: bool
    actions_executed: list[dict[str, Any]]
    actions_failed: list[dict[str, Any]]

    # ---- Results ----
    deals_moved: int                             # Stage transitions executed
    deals_won: int
    deals_lost: int

    # ---- Report ----
    report_summary: str
    report_generated_at: str
