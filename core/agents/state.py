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


# ══════════════════════════════════════════════════════════════════════
# Phase 18: Domain Expert Agents — Cybersecurity
# ══════════════════════════════════════════════════════════════════════


class VulnScannerAgentState(BaseAgentState, total=False):
    """State for Vulnerability Assessment Agent."""

    # ---- Target ----
    company_id: str
    company_domain: str
    company_name: str
    scan_targets: list[str]                 # domains, IPs to scan

    # ---- Scan Results ----
    scan_type: str                          # "full", "ssl_only", "headers_only", "network"
    ssl_findings: list[dict[str, Any]]
    header_findings: list[dict[str, Any]]
    port_findings: list[dict[str, Any]]
    dns_findings: list[dict[str, Any]]
    all_findings: list[dict[str, Any]]
    risk_score: float                       # 0-10

    # ---- Assessment ----
    assessment_id: str
    findings_saved: int

    # ---- Report ----
    executive_summary: str
    report_approved: bool
    report_delivered: bool
    report_summary: str
    report_generated_at: str


class NetworkAnalystAgentState(BaseAgentState, total=False):
    """State for Network Architecture Analyst."""

    # ---- Target ----
    company_id: str
    company_domain: str
    assessment_id: str

    # ---- Network Data ----
    exposed_services: list[dict[str, Any]]
    open_ports: list[dict[str, Any]]
    dns_records: list[dict[str, Any]]
    subdomains: list[str]
    attack_surface_score: float

    # ---- Analysis ----
    network_topology: dict[str, Any]
    risk_areas: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]

    # ---- Report ----
    analysis_approved: bool
    report_summary: str
    report_generated_at: str


class AppSecReviewerAgentState(BaseAgentState, total=False):
    """State for Application Security Reviewer."""

    # ---- Target ----
    company_id: str
    company_domain: str
    assessment_id: str
    target_urls: list[str]

    # ---- Scan Results ----
    header_analysis: list[dict[str, Any]]
    csp_analysis: dict[str, Any]
    cookie_analysis: list[dict[str, Any]]
    cors_analysis: dict[str, Any]
    tls_analysis: dict[str, Any]
    owasp_checks: list[dict[str, Any]]
    all_findings: list[dict[str, Any]]
    risk_score: float

    # ---- Report ----
    report_approved: bool
    report_summary: str
    report_generated_at: str


class ComplianceMapperAgentState(BaseAgentState, total=False):
    """State for Policy & Compliance Mapper."""

    # ---- Target ----
    company_id: str
    company_domain: str
    assessment_id: str

    # ---- Frameworks ----
    target_frameworks: list[str]            # ['SOC2', 'HIPAA', 'PCI', 'ISO27001']
    framework_requirements: list[dict[str, Any]]

    # ---- Mapping ----
    compliance_gaps: list[dict[str, Any]]
    compliant_controls: list[dict[str, Any]]
    gap_count: int
    compliance_score: float                 # 0-100 percentage

    # ---- Roadmap ----
    remediation_roadmap: list[dict[str, Any]]
    estimated_timeline_weeks: int

    # ---- Report ----
    report_approved: bool
    report_summary: str
    report_generated_at: str


class RiskReporterAgentState(BaseAgentState, total=False):
    """State for Risk Quantification & Executive Reporting."""

    # ---- Target ----
    company_id: str
    company_name: str
    assessment_id: str

    # ---- Input Data (aggregated from all assessments) ----
    all_findings: list[dict[str, Any]]
    compliance_data: dict[str, Any]
    network_data: dict[str, Any]

    # ---- Risk Quantification ----
    overall_risk_score: float
    risk_by_category: dict[str, float]
    estimated_breach_cost: float
    annualized_loss_expectancy: float

    # ---- Report Generation ----
    executive_summary: str
    detailed_sections: list[dict[str, Any]]
    risk_matrix: dict[str, Any]
    priority_actions: list[dict[str, Any]]

    # ---- Delivery ----
    report_format: str                      # 'markdown', 'pdf', 'presentation'
    report_approved: bool
    report_delivered: bool
    report_summary: str
    report_generated_at: str


class IAMAnalystAgentState(BaseAgentState, total=False):
    """State for IAM & Access Control Analyst."""

    # ---- Target ----
    company_id: str
    assessment_id: str

    # ---- Questionnaire ----
    questionnaire_responses: dict[str, Any]
    questionnaire_complete: bool

    # ---- Analysis ----
    mfa_status: dict[str, Any]
    password_policy: dict[str, Any]
    access_review_status: dict[str, Any]
    privileged_accounts: list[dict[str, Any]]
    iam_findings: list[dict[str, Any]]
    iam_risk_score: float

    # ---- Recommendations ----
    recommendations: list[dict[str, Any]]
    report_approved: bool
    report_summary: str
    report_generated_at: str


class IncidentReadinessAgentState(BaseAgentState, total=False):
    """State for Incident Response Readiness."""

    # ---- Target ----
    company_id: str
    assessment_id: str

    # ---- Evaluation ----
    questionnaire_responses: dict[str, Any]
    ir_plan_exists: bool
    ir_plan_score: float                    # 0-100
    backup_strategy: dict[str, Any]
    recovery_procedures: dict[str, Any]
    communication_plan: dict[str, Any]

    # ---- Scoring ----
    readiness_score: float                  # 0-100
    readiness_grade: str                    # A-F
    gaps: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]

    # ---- Report ----
    report_approved: bool
    report_summary: str
    report_generated_at: str


class CloudSecurityAgentState(BaseAgentState, total=False):
    """State for Cloud Security Posture Agent."""

    # ---- Target ----
    company_id: str
    assessment_id: str
    cloud_provider: str                     # 'aws', 'gcp', 'azure', 'multi'

    # ---- Config Analysis (from shared client data) ----
    s3_buckets: list[dict[str, Any]]
    iam_roles: list[dict[str, Any]]
    security_groups: list[dict[str, Any]]
    encryption_status: dict[str, Any]
    logging_status: dict[str, Any]

    # ---- Findings ----
    misconfigurations: list[dict[str, Any]]
    cloud_findings: list[dict[str, Any]]
    cloud_risk_score: float

    # ---- Recommendations ----
    recommendations: list[dict[str, Any]]
    report_approved: bool
    report_summary: str
    report_generated_at: str


class SecurityTrainerAgentState(BaseAgentState, total=False):
    """State for Security Awareness & Human Risk Agent."""

    # ---- Target ----
    company_id: str
    assessment_id: str

    # ---- Training Content ----
    training_modules: list[dict[str, Any]]
    phishing_scenarios: list[dict[str, Any]]
    awareness_topics: list[str]

    # ---- Human Risk Assessment ----
    human_risk_score: float
    risk_areas: list[dict[str, Any]]

    # ---- Delivery ----
    content_approved: bool
    training_delivered: bool
    report_summary: str
    report_generated_at: str


class RemediationGuideAgentState(BaseAgentState, total=False):
    """State for Remediation Guidance & Verification Agent."""

    # ---- Target ----
    company_id: str
    assessment_id: str

    # ---- Findings to Remediate ----
    open_findings: list[dict[str, Any]]
    priority_findings: list[dict[str, Any]]

    # ---- Remediation Plans ----
    remediation_steps: list[dict[str, Any]]  # [{finding_id, steps, estimated_hours, priority}]
    tasks_created: int

    # ---- Verification ----
    findings_to_verify: list[dict[str, Any]]
    verification_results: list[dict[str, Any]]
    findings_remediated: int
    findings_still_open: int

    # ---- Report ----
    progress_report: str
    report_approved: bool
    report_summary: str
    report_generated_at: str


# ══════════════════════════════════════════════════════════════════════
# Phase 19: Domain Expert Agents — PrintBiz (3D Printing)
# ══════════════════════════════════════════════════════════════════════


class FileAnalystAgentState(BaseAgentState, total=False):
    """State for the File Analyst Agent — 3D file geometry analysis."""

    # ---- Job Context ----
    print_job_id: str
    file_name: str
    file_format: str
    file_size_bytes: int
    file_url: str
    customer_id: str
    customer_name: str

    # ---- Geometry Analysis ----
    vertex_count: int
    face_count: int
    edge_count: int
    is_manifold: bool
    is_watertight: bool
    has_inverted_normals: bool
    bounding_box: dict[str, Any]           # {x, y, z} dimensions in mm
    volume_cm3: float
    surface_area_cm2: float
    center_of_mass: dict[str, float]

    # ---- Printability ----
    printability_score: float               # 0-100
    issues: list[dict[str, Any]]           # [{issue, severity, location, description}]
    warnings: list[dict[str, Any]]
    min_wall_thickness_mm: float
    min_detail_mm: float
    overhang_percentage: float

    # ---- Analysis Output ----
    file_analysis_id: str
    analysis_saved: bool
    all_findings: list[dict[str, Any]]

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class MeshRepairAgentState(BaseAgentState, total=False):
    """State for the Mesh Repair Agent — 3D mesh issue resolution."""

    # ---- Input Context ----
    file_analysis_id: str
    print_job_id: str
    file_name: str
    original_issues: list[dict[str, Any]]  # Issues from file analysis

    # ---- Repair Planning ----
    repair_plan: list[dict[str, Any]]      # [{issue, strategy, estimated_impact, priority}]
    repair_plan_summary: str

    # ---- Repair Execution ----
    repairs_applied: list[dict[str, Any]]  # [{issue, strategy, result, vertices_modified}]
    repairs_failed: list[dict[str, Any]]
    repaired_vertex_count: int
    repaired_face_count: int
    post_repair_manifold: bool
    post_repair_watertight: bool

    # ---- Results ----
    issues_resolved: int
    issues_remaining: int
    repair_success_rate: float

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ScaleOptimizerAgentState(BaseAgentState, total=False):
    """State for the Scale Optimizer Agent — architectural scale optimization."""

    # ---- Input Context ----
    file_analysis_id: str
    print_job_id: str
    file_name: str
    original_dimensions: dict[str, float]  # {x, y, z} in mm
    target_scale: str                       # e.g., "1:50", "1:100"
    print_technology: str                   # FDM, SLA, SLS, MJF

    # ---- Scale Analysis ----
    target_scale_factor: float
    scaled_dimensions: dict[str, float]    # Scaled {x, y, z} in mm
    min_feature_size_mm: float             # Smallest detail in model
    tech_min_feature_mm: float             # Min detail for chosen tech
    detail_loss_percentage: float           # How much detail is lost
    features_at_risk: list[dict[str, Any]] # [{feature, original_mm, scaled_mm, printable}]

    # ---- Recommendations ----
    recommended_scale_factor: float
    recommended_technology: str
    scale_adjustments: list[dict[str, Any]]  # [{adjustment, reason, impact}]
    fit_on_build_plate: bool
    build_plate_utilization: float          # 0-100%

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class MaterialAdvisorAgentState(BaseAgentState, total=False):
    """State for the Material Advisor Agent — 3D printing material recommendation."""

    # ---- Input Context ----
    print_job_id: str
    file_analysis_id: str
    customer_requirements: dict[str, Any]  # {detail_level, durability, budget, finish, ...}
    intended_use: str                       # display, functional, prototype, presentation
    volume_cm3: float
    surface_area_cm2: float

    # ---- Material Evaluation ----
    candidate_materials: list[str]
    material_scores: list[dict[str, Any]]  # [{material, score, strengths, weaknesses}]
    technology_options: list[str]

    # ---- Recommendation ----
    recommended_material: str
    recommended_technology: str
    material_cost_estimate: float           # In cents
    layer_height_um: int
    detail_rating: str                      # Low, Medium, High, Very High
    recommendation_reasoning: str

    # ---- Alternatives ----
    alternative_materials: list[dict[str, Any]]  # [{material, tech, cost, tradeoff}]

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class QuoteEngineAgentState(BaseAgentState, total=False):
    """State for the Quote Engine Agent — 3D printing cost calculation and quoting."""

    # ---- Input Context ----
    print_job_id: str
    file_analysis_id: str
    customer_id: str
    customer_name: str
    customer_email: str

    # ---- Geometry Data ----
    volume_cm3: float
    surface_area_cm2: float
    bounding_box: dict[str, float]
    print_technology: str
    material: str

    # ---- Cost Breakdown ----
    material_cost_cents: int
    time_cost_cents: int
    post_process_costs: list[dict[str, Any]]  # [{process, cost_cents}]
    post_process_total_cents: int
    shipping_cost_cents: int
    subtotal_cents: int
    markup_percent: float
    markup_cents: int
    total_cents: int
    estimated_print_hours: float

    # ---- Quote Document ----
    quote_id: str
    quote_document: str                     # Professional formatted quote markdown
    quote_valid_days: int
    quote_saved: bool

    # ---- Delivery ----
    quote_sent: bool
    sent_at: str

    # ---- Report ----
    report_summary: str
    report_generated_at: str


# ──────────────────────────────────────────────────────────────────────
# Phase 19 Batch 2: PrintBiz Operations Agents
# ──────────────────────────────────────────────────────────────────────


class PrintManagerAgentState(BaseAgentState, total=False):
    """State for the Print Manager Agent — 3D print farm job scheduling."""

    # ---- Queue Data ----
    pending_jobs: list[dict[str, Any]]          # Jobs awaiting assignment
    available_printers: list[dict[str, Any]]    # Printers online and idle
    active_prints: list[dict[str, Any]]         # Currently printing jobs
    queue_depth: int

    # ---- Assignment ----
    job_assignments: list[dict[str, Any]]       # [{job_id, printer_id, material, est_hours}]
    assignment_conflicts: list[dict[str, Any]]  # Jobs that couldn't be matched
    assignments_approved: bool

    # ---- Execution ----
    jobs_started: int
    jobs_failed_to_start: int

    # ---- Report ----
    throughput_summary: dict[str, Any]          # {jobs_today, avg_wait_hours, utilization_pct}
    report_summary: str
    report_generated_at: str


class PostProcessAgentState(BaseAgentState, total=False):
    """State for the Post-Processing Agent — finishing workflow management."""

    # ---- Job Context ----
    print_job_id: str
    print_technology: str                       # FDM, SLA, SLS, MJF
    material: str
    finish_requirement: str                     # raw, basic, standard, premium, exhibition

    # ---- Finishing Plan ----
    recommended_steps: list[dict[str, Any]]     # [{step, est_minutes, required, notes}]
    total_estimated_minutes: int
    finish_level_score: int                     # 0-4

    # ---- Work Order ----
    work_order: dict[str, Any]                  # Full structured work order
    work_order_id: str
    work_order_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class QCInspectorAgentState(BaseAgentState, total=False):
    """State for the QC Inspector Agent — quality control and defect detection."""

    # ---- Job Context ----
    print_job_id: str
    file_analysis_id: str
    expected_dimensions: dict[str, float]       # {x, y, z} in mm
    expected_material: str
    expected_technology: str

    # ---- Inspection Results ----
    measured_dimensions: dict[str, float]
    dimensional_deviations: list[dict[str, Any]]
    defects_found: list[dict[str, Any]]         # [{type, severity, description, location}]
    defect_count: int

    # ---- Scoring ----
    dimensional_accuracy_score: float           # 0-100
    surface_quality_score: float
    structural_integrity_score: float
    visual_appearance_score: float
    overall_qc_score: float                     # Weighted composite 0-100
    qc_pass: bool                               # True if above threshold

    # ---- Decision ----
    disposition: str                             # "ship", "rework", "reprint", "scrap"
    disposition_reasoning: str

    # ---- Report ----
    inspection_saved: bool
    report_summary: str
    report_generated_at: str


class CADAdvisorAgentState(BaseAgentState, total=False):
    """State for the CAD Advisor Agent — design-for-manufacturing guidance."""

    # ---- Consultation Context ----
    print_job_id: str
    file_analysis_id: str
    design_file_name: str
    consultation_notes: str
    target_technology: str
    target_scale: str

    # ---- Design Analysis ----
    printability_issues: list[dict[str, Any]]   # [{rule, violation, severity, location}]
    design_warnings: list[dict[str, Any]]
    printability_score: float                    # 0-100

    # ---- Advisory ----
    advisory_report: str                         # Full advisory markdown
    suggestions: list[dict[str, Any]]           # [{category, suggestion, impact, priority}]
    architecture_tips_applied: list[str]

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class LogisticsAgentState(BaseAgentState, total=False):
    """State for the Logistics Agent — packaging and shipping management."""

    # ---- Shipment Context ----
    print_job_id: str
    customer_id: str
    customer_name: str
    customer_email: str
    shipping_address: dict[str, Any]            # {line1, line2, city, state, zip, country}
    package_weight_kg: float
    package_dimensions: dict[str, float]        # {length, width, height} in cm
    is_fragile: bool

    # ---- Packaging ----
    selected_packaging: str                      # packaging type key
    packaging_cost_cents: int
    packaging_notes: str

    # ---- Carrier Selection ----
    carrier_options: list[dict[str, Any]]       # [{carrier, days, cost_cents, service}]
    selected_carrier: str
    shipping_cost_cents: int
    estimated_delivery_days: int

    # ---- Execution ----
    tracking_number: str
    shipment_status: str                         # "pending", "shipped", "in_transit", "delivered"
    shipped_at: str

    # ---- Report ----
    report_summary: str
    report_generated_at: str


# ──────────────────────────────────────────────────────────────────────
# Phase 19 Batch 3: Business Operations Agents
# ──────────────────────────────────────────────────────────────────────


class ContractManagerAgentState(BaseAgentState, total=False):
    """State for the Contract Manager Agent — contract lifecycle management."""

    # ---- Renewal Scanning ----
    expiring_contracts: list[dict[str, Any]]        # Contracts within renewal window
    total_expiring: int
    renewal_window_days: int

    # ---- Contract Generation ----
    contract_template: str                           # Template key
    company_name: str
    company_domain: str
    contact_name: str
    contact_email: str
    opportunity_id: str
    contract_terms: dict[str, Any]                   # Negotiated terms

    # ---- Draft ----
    draft_contract: str                              # Full contract markdown
    contract_type: str                               # service_agreement, msa, nda, sow
    contract_id: str
    contract_saved: bool

    # ---- Delivery ----
    contract_sent: bool
    sent_at: str

    # ---- Signature Tracking ----
    signature_status: str                            # "pending", "signed", "expired"
    signed_at: str

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class SupportAgentState(BaseAgentState, total=False):
    """State for the Support Agent — ticket classification and response drafting."""

    # ---- Ticket Context ----
    ticket_id: str
    ticket_subject: str
    ticket_body: str
    customer_email: str
    customer_name: str
    customer_id: str

    # ---- Classification ----
    category: str                                    # billing, technical, feature_request, bug, general
    priority: str                                    # critical, high, medium, low
    sentiment: str                                   # positive, neutral, negative, angry
    escalation_needed: bool
    classification_reasoning: str

    # ---- Knowledge Search ----
    relevant_articles: list[dict[str, Any]]          # [{title, content, relevance_score}]
    knowledge_sources_checked: int

    # ---- Response ----
    draft_response: str
    response_tone: str                               # empathetic, professional, technical
    suggested_actions: list[str]

    # ---- Execution ----
    response_sent: bool
    ticket_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class CompetitiveIntelAgentState(BaseAgentState, total=False):
    """State for the Competitive Intel Agent — competitor monitoring and analysis."""

    # ---- Scan Targets ----
    monitored_competitors: list[dict[str, Any]]      # [{name, domain, last_checked}]
    scan_results: list[dict[str, Any]]               # Raw scan findings

    # ---- Analysis ----
    intel_findings: list[dict[str, Any]]             # [{type, competitor, finding, severity, source}]
    finding_count: int
    critical_findings: int
    threat_score: float                              # 0-10

    # ---- Alerts ----
    alerts: list[dict[str, Any]]                     # [{competitor, alert_type, message, severity, action}]
    alerts_approved: bool

    # ---- Execution ----
    alerts_sent: int
    intel_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ReportingAgentState(BaseAgentState, total=False):
    """State for the Reporting Agent — business analytics and report generation."""

    # ---- Metrics Collection ----
    pipeline_metrics: dict[str, Any]                 # Pipeline stage data
    revenue_metrics: dict[str, Any]                  # Revenue and MRR
    outreach_metrics: dict[str, Any]                 # Email performance
    client_metrics: dict[str, Any]                   # Client health data
    period_start: str                                # ISO date
    period_end: str                                  # ISO date

    # ---- Trend Analysis ----
    trends: list[dict[str, Any]]                     # [{metric, direction, magnitude, insight}]
    forecasts: list[dict[str, Any]]                  # [{metric, forecast_value, confidence}]
    anomalies: list[dict[str, Any]]                  # [{metric, expected, actual, severity}]

    # ---- Report Generation ----
    report_format: str                               # "executive", "detailed", "weekly", "monthly"
    report_sections: list[dict[str, Any]]            # [{title, content, charts_data}]
    report_document: str                             # Full report markdown
    report_id: str
    report_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


# ══════════════════════════════════════════════════════════════════════
# Phase 21: Universal Business Agents v2
# ══════════════════════════════════════════════════════════════════════


class OnboardingAgentState(BaseAgentState, total=False):
    """
    State for the Onboarding Agent — The Welcome Committee.

    Flow: load_template → build_milestones → generate_welcome_package →
          schedule_kickoff → human_review → execute → track_progress →
          report → END

    The Onboarding Agent manages client onboarding workflows from
    contract signing through project kickoff, tracking milestones
    and ensuring nothing falls through the cracks.
    """

    # ---- Company & Contact ----
    company_name: str
    company_domain: str
    contact_name: str
    contact_email: str
    opportunity_id: str
    contract_id: str

    # ---- Template & Milestones ----
    template_name: str
    milestones: list[dict[str, Any]]
    total_milestones: int
    completed_milestones: int
    completion_percent: float

    # ---- Welcome Package ----
    welcome_package: str
    welcome_package_sent: bool

    # ---- Kickoff ----
    kickoff_scheduled: bool
    kickoff_date: str

    # ---- Persistence ----
    onboarding_id: str
    onboarding_status: str
    onboarding_saved: bool
    stalled_reason: str

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class InvoiceAgentState(BaseAgentState, total=False):
    """
    State for the Invoice Agent — The Billing Engine.

    Flow: generate_invoice → send_invoice → track_payments →
          detect_overdue → draft_reminders → human_review →
          send_reminders → report → END

    The Invoice Agent handles invoice generation from proposals/contracts,
    payment tracking, and automated overdue reminders with escalating tone.
    """

    # ---- Source Context ----
    proposal_id: str
    contract_id: str
    company_name: str
    contact_name: str
    contact_email: str

    # ---- Invoice Data ----
    invoice_id: str
    invoice_number: str
    line_items: list[dict[str, Any]]
    subtotal_cents: int
    tax_cents: int
    total_cents: int
    currency: str
    due_date: str
    due_days: int

    # ---- Payment Tracking ----
    payment_status: str
    paid_at: str
    stripe_invoice_id: str
    stripe_hosted_url: str

    # ---- Overdue & Reminders ----
    overdue_invoices: list[dict[str, Any]]
    reminder_drafts: list[dict[str, Any]]
    reminders_approved: bool
    reminders_sent: int

    # ---- Persistence ----
    invoice_saved: bool
    invoice_sent: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class KnowledgeBaseAgentState(BaseAgentState, total=False):
    """
    State for the Knowledge Base Agent — The Librarian.

    Flow: scan_sources → cluster_topics → identify_gaps →
          generate_articles → human_review → publish → report → END

    The Knowledge Base Agent automatically generates knowledge articles
    from support tickets, agent learnings, and identified gaps, keeping
    the internal and client-facing knowledge base up to date.
    """

    # ---- Source Data ----
    source_tickets: list[dict[str, Any]]
    source_insights: list[dict[str, Any]]
    scan_period_days: int

    # ---- Analysis ----
    topic_clusters: list[dict[str, Any]]
    knowledge_gaps: list[dict[str, Any]]
    total_gaps_found: int

    # ---- Article Generation ----
    draft_articles: list[dict[str, Any]]
    articles_generated: int
    articles_approved: bool
    articles_published: int

    # ---- Persistence ----
    articles_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class FeedbackAgentState(BaseAgentState, total=False):
    """
    State for the Feedback Agent — The Listener.

    Flow: identify_audience → send_surveys → collect_responses →
          analyze_sentiment → route_critical → report → END

    The Feedback Agent manages NPS/CSAT/CES surveys, collects and
    analyzes responses, identifies critical feedback for immediate
    follow-up, and generates sentiment reports.
    """

    # ---- Survey Config ----
    survey_type: str
    touchpoint: str
    target_audience: list[dict[str, Any]]

    # ---- Survey Execution ----
    surveys_to_send: list[dict[str, Any]]
    surveys_sent: int

    # ---- Responses ----
    responses_collected: list[dict[str, Any]]
    avg_nps_score: float
    promoters: int
    passives: int
    detractors: int

    # ---- Analysis ----
    sentiment_summary: str
    critical_feedback: list[dict[str, Any]]
    critical_routed: bool

    # ---- Persistence ----
    feedback_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ReferralAgentState(BaseAgentState, total=False):
    """
    State for the Referral Agent — The Network Builder.

    Flow: identify_happy_clients → generate_referral_requests →
          human_review → send_requests → track_referrals →
          report → END

    The Referral Agent identifies satisfied clients, generates
    personalized referral requests, tracks referral conversions,
    and manages commission payouts.
    """

    # ---- Client Identification ----
    happy_clients: list[dict[str, Any]]
    referral_candidates: list[dict[str, Any]]

    # ---- Referral Requests ----
    referral_requests: list[dict[str, Any]]
    requests_approved: bool

    # ---- Tracking ----
    active_referrals: list[dict[str, Any]]
    total_referrals: int
    converted_referrals: int
    total_commission_cents: int

    # ---- Execution ----
    requests_sent: int

    # ---- Persistence ----
    referrals_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class WinLossAgentState(BaseAgentState, total=False):
    """
    State for the Win/Loss Agent — The Strategist.

    Flow: load_recent_deals → analyze_outcomes → identify_patterns →
          generate_recommendations → report → END

    The Win/Loss Agent analyzes recently closed deals (won and lost),
    identifies common success/failure factors, competitive dynamics,
    and generates actionable recommendations to improve win rates.
    """

    # ---- Deal Data ----
    recent_deals: list[dict[str, Any]]
    analysis_period_days: int

    # ---- Outcome Counts ----
    total_won: int
    total_lost: int

    # ---- Analysis ----
    win_factors: list[dict[str, Any]]
    loss_factors: list[dict[str, Any]]
    competitive_gaps: list[dict[str, Any]]
    avg_sales_cycle_days: float

    # ---- Recommendations ----
    recommendations: list[dict[str, Any]]
    recommendations_count: int

    # ---- Persistence ----
    analyses_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class DataEnrichmentAgentState(BaseAgentState, total=False):
    """
    State for the Data Enrichment Agent — The Data Janitor.

    Flow: scan_tables → detect_issues → find_duplicates →
          enrich_records → human_review → apply_fixes →
          report → END

    The Data Enrichment Agent scans CRM data for quality issues
    (missing fields, invalid emails, duplicates, stale records),
    enriches incomplete records, and maintains data hygiene.
    """

    # ---- Scan Scope ----
    tables_scanned: list[str]
    records_scanned: int
    scan_mode: str

    # ---- Issues ----
    issues_found: list[dict[str, Any]]
    total_issues: int
    critical_issues: int

    # ---- Duplicates ----
    duplicate_groups: list[dict[str, Any]]
    duplicates_found: int

    # ---- Enrichment ----
    enrichment_tasks: list[dict[str, Any]]
    records_enriched: int

    # ---- Fixes ----
    fixes_applied: int
    fixes_approved: bool

    # ---- Persistence ----
    issues_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ComplianceAgentState(BaseAgentState, total=False):
    """
    State for the Compliance Agent — The Regulator.

    Flow: load_regulations → audit_consent → check_retention →
          identify_gaps → generate_findings → human_review →
          apply_actions → generate_report → END

    The Compliance Agent audits data handling practices against
    GDPR, CCPA, CAN-SPAM, and other regulations, tracks consent
    records, monitors retention expiries, and generates compliance
    reports.
    """

    # ---- Regulation Context ----
    active_regulations: list[str]
    regulation_requirements: list[dict[str, Any]]

    # ---- Consent Audit ----
    consent_records: list[dict[str, Any]]
    missing_consent: list[dict[str, Any]]
    total_records_audited: int
    consent_gaps: int

    # ---- Retention ----
    expiring_records: list[dict[str, Any]]
    retention_actions: list[dict[str, Any]]

    # ---- Scoring & Findings ----
    compliance_score: float
    findings: list[dict[str, Any]]
    findings_count: int

    # ---- Actions ----
    actions_approved: bool

    # ---- Persistence ----
    records_saved: bool

    # ---- Report ----
    report_document: str
    report_summary: str
    report_generated_at: str



class ProjectManagerAgentState(BaseAgentState, total=False):
    """
    State for the Project Manager Agent -- The Plan Architect.

    Flow: gather_requirements -> create_plan -> assess_risks ->
          human_review -> report -> END

    The Project Manager Agent creates project plans with phases,
    timelines, resource allocation, and risk assessments from
    gathered requirements and scope definitions.
    """

    # ---- Requirements ----
    requirements: list[dict[str, Any]]               # [{requirement_id, description, priority}]
    objectives: list[str]                            # Project objectives/goals
    scope_summary: str                               # Summarized project scope

    # ---- Plan ----
    plan_phases: list[dict[str, Any]]                # [{phase, tasks, duration_weeks, resources}]
    timeline_weeks: int
    resources: list[dict[str, Any]]                  # [{role, allocation_pct, phase}]
    budget_estimate_cents: int

    # ---- Risks ----
    risks: list[dict[str, Any]]                      # [{risk, severity, likelihood, mitigation}]
    total_risk_score: float
    high_risks: int

    # ---- Plan Approval ----
    plan_approved: bool

    # ---- Persistence ----
    plan_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str



class BrandMonitorAgentState(BaseAgentState, total=False):
    """
    State for the Brand Monitor Agent -- The Reputation Sentinel.

    Flow: scan_mentions -> analyze_sentiment -> generate_alerts ->
          human_review -> report -> END

    The Brand Monitor Agent scans platforms for brand mentions,
    analyzes sentiment, calculates brand health scores, and
    generates alerts for negative or critical mentions.
    """

    # ---- Mentions ----
    mentions: list[dict[str, Any]]
    mention_count: int
    platforms_scanned: list[str]

    # ---- Sentiment ----
    sentiment_results: list[dict[str, Any]]
    brand_health_score: float
    positive_count: int
    negative_count: int
    neutral_count: int

    # ---- Alerts ----
    alerts: list[dict[str, Any]]
    alert_count: int
    alerts_approved: bool

    # ---- Persistence ----
    mentions_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ThreatIntelAgentState(BaseAgentState, total=False):
    """
    State for the Threat Intel Agent -- The Cyber Watchdog.

    Flow: gather_feeds -> analyze_threats -> prioritize_risks ->
          human_review -> report -> END

    The Threat Intel Agent gathers threat intelligence from feeds,
    analyzes CVEs and IOCs, prioritizes risks, and generates
    security advisories with mitigation recommendations.
    """

    # ---- Feeds ----
    feed_data: list[dict[str, Any]]
    feed_count: int
    sources_queried: list[str]

    # ---- Analysis ----
    threat_findings: list[dict[str, Any]]
    threat_count: int
    cve_count: int
    ioc_count: int

    # ---- Prioritization ----
    prioritized_threats: list[dict[str, Any]]
    critical_count: int
    high_count: int
    overall_risk_score: float
    mitigations: list[dict[str, Any]]

    # ---- Review ----
    advisory_approved: bool

    # ---- Persistence ----
    threats_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str



# ============================================================================
# Phase 22: Universal + Creative Agents
# ============================================================================


class ProjectTrackerAgentState(BaseAgentState, total=False):
    """
    State for the Project Tracker Agent — The Taskmaster.

    Flow: load_projects → analyze_status → flag_blockers →
          human_review → report → END

    Tracks project tasks, milestones, blockers, and completion
    metrics across all active projects.
    """

    # ---- Project Data ----
    active_projects: list[dict[str, Any]]
    total_projects: int
    tasks: list[dict[str, Any]]
    total_tasks: int
    completed_tasks: int

    # ---- Milestones ----
    milestones: list[dict[str, Any]]
    upcoming_milestones: list[dict[str, Any]]

    # ---- Blockers ----
    blocked_items: list[dict[str, Any]]
    blockers_count: int
    overdue_tasks: list[dict[str, Any]]
    overdue_count: int

    # ---- Metrics ----
    completion_percent: float
    status_summary: dict[str, Any]

    # ---- Actions ----
    actions_approved: bool

    # ---- Persistence ----
    projects_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ProjectManagerAgentState(BaseAgentState, total=False):
    """
    State for the Project Manager Agent — The Strategist.

    Flow: gather_requirements → create_plan → assess_risks →
          human_review → report → END

    Creates project plans with timelines, risk assessments,
    resource allocation, and budget estimates.
    """

    # ---- Requirements ----
    project_scope: str
    objectives: list[str]
    constraints: list[str]

    # ---- Plan ----
    timeline_items: list[dict[str, Any]]
    total_phases: int
    plan_document: str

    # ---- Risk Assessment ----
    risk_register: list[dict[str, Any]]
    risk_count: int
    high_risks: int
    risk_score: float

    # ---- Resources ----
    resource_allocation: list[dict[str, Any]]
    budget_estimate_cents: int

    # ---- Recommendations ----
    recommendations: list[dict[str, Any]]

    # ---- Actions ----
    plan_approved: bool

    # ---- Persistence ----
    plan_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class BrandMonitorAgentState(BaseAgentState, total=False):
    """
    State for the Brand Monitor Agent — The Sentinel.

    Flow: scan_mentions → analyze_sentiment → generate_alerts →
          human_review → report → END

    Scans social media, news, and web for brand mentions,
    analyzes sentiment, and generates alerts for significant
    positive or negative mentions.
    """

    # ---- Mentions ----
    mentions_found: list[dict[str, Any]]
    total_mentions: int
    new_mentions: int

    # ---- Sentiment ----
    sentiment_breakdown: dict[str, Any]
    avg_sentiment: float
    brand_health_score: float

    # ---- Trending ----
    trending_topics: list[dict[str, Any]]
    competitor_mentions: list[dict[str, Any]]

    # ---- Alerts ----
    alert_items: list[dict[str, Any]]
    alerts_count: int
    alerts_approved: bool

    # ---- Persistence ----
    mentions_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class ThreatIntelAgentState(BaseAgentState, total=False):
    """
    State for the Threat Intelligence Agent — The Watchdog.

    Flow: gather_feeds → analyze_threats → prioritize_risks →
          human_review → report → END

    Gathers threat intelligence from feeds, analyzes CVEs
    and IOCs, prioritizes risks, and generates security
    advisories.
    """

    # ---- Feeds ----
    threat_feeds: list[dict[str, Any]]
    feeds_scanned: int

    # ---- Threats ----
    cve_items: list[dict[str, Any]]
    total_cves: int
    critical_cves: int
    ioc_indicators: list[dict[str, Any]]
    total_iocs: int

    # ---- Risk ----
    risk_scores: list[dict[str, Any]]
    max_risk_score: float
    severity_distribution: dict[str, int]

    # ---- Advisories ----
    advisories: list[dict[str, Any]]
    mitigation_actions: list[dict[str, Any]]

    # ---- Actions ----
    actions_approved: bool

    # ---- Persistence ----
    threats_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class WebDesignerAgentState(BaseAgentState, total=False):
    """
    State for the Web Designer Agent — The Architect.

    Flow: analyze_brief → generate_structure → create_pages →
          human_review → publish → report → END

    Analyzes design briefs, generates site structures, and
    creates actual HTML/CSS pages using MCP tools. Produces
    real web assets stored in storage/web_designs/.
    """

    # ---- Brief ----
    design_brief: dict[str, Any]
    site_type: str
    target_audience: str

    # ---- Structure ----
    site_structure: dict[str, Any]
    pages_planned: list[dict[str, Any]]
    total_pages: int

    # ---- Generation ----
    pages_generated: list[dict[str, Any]]
    generated_count: int
    html_output: str
    css_output: str
    preview_urls: list[str]

    # ---- Review ----
    design_feedback: list[dict[str, Any]]
    revisions_requested: int

    # ---- Publishing ----
    published: bool
    publish_url: str

    # ---- Persistence ----
    designs_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str


class GraphicsDesignerAgentState(BaseAgentState, total=False):
    """
    State for the Graphics Designer Agent — The Artist.

    Flow: analyze_brief → generate_concepts → create_assets →
          human_review → deliver → report → END

    Analyzes design briefs, generates creative concepts, and
    produces actual graphic assets (logos, banners, social graphics)
    using MCP tools. Outputs real image files to storage/graphics/.
    """

    # ---- Brief ----
    design_brief: dict[str, Any]
    asset_type: str
    brand_guidelines: dict[str, Any]

    # ---- Concepts ----
    concepts: list[dict[str, Any]]
    selected_concept: dict[str, Any]

    # ---- Generation ----
    assets_generated: list[dict[str, Any]]
    total_assets: int
    image_paths: list[str]
    variations: list[dict[str, Any]]

    # ---- Review ----
    design_feedback: list[dict[str, Any]]
    revisions_requested: int

    # ---- Delivery ----
    export_formats: list[str]
    delivered: bool
    delivery_paths: list[str]

    # ---- Persistence ----
    designs_saved: bool

    # ---- Report ----
    report_summary: str
    report_generated_at: str
