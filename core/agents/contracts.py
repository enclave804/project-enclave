"""
Standardized inter-agent data contracts for the Sovereign Venture Engine.

All inter-agent communication passes through these Pydantic models.
This prevents the "one agent says email, another says contact_email" problem.

Usage:
    from core.agents.contracts import LeadData, MeetingRequest, ContentBrief

    # When dispatching to another agent:
    payload = LeadData(
        contact_email="jane@acme.com",
        contact_name="Jane Doe",
        company_domain="acme.com",
    )
    event_bus.dispatch("new_lead_added", payload.model_dump())
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Lead & Contact ───────────────────────────────────────────────────

class LeadData(BaseModel):
    """Standardized lead/contact data passed between agents."""

    contact_email: str
    contact_name: str = ""
    contact_title: str = ""
    contact_seniority: str = ""
    company_name: str = ""
    company_domain: str = ""
    company_industry: str = ""
    company_size: Optional[int] = None
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnrichedLead(LeadData):
    """Lead data after enrichment (extends LeadData)."""

    tech_stack: dict[str, Any] = Field(default_factory=dict)
    vulnerabilities: list[dict[str, Any]] = Field(default_factory=list)
    qualification_score: float = 0.0
    qualified: bool = False
    matching_signals: list[str] = Field(default_factory=list)


# ─── Email & Communication ────────────────────────────────────────────

class EmailPayload(BaseModel):
    """Standardized email data for outreach and reply handling."""

    from_email: str
    to_email: str
    subject: str
    body: str
    direction: str = "outbound"  # outbound, inbound
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplyClassification(BaseModel):
    """Output from reply intent classification."""

    intent: str  # interested, objection, question, ooo, wrong_person, unsubscribe
    sentiment: str = "neutral"  # positive, negative, neutral
    confidence: float = 0.0
    objection_type: Optional[str] = None
    summary: str = ""


# ─── Meetings & Scheduling ────────────────────────────────────────────

class MeetingRequest(BaseModel):
    """Request to book a meeting with a prospect."""

    contact_email: str
    contact_name: str
    company_name: str = ""
    proposed_times: list[str] = Field(default_factory=list)
    meeting_type: str = "discovery"  # discovery, demo, follow_up
    duration_minutes: int = 30
    notes: str = ""


class MeetingBooked(BaseModel):
    """Confirmation of a booked meeting."""

    contact_email: str
    contact_name: str
    meeting_datetime: str
    meeting_url: Optional[str] = None
    calendar_event_id: Optional[str] = None


# ─── Content ──────────────────────────────────────────────────────────

class ContentBrief(BaseModel):
    """Brief for content generation (SEO, blog, case study)."""

    content_type: str  # blog_post, landing_page, case_study, ad_copy
    target_keywords: list[str] = Field(default_factory=list)
    target_topics: list[str] = Field(default_factory=list)
    target_word_count: int = 1500
    tone: str = "professional"
    audience: str = ""
    context: str = ""


class GeneratedContent(BaseModel):
    """Output from content generation agents."""

    title: str
    body: str
    content_type: str
    seo_score: Optional[float] = None
    meta_title: str = ""
    meta_description: str = ""
    keywords_used: list[str] = Field(default_factory=list)


# ─── Shared Insights ─────────────────────────────────────────────────

class InsightData(BaseModel):
    """Standardized insight to write to the shared brain."""

    insight_type: str  # winning_pattern, objection_rebuttal, keyword_performance, market_signal
    title: str = ""
    content: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    related_entity_id: Optional[str] = None
    related_entity_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Task Queue ───────────────────────────────────────────────────────

class TaskPayload(BaseModel):
    """Standardized task payload for cross-agent coordination."""

    task_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    source_context: str = ""  # why this task was created
    priority: int = Field(5, ge=1, le=10)


# ─── Commerce ────────────────────────────────────────────────────────

class OrderData(BaseModel):
    """Standardized order data passed between agents."""

    order_id: str
    customer_email: str
    customer_name: str = ""
    total_price: float
    currency: str = "USD"
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    financial_status: str = ""  # paid, pending, refunded
    fulfillment_status: str = ""  # fulfilled, unfulfilled, partial
    is_vip: bool = False
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaymentEvent(BaseModel):
    """Payment event from Stripe webhook."""

    payment_intent_id: str
    amount_cents: int
    currency: str = "usd"
    status: str  # succeeded, failed, pending
    customer_email: str = ""
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RefundRequest(BaseModel):
    """Request to process a refund."""

    order_id: str
    payment_intent_id: str
    amount_cents: Optional[int] = None  # None = full refund
    reason: str = ""
    customer_email: str = ""
    customer_name: str = ""
    approved: bool = False  # Requires human approval


# ─── Voice & Telephony ──────────────────────────────────────────────

class VoiceMessage(BaseModel):
    """Inbound voice message (voicemail) data."""

    call_sid: str
    caller_number: str
    caller_name: str = ""
    recording_url: str = ""
    recording_duration: int = 0  # seconds
    transcript: str = ""
    classified_intent: str = "unknown"  # sales, support, urgent, unknown
    intent_confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SMSMessage(BaseModel):
    """Inbound/outbound SMS message data."""

    message_sid: str = ""
    from_number: str
    to_number: str
    body: str
    direction: str = "inbound"  # inbound, outbound
    num_media: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Proposals & SOWs ──────────────────────────────────────────────────


class ProposalRequest(BaseModel):
    """Request to generate a proposal/SOW from meeting context."""

    company_name: str
    company_domain: str = ""
    contact_name: str
    contact_email: str
    contact_title: str = ""
    meeting_notes: str = ""
    meeting_date: str = ""  # ISO date
    proposal_type: str = "full_proposal"  # sow, one_pager, executive_summary, full_proposal
    pricing_tier: str = "professional"  # starter, professional, enterprise, custom
    custom_requirements: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedProposal(BaseModel):
    """Output from the Proposal Builder Agent."""

    title: str
    proposal_type: str
    sections: list[dict[str, Any]] = Field(default_factory=list)  # [{title, content, order}]
    full_markdown: str = ""
    pricing_tier: str = ""
    pricing_amount: float = 0.0
    pricing_breakdown: list[dict[str, Any]] = Field(default_factory=list)
    timeline_weeks: int = 0
    deliverables: list[str] = Field(default_factory=list)
    company_name: str = ""
    contact_name: str = ""


# ─── Social Media ──────────────────────────────────────────────────────


class SocialMediaPost(BaseModel):
    """A single social media post for scheduling."""

    platform: str  # "linkedin", "x"
    content: str
    hashtags: list[str] = Field(default_factory=list)
    media_suggestion: str = ""  # Description of ideal media to attach
    post_type: str = "thought_leadership"  # thought_leadership, case_study, industry_news, engagement
    scheduled_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentCalendarEntry(BaseModel):
    """A single entry in the content calendar."""

    date: str  # ISO date
    platform: str
    topic: str
    post_type: str = "thought_leadership"
    status: str = "planned"  # planned, drafted, approved, published


# ─── Ads & Campaigns ──────────────────────────────────────────────────


class AdCampaign(BaseModel):
    """A generated ad campaign definition."""

    platform: str  # "google", "meta", "linkedin"
    campaign_name: str
    objective: str = "lead_gen"
    budget_daily: float = 0.0
    ad_groups: list[dict[str, Any]] = Field(default_factory=list)
    target_audience: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdCreative(BaseModel):
    """A single ad creative (headline + description + CTA)."""

    headline: str
    description: str = ""
    display_url: str = ""
    final_url: str = ""
    call_to_action: str = "Learn More"
    variant_id: str = ""  # For A/B testing
    platform: str = "google"


# ─── Finance & Operations ────────────────────────────────────────────


class InvoiceRequest(BaseModel):
    """Request to generate an invoice from an accepted proposal."""

    proposal_id: str
    company_name: str
    contact_email: str
    contact_name: str = ""
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    total_amount: float = 0.0
    currency: str = "usd"
    due_days: int = 30  # Net 30 default
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvoiceData(BaseModel):
    """A generated/tracked invoice."""

    invoice_id: str  # Stripe invoice ID or mock ID
    proposal_id: str = ""
    company_name: str = ""
    contact_email: str = ""
    total_amount: float = 0.0
    currency: str = "usd"
    status: str = "draft"  # draft, open, paid, overdue, void
    due_date: str = ""  # ISO date
    created_at: str = ""
    paid_at: str = ""
    stripe_url: str = ""  # Hosted invoice URL
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaymentReminder(BaseModel):
    """A payment reminder for an overdue invoice."""

    invoice_id: str
    company_name: str
    contact_email: str
    amount_due: float = 0.0
    days_overdue: int = 0
    tone: str = "polite"  # polite, firm, final
    draft_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClientRecord(BaseModel):
    """A client record for the CS Agent."""

    client_id: str
    company_name: str
    contact_name: str = ""
    contact_email: str = ""
    proposal_id: str = ""
    onboarded_at: str = ""  # ISO date
    last_contact_at: str = ""  # ISO date
    sentiment_score: float = 0.5  # 0-1
    status: str = "active"  # active, at_risk, churned
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── 3D Printing Contracts ────────────────────────────────────────────


class PrintJobRequest(BaseModel):
    """Request to create/process a 3D print job."""

    file_name: str
    file_url: str = ""
    file_format: str = "STL"
    company_name: str = ""
    contact_email: str = ""
    use_case: str = ""
    material_preference: str = ""
    quantity: int = 1
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeometryAnalysis(BaseModel):
    """Output from file analysis — geometry metrics and printability."""

    is_manifold: bool = True
    is_watertight: bool = True
    vertex_count: int = 0
    face_count: int = 0
    volume_cm3: float = 0.0
    surface_area_cm2: float = 0.0
    bounding_box: dict[str, float] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    printability_score: float = 0.0


class MaterialRecommendation(BaseModel):
    """Material + technology recommendation from advisor."""

    material: str
    technology: str
    cost_per_cm3: float = 0.0
    layer_height_um: int = 200
    detail_level: str = "medium"
    reasoning: str = ""
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


class PrintQuote(BaseModel):
    """Generated quote for a print job."""

    quote_id: str = ""
    total_cents: int = 0
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    estimated_days: int = 5
    valid_until: str = ""
    company_name: str = ""
    contact_email: str = ""


# ─── Universal Business Contracts ────────────────────────────────────


class ContractRequest(BaseModel):
    """Request to generate a business contract."""

    contract_type: str = "service_agreement"
    company_name: str = ""
    contact_email: str = ""
    value_cents: int = 0
    start_date: str = ""
    duration_months: int = 12
    custom_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportTicketData(BaseModel):
    """Inbound support ticket data."""

    subject: str
    description: str = ""
    contact_email: str = ""
    company_name: str = ""
    category: str = "general"
    priority: str = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompetitorAlert(BaseModel):
    """Alert from competitive intelligence monitoring."""

    competitor_name: str
    intel_type: str = "news"
    title: str = ""
    content: str = ""
    source_url: str = ""
    severity: str = "info"
    actionable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Phase 21: Universal Business v2 ─────────────────────────────────


class OnboardingRequest(BaseModel):
    """Request to initiate client onboarding."""

    company_name: str
    contact_email: str
    contact_name: str = ""
    opportunity_id: str = ""
    contract_id: str = ""
    template_name: str = "default"
    custom_milestones: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeArticleData(BaseModel):
    """Data for a knowledge base article."""

    title: str
    body_markdown: str = ""
    category: str = "general"
    tags: list[str] = Field(default_factory=list)
    source_type: str = "manual"
    source_ticket_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackSurveyRequest(BaseModel):
    """Request to send a feedback survey."""

    survey_type: str = "nps"
    touchpoint: str = "post_project"
    contact_email: str
    contact_name: str = ""
    company_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponseData(BaseModel):
    """Data from a collected feedback response."""

    contact_email: str
    survey_type: str = "nps"
    nps_score: Optional[int] = None
    csat_score: Optional[int] = None
    comment: str = ""
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReferralData(BaseModel):
    """Data for a client referral submission."""

    referrer_email: str
    referrer_name: str = ""
    referrer_company: str = ""
    referee_email: str
    referee_name: str = ""
    referee_company: str = ""
    referee_domain: str = ""
    source: str = "client_referral"
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DealAnalysisData(BaseModel):
    """Data for a win/loss deal analysis."""

    opportunity_id: str
    outcome: str = "won"
    deal_value_cents: int = 0
    win_loss_factors: list[dict[str, Any]] = Field(default_factory=list)
    competitor_involved: str = ""
    sales_cycle_days: int = 0
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataQualityIssue(BaseModel):
    """Data quality issue detected during CRM scan."""

    target_table: str
    target_id: str = ""
    target_field: str = ""
    issue_type: str = "missing"
    severity: str = "medium"
    description: str = ""
    original_value: str = ""
    suggested_value: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComplianceRecordData(BaseModel):
    """Data for a regulatory compliance record."""

    regulation: str = "gdpr"
    record_type: str = "consent"
    contact_email: str = ""
    consent_given: bool = False
    consent_type: str = ""
    consent_timestamp: str = ""
    retention_expiry: str = ""
    data_categories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)



# --- Phase 22: Project Management ----------------------------------------


class ProjectData(BaseModel):
    """Data for a project tracking record."""

    project_id: str = ""
    project_name: str = ""
    status: str = "active"  # active, on_track, at_risk, blocked, completed
    completion_pct: float = 0.0
    milestones_total: int = 0
    milestones_completed: int = 0
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    health_score: float = 0.0
    owner: str = ""
    due_date: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectPlanData(BaseModel):
    """Data for a project plan with phases, timeline, and risks."""

    plan_id: str = ""
    project_name: str = ""
    phases: list[dict[str, Any]] = Field(default_factory=list)
    timeline_weeks: int = 0
    resources: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    total_risk_score: float = 0.0
    budget_estimate_cents: int = 0
    objectives: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)



# ─── Brand Monitoring ─────────────────────────────────────────────────

class BrandMentionData(BaseModel):
    """Data for a brand mention detected across platforms."""

    platform: str = ""
    source_url: str = ""
    author: str = ""
    mention_text: str = ""
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    brand_health_score: float = 0.0
    alert_triggered: bool = False
    alert_type: str = ""
    detected_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Threat Intelligence ─────────────────────────────────────────────

class ThreatIntelData(BaseModel):
    """Data for a threat intelligence finding."""

    threat_type: str = ""
    severity: str = "medium"
    severity_score: float = 0.0
    cve_id: str = ""
    ioc_indicators: list[str] = Field(default_factory=list)
    affected_systems: list[str] = Field(default_factory=list)
    mitigation: str = ""
    source_feed: str = ""
    detected_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)



# ─── Phase 22: Universal + Creative ──────────────────────────────────


class ProjectData(BaseModel):
    """Data for a tracked project."""

    project_name: str
    status: str = "not_started"
    priority: str = "medium"
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    milestones: list[dict[str, Any]] = Field(default_factory=list)
    assigned_to: str = ""
    due_date: str = ""
    completion_pct: float = 0.0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectPlanData(BaseModel):
    """Data for a project plan."""

    project_name: str
    scope: str = ""
    objectives: list[str] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    budget_estimate_cents: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrandMentionData(BaseModel):
    """Data for a brand mention detection."""

    source: str
    platform: str = ""
    content_snippet: str = ""
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    author: str = ""
    mention_url: str = ""
    detected_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreatIntelData(BaseModel):
    """Data for a threat intelligence item."""

    threat_type: str = "vulnerability"
    severity: str = "medium"
    source_feed: str = ""
    cve_id: str = ""
    cvss_score: float = 0.0
    ioc_type: str = ""
    ioc_value: str = ""
    affected_systems: list[str] = Field(default_factory=list)
    advisory_text: str = ""
    mitigation_steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebDesignBrief(BaseModel):
    """Data for a web design brief."""

    site_type: str = "landing_page"
    project_name: str = ""
    pages: list[dict[str, Any]] = Field(default_factory=list)
    brand_colors: list[str] = Field(default_factory=list)
    typography: dict[str, str] = Field(default_factory=dict)
    target_audience: str = ""
    content_sections: list[dict[str, Any]] = Field(default_factory=list)
    responsive: bool = True
    framework: str = "html5"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphicDesignBrief(BaseModel):
    """Data for a graphic design brief."""

    asset_type: str = "logo"
    project_name: str = ""
    dimensions: str = ""
    brand_colors: list[str] = Field(default_factory=list)
    style_guide: dict[str, Any] = Field(default_factory=dict)
    text_content: str = ""
    usage_context: str = ""
    file_formats: list[str] = Field(default_factory=lambda: ["png"])
    metadata: dict[str, Any] = Field(default_factory=dict)
