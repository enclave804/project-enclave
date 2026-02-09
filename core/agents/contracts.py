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
