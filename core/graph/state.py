"""
LangGraph state schema for Project Enclave sales pipeline.

Defines the typed state that flows through every node in the graph.
Each node reads from and writes to this shared state object.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class LeadState(TypedDict, total=False):
    """
    Complete state for processing a single lead through the pipeline.

    This state flows through all nodes:
    pull_leads → check_duplicate → enrich_company → enrich_contact
    → qualify_lead → select_strategy → draft_outreach
    → compliance_check → human_review → send_outreach → write_to_rag
    """

    # ---- Lead Identification ----
    lead_id: str                          # UUID assigned when lead enters pipeline
    company_id: str                       # Supabase company UUID
    contact_id: str                       # Supabase contact UUID
    company_name: str
    company_domain: str
    contact_name: str
    contact_email: str
    contact_title: str

    # ---- Apollo Data ----
    apollo_person_id: str
    apollo_org_id: str
    raw_apollo_data: dict[str, Any]       # preserved for debugging

    # ---- Enrichment Data ----
    tech_stack: dict[str, str]            # {technology: version_or_category}
    vulnerabilities: list[dict[str, Any]] # detected security findings
    company_intel: str                    # summarized company research
    company_industry: str
    company_size: int                     # employee count
    contact_seniority: str
    enrichment_sources: list[str]         # which APIs provided data

    # ---- Duplicate Check ----
    is_duplicate: bool
    last_contacted_at: Optional[str]      # ISO timestamp
    days_since_contact: Optional[int]
    previous_outreach: list[dict[str, Any]]  # past interactions from RAG

    # ---- Qualification ----
    qualification_score: float            # 0.0 to 1.0
    qualified: bool
    disqualification_reason: Optional[str]
    matching_signals: list[str]           # which ICP signals matched
    matching_disqualifiers: list[str]     # which disqualifiers matched

    # ---- Strategy Selection ----
    selected_persona: Optional[str]       # persona.id from config
    selected_approach: Optional[str]      # approach type
    rag_patterns: list[dict[str, Any]]    # relevant winning patterns from RAG
    vulnerability_context: list[dict]     # relevant vulnerability knowledge
    template_id: Optional[str]            # selected template UUID

    # ---- Outreach Draft ----
    draft_email_subject: Optional[str]
    draft_email_body: Optional[str]
    draft_reasoning: Optional[str]        # why the agent chose this approach

    # ---- Compliance ----
    compliance_passed: bool
    compliance_issues: list[str]          # list of compliance problems found
    is_suppressed: bool                   # on do-not-contact list

    # ---- Human Review ----
    human_review_status: Optional[Literal["approved", "rejected", "edited", "skipped"]]
    human_feedback: Optional[str]         # feedback if rejected
    edited_subject: Optional[str]         # human-edited subject
    edited_body: Optional[str]            # human-edited body
    review_attempts: int                  # how many times this was reviewed

    # ---- Sending ----
    email_sent: bool
    sending_provider_id: Optional[str]    # SendGrid/Mailgun message ID
    sent_at: Optional[str]               # ISO timestamp

    # ---- RAG Write-Back ----
    knowledge_written: bool

    # ---- Control Flow ----
    current_node: str                     # which node is executing
    error: Optional[str]                  # error message if any
    error_node: Optional[str]             # which node errored
    retry_count: int                      # retries for current node
    skip_reason: Optional[str]            # why the lead was skipped
    pipeline_run_id: Optional[str]        # for audit logging

    # ---- Config Reference ----
    vertical_id: str                      # which vertical this belongs to


class BatchState(TypedDict, total=False):
    """
    State for batch processing multiple leads.

    The outer graph processes leads in batches. Each lead
    spawns a sub-graph invocation with LeadState.
    """
    vertical_id: str
    batch_id: str
    leads: list[dict[str, Any]]           # raw leads from Apollo
    processed_count: int
    skipped_count: int
    error_count: int
    results: list[dict[str, Any]]         # summary per lead


def create_initial_lead_state(
    lead_data: dict[str, Any],
    vertical_id: str,
) -> LeadState:
    """
    Create an initial LeadState from raw Apollo lead data.

    This is called when a new lead enters the pipeline.
    """
    import uuid

    contact = lead_data.get("contact", {})
    company = lead_data.get("company", {})

    return LeadState(
        lead_id=str(uuid.uuid4()),
        company_name=company.get("name", ""),
        company_domain=company.get("domain", ""),
        contact_name=contact.get("name", ""),
        contact_email=contact.get("email", ""),
        contact_title=contact.get("title", ""),
        apollo_person_id=contact.get("apollo_id", ""),
        apollo_org_id=company.get("apollo_id", ""),
        raw_apollo_data=lead_data,
        tech_stack=company.get("tech_stack", {}),
        vulnerabilities=[],
        company_intel="",
        company_industry=company.get("industry", ""),
        company_size=company.get("employee_count", 0) or 0,
        contact_seniority=contact.get("seniority", ""),
        enrichment_sources=["apollo"],
        is_duplicate=False,
        last_contacted_at=None,
        days_since_contact=None,
        previous_outreach=[],
        qualification_score=0.0,
        qualified=False,
        disqualification_reason=None,
        matching_signals=[],
        matching_disqualifiers=[],
        selected_persona=None,
        selected_approach=None,
        rag_patterns=[],
        vulnerability_context=[],
        template_id=None,
        draft_email_subject=None,
        draft_email_body=None,
        draft_reasoning=None,
        compliance_passed=False,
        compliance_issues=[],
        is_suppressed=False,
        human_review_status=None,
        human_feedback=None,
        edited_subject=None,
        edited_body=None,
        review_attempts=0,
        email_sent=False,
        sending_provider_id=None,
        sent_at=None,
        knowledge_written=False,
        current_node="start",
        error=None,
        error_node=None,
        retry_count=0,
        skip_reason=None,
        pipeline_run_id=None,
        vertical_id=vertical_id,
    )
