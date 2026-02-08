"""
Pydantic configuration schema for Project Enclave verticals.

Each vertical (Enclave Guard, ChokPro, PrintBiz, etc.) is defined by a
config.yaml file that conforms to these models. The core platform reads
the config and adapts its behavior accordingly — no code changes needed
to launch a new vertical.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Channel(str, Enum):
    EMAIL = "email"
    LINKEDIN = "linkedin"
    PHONE = "phone"
    REFERRAL = "referral"


class LeadStage(str, Enum):
    PROSPECT = "prospect"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class Jurisdiction(str, Enum):
    US_CAN_SPAM = "US_CAN_SPAM"
    EU_GDPR = "EU_GDPR"
    UK_PECR = "UK_PECR"
    CA_CASL = "CA_CASL"


class ChunkType(str, Enum):
    COMPANY_INTEL = "company_intel"
    OUTREACH_RESULT = "outreach_result"
    WINNING_PATTERN = "winning_pattern"
    VULNERABILITY_KNOWLEDGE = "vulnerability_knowledge"
    INDUSTRY_INSIGHT = "industry_insight"
    OBJECTION_HANDLING = "objection_handling"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class BusinessConfig(BaseModel):
    """Core business parameters for the vertical."""
    ticket_range: tuple[int, int] = Field(
        ..., description="Min and max ticket price in the base currency"
    )
    currency: str = "USD"
    sales_cycle_days: int = Field(
        30, description="Expected average sales cycle length"
    )


class TargetPersona(BaseModel):
    """A buyer persona to target within the ICP."""
    id: str
    title_patterns: list[str] = Field(
        ..., description="Job title patterns to match (e.g. 'CTO', 'VP Engineering')"
    )
    company_size: tuple[int, int] = Field(
        ..., description="Employee count range for this persona"
    )
    approach: str = Field(
        ..., description="Default outreach approach ID for this persona"
    )


class ICPConfig(BaseModel):
    """Ideal Customer Profile definition."""
    company_size: tuple[int, int] = Field(
        ..., description="Target employee count range"
    )
    industries: list[str] = Field(
        ..., description="Target industries"
    )
    signals: list[str] = Field(
        default_factory=list,
        description="Positive signals that indicate a good fit",
    )
    disqualifiers: list[str] = Field(
        default_factory=list,
        description="Signals that disqualify a lead",
    )


class TargetingConfig(BaseModel):
    """Who to target and how to segment them."""
    ideal_customer_profile: ICPConfig
    personas: list[TargetPersona] = Field(
        ..., min_length=1, description="At least one persona is required"
    )


class EmailSequence(BaseModel):
    """A multi-step email sequence definition."""
    name: str
    steps: int = Field(..., ge=1, le=10)
    delay_days: list[int] = Field(
        ..., description="Delay in days between each step"
    )

    @field_validator("delay_days")
    @classmethod
    def validate_delay_length(cls, v: list[int], info) -> list[int]:
        steps = info.data.get("steps")
        if steps is not None and len(v) != steps:
            raise ValueError(
                f"delay_days length ({len(v)}) must match steps ({steps})"
            )
        return v


class EmailConfig(BaseModel):
    """Email outreach configuration."""
    daily_limit: int = Field(25, ge=1, le=100)
    warmup_days: int = Field(14, ge=7)
    sending_domain: str
    reply_to: str
    sequences: list[EmailSequence] = Field(default_factory=list)


class ComplianceConfig(BaseModel):
    """Legal compliance settings."""
    jurisdictions: list[Jurisdiction] = Field(
        default_factory=lambda: [Jurisdiction.US_CAN_SPAM]
    )
    unsubscribe_mechanism: str = "one_click_rfc8058"
    physical_address: str = Field(
        ..., description="Required by CAN-SPAM"
    )
    suppress_list_path: Optional[str] = None
    exclude_countries: list[str] = Field(
        default_factory=list,
        description="ISO country codes to exclude from outreach",
    )


class OutreachConfig(BaseModel):
    """All outreach-related configuration."""
    email: EmailConfig
    compliance: ComplianceConfig


class EnrichmentSource(BaseModel):
    """An external data source for lead/company enrichment."""
    type: str = Field(..., description="'web_scraper', 'api', or 'database'")
    provider: Optional[str] = None
    api_key_env: Optional[str] = Field(
        None, description="Env var name holding the API key"
    )
    targets: list[str] = Field(default_factory=list)


class EnrichmentConfig(BaseModel):
    """Configuration for lead enrichment sources."""
    sources: list[EnrichmentSource] = Field(default_factory=list)


class ApolloFilters(BaseModel):
    """Apollo.io search filters derived from ICP."""
    person_titles: list[str] = Field(default_factory=list)
    person_seniorities: list[str] = Field(
        default_factory=lambda: ["c_suite", "vp", "director"]
    )
    organization_num_employees_ranges: list[str] = Field(
        default_factory=list,
        description="Apollo format: ['11,50', '51,200', '201,500']",
    )
    organization_industry_tag_ids: list[str] = Field(default_factory=list)
    person_locations: list[str] = Field(default_factory=list)
    per_page: int = Field(25, ge=1, le=100)


class ApolloConfig(BaseModel):
    """Apollo.io integration settings."""
    api_key_env: str = "APOLLO_API_KEY"
    filters: ApolloFilters = Field(default_factory=ApolloFilters)
    daily_lead_pull: int = Field(
        25, ge=1, le=100, description="Leads to pull per daily run"
    )


class RAGConfig(BaseModel):
    """RAG knowledge base configuration."""
    chunk_types: list[ChunkType] = Field(
        default_factory=lambda: [
            ChunkType.COMPANY_INTEL,
            ChunkType.OUTREACH_RESULT,
            ChunkType.WINNING_PATTERN,
        ]
    )
    seed_data_path: Optional[str] = None
    learning_threshold: int = Field(
        100,
        description="Minimum outreach events before activating the learning loop",
    )


class ModelRouting(BaseModel):
    """Which Claude model to use for each task type."""
    enrichment: str = "claude-haiku-4-5-20250514"
    strategy: str = "claude-sonnet-4-5-20250514"
    email_drafting: str = "claude-sonnet-4-5-20250514"
    classification: str = "claude-haiku-4-5-20250514"
    quarterly_review: str = "claude-opus-4-5-20250514"


class TemperatureConfig(BaseModel):
    """Temperature settings per task type."""
    email_drafting: float = Field(0.7, ge=0.0, le=2.0)
    analysis: float = Field(0.3, ge=0.0, le=2.0)
    classification: float = Field(0.1, ge=0.0, le=2.0)


class AgentConfig(BaseModel):
    """AI agent behavior configuration."""
    model_routing: ModelRouting = Field(default_factory=ModelRouting)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)


class PipelineConfig(BaseModel):
    """Pipeline behavior controls."""
    duplicate_cooldown_days: int = Field(
        90, description="Days to wait before re-contacting a lead"
    )
    max_retries_per_node: int = Field(3, ge=1, le=10)
    human_review_required: bool = Field(
        True, description="Whether every email must be human-approved"
    )
    human_review_score_threshold: Optional[float] = Field(
        None,
        description="If set and human_review_required=False, only review leads above this score",
    )


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class VerticalConfig(BaseModel):
    """
    Complete configuration for a Project Enclave vertical.

    This is the top-level model that gets loaded from config.yaml.
    Every vertical (Enclave Guard, ChokPro, PrintBiz, etc.) has one of these.
    """
    vertical_id: str = Field(
        ..., pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique snake_case identifier for this vertical",
    )
    vertical_name: str = Field(..., description="Human-readable name")
    industry: str

    business: BusinessConfig
    targeting: TargetingConfig
    outreach: OutreachConfig
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    apollo: ApolloConfig = Field(default_factory=ApolloConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    class Config:
        use_enum_values = True
