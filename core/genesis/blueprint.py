"""
Genesis Blueprint — The DNA of a business vertical.

Pydantic models that represent the complete strategic specification for
a new business vertical. These models capture everything the ConfigGenerator
needs to produce valid VerticalConfig + AgentInstanceConfig YAML files.

The models are designed to:
1. Be serializable to/from JSON (for storage in genesis_sessions table)
2. Contain all information needed to generate valid YAML configs
3. Validate at construction time — invalid blueprints cannot exist
4. Be human-readable when displayed in the dashboard

Data flow:
    ArchitectAgent interview → BusinessContext (raw answers)
    ArchitectAgent analysis  → BusinessBlueprint (strategic plan)
    ConfigGenerator          → YAML files (validated against VerticalConfig)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BlueprintStatus(str, Enum):
    """Lifecycle stages of a business blueprint."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    CONFIG_GENERATED = "config_generated"
    LAUNCHED = "launched"
    ARCHIVED = "archived"


class AgentRole(str, Enum):
    """Standard agent types the platform supports."""
    OUTREACH = "outreach"
    SEO_CONTENT = "seo_content"
    APPOINTMENT_SETTER = "appointment_setter"
    JANITOR = "janitor"
    COMMERCE = "commerce"


class IntegrationType(str, Enum):
    """Types of external integrations a vertical may need."""
    EMAIL_PROVIDER = "email_provider"
    CRM = "crm"
    LEAD_DATABASE = "lead_database"
    ENRICHMENT = "enrichment"
    ANALYTICS = "analytics"
    CALENDAR = "calendar"
    PAYMENT = "payment"
    SOCIAL_MEDIA = "social_media"
    DOMAIN_SPECIFIC = "domain_specific"


class ComplianceJurisdiction(str, Enum):
    """Legal jurisdictions for outreach compliance."""
    US_CAN_SPAM = "US_CAN_SPAM"
    EU_GDPR = "EU_GDPR"
    UK_PECR = "UK_PECR"
    CA_CASL = "CA_CASL"


# ---------------------------------------------------------------------------
# Sub-models: Business Context (Interview Outputs)
# ---------------------------------------------------------------------------

class BusinessContext(BaseModel):
    """
    Raw business context gathered during the interview stage.

    This is what the ArchitectAgent collects from the user through
    adaptive Q&A. It's the INPUT to blueprint generation.
    """
    # Core identity
    business_name: str = Field(
        ..., min_length=1, max_length=200,
        description="The name of the business",
    )
    business_description: str = Field(
        ..., min_length=10, max_length=2000,
        description="What the business does, in the user's own words",
    )
    website: Optional[str] = Field(
        None, description="Business website URL (if exists)",
    )
    region: str = Field(
        "United States",
        description="Primary operating region / market",
    )

    # Business model
    business_model: str = Field(
        ..., min_length=5, max_length=500,
        description="How the business makes money (B2B, B2C, SaaS, service, etc.)",
    )
    price_range: tuple[int, int] = Field(
        ..., description="Min and max ticket price in USD",
    )
    sales_cycle_days: int = Field(
        30, ge=1, le=730,
        description="Expected average sales cycle in days",
    )
    currency: str = Field("USD", max_length=3)

    # Target market
    target_industries: list[str] = Field(
        ..., min_length=1,
        description="Industries this business sells to",
    )
    target_company_sizes: tuple[int, int] = Field(
        ..., description="Target company employee count range",
    )
    target_titles: list[str] = Field(
        ..., min_length=1,
        description="Job titles of the decision makers",
    )
    target_locations: list[str] = Field(
        default_factory=lambda: ["United States"],
        description="Geographic locations to target",
    )

    # Value proposition
    pain_points: list[str] = Field(
        ..., min_length=1,
        description="Customer pain points this business solves",
    )
    value_propositions: list[str] = Field(
        ..., min_length=1,
        description="Key value propositions",
    )
    differentiators: list[str] = Field(
        default_factory=list,
        description="What makes this business unique vs. competitors",
    )

    # Outreach
    sending_domain: Optional[str] = Field(
        None, description="Email domain for outreach (e.g., 'mail.company.com')",
    )
    reply_to_email: Optional[str] = Field(
        None, description="Reply-to email address",
    )
    physical_address: Optional[str] = Field(
        None, description="Physical address for CAN-SPAM compliance",
    )
    daily_outreach_limit: int = Field(
        25, ge=1, le=100,
        description="Max emails per day",
    )

    # Enrichment & signals
    positive_signals: list[str] = Field(
        default_factory=list,
        description="Signals that indicate a lead is a good fit",
    )
    disqualifiers: list[str] = Field(
        default_factory=list,
        description="Signals that disqualify a lead",
    )

    # Content
    tone: str = Field(
        "Professional, approachable",
        description="Desired communication tone/style",
    )
    content_topics: list[str] = Field(
        default_factory=list,
        description="Blog/content topics aligned with the business",
    )

    @field_validator("price_range")
    @classmethod
    def validate_price_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("price_range min must be <= max")
        if v[0] < 0:
            raise ValueError("price_range values must be non-negative")
        return v

    @field_validator("target_company_sizes")
    @classmethod
    def validate_company_sizes(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("target_company_sizes min must be <= max")
        if v[0] < 1:
            raise ValueError("target_company_sizes min must be >= 1")
        return v


# ---------------------------------------------------------------------------
# Sub-models: Blueprint Components
# ---------------------------------------------------------------------------

class ICPSpec(BaseModel):
    """Ideal Customer Profile specification."""
    company_size: tuple[int, int] = Field(
        ..., description="Employee count range",
    )
    industries: list[str] = Field(
        ..., min_length=1,
        description="Target industries",
    )
    signals: list[str] = Field(
        default_factory=list,
        description="Positive buying signals",
    )
    disqualifiers: list[str] = Field(
        default_factory=list,
        description="Lead disqualification signals",
    )

    @field_validator("company_size")
    @classmethod
    def validate_company_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("company_size min must be <= max")
        return v


class PersonaSpec(BaseModel):
    """A buyer persona within the ICP."""
    id: str = Field(
        ..., pattern=r"^[a-z][a-z0-9_]*$",
        description="Snake_case persona identifier",
    )
    title_patterns: list[str] = Field(
        ..., min_length=1,
        description="Job title patterns to match",
    )
    company_size: tuple[int, int] = Field(
        ..., description="Relevant company size range for this persona",
    )
    approach: str = Field(
        ..., description="Outreach approach/sequence name for this persona",
    )
    seniorities: list[str] = Field(
        default_factory=lambda: ["c_suite", "vp", "director"],
        description="Apollo seniority levels",
    )

    @field_validator("company_size")
    @classmethod
    def validate_company_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("company_size min must be <= max")
        return v


class EmailSequenceSpec(BaseModel):
    """An email sequence for outreach."""
    name: str = Field(..., min_length=1, description="Sequence identifier")
    steps: int = Field(..., ge=1, le=10, description="Number of emails")
    delay_days: list[int] = Field(
        ..., description="Delay in days between each step",
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


class OutreachSpec(BaseModel):
    """Outreach configuration specification."""
    daily_limit: int = Field(25, ge=1, le=100)
    warmup_days: int = Field(14, ge=7)
    sending_domain: str = Field(
        ..., description="Email sending domain",
    )
    reply_to: str = Field(
        ..., description="Reply-to email address",
    )
    sequences: list[EmailSequenceSpec] = Field(
        default_factory=list,
        description="Email sequences for different approaches",
    )
    # Compliance
    jurisdictions: list[ComplianceJurisdiction] = Field(
        default_factory=lambda: [ComplianceJurisdiction.US_CAN_SPAM],
    )
    physical_address: str = Field(
        ..., description="Required by CAN-SPAM",
    )
    exclude_countries: list[str] = Field(
        default_factory=list,
        description="ISO country codes to exclude",
    )


class AgentSpec(BaseModel):
    """Specification for an agent to be created for this vertical."""
    agent_type: AgentRole = Field(
        ..., description="Type of agent (maps to @register_agent_type)",
    )
    name: str = Field(
        ..., description="Human-readable agent name",
    )
    description: str = Field(
        "", description="What this agent does for this vertical",
    )
    enabled: bool = True
    browser_enabled: bool = False
    tools: list[str] = Field(
        default_factory=list,
        description="MCP tool names this agent needs",
    )
    human_gate_nodes: list[str] = Field(
        default_factory=list,
        description="Graph nodes requiring human approval",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific parameters",
    )
    system_prompt_template: Optional[str] = Field(
        None,
        description="Optional system prompt for this agent",
    )


class IntegrationSpec(BaseModel):
    """An external integration the vertical requires."""
    name: str = Field(
        ..., description="Integration name (e.g. 'Apollo', 'Shopify')",
    )
    type: IntegrationType
    env_var: str = Field(
        ..., description="Environment variable name for the API key",
    )
    required: bool = True
    instructions: str = Field(
        "", description="Human-readable setup instructions",
    )


class EnrichmentSourceSpec(BaseModel):
    """An enrichment data source for the vertical."""
    type: str = Field(
        ..., description="'web_scraper', 'api', or 'database'",
    )
    provider: Optional[str] = None
    api_key_env: Optional[str] = None
    targets: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root Model: BusinessBlueprint
# ---------------------------------------------------------------------------

class BusinessBlueprint(BaseModel):
    """
    The complete strategic blueprint for a new business vertical.

    This is the OUTPUT of the ArchitectAgent's analysis and the INPUT
    to the ConfigGenerator. It contains everything needed to produce
    valid VerticalConfig + AgentInstanceConfig YAML files.

    Lifecycle:
        Interview → BusinessContext → ArchitectAgent → BusinessBlueprint
        BusinessBlueprint → ConfigGenerator → YAML files → Launch
    """
    # --- Identity ---
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique blueprint ID",
    )
    vertical_id: str = Field(
        ..., pattern=r"^[a-z][a-z0-9_]*$",
        description="Snake_case vertical identifier (used as directory name)",
    )
    vertical_name: str = Field(
        ..., min_length=1, max_length=200,
        description="Human-readable vertical name",
    )
    industry: str = Field(
        ..., min_length=1,
        description="Industry/sector",
    )
    status: BlueprintStatus = BlueprintStatus.DRAFT

    # --- Source Context ---
    context: BusinessContext = Field(
        ..., description="Original interview answers",
    )

    # --- Strategic Components ---
    icp: ICPSpec = Field(
        ..., description="Ideal Customer Profile",
    )
    personas: list[PersonaSpec] = Field(
        ..., min_length=1,
        description="At least one buyer persona",
    )
    outreach: OutreachSpec = Field(
        ..., description="Outreach strategy and compliance",
    )
    agents: list[AgentSpec] = Field(
        ..., min_length=1,
        description="Agents to deploy for this vertical",
    )
    integrations: list[IntegrationSpec] = Field(
        default_factory=list,
        description="External integrations required",
    )
    enrichment_sources: list[EnrichmentSourceSpec] = Field(
        default_factory=list,
        description="Data enrichment sources",
    )

    # --- AI Reasoning ---
    strategy_reasoning: str = Field(
        "", description="Why the AI chose this configuration",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risks and mitigations",
    )
    success_metrics: list[str] = Field(
        default_factory=list,
        description="KPIs to track success",
    )

    # --- Metadata ---
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    created_by: str = Field(
        "architect_agent",
        description="Who/what created this blueprint",
    )
    version: int = Field(
        1, ge=1,
        description="Blueprint version (incremented on revision)",
    )

    # --- Apollo Config ---
    apollo_filters: Optional[dict[str, Any]] = Field(
        None,
        description="Apollo.io search filter overrides",
    )

    # --- Content / SEO ---
    content_topics: list[str] = Field(
        default_factory=list,
        description="Target content/SEO topics",
    )
    tone: str = Field(
        "Professional, approachable",
        description="Communication tone/style",
    )

    model_config = ConfigDict(use_enum_values=True)

    # --- Validators ---

    @model_validator(mode="after")
    def validate_agent_types(self) -> "BusinessBlueprint":
        """Ensure all agents have valid and unique IDs."""
        agent_types = [a.agent_type for a in self.agents]
        # At minimum, a business vertical should have an outreach agent
        if AgentRole.OUTREACH.value not in agent_types:
            raise ValueError(
                "Blueprint must include at least an 'outreach' agent. "
                "This is the core revenue-generating agent."
            )
        return self

    @model_validator(mode="after")
    def validate_persona_approaches(self) -> "BusinessBlueprint":
        """
        Ensure persona approaches match sequence names (if sequences defined).
        """
        if self.outreach.sequences:
            sequence_names = {s.name for s in self.outreach.sequences}
            for persona in self.personas:
                if persona.approach not in sequence_names:
                    raise ValueError(
                        f"Persona '{persona.id}' references approach "
                        f"'{persona.approach}' but no matching sequence "
                        f"exists. Available: {sequence_names}"
                    )
        return self

    # --- Utility Methods ---

    def get_required_env_vars(self) -> list[str]:
        """Return all environment variables this vertical needs."""
        env_vars = [
            "ANTHROPIC_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_KEY",
            "APOLLO_API_KEY",
        ]
        for integration in self.integrations:
            if integration.env_var not in env_vars:
                env_vars.append(integration.env_var)
        for source in self.enrichment_sources:
            if source.api_key_env and source.api_key_env not in env_vars:
                env_vars.append(source.api_key_env)
        return sorted(env_vars)

    def get_agent_by_type(self, agent_type: str) -> Optional[AgentSpec]:
        """Find an agent spec by its type (accepts string or enum)."""
        for agent in self.agents:
            agent_val = (
                agent.agent_type.value
                if hasattr(agent.agent_type, "value")
                else str(agent.agent_type)
            )
            if agent_val == agent_type or agent.agent_type == agent_type:
                return agent
        return None

    def to_summary(self) -> dict[str, Any]:
        """Produce a human-readable summary for dashboard display."""
        return {
            "vertical_id": self.vertical_id,
            "vertical_name": self.vertical_name,
            "industry": self.industry,
            "status": self.status,
            "target_industries": self.icp.industries,
            "num_personas": len(self.personas),
            "num_agents": len(self.agents),
            "agent_types": [
                a.agent_type.value if hasattr(a.agent_type, "value") else str(a.agent_type)
                for a in self.agents
            ],
            "num_integrations": len(self.integrations),
            "required_env_vars": self.get_required_env_vars(),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Factory: Build blueprint from BusinessContext
# ---------------------------------------------------------------------------

def generate_vertical_id(business_name: str) -> str:
    """
    Convert a business name to a valid vertical_id (snake_case).

    Examples:
        "PrintBiz" → "print_biz"
        "Enclave Guard" → "enclave_guard"
        "My 3D Printing Co." → "my_3d_printing_co"
    """
    s = business_name.strip()
    # Insert underscore before uppercase letters (CamelCase → Camel_Case)
    # Only between lowercase→uppercase to avoid splitting "3D" into "3_D"
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    # Lowercase
    s = s.lower()
    # Replace non-alnum with underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Remove leading/trailing underscores
    s = s.strip("_")
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # Ensure starts with letter
    if s and not s[0].isalpha():
        s = "v_" + s
    return s or "unnamed_vertical"
