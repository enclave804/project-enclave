"""
Tests for Genesis Blueprint — The DNA models.

Validates that BusinessBlueprint and all sub-models:
1. Construct correctly with valid data
2. Reject invalid data with clear errors
3. Serialize to/from JSON (for DB storage)
4. Enforce business rules (outreach agent required, persona-sequence alignment)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

from core.genesis.blueprint import (
    AgentRole,
    AgentSpec,
    BlueprintStatus,
    BusinessBlueprint,
    BusinessContext,
    ComplianceJurisdiction,
    EmailSequenceSpec,
    EnrichmentSourceSpec,
    ICPSpec,
    IntegrationSpec,
    IntegrationType,
    OutreachSpec,
    PersonaSpec,
    generate_vertical_id,
)


# ---------------------------------------------------------------------------
# Fixtures: Reusable building blocks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_context() -> BusinessContext:
    """Minimal valid business context (like interview output)."""
    return BusinessContext(
        business_name="TestBiz",
        business_description="A test business for unit testing the Genesis Engine.",
        region="United States",
        business_model="B2B SaaS — monthly subscription for compliance software",
        price_range=(1000, 5000),
        sales_cycle_days=30,
        target_industries=["Healthcare", "Fintech"],
        target_company_sizes=(10, 500),
        target_titles=["CTO", "VP Engineering"],
        pain_points=["Manual compliance checks", "Audit failures"],
        value_propositions=["Automated compliance", "Real-time monitoring"],
    )


@pytest.fixture
def sample_icp() -> ICPSpec:
    return ICPSpec(
        company_size=(10, 500),
        industries=["Healthcare", "Fintech"],
        signals=["no_compliance_tool", "recent_audit_failure"],
        disqualifiers=["has_compliance_team_10plus"],
    )


@pytest.fixture
def sample_personas() -> list[PersonaSpec]:
    return [
        PersonaSpec(
            id="cto_small",
            title_patterns=["CTO", "VP Engineering"],
            company_size=(10, 200),
            approach="tech_demo",
        ),
        PersonaSpec(
            id="compliance_head",
            title_patterns=["Compliance Officer", "Head of Compliance"],
            company_size=(100, 500),
            approach="compliance_gap",
        ),
    ]


@pytest.fixture
def sample_outreach() -> OutreachSpec:
    return OutreachSpec(
        daily_limit=25,
        warmup_days=14,
        sending_domain="mail.testbiz.com",
        reply_to="hello@testbiz.com",
        sequences=[
            EmailSequenceSpec(name="tech_demo", steps=3, delay_days=[0, 3, 7]),
            EmailSequenceSpec(name="compliance_gap", steps=2, delay_days=[0, 5]),
        ],
        physical_address="123 Test St, Austin TX 78701",
    )


@pytest.fixture
def sample_agents() -> list[AgentSpec]:
    return [
        AgentSpec(
            agent_type=AgentRole.OUTREACH,
            name="Outreach Agent",
            description="B2B outreach for compliance software",
            tools=["apollo_search", "apollo_enrich", "send_email"],
            human_gate_nodes=["send_outreach"],
        ),
        AgentSpec(
            agent_type=AgentRole.SEO_CONTENT,
            name="SEO Content Agent",
            description="Blog posts about compliance topics",
            browser_enabled=True,
            human_gate_nodes=["human_review"],
        ),
    ]


@pytest.fixture
def full_blueprint(
    sample_context, sample_icp, sample_personas, sample_outreach, sample_agents
) -> BusinessBlueprint:
    """A fully populated blueprint for testing."""
    return BusinessBlueprint(
        vertical_id="test_biz",
        vertical_name="TestBiz Compliance",
        industry="Compliance Software",
        context=sample_context,
        icp=sample_icp,
        personas=sample_personas,
        outreach=sample_outreach,
        agents=sample_agents,
        integrations=[
            IntegrationSpec(
                name="Apollo",
                type=IntegrationType.LEAD_DATABASE,
                env_var="APOLLO_API_KEY",
            ),
        ],
        content_topics=["compliance automation", "HIPAA compliance"],
        tone="Professional, technical",
    )


# ---------------------------------------------------------------------------
# BusinessContext Tests
# ---------------------------------------------------------------------------

class TestBusinessContext:
    """Tests for the interview output model."""

    def test_minimal_valid_context(self, sample_context):
        """Should create with all required fields."""
        assert sample_context.business_name == "TestBiz"
        assert sample_context.currency == "USD"
        assert sample_context.daily_outreach_limit == 25

    def test_defaults_applied(self, sample_context):
        """Should fill in sensible defaults."""
        assert sample_context.region == "United States"
        assert sample_context.target_locations == ["United States"]
        assert sample_context.tone == "Professional, approachable"
        assert sample_context.differentiators == []

    def test_rejects_empty_business_name(self):
        """Business name is required."""
        with pytest.raises(Exception):
            BusinessContext(
                business_name="",
                business_description="A test business",
                business_model="B2B SaaS",
                price_range=(1000, 5000),
                target_industries=["Tech"],
                target_company_sizes=(10, 500),
                target_titles=["CTO"],
                pain_points=["Problem"],
                value_propositions=["Solution"],
            )

    def test_rejects_inverted_price_range(self):
        """Max price must be >= min price."""
        with pytest.raises(Exception):
            BusinessContext(
                business_name="TestBiz",
                business_description="A test business",
                business_model="B2B SaaS",
                price_range=(5000, 1000),  # Inverted!
                target_industries=["Tech"],
                target_company_sizes=(10, 500),
                target_titles=["CTO"],
                pain_points=["Problem"],
                value_propositions=["Solution"],
            )

    def test_rejects_negative_price(self):
        """Price must be non-negative."""
        with pytest.raises(Exception):
            BusinessContext(
                business_name="TestBiz",
                business_description="A test business",
                business_model="B2B SaaS",
                price_range=(-100, 1000),
                target_industries=["Tech"],
                target_company_sizes=(10, 500),
                target_titles=["CTO"],
                pain_points=["Problem"],
                value_propositions=["Solution"],
            )

    def test_rejects_inverted_company_sizes(self):
        """Company size min must be <= max."""
        with pytest.raises(Exception):
            BusinessContext(
                business_name="TestBiz",
                business_description="A test business",
                business_model="B2B SaaS",
                price_range=(1000, 5000),
                target_industries=["Tech"],
                target_company_sizes=(500, 10),  # Inverted!
                target_titles=["CTO"],
                pain_points=["Problem"],
                value_propositions=["Solution"],
            )

    def test_rejects_empty_industries(self):
        """At least one target industry required."""
        with pytest.raises(Exception):
            BusinessContext(
                business_name="TestBiz",
                business_description="A test business",
                business_model="B2B SaaS",
                price_range=(1000, 5000),
                target_industries=[],
                target_company_sizes=(10, 500),
                target_titles=["CTO"],
                pain_points=["Problem"],
                value_propositions=["Solution"],
            )

    def test_json_serialization(self, sample_context):
        """Must be serializable for DB storage."""
        json_str = sample_context.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["business_name"] == "TestBiz"
        # Round-trip
        restored = BusinessContext(**parsed)
        assert restored.business_name == sample_context.business_name
        assert restored.price_range == sample_context.price_range


# ---------------------------------------------------------------------------
# ICPSpec Tests
# ---------------------------------------------------------------------------

class TestICPSpec:

    def test_valid_icp(self, sample_icp):
        assert sample_icp.company_size == (10, 500)
        assert len(sample_icp.industries) == 2

    def test_rejects_inverted_company_size(self):
        with pytest.raises(Exception):
            ICPSpec(
                company_size=(500, 10),
                industries=["Tech"],
            )

    def test_defaults(self):
        icp = ICPSpec(company_size=(10, 100), industries=["Tech"])
        assert icp.signals == []
        assert icp.disqualifiers == []


# ---------------------------------------------------------------------------
# PersonaSpec Tests
# ---------------------------------------------------------------------------

class TestPersonaSpec:

    def test_valid_persona(self):
        p = PersonaSpec(
            id="cto_small",
            title_patterns=["CTO"],
            company_size=(10, 100),
            approach="tech_demo",
        )
        assert p.id == "cto_small"
        assert p.seniorities == ["c_suite", "vp", "director"]

    def test_rejects_invalid_id_format(self):
        """Persona IDs must be snake_case starting with letter."""
        with pytest.raises(Exception):
            PersonaSpec(
                id="CTO-Small",  # Invalid: uppercase and hyphen
                title_patterns=["CTO"],
                company_size=(10, 100),
                approach="tech_demo",
            )

    def test_rejects_empty_titles(self):
        with pytest.raises(Exception):
            PersonaSpec(
                id="cto_small",
                title_patterns=[],  # Must have at least 1
                company_size=(10, 100),
                approach="tech_demo",
            )

    def test_custom_seniorities(self):
        p = PersonaSpec(
            id="manager_mid",
            title_patterns=["Manager"],
            company_size=(50, 200),
            approach="efficiency",
            seniorities=["manager", "director"],
        )
        assert p.seniorities == ["manager", "director"]


# ---------------------------------------------------------------------------
# EmailSequenceSpec Tests
# ---------------------------------------------------------------------------

class TestEmailSequenceSpec:

    def test_valid_sequence(self):
        seq = EmailSequenceSpec(
            name="intro_series",
            steps=3,
            delay_days=[0, 3, 7],
        )
        assert seq.steps == 3
        assert len(seq.delay_days) == 3

    def test_rejects_mismatched_delays(self):
        """delay_days length must match steps."""
        with pytest.raises(Exception):
            EmailSequenceSpec(
                name="broken",
                steps=3,
                delay_days=[0, 3],  # Only 2, need 3
            )

    def test_rejects_zero_steps(self):
        with pytest.raises(Exception):
            EmailSequenceSpec(name="zero", steps=0, delay_days=[])


# ---------------------------------------------------------------------------
# OutreachSpec Tests
# ---------------------------------------------------------------------------

class TestOutreachSpec:

    def test_valid_outreach(self, sample_outreach):
        assert sample_outreach.daily_limit == 25
        assert sample_outreach.sending_domain == "mail.testbiz.com"
        assert len(sample_outreach.sequences) == 2

    def test_defaults(self):
        outreach = OutreachSpec(
            sending_domain="mail.test.com",
            reply_to="test@test.com",
            physical_address="123 Main St",
        )
        assert outreach.daily_limit == 25
        assert outreach.warmup_days == 14
        assert outreach.jurisdictions == [ComplianceJurisdiction.US_CAN_SPAM]

    def test_rejects_excessive_daily_limit(self):
        with pytest.raises(Exception):
            OutreachSpec(
                daily_limit=200,  # Max is 100
                sending_domain="mail.test.com",
                reply_to="test@test.com",
                physical_address="123 Main St",
            )


# ---------------------------------------------------------------------------
# AgentSpec Tests
# ---------------------------------------------------------------------------

class TestAgentSpec:

    def test_valid_agent(self):
        agent = AgentSpec(
            agent_type=AgentRole.OUTREACH,
            name="Outreach Agent",
            tools=["apollo_search", "send_email"],
            human_gate_nodes=["send_outreach"],
        )
        assert agent.agent_type == AgentRole.OUTREACH
        assert agent.enabled is True

    def test_defaults(self):
        agent = AgentSpec(
            agent_type=AgentRole.SEO_CONTENT,
            name="SEO Agent",
        )
        assert agent.tools == []
        assert agent.human_gate_nodes == []
        assert agent.browser_enabled is False
        assert agent.params == {}

    def test_custom_params(self):
        agent = AgentSpec(
            agent_type=AgentRole.OUTREACH,
            name="Custom Outreach",
            params={"daily_lead_limit": 50, "custom_flag": True},
        )
        assert agent.params["daily_lead_limit"] == 50


# ---------------------------------------------------------------------------
# IntegrationSpec Tests
# ---------------------------------------------------------------------------

class TestIntegrationSpec:

    def test_valid_integration(self):
        i = IntegrationSpec(
            name="Apollo",
            type=IntegrationType.LEAD_DATABASE,
            env_var="APOLLO_API_KEY",
        )
        assert i.required is True
        assert i.instructions == ""

    def test_optional_integration(self):
        i = IntegrationSpec(
            name="Shopify",
            type=IntegrationType.PAYMENT,
            env_var="SHOPIFY_API_KEY",
            required=False,
            instructions="Create a Shopify store first",
        )
        assert i.required is False


# ---------------------------------------------------------------------------
# BusinessBlueprint Tests
# ---------------------------------------------------------------------------

class TestBusinessBlueprint:
    """Tests for the root blueprint model."""

    def test_full_construction(self, full_blueprint):
        """Should construct with all fields."""
        assert full_blueprint.vertical_id == "test_biz"
        assert full_blueprint.status == BlueprintStatus.DRAFT
        assert len(full_blueprint.agents) == 2
        assert len(full_blueprint.personas) == 2

    def test_default_status(self, full_blueprint):
        assert full_blueprint.status == "draft"

    def test_auto_generated_id(self, full_blueprint):
        """Blueprint ID should be auto-generated UUID."""
        assert len(full_blueprint.id) == 36  # UUID format

    def test_created_at_auto(self, full_blueprint):
        """Timestamp should be auto-set."""
        assert isinstance(full_blueprint.created_at, datetime)
        # Should be UTC
        now = datetime.now(timezone.utc)
        delta = (now - full_blueprint.created_at).total_seconds()
        assert delta < 5  # Created within last 5 seconds

    def test_rejects_missing_outreach_agent(
        self, sample_context, sample_icp, sample_personas, sample_outreach
    ):
        """Every blueprint must include an outreach agent."""
        with pytest.raises(Exception, match="outreach"):
            BusinessBlueprint(
                vertical_id="bad_vertical",
                vertical_name="Bad Vertical",
                industry="Testing",
                context=sample_context,
                icp=sample_icp,
                personas=sample_personas,
                outreach=sample_outreach,
                agents=[
                    AgentSpec(
                        agent_type=AgentRole.SEO_CONTENT,
                        name="SEO Only",
                    ),
                ],
            )

    def test_rejects_invalid_vertical_id(
        self, sample_context, sample_icp, sample_personas,
        sample_outreach, sample_agents,
    ):
        """vertical_id must be snake_case starting with letter."""
        with pytest.raises(Exception):
            BusinessBlueprint(
                vertical_id="Invalid-ID",  # Uppercase + hyphen
                vertical_name="Test",
                industry="Test",
                context=sample_context,
                icp=sample_icp,
                personas=sample_personas,
                outreach=sample_outreach,
                agents=sample_agents,
            )

    def test_rejects_empty_personas(
        self, sample_context, sample_icp, sample_outreach, sample_agents,
    ):
        """At least one persona required."""
        with pytest.raises(Exception):
            BusinessBlueprint(
                vertical_id="test",
                vertical_name="Test",
                industry="Test",
                context=sample_context,
                icp=sample_icp,
                personas=[],
                outreach=sample_outreach,
                agents=sample_agents,
            )

    def test_validates_persona_approach_alignment(
        self, sample_context, sample_icp, sample_outreach,
    ):
        """Persona approaches must match defined sequences."""
        with pytest.raises(Exception, match="approach"):
            BusinessBlueprint(
                vertical_id="test",
                vertical_name="Test",
                industry="Test",
                context=sample_context,
                icp=sample_icp,
                personas=[
                    PersonaSpec(
                        id="cto",
                        title_patterns=["CTO"],
                        company_size=(10, 100),
                        approach="nonexistent_sequence",  # Not in outreach.sequences!
                    ),
                ],
                outreach=sample_outreach,
                agents=[
                    AgentSpec(
                        agent_type=AgentRole.OUTREACH,
                        name="Outreach",
                    ),
                ],
            )

    def test_json_round_trip(self, full_blueprint):
        """Blueprint must survive JSON serialization for DB storage."""
        json_str = full_blueprint.model_dump_json()
        parsed = json.loads(json_str)
        restored = BusinessBlueprint(**parsed)
        assert restored.vertical_id == full_blueprint.vertical_id
        assert restored.vertical_name == full_blueprint.vertical_name
        assert len(restored.agents) == len(full_blueprint.agents)
        assert len(restored.personas) == len(full_blueprint.personas)

    def test_get_required_env_vars(self, full_blueprint):
        """Should collect all required env vars."""
        env_vars = full_blueprint.get_required_env_vars()
        assert "ANTHROPIC_API_KEY" in env_vars
        assert "SUPABASE_URL" in env_vars
        assert "APOLLO_API_KEY" in env_vars

    def test_get_agent_by_type(self, full_blueprint):
        """Should find agents by type."""
        outreach = full_blueprint.get_agent_by_type("outreach")
        assert outreach is not None
        assert outreach.name == "Outreach Agent"

        missing = full_blueprint.get_agent_by_type("nonexistent")
        assert missing is None

    def test_to_summary(self, full_blueprint):
        """Should produce a clean summary dict."""
        summary = full_blueprint.to_summary()
        assert summary["vertical_id"] == "test_biz"
        assert summary["num_agents"] == 2
        assert summary["num_personas"] == 2
        assert "outreach" in summary["agent_types"]

    def test_version_field(self, full_blueprint):
        """Version defaults to 1."""
        assert full_blueprint.version == 1

    def test_enrichment_sources(self):
        """Enrichment sources should be optional."""
        src = EnrichmentSourceSpec(
            type="api",
            provider="shodan",
            api_key_env="SHODAN_API_KEY",
        )
        assert src.provider == "shodan"
        assert src.targets == []

    def test_blueprint_with_enrichment_env_vars(
        self, sample_context, sample_icp, sample_personas,
        sample_outreach, sample_agents,
    ):
        """Enrichment source env vars should appear in get_required_env_vars."""
        bp = BusinessBlueprint(
            vertical_id="enriched_biz",
            vertical_name="Enriched Biz",
            industry="Testing",
            context=sample_context,
            icp=sample_icp,
            personas=sample_personas,
            outreach=sample_outreach,
            agents=sample_agents,
            enrichment_sources=[
                EnrichmentSourceSpec(
                    type="api",
                    provider="shodan",
                    api_key_env="SHODAN_API_KEY",
                ),
            ],
        )
        env_vars = bp.get_required_env_vars()
        assert "SHODAN_API_KEY" in env_vars


# ---------------------------------------------------------------------------
# generate_vertical_id Tests
# ---------------------------------------------------------------------------

class TestGenerateVerticalId:

    def test_simple_name(self):
        assert generate_vertical_id("PrintBiz") == "print_biz"

    def test_spaces(self):
        assert generate_vertical_id("Enclave Guard") == "enclave_guard"

    def test_special_characters(self):
        assert generate_vertical_id("My 3D Printing Co.") == "my_3d_printing_co"

    def test_leading_number(self):
        """IDs starting with numbers get prefixed."""
        result = generate_vertical_id("3D Printers")
        assert result.startswith("v_")
        assert result[0].isalpha()

    def test_empty_string(self):
        assert generate_vertical_id("") == "unnamed_vertical"

    def test_multiple_spaces(self):
        assert generate_vertical_id("  Super  Cool  Biz  ") == "super_cool_biz"

    def test_already_snake_case(self):
        assert generate_vertical_id("test_vertical") == "test_vertical"

    def test_hyphens_converted(self):
        assert generate_vertical_id("my-cool-biz") == "my_cool_biz"

    def test_valid_pattern(self):
        """All generated IDs must match the VerticalConfig pattern."""
        names = [
            "PrintBiz", "Enclave Guard", "My 3D Co", "test",
            "UPPERCASE", "a-b-c", "already_snake",
        ]
        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for name in names:
            vid = generate_vertical_id(name)
            assert pattern.match(vid), f"'{name}' → '{vid}' doesn't match pattern"


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestEnums:

    def test_blueprint_status_values(self):
        assert BlueprintStatus.DRAFT == "draft"
        assert BlueprintStatus.LAUNCHED == "launched"

    def test_agent_role_values(self):
        assert AgentRole.OUTREACH == "outreach"
        assert AgentRole.SEO_CONTENT == "seo_content"

    def test_integration_type_values(self):
        assert IntegrationType.EMAIL_PROVIDER == "email_provider"
        assert IntegrationType.CRM == "crm"

    def test_compliance_jurisdiction_values(self):
        assert ComplianceJurisdiction.US_CAN_SPAM == "US_CAN_SPAM"
        assert ComplianceJurisdiction.EU_GDPR == "EU_GDPR"
