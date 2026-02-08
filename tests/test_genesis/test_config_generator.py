"""
Tests for Genesis Config Generator — The Ribosome.

Validates that the ConfigGenerator:
1. Produces valid VerticalConfig YAML from a BusinessBlueprint
2. Produces valid AgentInstanceConfig YAML for each agent
3. Rejects invalid blueprints without writing any files
4. Can generate configs equivalent to manually-created verticals
5. Generated configs can be loaded by the existing platform

The "Gold Standard" test: Generate a PrintBiz-equivalent from a blueprint
and verify it matches the manually-curated config in structure and validity.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from core.config.agent_schema import AgentInstanceConfig
from core.config.loader import clear_cache, load_vertical_config
from core.config.schema import VerticalConfig
from core.genesis.blueprint import (
    AgentRole,
    AgentSpec,
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
)
from core.genesis.config_generator import (
    ConfigGenerationError,
    ConfigGenerator,
    GenerationResult,
    validate_existing_vertical,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return ConfigGenerator()


@pytest.fixture
def tmp_dir():
    """Create a temp directory for generated files, clean up after test."""
    d = tempfile.mkdtemp(prefix="genesis_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def minimal_blueprint() -> BusinessBlueprint:
    """Smallest valid blueprint that produces valid configs."""
    return BusinessBlueprint(
        vertical_id="test_minimal",
        vertical_name="Test Minimal",
        industry="Testing",
        context=BusinessContext(
            business_name="Test Minimal",
            business_description="Minimal business for testing the config generator.",
            business_model="B2B service — consulting",
            price_range=(1000, 5000),
            target_industries=["Technology"],
            target_company_sizes=(10, 200),
            target_titles=["CTO"],
            pain_points=["Manual processes"],
            value_propositions=["Automation"],
            sending_domain="mail.test.com",
            reply_to="hello@test.com",
            physical_address="123 Test St, Austin TX 78701",
        ),
        icp=ICPSpec(
            company_size=(10, 200),
            industries=["Technology"],
        ),
        personas=[
            PersonaSpec(
                id="cto_small",
                title_patterns=["CTO"],
                company_size=(10, 200),
                approach="direct_pitch",
            ),
        ],
        outreach=OutreachSpec(
            daily_limit=25,
            sending_domain="mail.test.com",
            reply_to="hello@test.com",
            sequences=[
                EmailSequenceSpec(
                    name="direct_pitch", steps=2, delay_days=[0, 3],
                ),
            ],
            physical_address="123 Test St, Austin TX 78701",
        ),
        agents=[
            AgentSpec(
                agent_type=AgentRole.OUTREACH,
                name="Test Outreach Agent",
            ),
        ],
    )


@pytest.fixture
def print_biz_blueprint() -> BusinessBlueprint:
    """
    Blueprint that should produce configs equivalent to
    the manually-curated verticals/print_biz/ directory.

    This is the Gold Standard test: if this blueprint generates
    configs that pass VerticalConfig validation, the Genesis Engine works.
    """
    return BusinessBlueprint(
        vertical_id="print_biz_gen",
        vertical_name="PrintBiz — 3D Printing Services",
        industry="3D Printing / Additive Manufacturing",
        context=BusinessContext(
            business_name="PrintBiz",
            business_description="B2B 3D printing and rapid prototyping for architecture firms.",
            website="https://printbiz3d.com",
            business_model="B2B service — 3D printing and rapid prototyping",
            price_range=(500, 5000),
            sales_cycle_days=21,
            target_industries=[
                "Architecture", "Interior Design",
                "Real Estate Development", "Urban Planning",
            ],
            target_company_sizes=(10, 500),
            target_titles=[
                "Design Director", "Principal Architect",
                "Head of Design", "VP Operations",
                "Studio Manager", "Creative Director",
            ],
            target_locations=["United States", "Canada"],
            pain_points=[
                "Physical models take weeks to produce manually",
                "Client presentations lack tangible 3D context",
                "Iterating on architectural designs is expensive",
            ],
            value_propositions=[
                "CAD to 3D print in 48 hours",
                "Architectural-grade detail at 0.1mm resolution",
                "Bulk pricing for firms with recurring projects",
            ],
            sending_domain="mail.printbiz3d.com",
            reply_to="hello@printbiz3d.com",
            physical_address="456 Maker Lane, San Francisco CA 94105",
            daily_outreach_limit=30,
            positive_signals=[
                "active_building_projects",
                "uses_cad_software",
                "recent_design_competition",
            ],
            disqualifiers=[
                "has_in_house_3d_printing",
                "less_than_10_employees",
            ],
            tone="Friendly, creative, solution-oriented",
            content_topics=[
                "3D printing for architecture",
                "rapid prototyping benefits",
            ],
        ),
        icp=ICPSpec(
            company_size=(10, 500),
            industries=[
                "Architecture", "Interior Design",
                "Real Estate Development", "Urban Planning",
            ],
            signals=[
                "active_building_projects",
                "uses_cad_software",
                "recent_design_competition",
                "expanding_office_locations",
            ],
            disqualifiers=[
                "has_in_house_3d_printing",
                "less_than_10_employees",
                "no_design_department",
            ],
        ),
        personas=[
            PersonaSpec(
                id="design_director",
                title_patterns=[
                    "Design Director", "Principal Architect",
                    "Head of Design", "Studio Manager", "Creative Director",
                ],
                company_size=(10, 200),
                approach="design_showcase",
                seniorities=["director", "vp", "c_suite"],
            ),
            PersonaSpec(
                id="vp_ops",
                title_patterns=[
                    "VP Operations", "COO",
                    "Director of Operations", "Office Manager",
                ],
                company_size=(50, 500),
                approach="efficiency_pitch",
                seniorities=["vp", "c_suite", "director", "manager"],
            ),
        ],
        outreach=OutreachSpec(
            daily_limit=30,
            warmup_days=14,
            sending_domain="mail.printbiz3d.com",
            reply_to="hello@printbiz3d.com",
            sequences=[
                EmailSequenceSpec(
                    name="design_showcase", steps=3, delay_days=[0, 3, 7],
                ),
                EmailSequenceSpec(
                    name="efficiency_pitch", steps=2, delay_days=[0, 5],
                ),
            ],
            jurisdictions=[
                ComplianceJurisdiction.US_CAN_SPAM,
                ComplianceJurisdiction.CA_CASL,
            ],
            physical_address="456 Maker Lane, San Francisco CA 94105",
        ),
        agents=[
            AgentSpec(
                agent_type=AgentRole.OUTREACH,
                name="PrintBiz Outreach Agent",
                description="B2B outreach for 3D printing services targeting architecture firms.",
                tools=["apollo_search", "apollo_enrich", "send_email", "rag_search"],
                human_gate_nodes=["send_outreach"],
                params={
                    "daily_lead_limit": 30,
                    "duplicate_cooldown_days": 60,
                },
            ),
        ],
        integrations=[
            IntegrationSpec(
                name="Apollo",
                type=IntegrationType.LEAD_DATABASE,
                env_var="APOLLO_API_KEY",
            ),
        ],
        enrichment_sources=[
            EnrichmentSourceSpec(
                type="web_scraper",
                targets=["company_website"],
            ),
        ],
        content_topics=["3D printing for architecture", "rapid prototyping benefits"],
        tone="Friendly, creative, solution-oriented",
    )


@pytest.fixture
def multi_agent_blueprint() -> BusinessBlueprint:
    """Blueprint with multiple agent types."""
    return BusinessBlueprint(
        vertical_id="multi_agent_test",
        vertical_name="Multi Agent Test",
        industry="Testing",
        context=BusinessContext(
            business_name="MultiAgent Co",
            business_description="Tests multi-agent generation capabilities.",
            business_model="B2B SaaS",
            price_range=(2000, 10000),
            target_industries=["SaaS"],
            target_company_sizes=(50, 1000),
            target_titles=["CTO", "VP Engineering"],
            pain_points=["Slow development"],
            value_propositions=["Faster delivery"],
            sending_domain="mail.multi.com",
            reply_to="hi@multi.com",
            physical_address="789 Multi Ave, NYC NY 10001",
        ),
        icp=ICPSpec(
            company_size=(50, 1000),
            industries=["SaaS"],
        ),
        personas=[
            PersonaSpec(
                id="cto",
                title_patterns=["CTO", "VP Engineering"],
                company_size=(50, 500),
                approach="tech_pitch",
            ),
        ],
        outreach=OutreachSpec(
            sending_domain="mail.multi.com",
            reply_to="hi@multi.com",
            sequences=[
                EmailSequenceSpec(name="tech_pitch", steps=3, delay_days=[0, 3, 7]),
            ],
            physical_address="789 Multi Ave, NYC NY 10001",
        ),
        agents=[
            AgentSpec(
                agent_type=AgentRole.OUTREACH,
                name="Outreach Agent",
                tools=["apollo_search", "send_email"],
                human_gate_nodes=["send_outreach"],
            ),
            AgentSpec(
                agent_type=AgentRole.SEO_CONTENT,
                name="SEO Content Agent",
                browser_enabled=True,
                human_gate_nodes=["human_review"],
            ),
            AgentSpec(
                agent_type=AgentRole.APPOINTMENT_SETTER,
                name="Appointment Setter",
                human_gate_nodes=["human_review"],
            ),
        ],
        content_topics=["developer tools", "engineering productivity"],
    )


# ---------------------------------------------------------------------------
# GenerationResult Tests
# ---------------------------------------------------------------------------

class TestGenerationResult:

    def test_config_path_property(self):
        r = GenerationResult(
            success=True,
            vertical_id="test",
            paths=["verticals/test/config.yaml", "verticals/test/agents/outreach.yaml"],
        )
        assert r.config_path == "verticals/test/config.yaml"

    def test_agent_paths_property(self):
        r = GenerationResult(
            success=True,
            vertical_id="test",
            paths=[
                "verticals/test/config.yaml",
                "verticals/test/agents/outreach.yaml",
                "verticals/test/agents/seo_content.yaml",
            ],
        )
        assert len(r.agent_paths) == 2

    def test_no_config_path(self):
        r = GenerationResult(success=False, vertical_id="test")
        assert r.config_path is None


# ---------------------------------------------------------------------------
# ConfigGenerator: Minimal Generation
# ---------------------------------------------------------------------------

class TestConfigGeneratorMinimal:
    """Test with the smallest valid blueprint."""

    def test_generates_valid_vertical_config(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Generated config.yaml must pass VerticalConfig validation."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success, f"Generation failed: {result.errors}"
        assert result.config_path is not None

        # Load and validate
        with open(result.config_path) as f:
            raw = yaml.safe_load(f)
        config = VerticalConfig(**raw)
        assert config.vertical_id == "test_minimal"
        assert config.vertical_name == "Test Minimal"

    def test_generates_valid_agent_config(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Generated agent YAML must pass AgentInstanceConfig validation."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        agent_paths = result.agent_paths
        assert len(agent_paths) == 1

        with open(agent_paths[0]) as f:
            raw = yaml.safe_load(f)
        agent_config = AgentInstanceConfig(**raw)
        assert agent_config.agent_type == "outreach"
        assert agent_config.shadow_mode is True  # New verticals always shadow

    def test_creates_directory_structure(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Must create proper directory hierarchy."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        vertical_dir = Path(tmp_dir) / "test_minimal"
        assert vertical_dir.is_dir()
        assert (vertical_dir / "agents").is_dir()
        assert (vertical_dir / "config.yaml").is_file()
        assert (vertical_dir / "agents" / "outreach.yaml").is_file()
        assert (vertical_dir / "__init__.py").is_file()
        assert (vertical_dir / "agents" / "__init__.py").is_file()

    def test_dry_run_no_files(self, generator, minimal_blueprint, tmp_dir):
        """Dry run validates but doesn't write files."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir, dry_run=True,
        )
        assert result.success
        assert len(result.paths) > 0
        assert "Dry run" in result.warnings[0]
        assert not (Path(tmp_dir) / "test_minimal").exists()

    def test_idempotent_generation(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Running generation twice should work (overwrite)."""
        result1 = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result1.success

        result2 = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result2.success

    def test_generated_config_loadable_by_platform(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """
        Ultimate test: generated config must be loadable by the
        existing platform loader (core.config.loader).
        """
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        clear_cache()
        config = load_vertical_config(
            "test_minimal",
            config_path=Path(tmp_dir) / "test_minimal" / "config.yaml",
        )
        assert config.vertical_id == "test_minimal"
        assert len(config.targeting.personas) == 1
        clear_cache()


# ---------------------------------------------------------------------------
# ConfigGenerator: PrintBiz Gold Standard
# ---------------------------------------------------------------------------

class TestConfigGeneratorPrintBiz:
    """
    Gold Standard: Generate a PrintBiz-equivalent and validate it
    matches the manually-curated config in structure and validity.
    """

    def test_generates_valid_config(
        self, generator, print_biz_blueprint, tmp_dir
    ):
        """PrintBiz-equivalent must pass VerticalConfig validation."""
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success, f"Errors: {result.errors}"

        with open(result.config_path) as f:
            raw = yaml.safe_load(f)
        config = VerticalConfig(**raw)

        # Verify key structural elements match the real PrintBiz
        assert config.industry == "3D Printing / Additive Manufacturing"
        assert config.business.ticket_range == (500, 5000)
        assert config.business.sales_cycle_days == 21
        assert len(config.targeting.personas) == 2
        assert config.outreach.email.daily_limit == 30
        assert config.outreach.email.sending_domain == "mail.printbiz3d.com"
        assert config.outreach.compliance.physical_address == "456 Maker Lane, San Francisco CA 94105"

    def test_personas_match(self, generator, print_biz_blueprint, tmp_dir):
        """Generated personas should match the manual config structure."""
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        with open(result.config_path) as f:
            raw = yaml.safe_load(f)
        config = VerticalConfig(**raw)

        persona_ids = {p.id for p in config.targeting.personas}
        assert "design_director" in persona_ids
        assert "vp_ops" in persona_ids

    def test_sequences_match(self, generator, print_biz_blueprint, tmp_dir):
        """Email sequences should match the manual config."""
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        with open(result.config_path) as f:
            raw = yaml.safe_load(f)
        config = VerticalConfig(**raw)

        seq_names = {s.name for s in config.outreach.email.sequences}
        assert "design_showcase" in seq_names
        assert "efficiency_pitch" in seq_names

    def test_agent_yaml_valid(
        self, generator, print_biz_blueprint, tmp_dir
    ):
        """Generated agent YAML must be valid."""
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        for agent_path in result.agent_paths:
            with open(agent_path) as f:
                raw = yaml.safe_load(f)
            agent = AgentInstanceConfig(**raw)
            assert agent.agent_type == "outreach"
            assert agent.shadow_mode is True

    def test_loadable_by_platform(
        self, generator, print_biz_blueprint, tmp_dir
    ):
        """Generated config loadable by the standard platform loader."""
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        clear_cache()
        config = load_vertical_config(
            "print_biz_gen",
            config_path=Path(tmp_dir) / "print_biz_gen" / "config.yaml",
        )
        assert config.vertical_id == "print_biz_gen"
        assert config.industry == "3D Printing / Additive Manufacturing"
        clear_cache()

    def test_enrichment_sources_generated(
        self, generator, print_biz_blueprint, tmp_dir
    ):
        """Enrichment sources should be in the config."""
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        with open(result.config_path) as f:
            raw = yaml.safe_load(f)
        config = VerticalConfig(**raw)
        assert len(config.enrichment.sources) >= 1
        assert config.enrichment.sources[0].type == "web_scraper"


# ---------------------------------------------------------------------------
# ConfigGenerator: Multi-Agent
# ---------------------------------------------------------------------------

class TestConfigGeneratorMultiAgent:
    """Test generation with multiple agent types."""

    def test_generates_all_agent_yamls(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """Should create YAML files for each agent."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success
        assert len(result.agent_paths) == 3

    def test_seo_agent_has_browser(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """SEO content agent should have browser_enabled=True."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        seo_path = [p for p in result.agent_paths if "seo_content" in p][0]
        with open(seo_path) as f:
            raw = yaml.safe_load(f)
        agent = AgentInstanceConfig(**raw)
        assert agent.browser_enabled is True

    def test_each_agent_valid(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """Every agent YAML must individually pass validation."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        for agent_path in result.agent_paths:
            with open(agent_path) as f:
                raw = yaml.safe_load(f)
            # Must not raise
            AgentInstanceConfig(**raw)

    def test_all_agents_shadow_mode(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """All agents from Genesis start in shadow mode."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        for agent_path in result.agent_paths:
            with open(agent_path) as f:
                raw = yaml.safe_load(f)
            assert raw.get("shadow_mode") is True

    def test_default_tools_applied(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """Agents without explicit tools should get defaults."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        # SEO agent had no tools specified — should get defaults
        seo_path = [p for p in result.agent_paths if "seo_content" in p][0]
        with open(seo_path) as f:
            raw = yaml.safe_load(f)
        agent = AgentInstanceConfig(**raw)
        assert "rag_search" in [t.name for t in agent.tools]

    def test_agent_params_populated(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """Agent-specific params should be auto-populated."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        # SEO agent should have content params
        seo_path = [p for p in result.agent_paths if "seo_content" in p][0]
        with open(seo_path) as f:
            raw = yaml.safe_load(f)
        assert "target_word_count" in raw.get("params", {})
        assert "target_topics" in raw.get("params", {})


# ---------------------------------------------------------------------------
# ConfigGenerator: Edge Cases
# ---------------------------------------------------------------------------

class TestConfigGeneratorEdgeCases:

    def test_yaml_header_comments(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Generated YAML should have helpful header comments."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        with open(result.config_path) as f:
            content = f.read()
        assert "# Test Minimal — Vertical Configuration" in content
        assert "Genesis Engine" in content

    def test_agent_yaml_header(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Agent YAMLs should have header comments."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        with open(result.agent_paths[0]) as f:
            content = f.read()
        assert "# Test Outreach Agent" in content

    def test_prompts_directory_created(
        self, generator, minimal_blueprint, tmp_dir
    ):
        """Prompts directory should be created even if empty."""
        result = generator.generate_vertical(
            minimal_blueprint, output_dir=tmp_dir,
        )
        assert result.success
        prompts_dir = Path(tmp_dir) / "test_minimal" / "prompts" / "agent_prompts"
        assert prompts_dir.is_dir()

    def test_system_prompt_written(self, generator, tmp_dir):
        """If agent has system_prompt_template, it should be written."""
        bp = BusinessBlueprint(
            vertical_id="prompt_test",
            vertical_name="Prompt Test",
            industry="Testing",
            context=BusinessContext(
                business_name="PromptTest",
                business_description="Testing system prompt generation.",
                business_model="B2B SaaS",
                price_range=(500, 2000),
                target_industries=["Tech"],
                target_company_sizes=(5, 100),
                target_titles=["CTO"],
                pain_points=["Testing"],
                value_propositions=["Testing"],
                sending_domain="mail.test.com",
                reply_to="test@test.com",
                physical_address="123 Test St",
            ),
            icp=ICPSpec(company_size=(5, 100), industries=["Tech"]),
            personas=[
                PersonaSpec(
                    id="cto", title_patterns=["CTO"],
                    company_size=(5, 100), approach="direct",
                ),
            ],
            outreach=OutreachSpec(
                sending_domain="mail.test.com",
                reply_to="test@test.com",
                sequences=[
                    EmailSequenceSpec(name="direct", steps=1, delay_days=[0]),
                ],
                physical_address="123 Test St",
            ),
            agents=[
                AgentSpec(
                    agent_type=AgentRole.OUTREACH,
                    name="Outreach",
                    system_prompt_template="You are an outreach agent for {{vertical_name}}.",
                ),
            ],
        )
        result = generator.generate_vertical(bp, output_dir=tmp_dir)
        assert result.success

        prompt_path = (
            Path(tmp_dir) / "prompt_test" / "prompts" / "agent_prompts"
            / "outreach_system.md"
        )
        assert prompt_path.is_file()
        assert "outreach agent" in prompt_path.read_text()


# ---------------------------------------------------------------------------
# ConfigGenerator: Employee Range Formatting
# ---------------------------------------------------------------------------

class TestEmployeeRangeFormatting:
    """Test Apollo employee range conversion."""

    def test_small_range(self, generator):
        ranges = generator._format_employee_ranges((10, 50))
        assert "11,50" in ranges

    def test_medium_range(self, generator):
        ranges = generator._format_employee_ranges((10, 500))
        assert len(ranges) >= 2

    def test_large_range(self, generator):
        ranges = generator._format_employee_ranges((10, 5000))
        assert len(ranges) >= 3

    def test_single_bucket(self, generator):
        ranges = generator._format_employee_ranges((1, 10))
        assert len(ranges) >= 1

    def test_fallback_on_unusual_range(self, generator):
        """Unusual ranges should still produce something valid."""
        ranges = generator._format_employee_ranges((7500, 8000))
        assert len(ranges) >= 1


# ---------------------------------------------------------------------------
# Validate Existing Verticals
# ---------------------------------------------------------------------------

class TestValidateExistingVertical:

    def test_enclave_guard_valid(self):
        """The gold standard manual config should validate."""
        result = validate_existing_vertical("verticals/enclave_guard")
        assert result.success, f"Errors: {result.errors}"
        assert any("config.yaml" in p for p in result.paths)

    def test_print_biz_valid(self):
        """The upgraded PrintBiz config should validate."""
        result = validate_existing_vertical("verticals/print_biz")
        assert result.success, f"Errors: {result.errors}"

    def test_nonexistent_vertical(self):
        """Non-existent vertical directory should fail gracefully."""
        result = validate_existing_vertical("verticals/nonexistent_12345")
        assert not result.success

    def test_empty_directory(self, tmp_dir):
        """Empty directory (no config.yaml) should fail."""
        empty_dir = Path(tmp_dir) / "empty_vertical"
        empty_dir.mkdir()
        result = validate_existing_vertical(str(empty_dir))
        assert not result.success

    def test_validates_agent_yamls(self):
        """Should also validate agent YAML files."""
        result = validate_existing_vertical("verticals/enclave_guard")
        agent_paths = [p for p in result.paths if "/agents/" in p]
        assert len(agent_paths) >= 1  # At least outreach.yaml


# ---------------------------------------------------------------------------
# Integration: Full Round-Trip
# ---------------------------------------------------------------------------

class TestFullRoundTrip:
    """End-to-end: Blueprint → Generate → Load → Verify."""

    def test_blueprint_to_running_config(
        self, generator, print_biz_blueprint, tmp_dir
    ):
        """
        The full Genesis pipeline:
        1. Create BusinessBlueprint (from interview)
        2. Generate YAML configs
        3. Load through platform loader
        4. Verify all fields are correct
        """
        # Step 2: Generate
        result = generator.generate_vertical(
            print_biz_blueprint, output_dir=tmp_dir,
        )
        assert result.success, f"Generation failed: {result.errors}"

        # Step 3: Load through standard platform loader
        clear_cache()
        config = load_vertical_config(
            "print_biz_gen",
            config_path=Path(tmp_dir) / "print_biz_gen" / "config.yaml",
        )

        # Step 4: Verify
        assert config.vertical_id == "print_biz_gen"
        assert config.vertical_name == "PrintBiz — 3D Printing Services"
        assert config.industry == "3D Printing / Additive Manufacturing"

        # Business
        assert config.business.ticket_range == (500, 5000)
        assert config.business.currency == "USD"
        assert config.business.sales_cycle_days == 21

        # Targeting
        assert len(config.targeting.personas) == 2
        assert len(config.targeting.ideal_customer_profile.industries) == 4

        # Outreach
        assert config.outreach.email.daily_limit == 30
        assert config.outreach.email.sending_domain == "mail.printbiz3d.com"
        assert len(config.outreach.email.sequences) == 2

        # Compliance
        assert config.outreach.compliance.physical_address == "456 Maker Lane, San Francisco CA 94105"

        clear_cache()

    def test_generated_agents_loadable(
        self, generator, multi_agent_blueprint, tmp_dir
    ):
        """All generated agent YAMLs should be individually loadable."""
        result = generator.generate_vertical(
            multi_agent_blueprint, output_dir=tmp_dir,
        )
        assert result.success

        for agent_path in result.agent_paths:
            with open(agent_path) as f:
                raw = yaml.safe_load(f)
            config = AgentInstanceConfig(**raw)
            assert config.enabled is True
            assert config.shadow_mode is True
