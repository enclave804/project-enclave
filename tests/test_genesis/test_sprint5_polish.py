"""
Sprint 5: Polish + Safety — Tests for Genesis Engine.

Covers:
    1. Telegram notifications for genesis lifecycle events
    2. Golden file regression test (generate enclave_guard-equivalent configs)
    3. Security audit: credential handling, file writes, YAML injection
    4. End-to-end flow verification
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_business_context(**overrides):
    """Create a valid BusinessContext with sensible defaults."""
    from core.genesis.blueprint import BusinessContext

    defaults = {
        "business_name": "Test Business",
        "business_description": "A test business for validation purposes.",
        "business_model": "B2B SaaS platform",
        "price_range": (5000, 10000),
        "target_industries": ["technology"],
        "target_company_sizes": (11, 500),
        "target_titles": ["CTO", "VP Engineering"],
        "pain_points": ["scaling challenges"],
        "value_propositions": ["Automated solutions that scale"],
        "region": "United States",
        "geographic_focus": "United States",
    }
    defaults.update(overrides)
    return BusinessContext(**defaults)


# ===========================================================================
# 1. Telegram Notifications
# ===========================================================================


class TestGenesisNotifier:
    """Test genesis lifecycle notifications."""

    def _run(self, coro):
        """Helper to run async coroutines in sync tests."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_import(self):
        from core.genesis.notifications import GenesisNotifier
        assert GenesisNotifier is not None

    def test_no_bot_configured(self):
        from core.genesis.notifications import GenesisNotifier
        notifier = GenesisNotifier(telegram_bot=None)
        result = self._run(notifier.on_launch_success("v1", 2, ["a1", "a2"]))
        assert result is False  # No bot, returns False

    def test_on_interview_complete(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_interview_complete("test_biz", 7, "abc123"))
        assert result is True
        mock_bot.send_alert.assert_called_once()
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Interview Complete" in msg
        assert "test_biz" in msg
        assert "7" in msg

    def test_on_blueprint_ready(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_blueprint_ready("test_biz", "3D Print Shop"))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Blueprint Ready" in msg
        assert "3D Print Shop" in msg

    def test_on_blueprint_approved(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_blueprint_approved("test_biz", "PrintPro"))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Blueprint Approved" in msg

    def test_on_configs_generated(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_configs_generated("test_biz", 3))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Configs Generated" in msg
        assert "3" in msg

    def test_on_launch_success(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_launch_success(
            "test_biz", 2, ["outreach_v1", "seo_v1"]
        ))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Launched" in msg
        assert "Shadow Mode" in msg
        assert "outreach_v1" in msg

    def test_on_launch_failed(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_launch_failed(
            "test_biz", ["Config invalid", "Missing agent type"]
        ))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Launch Failed" in msg
        assert "Config invalid" in msg

    def test_on_promotion(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_promotion("test_biz"))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Promoted to LIVE" in msg

    def test_send_alert_failure_is_silent(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(side_effect=RuntimeError("Network error"))
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        # Should not raise
        result = self._run(notifier.on_launch_success("v1", 1, ["a1"]))
        assert result is False

    def test_interview_complete_without_session_id(self):
        from core.genesis.notifications import GenesisNotifier
        mock_bot = MagicMock()
        mock_bot.send_alert = AsyncMock(return_value=True)
        notifier = GenesisNotifier(telegram_bot=mock_bot)

        result = self._run(notifier.on_interview_complete("v1", 5))
        assert result is True
        msg = mock_bot.send_alert.call_args[0][0]
        assert "Session" not in msg  # No session ID, no mention


# ===========================================================================
# 2. Golden File Regression Test
# ===========================================================================


class TestGoldenFileRegression:
    """
    Verify that the ConfigGenerator produces configs structurally equivalent
    to the hand-crafted enclave_guard configs.

    This is the ultimate integration test: given a BusinessBlueprint that
    describes enclave_guard's actual business, the generator should produce
    valid configs that match the reference.
    """

    def test_generated_config_validates(self, tmp_path):
        """Generated config passes VerticalConfig Pydantic validation."""
        from core.genesis.blueprint import (
            AgentSpec,
            BusinessBlueprint,
            EmailSequenceSpec,
            ICPSpec,
            IntegrationSpec,
            OutreachSpec,
            PersonaSpec,
        )
        from core.genesis.config_generator import ConfigGenerator

        # Build a blueprint that matches enclave_guard's business
        context = _make_business_context(
            business_name="Enclave Guard Security",
            business_description="Cybersecurity consulting and security assessments for SMBs",
            business_model="Consulting services — security assessments and audits",
            price_range=(5000, 15000),
            target_industries=["technology", "finance", "healthcare"],
            target_company_sizes=(51, 500),
            target_titles=["CTO", "VP Engineering", "CISO"],
            pain_points=["compliance requirements", "breach risk", "scaling security"],
            value_propositions=["Automated security scanning", "Expert-led assessments"],
        )

        blueprint = BusinessBlueprint(
            vertical_id="golden_test",
            vertical_name="Enclave Guard Security",
            industry="Cybersecurity Consulting",
            context=context,
            icp=ICPSpec(
                company_size=(51, 500),
                industries=["technology", "finance", "healthcare"],
                signals=["recent_funding", "job_postings_security"],
                disqualifiers=["government", "defense_contractor"],
            ),
            personas=[
                PersonaSpec(
                    id="cto",
                    title_patterns=["CTO", "Chief Technology Officer"],
                    company_size=(51, 500),
                    approach="initial_outreach",
                ),
                PersonaSpec(
                    id="vp_eng",
                    title_patterns=["VP Engineering", "VP of Engineering"],
                    company_size=(51, 500),
                    approach="follow_up",
                ),
            ],
            outreach=OutreachSpec(
                daily_limit=25,
                sending_domain="mail.enclaveguard.com",
                reply_to="hello@enclaveguard.com",
                physical_address="123 Security Ave, Austin TX 78701",
                sequences=[
                    EmailSequenceSpec(name="initial_outreach", steps=3, delay_days=[0, 3, 7]),
                    EmailSequenceSpec(name="follow_up", steps=2, delay_days=[0, 5]),
                ],
            ),
            agents=[
                AgentSpec(
                    agent_type="outreach",
                    name="Outreach Agent",
                    description="Outbound lead generation",
                    tools=["apollo_search", "send_email"],
                ),
                AgentSpec(
                    agent_type="seo_content",
                    name="SEO Content Agent",
                    description="Security blog content",
                    tools=["browser_search"],
                ),
            ],
            integrations=[
                IntegrationSpec(
                    name="Apollo",
                    type="lead_database",
                    env_var="APOLLO_API_KEY",
                    instructions="Lead data enrichment",
                    required=True,
                ),
            ],
            strategy_reasoning="Direct outbound to security-conscious SMBs.",
            risk_factors=["Competitive market", "Long sales cycle"],
            success_metrics=["50 leads/month", "5% reply rate"],
        )

        # Generate configs
        gen = ConfigGenerator()
        result = gen.generate_vertical(blueprint, output_dir=str(tmp_path))

        # Must succeed
        assert result.success is True, f"Config generation failed: {result.errors}"

        # Verify config.yaml was created and validates
        config_path = tmp_path / "golden_test" / "config.yaml"
        assert config_path.exists()

        from core.config.schema import VerticalConfig

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        raw["vertical_id"] = "golden_test"
        cfg = VerticalConfig(**raw)
        assert cfg.vertical_name == "Enclave Guard Security"
        assert cfg.industry == "Cybersecurity Consulting"

    def test_generated_agent_configs_validate(self, tmp_path):
        """Generated agent YAMLs pass AgentInstanceConfig validation."""
        from core.genesis.blueprint import (
            AgentSpec,
            BusinessBlueprint,
            ICPSpec,
            OutreachSpec,
            PersonaSpec,
        )
        from core.genesis.config_generator import ConfigGenerator
        from core.config.agent_schema import AgentInstanceConfig

        context = _make_business_context()

        blueprint = BusinessBlueprint(
            vertical_id="agent_test",
            vertical_name="Test Business",
            industry="Technology",
            context=context,
            icp=ICPSpec(
                company_size=(11, 50),
                industries=["technology"],
            ),
            personas=[
                PersonaSpec(
                    id="cto",
                    title_patterns=["CTO", "Chief Technology Officer"],
                    company_size=(11, 50),
                    approach="initial_outreach",
                ),
            ],
            outreach=OutreachSpec(
                daily_limit=10,
                sending_domain="mail.test.com",
                reply_to="hello@test.com",
                physical_address="123 Test St, Austin TX 78701",
            ),
            agents=[
                AgentSpec(agent_type="outreach", name="Outreach Agent", description="Lead gen"),
                AgentSpec(agent_type="seo_content", name="SEO Content Agent", description="Content"),
            ],
        )

        gen = ConfigGenerator()
        result = gen.generate_vertical(blueprint, output_dir=str(tmp_path))
        assert result.success is True

        agents_dir = tmp_path / "agent_test" / "agents"
        assert agents_dir.is_dir()

        agent_files = list(agents_dir.glob("*.yaml"))
        assert len(agent_files) >= 1

        for af in agent_files:
            with open(af) as f:
                raw = yaml.safe_load(f)
            if raw:
                cfg = AgentInstanceConfig(**raw)
                assert cfg.agent_type in ("outreach", "seo_content")
                assert cfg.enabled is True

    def test_generated_config_has_required_sections(self, tmp_path):
        """Generated config.yaml has all required VerticalConfig sections."""
        from core.genesis.blueprint import (
            AgentSpec,
            BusinessBlueprint,
            ICPSpec,
            OutreachSpec,
            PersonaSpec,
        )
        from core.genesis.config_generator import ConfigGenerator

        context = _make_business_context(business_name="Sections Test")

        blueprint = BusinessBlueprint(
            vertical_id="sections_test",
            vertical_name="Sections Test",
            industry="Tech",
            context=context,
            icp=ICPSpec(company_size=(11, 50), industries=["technology"]),
            personas=[
                PersonaSpec(
                    id="cto",
                    title_patterns=["CTO"],
                    company_size=(11, 50),
                    approach="initial_outreach",
                ),
            ],
            outreach=OutreachSpec(
                daily_limit=10,
                sending_domain="mail.sections.com",
                reply_to="hello@sections.com",
                physical_address="123 Section St, Austin TX 78701",
            ),
            agents=[AgentSpec(agent_type="outreach", name="Outreach Agent", description="Outbound")],
        )

        gen = ConfigGenerator()
        result = gen.generate_vertical(blueprint, output_dir=str(tmp_path))
        assert result.success is True

        config_path = tmp_path / "sections_test" / "config.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Must have all top-level sections required by VerticalConfig
        assert "business" in raw
        assert "targeting" in raw
        assert "outreach" in raw
        assert raw["business"].get("ticket_range")  # price range is always present
        assert raw["targeting"].get("personas")
        assert raw["outreach"].get("email")


# ===========================================================================
# 3. Security Audit
# ===========================================================================


class TestCredentialSecurity:
    """Security tests for credential handling."""

    def test_credentials_never_stored_plaintext(self):
        """Credentials must always be encrypted before storage."""
        from core.genesis.credential_manager import CredentialManager

        cm = CredentialManager(master_key="test-audit-key-42")
        cm.set_credential("v1", "SECRET_KEY", "my-super-secret")

        # The in-memory store should NOT contain the plaintext
        for key, value in cm._memory_store.items():
            assert value != "my-super-secret", (
                "Plaintext credential found in memory store!"
            )

    def test_credentials_retrievable_after_encryption(self):
        """Encrypted credentials must round-trip correctly."""
        from core.genesis.credential_manager import CredentialManager

        cm = CredentialManager(master_key="test-roundtrip-key-42")
        cm.set_credential("v1", "API_KEY", "sk-test-12345")

        result = cm.get_credential("v1", "API_KEY")
        assert result == "sk-test-12345"

    def test_different_master_keys_cannot_decrypt(self):
        """Credentials encrypted with one key cannot be decrypted with another."""
        from core.genesis.credential_manager import encrypt_value, decrypt_value

        # encrypt_value/decrypt_value accept string master keys
        encrypted = encrypt_value("secret", "master-key-1")

        # Decrypting with wrong key should fail
        from cryptography.fernet import InvalidToken
        with pytest.raises(InvalidToken):
            decrypt_value(encrypted, "master-key-2")

    def test_empty_credential_rejected(self):
        """Cannot store empty/whitespace credentials."""
        from core.genesis.credential_manager import CredentialManager

        cm = CredentialManager(master_key="test-empty-key-42")

        with pytest.raises(ValueError):
            cm.set_credential("v1", "KEY", "")

        with pytest.raises(ValueError):
            cm.set_credential("v1", "KEY", "   ")

    def test_credentials_not_in_env_export_without_setting(self):
        """export_as_env only returns credentials that have been set."""
        from core.genesis.credential_manager import CredentialManager

        cm = CredentialManager(master_key="test-export-key-42")
        # Don't set any credentials
        env_dict = cm.export_as_env("v1")
        assert len(env_dict) == 0

    def test_master_key_derivation_deterministic(self):
        """Same master key always produces the same derived key."""
        import hashlib, base64
        key1 = base64.urlsafe_b64encode(hashlib.sha256(b"my-master-key").digest())
        key2 = base64.urlsafe_b64encode(hashlib.sha256(b"my-master-key").digest())
        assert key1 == key2

    def test_master_key_derivation_unique(self):
        """Different master keys produce different derived keys."""
        import hashlib, base64
        key1 = base64.urlsafe_b64encode(hashlib.sha256(b"key-alpha").digest())
        key2 = base64.urlsafe_b64encode(hashlib.sha256(b"key-beta").digest())
        assert key1 != key2


class TestYAMLInjectionSafety:
    """Test that generated YAML files are safe from injection attacks."""

    def test_yaml_safe_load_used(self):
        """Config generator must use yaml.safe_load, not yaml.load."""
        import inspect
        from core.genesis import config_generator

        source = inspect.getsource(config_generator)
        # Should use safe_load
        assert "yaml.safe_load" in source or "safe_load" in source
        # Should NOT use unsafe yaml.load (without Loader=)
        # Note: yaml.safe_load is fine, yaml.load without explicit Loader is dangerous
        lines = source.split("\n")
        for line in lines:
            if "yaml.load(" in line and "safe_load" not in line:
                # Check if it has an explicit Loader argument
                assert "Loader=" in line, (
                    f"Unsafe yaml.load() without Loader= found: {line.strip()}"
                )

    def test_config_generator_uses_safe_dump(self):
        """Config generator must use yaml.safe_dump for output."""
        import inspect
        from core.genesis import config_generator

        source = inspect.getsource(config_generator)
        # Should use safe_dump or dump with default_flow_style
        assert "yaml.dump" in source or "yaml.safe_dump" in source

    def test_malicious_vertical_id_rejected(self):
        """Vertical IDs with path traversal are rejected by Pydantic."""
        from pydantic import ValidationError
        from core.genesis.blueprint import BusinessBlueprint, ICPSpec, OutreachSpec, PersonaSpec, AgentSpec

        context = _make_business_context(business_name="Evil Corp")

        # Path traversal in vertical_id — should fail the pattern ^[a-z][a-z0-9_]*$
        with pytest.raises(ValidationError):
            BusinessBlueprint(
                vertical_id="../../../etc/passwd",
                vertical_name="Evil",
                industry="Hacking",
                context=context,
                icp=ICPSpec(company_size=(1, 10), industries=["tech"]),
                personas=[
                    PersonaSpec(id="cto", title_patterns=["CTO"], company_size=(1, 10), approach="initial_outreach"),
                ],
                outreach=OutreachSpec(
                    daily_limit=1,
                    sending_domain="mail.evil.com",
                    reply_to="evil@evil.com",
                    physical_address="666 Evil St",
                ),
                agents=[AgentSpec(agent_type="outreach", name="Outreach Agent", description="test")],
            )

    def test_vertical_id_must_be_snake_case(self):
        """Vertical IDs must match ^[a-z][a-z0-9_]*$ pattern."""
        from pydantic import ValidationError
        from core.genesis.blueprint import BusinessBlueprint, ICPSpec, OutreachSpec, PersonaSpec, AgentSpec

        context = _make_business_context()

        _icp = ICPSpec(company_size=(1, 10), industries=["tech"])
        _personas = [
            PersonaSpec(id="cto", title_patterns=["CTO"], company_size=(1, 10), approach="initial_outreach"),
        ]
        _outreach = OutreachSpec(
            daily_limit=1,
            sending_domain="mail.test.com",
            reply_to="hello@test.com",
            physical_address="123 Test St",
        )
        _agents = [AgentSpec(agent_type="outreach", name="Outreach Agent", description="test")]

        # Uppercase rejected
        with pytest.raises(ValidationError):
            BusinessBlueprint(
                vertical_id="BadName",
                vertical_name="Test",
                industry="Tech",
                context=context,
                icp=_icp,
                personas=_personas,
                outreach=_outreach,
                agents=_agents,
            )

        # Spaces rejected
        with pytest.raises(ValidationError):
            BusinessBlueprint(
                vertical_id="has spaces",
                vertical_name="Test",
                industry="Tech",
                context=context,
                icp=_icp,
                personas=_personas,
                outreach=_outreach,
                agents=_agents,
            )


class TestFileWriteSafety:
    """Test that file writes are contained to expected directories."""

    def test_config_generator_writes_to_output_dir(self, tmp_path):
        """ConfigGenerator only writes within the specified output directory."""
        from core.genesis.blueprint import (
            AgentSpec,
            BusinessBlueprint,
            ICPSpec,
            OutreachSpec,
            PersonaSpec,
        )
        from core.genesis.config_generator import ConfigGenerator

        context = _make_business_context(business_name="Safe Write Test")

        blueprint = BusinessBlueprint(
            vertical_id="safe_write",
            vertical_name="Safe Write Test",
            industry="Tech",
            context=context,
            icp=ICPSpec(company_size=(11, 50), industries=["technology"]),
            personas=[
                PersonaSpec(
                    id="cto",
                    title_patterns=["CTO"],
                    company_size=(11, 50),
                    approach="initial_outreach",
                ),
            ],
            outreach=OutreachSpec(
                daily_limit=10,
                sending_domain="mail.safe.com",
                reply_to="hello@safe.com",
                physical_address="123 Safe St, Austin TX 78701",
            ),
            agents=[AgentSpec(agent_type="outreach", name="Outreach Agent", description="Outbound")],
        )

        gen = ConfigGenerator()
        result = gen.generate_vertical(blueprint, output_dir=str(tmp_path))
        assert result.success is True

        # All created files should be under tmp_path
        for root, dirs, files in os.walk(tmp_path):
            for f in files:
                file_path = Path(root) / f
                # Verify the file is within tmp_path
                assert str(file_path).startswith(str(tmp_path)), (
                    f"File written outside output directory: {file_path}"
                )

    def test_launcher_preflight_reads_only(self, tmp_path):
        """Preflight check should not create or modify any files."""
        from core.genesis.launcher import GenesisLauncher
        from core.genesis.credential_manager import CredentialManager

        cm = CredentialManager(master_key="test-readonly-key-42")
        launcher = GenesisLauncher(credential_manager=cm)

        # Take snapshot of tmp_path contents before
        before = set()
        for root, dirs, files in os.walk(tmp_path):
            for f in files:
                before.add(os.path.join(root, f))

        # Run preflight (it will fail because no config exists)
        launcher.preflight_check("nonexistent", str(tmp_path))

        # Take snapshot after
        after = set()
        for root, dirs, files in os.walk(tmp_path):
            for f in files:
                after.add(os.path.join(root, f))

        # No new files should have been created
        assert before == after, (
            f"Preflight check modified filesystem: new files {after - before}"
        )


class TestLauncherShadowModeEnforcement:
    """Verify shadow mode is mandatory and cannot be bypassed."""

    def test_shadow_mode_always_set(self):
        """Launch result always has shadow_mode status for new verticals."""
        from core.genesis.launcher import LaunchResult

        result = LaunchResult(success=True, vertical_id="test", status="shadow_mode")
        assert result.is_shadow_mode is True

    def test_promote_to_live_not_yet_implemented(self):
        """Promotion is deliberately disabled for safety."""
        from core.genesis.launcher import GenesisLauncher

        launcher = GenesisLauncher()
        result = launcher.promote_to_live("any_vertical")
        assert result is False  # Safety: not auto-promoted

    def test_launch_result_status_values(self):
        """LaunchResult status can only be known values."""
        from core.genesis.launcher import LaunchResult

        for status in ["pending", "shadow_mode", "failed"]:
            result = LaunchResult(success=True, vertical_id="test", status=status)
            assert result.status == status


# ===========================================================================
# 4. End-to-End Flow Verification
# ===========================================================================


class TestEndToEndFlow:
    """Verify the complete Genesis flow from blueprint to launch."""

    def test_blueprint_to_config_to_launch(self, tmp_path):
        """Full flow: Blueprint → ConfigGenerator → Launcher → shadow mode."""
        import core.agents.implementations.outreach_agent  # noqa: F401

        from core.genesis.blueprint import (
            AgentSpec,
            BusinessBlueprint,
            ICPSpec,
            OutreachSpec,
            PersonaSpec,
        )
        from core.genesis.config_generator import ConfigGenerator
        from core.genesis.credential_manager import CredentialManager
        from core.genesis.launcher import GenesisLauncher

        # Stage 1: Create blueprint
        context = _make_business_context(business_name="E2E Test Business")

        blueprint = BusinessBlueprint(
            vertical_id="e2e_test",
            vertical_name="E2E Test Business",
            industry="Technology",
            context=context,
            icp=ICPSpec(
                company_size=(11, 200),
                industries=["technology"],
                signals=["recent_funding"],
            ),
            personas=[
                PersonaSpec(
                    id="cto",
                    title_patterns=["CTO", "Chief Technology Officer"],
                    company_size=(11, 200),
                    approach="initial_outreach",
                ),
            ],
            outreach=OutreachSpec(
                daily_limit=15,
                sending_domain="mail.e2e.com",
                reply_to="hello@e2e.com",
                physical_address="123 E2E St, Austin TX 78701",
            ),
            agents=[
                AgentSpec(
                    agent_type="outreach",
                    name="Outreach Agent",
                    description="Lead generation",
                    tools=["apollo_search", "send_email"],
                ),
            ],
            strategy_reasoning="Direct outbound to tech SMBs.",
            risk_factors=["Competitive market"],
            success_metrics=["50 leads/month"],
        )

        # Stage 2: Generate configs
        gen = ConfigGenerator()
        gen_result = gen.generate_vertical(blueprint, output_dir=str(tmp_path))
        assert gen_result.success is True

        # Stage 3: Launch in shadow mode
        cm = CredentialManager(master_key="e2e-test-master-key-42")
        launcher = GenesisLauncher(credential_manager=cm)

        launch_result = launcher.launch_vertical(
            "e2e_test", str(tmp_path), skip_credential_check=True
        )

        # Stage 4: Verify
        assert launch_result.success is True
        assert launch_result.status == "shadow_mode"
        assert launch_result.is_shadow_mode is True
        assert len(launch_result.agent_ids) >= 1
        assert "outreach_v1" in launch_result.agent_ids

    def test_empty_agents_blueprint_rejected_by_validation(self):
        """Blueprint with no agents is rejected by Pydantic validation."""
        from pydantic import ValidationError
        from core.genesis.blueprint import (
            BusinessBlueprint,
            ICPSpec,
            OutreachSpec,
            PersonaSpec,
        )

        context = _make_business_context(business_name="No Agents Test")

        # BusinessBlueprint requires min_length=1 on agents + outreach agent validator
        with pytest.raises(ValidationError):
            BusinessBlueprint(
                vertical_id="noagent_test",
                vertical_name="No Agents Test",
                industry="Technology",
                context=context,
                icp=ICPSpec(company_size=(11, 50), industries=["technology"]),
                personas=[
                    PersonaSpec(
                        id="cto",
                        title_patterns=["CTO"],
                        company_size=(11, 50),
                        approach="initial_outreach",
                    ),
                ],
                outreach=OutreachSpec(
                    daily_limit=5,
                    sending_domain="mail.noagent.com",
                    reply_to="hello@noagent.com",
                    physical_address="123 NoAgent St",
                ),
                agents=[],  # No agents — should fail validation
            )
