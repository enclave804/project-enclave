"""
Tests for the Genesis Launcher.

Covers:
- Pre-flight checks (directory, config, agents, types, credentials)
- Launch flow (shadow mode, agent registration)
- Error handling (missing directory, invalid config, no agents)
- Credential injection during launch
- Launch status tracking
- Promotion to live (safety)
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import yaml

from core.genesis.credential_manager import CredentialManager
from core.genesis.launcher import (
    GenesisLauncher,
    LaunchResult,
    PreflightResult,
)


# ---------------------------------------------------------------------------
# Fixtures: Create real vertical directories with valid YAMLs
# ---------------------------------------------------------------------------

TEST_MASTER_KEY = "test-launcher-master-key-42"


def _write_valid_config(vertical_dir: Path) -> None:
    """Write a minimal valid config.yaml to a directory."""
    config = {
        "vertical_id": vertical_dir.name,
        "vertical_name": "Test Vertical",
        "industry": "Technology",
        "business": {
            "ticket_range": [500, 5000],
            "currency": "USD",
            "sales_cycle_days": 30,
        },
        "targeting": {
            "ideal_customer_profile": {
                "company_size": [10, 500],
                "industries": ["Technology"],
                "signals": ["hiring"],
                "disqualifiers": ["bankrupt"],
            },
            "personas": [
                {
                    "id": "cto",
                    "title_patterns": ["CTO"],
                    "company_size": [10, 500],
                    "approach": "tech_pitch",
                }
            ],
        },
        "outreach": {
            "email": {
                "daily_limit": 25,
                "warmup_days": 14,
                "sending_domain": "mail.test.com",
                "reply_to": "hi@test.com",
                "sequences": [
                    {"name": "tech_pitch", "steps": 3, "delay_days": [0, 3, 7]},
                ],
            },
            "compliance": {
                "jurisdictions": ["US_CAN_SPAM"],
                "physical_address": "123 Test St",
                "exclude_countries": [],
            },
        },
        "enrichment": {
            "sources": [{"type": "web_scraper"}],
        },
        "apollo": {
            "filters": {
                "person_titles": ["CTO"],
                "person_seniorities": ["c_suite"],
                "organization_num_employees_ranges": ["11,50"],
                "person_locations": ["United States"],
            },
            "daily_lead_pull": 25,
        },
        "rag": {
            "chunk_types": ["company_intel"],
        },
    }

    config_path = vertical_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, sort_keys=False))


def _write_valid_agent(vertical_dir: Path, agent_type: str = "outreach") -> None:
    """Write a minimal valid agent YAML to a directory."""
    agents_dir = vertical_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    agent_config = {
        "agent_id": f"{agent_type}_v1",
        "agent_type": agent_type,
        "name": f"{agent_type.title()} Agent",
        "enabled": True,
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.5,
            "max_tokens": 4096,
        },
        "tools": [],
        "browser_enabled": False,
        "human_gates": {
            "enabled": True,
            "gate_before": ["send_outreach"],
        },
        "schedule": {"trigger": "manual"},
        "shadow_mode": True,
        "params": {},
    }

    agent_path = agents_dir / f"{agent_type}.yaml"
    agent_path.write_text(yaml.dump(agent_config, sort_keys=False))


@pytest.fixture
def valid_vertical(tmp_path) -> Path:
    """Create a complete valid vertical directory."""
    vertical_dir = tmp_path / "test_biz"
    vertical_dir.mkdir()
    _write_valid_config(vertical_dir)
    _write_valid_agent(vertical_dir, "outreach")
    return tmp_path


@pytest.fixture
def launcher() -> GenesisLauncher:
    """Create a launcher with mocked DB and in-memory creds."""
    cm = CredentialManager(master_key=TEST_MASTER_KEY)
    return GenesisLauncher(credential_manager=cm)


@pytest.fixture
def launcher_with_creds(valid_vertical) -> tuple[GenesisLauncher, Path]:
    """Launcher with credentials pre-populated for the test vertical."""
    cm = CredentialManager(master_key=TEST_MASTER_KEY)
    # Set all platform credentials
    cm.set_credential("test_biz", "ANTHROPIC_API_KEY", "sk-ant-test")
    cm.set_credential("test_biz", "SUPABASE_URL", "https://test.supabase.co")
    cm.set_credential("test_biz", "SUPABASE_SERVICE_KEY", "sbp_test")
    cm.set_credential("test_biz", "OPENAI_API_KEY", "sk-test")
    cm.set_credential("test_biz", "APOLLO_API_KEY", "apollo-test")
    launcher = GenesisLauncher(credential_manager=cm)
    return launcher, valid_vertical


# ---------------------------------------------------------------------------
# Test: PreflightResult
# ---------------------------------------------------------------------------

class TestPreflightResult:
    """Test the PreflightResult data structure."""

    def test_empty_result_passes(self):
        """Empty preflight result (no checks) passes."""
        result = PreflightResult()
        assert result.passed is True
        assert result.failed_checks == []

    def test_passing_check(self):
        """Adding a passing check keeps result passing."""
        result = PreflightResult()
        result.add_check("test", True)
        assert result.passed is True

    def test_failing_check(self):
        """Adding a failing check marks result as failed."""
        result = PreflightResult()
        result.add_check("test", False, "Something broke")
        assert result.passed is False
        assert len(result.failed_checks) == 1
        assert result.failed_checks[0]["detail"] == "Something broke"

    def test_mixed_checks(self):
        """Mix of passing and failing checks → fails overall."""
        result = PreflightResult()
        result.add_check("good", True)
        result.add_check("bad", False, "Failed")
        result.add_check("also_good", True)
        assert result.passed is False
        assert len(result.failed_checks) == 1


# ---------------------------------------------------------------------------
# Test: LaunchResult
# ---------------------------------------------------------------------------

class TestLaunchResult:
    """Test the LaunchResult data structure."""

    def test_default_values(self):
        """Default launch result is not successful."""
        result = LaunchResult(success=False, vertical_id="v1")
        assert result.success is False
        assert result.status == "pending"
        assert result.agent_ids == []
        assert result.errors == []

    def test_shadow_mode_property(self):
        """is_shadow_mode property works."""
        result = LaunchResult(success=True, vertical_id="v1", status="shadow_mode")
        assert result.is_shadow_mode is True

        result2 = LaunchResult(success=True, vertical_id="v1", status="failed")
        assert result2.is_shadow_mode is False


# ---------------------------------------------------------------------------
# Test: Preflight Checks
# ---------------------------------------------------------------------------

class TestPreflightChecks:
    """Test pre-launch validation."""

    def test_missing_directory(self, launcher, tmp_path):
        """Preflight fails when vertical directory doesn't exist."""
        result = launcher.preflight_check("nonexistent", tmp_path)
        assert result.passed is False
        assert any("directory" in c["name"] for c in result.failed_checks)

    def test_missing_config_yaml(self, launcher, tmp_path):
        """Preflight fails when config.yaml is missing."""
        vertical_dir = tmp_path / "test_biz"
        vertical_dir.mkdir()
        # No config.yaml
        result = launcher.preflight_check("test_biz", tmp_path)
        assert result.passed is False

    def test_empty_config_yaml(self, launcher, tmp_path):
        """Preflight fails when config.yaml is empty."""
        vertical_dir = tmp_path / "test_biz"
        vertical_dir.mkdir()
        (vertical_dir / "config.yaml").write_text("")
        result = launcher.preflight_check("test_biz", tmp_path)
        assert result.passed is False

    def test_invalid_config_yaml(self, launcher, tmp_path):
        """Preflight fails when config.yaml has validation errors."""
        vertical_dir = tmp_path / "test_biz"
        vertical_dir.mkdir()
        (vertical_dir / "config.yaml").write_text(
            yaml.dump({"vertical_id": "test_biz"})
        )
        result = launcher.preflight_check("test_biz", tmp_path)
        assert result.passed is False

    def test_no_agents_directory(self, launcher, tmp_path):
        """Preflight fails when agents/ directory is missing."""
        vertical_dir = tmp_path / "test_biz"
        vertical_dir.mkdir()
        _write_valid_config(vertical_dir)
        # No agents/ directory
        result = launcher.preflight_check("test_biz", tmp_path)
        assert result.passed is False

    def test_empty_agents_directory(self, launcher, tmp_path):
        """Preflight fails when agents/ directory is empty."""
        vertical_dir = tmp_path / "test_biz"
        vertical_dir.mkdir()
        _write_valid_config(vertical_dir)
        (vertical_dir / "agents").mkdir()
        # No YAML files
        result = launcher.preflight_check("test_biz", tmp_path)
        assert result.passed is False

    def test_valid_vertical_passes_structure(self, launcher, valid_vertical):
        """Valid vertical passes structure checks."""
        result = launcher.preflight_check("test_biz", valid_vertical)
        check_names = [c["name"] for c in result.checks if c["passed"]]
        assert "directory_exists" in check_names
        assert "config_valid" in check_names
        assert "agents_valid" in check_names

    def test_credential_check_fails_when_missing(self, launcher, valid_vertical):
        """Preflight reports missing credentials."""
        result = launcher.preflight_check("test_biz", valid_vertical)
        cred_check = next(
            c for c in result.checks if c["name"] == "credentials_available"
        )
        assert cred_check["passed"] is False

    def test_credential_check_passes_with_env(self, launcher, valid_vertical):
        """Preflight passes when credentials are in os.environ."""
        env_override = {
            "ANTHROPIC_API_KEY": "test",
            "SUPABASE_URL": "test",
            "SUPABASE_SERVICE_KEY": "test",
            "OPENAI_API_KEY": "test",
            "APOLLO_API_KEY": "test",
        }
        with patch.dict(os.environ, env_override, clear=False):
            result = launcher.preflight_check("test_biz", valid_vertical)
            cred_check = next(
                c for c in result.checks if c["name"] == "credentials_available"
            )
            assert cred_check["passed"] is True

    def test_agent_type_registration_check(self, launcher, valid_vertical):
        """Preflight checks that agent types are registered."""
        # outreach is registered (imported in conftest or via decorator)
        # Force import of outreach agent
        import core.agents.implementations.outreach_agent  # noqa: F401

        result = launcher.preflight_check("test_biz", valid_vertical)
        type_check = next(
            (c for c in result.checks if c["name"] == "agent_types_registered"),
            None,
        )
        if type_check:
            assert type_check["passed"] is True


# ---------------------------------------------------------------------------
# Test: Launch Flow
# ---------------------------------------------------------------------------

class TestLaunchFlow:
    """Test the full launch flow."""

    def test_launch_success_shadow_mode(self, launcher_with_creds):
        """Successful launch puts vertical in shadow mode."""
        launcher, output_dir = launcher_with_creds

        # Force import so agent type is registered
        import core.agents.implementations.outreach_agent  # noqa: F401

        result = launcher.launch_vertical(
            "test_biz", output_dir, skip_credential_check=True
        )
        assert result.success is True
        assert result.status == "shadow_mode"
        assert "outreach_v1" in result.agent_ids

    def test_launch_skips_disabled_agents(self, tmp_path):
        """Disabled agents are not included in launch."""
        vertical_dir = tmp_path / "test_biz"
        vertical_dir.mkdir()
        _write_valid_config(vertical_dir)

        # Write a disabled agent
        agents_dir = vertical_dir / "agents"
        agents_dir.mkdir()
        agent_config = {
            "agent_id": "outreach_v1",
            "agent_type": "outreach",
            "name": "Outreach Agent",
            "enabled": False,  # Disabled
            "model": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
            "tools": [],
            "schedule": {"trigger": "manual"},
            "shadow_mode": True,
            "params": {},
        }
        (agents_dir / "outreach.yaml").write_text(
            yaml.dump(agent_config, sort_keys=False)
        )

        cm = CredentialManager(master_key=TEST_MASTER_KEY)
        launcher = GenesisLauncher(credential_manager=cm)

        result = launcher.launch_vertical(
            "test_biz", tmp_path, skip_credential_check=True
        )
        assert result.success is True
        assert "outreach_v1" not in result.agent_ids

    def test_launch_fails_without_directory(self, launcher, tmp_path):
        """Launch fails when vertical directory doesn't exist."""
        result = launcher.launch_vertical("nonexistent", tmp_path)
        assert result.success is False
        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_launch_fails_with_invalid_config(self, launcher, tmp_path):
        """Launch fails when config.yaml is invalid."""
        vertical_dir = tmp_path / "bad_biz"
        vertical_dir.mkdir()
        (vertical_dir / "config.yaml").write_text("invalid: true")
        result = launcher.launch_vertical("bad_biz", tmp_path)
        assert result.success is False

    def test_launch_with_multiple_agents(self, tmp_path):
        """Launch with multiple agent types."""
        import core.agents.implementations.outreach_agent  # noqa: F401
        import core.agents.implementations.seo_content_agent  # noqa: F401

        vertical_dir = tmp_path / "multi_biz"
        vertical_dir.mkdir()
        _write_valid_config(vertical_dir)
        _write_valid_agent(vertical_dir, "outreach")

        # Add a second agent
        agents_dir = vertical_dir / "agents"
        agent2 = {
            "agent_id": "seo_content_v1",
            "agent_type": "seo_content",
            "name": "SEO Agent",
            "enabled": True,
            "model": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
            "tools": [],
            "browser_enabled": True,
            "schedule": {"trigger": "manual"},
            "shadow_mode": True,
            "params": {},
        }
        (agents_dir / "seo_content.yaml").write_text(
            yaml.dump(agent2, sort_keys=False)
        )

        cm = CredentialManager(master_key=TEST_MASTER_KEY)
        launcher = GenesisLauncher(credential_manager=cm)

        result = launcher.launch_vertical(
            "multi_biz", tmp_path, skip_credential_check=True
        )
        assert result.success is True
        assert len(result.agent_ids) == 2
        assert "outreach_v1" in result.agent_ids
        assert "seo_content_v1" in result.agent_ids


# ---------------------------------------------------------------------------
# Test: Database Integration (mocked)
# ---------------------------------------------------------------------------

class TestDatabaseIntegration:
    """Test launch with mocked database."""

    def test_launch_registers_agents_in_db(self, valid_vertical):
        """Launch registers each agent in the database."""
        import core.agents.implementations.outreach_agent  # noqa: F401

        mock_db = MagicMock()
        mock_db.register_agent = MagicMock()

        cm = CredentialManager(master_key=TEST_MASTER_KEY)
        launcher = GenesisLauncher(db=mock_db, credential_manager=cm)

        result = launcher.launch_vertical(
            "test_biz", valid_vertical, skip_credential_check=True
        )
        assert result.success is True
        mock_db.register_agent.assert_called_once()

        call_args = mock_db.register_agent.call_args[0][0]  # First positional arg (dict)
        assert call_args["agent_id"] == "outreach_v1"
        assert call_args["vertical_id"] == "test_biz"

    def test_launch_updates_genesis_session(self, valid_vertical):
        """Launch updates genesis session when session_id provided."""
        import core.agents.implementations.outreach_agent  # noqa: F401

        mock_db = MagicMock()
        mock_rpc_result = MagicMock()
        mock_db.client.rpc.return_value = mock_rpc_result
        mock_rpc_result.execute.return_value = MagicMock()
        mock_db.register_agent = MagicMock()

        cm = CredentialManager(master_key=TEST_MASTER_KEY)
        launcher = GenesisLauncher(db=mock_db, credential_manager=cm)

        result = launcher.launch_vertical(
            "test_biz",
            valid_vertical,
            skip_credential_check=True,
            session_id="sess-123",
        )
        assert result.success is True
        mock_db.client.rpc.assert_called_once_with(
            "upsert_genesis_session",
            {
                "p_session_id": "sess-123",
                "p_status": "launched",
                "p_vertical_id": "test_biz",
            },
        )

    def test_db_failure_continues_launch(self, valid_vertical):
        """Database failure doesn't prevent launch."""
        import core.agents.implementations.outreach_agent  # noqa: F401

        mock_db = MagicMock()
        mock_db.register_agent = MagicMock(side_effect=Exception("DB down"))

        cm = CredentialManager(master_key=TEST_MASTER_KEY)
        launcher = GenesisLauncher(db=mock_db, credential_manager=cm)

        result = launcher.launch_vertical(
            "test_biz", valid_vertical, skip_credential_check=True
        )
        assert result.success is True  # Launch succeeds despite DB error
        assert len(result.warnings) > 0  # But warning is recorded


# ---------------------------------------------------------------------------
# Test: Credential Injection
# ---------------------------------------------------------------------------

class TestCredentialInjection:
    """Test credential injection during launch."""

    def test_credentials_injected_on_launch(self, valid_vertical):
        """Credentials are injected into os.environ during launch."""
        import core.agents.implementations.outreach_agent  # noqa: F401

        cm = CredentialManager(master_key=TEST_MASTER_KEY)
        cm.set_credential("test_biz", "TEST_LAUNCH_KEY", "injected-val")

        launcher = GenesisLauncher(credential_manager=cm)

        # Remove if exists
        os.environ.pop("TEST_LAUNCH_KEY", None)

        result = launcher.launch_vertical(
            "test_biz", valid_vertical, skip_credential_check=True
        )
        assert result.success is True

        # Check that the credential was injected
        assert os.environ.get("TEST_LAUNCH_KEY") == "injected-val"

        # Clean up
        os.environ.pop("TEST_LAUNCH_KEY", None)


# ---------------------------------------------------------------------------
# Test: Promote to Live
# ---------------------------------------------------------------------------

class TestPromoteToLive:
    """Test shadow → live promotion (safety)."""

    def test_promotion_not_yet_implemented(self, launcher):
        """Promotion returns False (safety — not auto-promoted)."""
        result = launcher.promote_to_live("any_vertical")
        assert result is False


# ---------------------------------------------------------------------------
# Test: Launch Status
# ---------------------------------------------------------------------------

class TestLaunchStatus:
    """Test launch status checking."""

    def test_status_returns_none_without_db(self, launcher):
        """Status returns None when no DB is available."""
        result = launcher.get_launch_status("v1")
        assert result is None

    def test_status_queries_db(self):
        """Status queries the genesis_sessions table."""
        mock_db = MagicMock()
        mock_chain = MagicMock()
        mock_chain.select.return_value = mock_chain
        mock_chain.eq.return_value = mock_chain
        mock_chain.order.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        mock_chain.execute.return_value = MagicMock(
            data=[{"status": "launched", "vertical_id": "v1"}]
        )
        mock_db.client.table.return_value = mock_chain

        launcher = GenesisLauncher(db=mock_db)
        result = launcher.get_launch_status("v1")
        assert result is not None
        assert result["status"] == "launched"
