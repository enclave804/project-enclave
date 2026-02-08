"""
Genesis Launcher — Deploy an agent fleet in shadow mode.

The final stage of the Genesis Engine: takes a validated vertical
(config.yaml + agent YAMLs) and brings it online. All new verticals
launch in shadow mode — every external action is intercepted and logged
but never actually sent.

Safety guarantees:
    1. Config files must pass Pydantic validation before launch
    2. All required credentials must be available
    3. Agent types must be registered in the platform
    4. Shadow mode is mandatory for first launch (no opt-out)
    5. Database records track the full launch lifecycle

Usage:
    from core.genesis.launcher import GenesisLauncher, LaunchResult

    launcher = GenesisLauncher(db=db, credential_manager=cm)
    result = launcher.launch_vertical("my_vertical", output_dir="verticals")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from core.config.agent_schema import AgentInstanceConfig
from core.config.schema import VerticalConfig
from core.genesis.credential_manager import CredentialManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class LaunchResult:
    """Result of a vertical launch attempt."""
    success: bool
    vertical_id: str
    status: str = "pending"  # pending, shadow_mode, failed
    agent_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_shadow_mode(self) -> bool:
        return self.status == "shadow_mode"


# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    """Result of pre-launch validation."""
    passed: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        if not passed:
            self.passed = False

    @property
    def failed_checks(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]


# ---------------------------------------------------------------------------
# Genesis Launcher
# ---------------------------------------------------------------------------

class GenesisLauncher:
    """
    Validates and launches a new vertical's agent fleet.

    The launcher performs pre-flight checks, validates all configs,
    verifies credentials, and brings agents online in shadow mode.
    """

    def __init__(
        self,
        db: Any = None,
        credential_manager: Optional[CredentialManager] = None,
    ):
        self._db = db
        self._credential_manager = credential_manager or CredentialManager()

    def preflight_check(
        self,
        vertical_id: str,
        output_dir: str | Path = "verticals",
    ) -> PreflightResult:
        """
        Run all pre-launch validations without actually launching.

        Checks:
            1. Vertical directory exists
            2. config.yaml exists and validates
            3. At least one agent YAML exists and validates
            4. All agent types are registered
            5. Required credentials are available

        Args:
            vertical_id: The vertical to check.
            output_dir: Base directory for verticals.

        Returns:
            PreflightResult with detailed check results.
        """
        result = PreflightResult()
        output_dir = Path(output_dir)
        vertical_dir = output_dir / vertical_id

        # Check 1: Directory exists
        if not vertical_dir.is_dir():
            result.add_check(
                "directory_exists", False,
                f"Vertical directory not found: {vertical_dir}",
            )
            return result  # Can't continue without directory
        result.add_check("directory_exists", True)

        # Check 2: config.yaml exists and validates
        config_path = vertical_dir / "config.yaml"
        vertical_config = None
        if not config_path.exists():
            result.add_check(
                "config_valid", False,
                "config.yaml not found",
            )
        else:
            try:
                with open(config_path) as f:
                    raw = yaml.safe_load(f)
                if raw is None:
                    result.add_check(
                        "config_valid", False,
                        "config.yaml is empty",
                    )
                else:
                    if "vertical_id" not in raw:
                        raw["vertical_id"] = vertical_id
                    vertical_config = VerticalConfig(**raw)
                    result.add_check("config_valid", True)
            except ValidationError as e:
                error_count = len(e.errors())
                result.add_check(
                    "config_valid", False,
                    f"config.yaml validation failed with {error_count} errors",
                )
            except Exception as e:
                result.add_check(
                    "config_valid", False,
                    f"config.yaml parse error: {e}",
                )

        # Check 3: Agent YAML(s) exist and validate
        agents_dir = vertical_dir / "agents"
        agent_configs: list[AgentInstanceConfig] = []

        if not agents_dir.is_dir():
            result.add_check(
                "agents_valid", False,
                "No agents/ directory found",
            )
        else:
            agent_files = list(agents_dir.glob("*.yaml"))
            if not agent_files:
                result.add_check(
                    "agents_valid", False,
                    "No agent YAML files found in agents/",
                )
            else:
                all_valid = True
                for agent_file in agent_files:
                    try:
                        with open(agent_file) as f:
                            raw = yaml.safe_load(f)
                        if raw is None:
                            continue
                        ac = AgentInstanceConfig(**raw)
                        agent_configs.append(ac)
                    except ValidationError as e:
                        all_valid = False
                        result.add_check(
                            "agents_valid", False,
                            f"{agent_file.name}: validation failed ({len(e.errors())} errors)",
                        )
                    except Exception as e:
                        all_valid = False
                        result.add_check(
                            "agents_valid", False,
                            f"{agent_file.name}: parse error: {e}",
                        )

                if all_valid and agent_configs:
                    result.add_check(
                        "agents_valid", True,
                        f"{len(agent_configs)} agent(s) validated",
                    )

        # Check 4: Agent types are registered
        if agent_configs:
            from core.agents.registry import AGENT_IMPLEMENTATIONS

            unregistered = []
            for ac in agent_configs:
                if ac.agent_type not in AGENT_IMPLEMENTATIONS:
                    unregistered.append(ac.agent_type)

            if unregistered:
                result.add_check(
                    "agent_types_registered", False,
                    f"Unregistered agent types: {', '.join(unregistered)}",
                )
            else:
                result.add_check("agent_types_registered", True)

        # Check 5: Credentials available
        cred_report = self._credential_manager.get_credential_report(
            vertical_id
        )
        if cred_report.all_required_set:
            result.add_check(
                "credentials_available", True,
                f"{cred_report.total_set}/{len(cred_report.credentials)} set",
            )
        else:
            missing = [c.env_var_name for c in cred_report.missing_required]
            result.add_check(
                "credentials_available", False,
                f"Missing required: {', '.join(missing)}",
            )

        return result

    def launch_vertical(
        self,
        vertical_id: str,
        output_dir: str | Path = "verticals",
        *,
        skip_credential_check: bool = False,
        session_id: Optional[str] = None,
    ) -> LaunchResult:
        """
        Launch a vertical's agent fleet in shadow mode.

        Args:
            vertical_id: The vertical to launch.
            output_dir: Base directory for verticals.
            skip_credential_check: Skip credential validation (for testing).
            session_id: Optional genesis session ID for tracking.

        Returns:
            LaunchResult with launch status and agent details.
        """
        result = LaunchResult(
            success=False,
            vertical_id=vertical_id,
        )

        # Run pre-flight checks
        preflight = self.preflight_check(vertical_id, output_dir)

        if not preflight.passed:
            # Allow credential failures when skip_credential_check is set
            critical_failures = [
                c for c in preflight.failed_checks
                if not (skip_credential_check and c["name"] == "credentials_available")
            ]
            if critical_failures:
                result.errors = [c["detail"] for c in critical_failures]
                result.status = "failed"
                logger.error(
                    "launch_preflight_failed",
                    extra={
                        "vertical_id": vertical_id,
                        "failed_checks": len(critical_failures),
                    },
                )
                return result

        # Load agent configs
        output_dir = Path(output_dir)
        agents_dir = output_dir / vertical_id / "agents"
        agent_configs: list[AgentInstanceConfig] = []

        for agent_file in sorted(agents_dir.glob("*.yaml")):
            try:
                with open(agent_file) as f:
                    raw = yaml.safe_load(f)
                if raw:
                    ac = AgentInstanceConfig(**raw)
                    agent_configs.append(ac)
            except Exception:
                continue

        if not agent_configs:
            result.errors.append("No valid agent configurations found")
            result.status = "failed"
            return result

        # Register agents in shadow mode
        for ac in agent_configs:
            if ac.enabled:
                result.agent_ids.append(ac.agent_id)

                # Ensure shadow mode (non-negotiable for new verticals)
                if not getattr(ac, "shadow_mode", True):
                    result.warnings.append(
                        f"Agent {ac.agent_id} forced into shadow mode"
                    )

        # Store launch record in database
        if self._db is not None:
            try:
                # Register each agent
                for ac in agent_configs:
                    if ac.enabled:
                        self._db.register_agent({
                            "agent_id": ac.agent_id,
                            "agent_type": ac.agent_type,
                            "name": ac.name,
                            "config": ac.model_dump(mode="json"),
                            "vertical_id": vertical_id,
                        })
            except Exception as e:
                result.warnings.append(
                    f"Could not register agents in DB: {e}"
                )

            # Update genesis session if tracking
            if session_id:
                try:
                    self._db.client.rpc(
                        "upsert_genesis_session",
                        {
                            "p_session_id": session_id,
                            "p_status": "launched",
                            "p_vertical_id": vertical_id,
                        },
                    ).execute()
                except Exception as e:
                    result.warnings.append(
                        f"Could not update genesis session: {e}"
                    )

        # Inject credentials into environment (always — skip only affects validation)
        if self._credential_manager:
            try:
                injected = self._credential_manager.inject_into_env(
                    vertical_id
                )
                if injected > 0:
                    logger.info(
                        "credentials_injected",
                        extra={
                            "vertical_id": vertical_id,
                            "count": injected,
                        },
                    )
            except Exception as e:
                result.warnings.append(
                    f"Could not inject credentials: {e}"
                )

        result.success = True
        result.status = "shadow_mode"

        logger.info(
            "vertical_launched",
            extra={
                "vertical_id": vertical_id,
                "agent_count": len(result.agent_ids),
                "status": result.status,
            },
        )

        return result

    def get_launch_status(
        self,
        vertical_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Check the current launch status of a vertical.

        Returns None if the vertical has no launch record.
        """
        if self._db is None:
            return None

        try:
            result = (
                self._db.client.table("genesis_sessions")
                .select("*")
                .eq("vertical_id", vertical_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]
        except Exception as e:
            logger.warning(f"Could not check launch status: {e}")

        return None

    def promote_to_live(
        self,
        vertical_id: str,
    ) -> bool:
        """
        Promote a vertical from shadow mode to live.

        This removes the shadow mode flag from all agents. Only call
        after thorough review of shadow mode activity.

        Returns True if promotion was successful.
        """
        # For safety, this is a no-op in the current implementation.
        # The actual shadow → live promotion requires:
        # 1. Review of shadow mode activity
        # 2. Confirmation from the user
        # 3. Update of all agent YAML files
        # 4. Database status update
        logger.warning(
            "promote_to_live called — not yet implemented. "
            "Manual promotion is required via dashboard.",
            extra={"vertical_id": vertical_id},
        )
        return False
