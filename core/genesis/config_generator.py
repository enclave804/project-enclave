"""
Genesis Config Generator — Blueprint → Validated YAML Files.

The "ribosome" of the platform: takes a BusinessBlueprint (DNA) and
translates it into functioning YAML config files (proteins) that the
existing platform can execute.

Critical safety guarantee:
    Every generated config is validated through the SAME Pydantic models
    (VerticalConfig, AgentInstanceConfig) that the platform uses at runtime.
    If validation fails, NO files are written. This eliminates "hallucinated
    configs" — the Architect cannot produce configs the engine can't run.

Usage:
    from core.genesis.blueprint import BusinessBlueprint
    from core.genesis.config_generator import ConfigGenerator

    blueprint = BusinessBlueprint(...)
    generator = ConfigGenerator()
    result = generator.generate_vertical(blueprint, output_dir="verticals")
    # result.success == True, result.paths contains generated file paths
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from core.config.agent_schema import AgentInstanceConfig
from core.config.schema import VerticalConfig
from core.genesis.blueprint import (
    AgentRole,
    AgentSpec,
    BusinessBlueprint,
    EnrichmentSourceSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result of a config generation attempt."""
    success: bool
    vertical_id: str
    paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def config_path(self) -> Optional[str]:
        """Path to the main config.yaml, if generated."""
        for p in self.paths:
            if p.endswith("config.yaml"):
                return p
        return None

    @property
    def agent_paths(self) -> list[str]:
        """Paths to generated agent YAML files."""
        return [p for p in self.paths if "/agents/" in p]


# ---------------------------------------------------------------------------
# Config Generator
# ---------------------------------------------------------------------------

class ConfigGenerationError(Exception):
    """Raised when config generation fails validation."""
    def __init__(self, message: str, validation_errors: Optional[list[str]] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or []


class ConfigGenerator:
    """
    Generates validated YAML config files from a BusinessBlueprint.

    The generator produces two types of files:
    1. verticals/{id}/config.yaml — VerticalConfig (business settings)
    2. verticals/{id}/agents/{type}.yaml — AgentInstanceConfig (per agent)

    Both are validated against their Pydantic schemas BEFORE writing.
    If validation fails, no files are written and errors are returned.
    """

    # Default tools for each agent type
    DEFAULT_AGENT_TOOLS: dict[str, list[str]] = {
        AgentRole.OUTREACH.value: [
            "apollo_search", "apollo_enrich", "send_email",
            "rag_search", "compliance_check",
        ],
        AgentRole.SEO_CONTENT.value: [
            "rag_search",
        ],
        AgentRole.APPOINTMENT_SETTER.value: [
            "rag_search", "send_email",
        ],
        AgentRole.JANITOR.value: [],
        AgentRole.COMMERCE.value: [
            "shopify_get_products", "shopify_get_recent_orders",
            "shopify_update_inventory", "stripe_create_payment_link",
            "stripe_check_payment", "stripe_process_refund",
            "rag_search",
        ],
        AgentRole.VOICE.value: [
            "send_sms", "make_call", "get_call_logs", "get_sms_logs",
            "transcribe_audio", "buy_phone_number", "rag_search",
        ],
    }

    # Default human gates for each agent type
    DEFAULT_HUMAN_GATES: dict[str, list[str]] = {
        AgentRole.OUTREACH.value: ["send_outreach"],
        AgentRole.SEO_CONTENT.value: ["human_review"],
        AgentRole.APPOINTMENT_SETTER.value: ["human_review"],
        AgentRole.JANITOR.value: [],
        AgentRole.COMMERCE.value: ["human_review"],
        AgentRole.VOICE.value: ["human_review"],
    }

    # Browser requirement by agent type
    BROWSER_AGENTS: set[str] = {AgentRole.SEO_CONTENT.value}

    def generate_vertical(
        self,
        blueprint: BusinessBlueprint,
        output_dir: str | Path = "verticals",
        *,
        dry_run: bool = False,
    ) -> GenerationResult:
        """
        Generate all config files for a vertical from a blueprint.

        Args:
            blueprint: The validated BusinessBlueprint.
            output_dir: Base directory for verticals (default: "verticals").
            dry_run: If True, validate but don't write files.

        Returns:
            GenerationResult with success status, paths, and any errors.

        Raises:
            ConfigGenerationError: If validation fails (with details).
        """
        result = GenerationResult(
            success=False,
            vertical_id=blueprint.vertical_id,
        )

        output_dir = Path(output_dir)
        vertical_dir = output_dir / blueprint.vertical_id
        agents_dir = vertical_dir / "agents"
        prompts_dir = vertical_dir / "prompts" / "agent_prompts"

        # --- Step 1: Build VerticalConfig dict ---
        try:
            vertical_dict = self._build_vertical_config(blueprint)
        except Exception as e:
            result.errors.append(f"Failed to build vertical config: {e}")
            return result

        # --- Step 2: Validate VerticalConfig ---
        try:
            validated_vertical = VerticalConfig(**vertical_dict)
            logger.info(
                "Vertical config validated",
                extra={"vertical_id": blueprint.vertical_id},
            )
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            result.errors.append(f"VerticalConfig validation failed: {e}")
            raise ConfigGenerationError(
                f"Generated VerticalConfig for '{blueprint.vertical_id}' "
                f"failed validation with {len(errors)} errors",
                validation_errors=errors,
            ) from e

        # --- Step 3: Build and validate AgentInstanceConfigs ---
        agent_configs: list[tuple[str, dict[str, Any]]] = []

        for agent_spec in blueprint.agents:
            try:
                agent_dict = self._build_agent_config(blueprint, agent_spec)
                # Validate through Pydantic
                AgentInstanceConfig(**agent_dict)
                agent_type_val = (
                    agent_spec.agent_type.value
                    if hasattr(agent_spec.agent_type, "value")
                    else str(agent_spec.agent_type)
                )
                agent_configs.append((agent_type_val, agent_dict))
                logger.info(
                    "Agent config validated",
                    extra={
                        "vertical_id": blueprint.vertical_id,
                        "agent_type": agent_spec.agent_type,
                    },
                )
            except ValidationError as e:
                errors = [str(err) for err in e.errors()]
                result.errors.append(
                    f"AgentInstanceConfig validation failed for "
                    f"'{agent_spec.agent_type}': {e}"
                )
                raise ConfigGenerationError(
                    f"Generated AgentInstanceConfig for "
                    f"'{agent_spec.agent_type}' in vertical "
                    f"'{blueprint.vertical_id}' failed validation",
                    validation_errors=errors,
                ) from e

        # --- Step 4: Write files (if not dry_run) ---
        if dry_run:
            result.success = True
            result.paths.append(str(vertical_dir / "config.yaml"))
            for agent_type, _ in agent_configs:
                result.paths.append(str(agents_dir / f"{agent_type}.yaml"))
            result.warnings.append("Dry run — no files written")
            return result

        try:
            # Create directories
            agents_dir.mkdir(parents=True, exist_ok=True)
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # Write __init__.py for Python package
            init_path = vertical_dir / "__init__.py"
            if not init_path.exists():
                init_path.write_text("")

            agents_init = agents_dir / "__init__.py"
            if not agents_init.exists():
                agents_init.write_text("")

            # Write config.yaml
            config_path = vertical_dir / "config.yaml"
            self._write_yaml(
                config_path,
                vertical_dict,
                header=self._vertical_header(blueprint),
            )
            result.paths.append(str(config_path))

            # Write agent YAMLs
            for agent_type, agent_dict in agent_configs:
                agent_path = agents_dir / f"{agent_type}.yaml"
                agent_spec = blueprint.get_agent_by_type(agent_type)
                self._write_yaml(
                    agent_path,
                    agent_dict,
                    header=self._agent_header(blueprint, agent_spec),
                )
                result.paths.append(str(agent_path))

            # Write system prompts if specified
            for agent_spec in blueprint.agents:
                if agent_spec.system_prompt_template:
                    at_val = (
                        agent_spec.agent_type.value
                        if hasattr(agent_spec.agent_type, "value")
                        else str(agent_spec.agent_type)
                    )
                    prompt_path = prompts_dir / f"{at_val}_system.md"
                    prompt_path.write_text(agent_spec.system_prompt_template)
                    result.paths.append(str(prompt_path))

            result.success = True
            logger.info(
                "Vertical generated successfully",
                extra={
                    "vertical_id": blueprint.vertical_id,
                    "files_created": len(result.paths),
                },
            )

        except OSError as e:
            result.errors.append(f"File write error: {e}")
            # Clean up on failure
            if vertical_dir.exists():
                shutil.rmtree(vertical_dir, ignore_errors=True)

        return result

    # --- Private: Build Config Dicts ---

    def _build_vertical_config(self, bp: BusinessBlueprint) -> dict[str, Any]:
        """
        Map a BusinessBlueprint to a VerticalConfig-compatible dict.

        This is the critical translation layer. The dict MUST pass
        VerticalConfig(**dict) validation.
        """
        # Build personas for targeting
        personas = []
        for p in bp.personas:
            personas.append({
                "id": p.id,
                "title_patterns": p.title_patterns,
                "company_size": list(p.company_size),
                "approach": p.approach,
            })

        # Build email sequences
        sequences = []
        for seq in bp.outreach.sequences:
            sequences.append({
                "name": seq.name,
                "steps": seq.steps,
                "delay_days": seq.delay_days,
            })

        # Build enrichment sources
        enrichment_sources = []
        for src in bp.enrichment_sources:
            source_dict: dict[str, Any] = {"type": src.type}
            if src.provider:
                source_dict["provider"] = src.provider
            if src.api_key_env:
                source_dict["api_key_env"] = src.api_key_env
            if src.targets:
                source_dict["targets"] = src.targets
            enrichment_sources.append(source_dict)

        # Build Apollo filters
        apollo_filters: dict[str, Any] = {
            "person_titles": bp.context.target_titles,
            "person_seniorities": self._extract_seniorities(bp),
            "organization_num_employees_ranges": self._format_employee_ranges(
                bp.icp.company_size
            ),
            "person_locations": bp.context.target_locations,
        }

        # Override with explicit Apollo filters if provided
        if bp.apollo_filters:
            apollo_filters.update(bp.apollo_filters)

        config = {
            "vertical_id": bp.vertical_id,
            "vertical_name": bp.vertical_name,
            "industry": bp.industry,
            "business": {
                "ticket_range": list(bp.context.price_range),
                "currency": bp.context.currency,
                "sales_cycle_days": bp.context.sales_cycle_days,
            },
            "targeting": {
                "ideal_customer_profile": {
                    "company_size": list(bp.icp.company_size),
                    "industries": bp.icp.industries,
                    "signals": bp.icp.signals,
                    "disqualifiers": bp.icp.disqualifiers,
                },
                "personas": personas,
            },
            "outreach": {
                "email": {
                    "daily_limit": bp.outreach.daily_limit,
                    "warmup_days": bp.outreach.warmup_days,
                    "sending_domain": bp.outreach.sending_domain,
                    "reply_to": bp.outreach.reply_to,
                    "sequences": sequences,
                },
                "compliance": {
                    "jurisdictions": [
                        j.value if hasattr(j, "value") else str(j)
                        for j in bp.outreach.jurisdictions
                    ],
                    "physical_address": bp.outreach.physical_address,
                    "exclude_countries": bp.outreach.exclude_countries,
                },
            },
            "enrichment": {
                "sources": enrichment_sources,
            },
            "apollo": {
                "filters": apollo_filters,
                "daily_lead_pull": bp.outreach.daily_limit,
            },
            "rag": {
                "chunk_types": [
                    "company_intel",
                    "outreach_result",
                    "winning_pattern",
                ],
            },
        }

        return config

    def _build_agent_config(
        self, bp: BusinessBlueprint, agent: AgentSpec
    ) -> dict[str, Any]:
        """
        Map an AgentSpec to an AgentInstanceConfig-compatible dict.
        """
        # Extract string value — AgentRole is a str enum, so str() gives
        # the repr (e.g. "AgentRole.OUTREACH"), but .value gives "outreach".
        agent_type_str = (
            agent.agent_type.value
            if hasattr(agent.agent_type, "value")
            else str(agent.agent_type)
        )

        # Determine agent_id
        agent_id = f"{agent_type_str}_v1"

        # Determine tools
        tools = agent.tools or self.DEFAULT_AGENT_TOOLS.get(agent_type_str, [])

        # Determine human gates
        human_gate_nodes = (
            agent.human_gate_nodes
            or self.DEFAULT_HUMAN_GATES.get(agent_type_str, [])
        )

        # Determine browser
        browser_enabled = (
            agent.browser_enabled
            or agent_type_str in self.BROWSER_AGENTS
        )

        config: dict[str, Any] = {
            "agent_id": agent_id,
            "agent_type": agent_type_str,
            "name": agent.name,
            "description": agent.description or f"{agent.name} for {bp.vertical_name}",
            "enabled": agent.enabled,
            "model": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
            "tools": tools,
            "browser_enabled": browser_enabled,
            "human_gates": {
                "enabled": len(human_gate_nodes) > 0,
                "gate_before": human_gate_nodes,
            },
            "schedule": {
                "trigger": "manual",
            },
            "max_consecutive_errors": 5,
            "rag_write_confidence_threshold": 0.7,
            "shadow_mode": True,  # ALL new verticals launch in shadow mode
            "params": self._build_agent_params(bp, agent),
        }

        # Set system prompt path if template provided
        if agent.system_prompt_template:
            config["system_prompt_path"] = (
                f"verticals/{bp.vertical_id}/prompts/agent_prompts/"
                f"{agent_type_str}_system.md"
            )

        return config

    def _build_agent_params(
        self, bp: BusinessBlueprint, agent: AgentSpec
    ) -> dict[str, Any]:
        """Build agent-specific params from blueprint context."""
        params = dict(agent.params)  # Start with explicit params

        agent_type_str = (
            agent.agent_type.value
            if hasattr(agent.agent_type, "value")
            else str(agent.agent_type)
        )

        if agent_type_str == AgentRole.OUTREACH.value:
            params.setdefault("daily_lead_limit", bp.outreach.daily_limit)
            params.setdefault(
                "duplicate_cooldown_days",
                90,
            )

        elif agent_type_str == AgentRole.SEO_CONTENT.value:
            params.setdefault("target_word_count", 1500)
            params.setdefault("tone", bp.tone)
            params.setdefault("content_type", "blog_post")
            if bp.content_topics:
                params.setdefault("target_topics", bp.content_topics)

        elif agent_type_str == AgentRole.APPOINTMENT_SETTER.value:
            params.setdefault("response_tone", bp.tone)

        elif agent_type_str == AgentRole.COMMERCE.value:
            params.setdefault("vip_threshold", 500.0)
            params.setdefault("low_stock_threshold", 5)
            params.setdefault("check_interval_hours", 6)

        return params

    # --- Private: Helpers ---

    def _extract_seniorities(self, bp: BusinessBlueprint) -> list[str]:
        """Extract unique seniorities from all personas."""
        seniorities: set[str] = set()
        for persona in bp.personas:
            seniorities.update(persona.seniorities)
        return sorted(seniorities) or ["c_suite", "vp", "director"]

    def _format_employee_ranges(
        self, company_size: tuple[int, int]
    ) -> list[str]:
        """
        Convert a company_size range into Apollo-format employee ranges.

        Example: (10, 500) → ["11,50", "51,200", "201,500"]
        """
        ranges = []
        boundaries = [1, 10, 50, 200, 500, 1000, 5000, 10000]
        min_emp, max_emp = company_size

        for i in range(len(boundaries) - 1):
            range_start = boundaries[i] + 1 if boundaries[i] > 1 else 1
            range_end = boundaries[i + 1]

            # Check if this range overlaps with our target
            if range_end >= min_emp and range_start <= max_emp:
                ranges.append(f"{boundaries[i] + 1},{boundaries[i + 1]}")

        return ranges or [f"{min_emp},{max_emp}"]

    @staticmethod
    def _sanitize_for_yaml(obj: Any) -> Any:
        """
        Deep-convert Python objects to YAML-safe primitives.

        Handles:
        - Enums → their .value (string/int)
        - Tuples → lists
        - Nested dicts/lists → recursively sanitized
        """
        from enum import Enum as _Enum

        if isinstance(obj, _Enum):
            return obj.value
        if isinstance(obj, tuple):
            return [ConfigGenerator._sanitize_for_yaml(item) for item in obj]
        if isinstance(obj, list):
            return [ConfigGenerator._sanitize_for_yaml(item) for item in obj]
        if isinstance(obj, dict):
            return {
                k: ConfigGenerator._sanitize_for_yaml(v)
                for k, v in obj.items()
            }
        return obj

    def _write_yaml(
        self,
        path: Path,
        data: dict[str, Any],
        header: str = "",
    ) -> None:
        """Write a dict to a YAML file with an optional header comment."""
        content = ""
        if header:
            content = header + "\n\n"

        # Sanitize: convert enums, tuples, etc. to YAML-safe primitives
        clean_data = self._sanitize_for_yaml(data)

        yaml_str = yaml.dump(
            clean_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        content += yaml_str
        path.write_text(content)

    def _vertical_header(self, bp: BusinessBlueprint) -> str:
        """Generate a header comment for config.yaml."""
        return (
            f"# {bp.vertical_name} — Vertical Configuration\n"
            f"#\n"
            f"# Industry: {bp.industry}\n"
            f"# Generated by: Genesis Engine (v1)\n"
            f"# Blueprint ID: {bp.id}\n"
            f"#\n"
            f"# This file conforms to core.config.schema.VerticalConfig.\n"
            f"# Modify with care — changes affect all agents in this vertical."
        )

    def _agent_header(
        self, bp: BusinessBlueprint, agent: Optional[AgentSpec]
    ) -> str:
        """Generate a header comment for an agent YAML."""
        if agent is None:
            return "# Agent Configuration"
        return (
            f"# {agent.name} — {bp.vertical_name}\n"
            f"#\n"
            f"# Type: {agent.agent_type}\n"
            f"# Generated by: Genesis Engine (v1)\n"
            f"#\n"
            f"# This file conforms to core.config.agent_schema.AgentInstanceConfig."
        )


# ---------------------------------------------------------------------------
# Convenience: Validate existing configs
# ---------------------------------------------------------------------------

def validate_existing_vertical(vertical_dir: str | Path) -> GenerationResult:
    """
    Validate that an existing vertical's YAML files conform to schemas.

    Useful for auditing manually-created configs or post-upgrade checks.
    """
    vertical_dir = Path(vertical_dir)
    vertical_id = vertical_dir.name
    result = GenerationResult(success=True, vertical_id=vertical_id)

    # Validate config.yaml
    config_path = vertical_dir / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            if raw is None:
                result.errors.append("config.yaml is empty")
                result.success = False
            else:
                if "vertical_id" not in raw:
                    raw["vertical_id"] = vertical_id
                VerticalConfig(**raw)
                result.paths.append(str(config_path))
        except ValidationError as e:
            result.errors.append(f"config.yaml: {e}")
            result.success = False
    else:
        result.errors.append("config.yaml not found")
        result.success = False

    # Validate agent YAMLs
    agents_dir = vertical_dir / "agents"
    if agents_dir.is_dir():
        for agent_file in sorted(agents_dir.glob("*.yaml")):
            try:
                with open(agent_file) as f:
                    raw = yaml.safe_load(f)
                if raw is None:
                    result.warnings.append(f"{agent_file.name}: empty file")
                    continue
                AgentInstanceConfig(**raw)
                result.paths.append(str(agent_file))
            except ValidationError as e:
                result.errors.append(f"{agent_file.name}: {e}")
                result.success = False

    return result
