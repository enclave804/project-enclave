"""
Config-driven agent registry for the Sovereign Venture Engine.

Discovers agent YAML configs, maps them to implementations, and
instantiates agents. New agent = YAML config + @register_agent_type decorator.

Usage:
    from core.agents.registry import AgentRegistry, register_agent_type

    @register_agent_type("my_agent")
    class MyAgent(BaseAgent):
        ...

    registry = AgentRegistry("enclave_guard", verticals_dir)
    registry.discover_agents()
    agents = registry.instantiate_all(db=db, embedder=embedder, ...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Type

import yaml

from core.agents.base import BaseAgent
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# Global map: agent_type string -> implementation class
AGENT_IMPLEMENTATIONS: dict[str, Type[BaseAgent]] = {}


def register_agent_type(agent_type: str):
    """
    Decorator to register an agent implementation.

    Usage:
        @register_agent_type("outreach")
        class OutreachAgent(BaseAgent):
            ...
    """

    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        if agent_type in AGENT_IMPLEMENTATIONS:
            logger.warning(
                f"Overwriting existing agent type registration: {agent_type}"
            )
        AGENT_IMPLEMENTATIONS[agent_type] = cls
        cls.agent_type = agent_type
        return cls

    return decorator


def get_registered_types() -> list[str]:
    """Return all registered agent type names."""
    return sorted(AGENT_IMPLEMENTATIONS.keys())


class AgentRegistry:
    """
    Discovers, loads, and manages agent instances for a vertical.

    Reads agent YAML configs from verticals/{vertical_id}/agents/*.yaml
    and instantiates the corresponding BaseAgent subclasses.
    """

    def __init__(self, vertical_id: str, verticals_dir: Optional[Path] = None):
        self.vertical_id = vertical_id
        if verticals_dir is None:
            verticals_dir = Path(__file__).parent.parent.parent / "verticals"
        self.verticals_dir = verticals_dir
        self.agents_dir = verticals_dir / vertical_id / "agents"
        self._configs: dict[str, AgentInstanceConfig] = {}
        self._agents: dict[str, BaseAgent] = {}

    def discover_agents(self) -> list[AgentInstanceConfig]:
        """
        Discover all agent configs in the vertical's agents/ directory.

        Returns list of validated AgentInstanceConfig objects.
        """
        if not self.agents_dir.exists():
            logger.info(f"No agents directory found: {self.agents_dir}")
            return []

        configs: list[AgentInstanceConfig] = []
        for yaml_file in sorted(self.agents_dir.glob("*.yaml")):
            try:
                with open(yaml_file) as f:
                    raw = yaml.safe_load(f)

                if not raw:
                    logger.warning(f"Empty agent config: {yaml_file.name}")
                    continue

                # Inject vertical_id
                raw["vertical_id"] = self.vertical_id
                config = AgentInstanceConfig(**raw)

                if not config.enabled:
                    logger.info(f"Agent disabled: {config.agent_id}")
                    continue

                self._configs[config.agent_id] = config
                configs.append(config)
                logger.info(
                    f"Discovered agent: {config.agent_id} "
                    f"(type={config.agent_type}, name={config.name!r})"
                )

            except Exception as e:
                logger.error(f"Failed to load agent config {yaml_file.name}: {e}")

        return configs

    def instantiate_agent(
        self,
        agent_id: str,
        db: Any,
        embedder: Any,
        anthropic_client: Any,
        **kwargs: Any,
    ) -> BaseAgent:
        """Instantiate a single agent by ID."""
        config = self._configs.get(agent_id)
        if not config:
            raise ValueError(
                f"Agent config not found: {agent_id!r}. "
                f"Available: {list(self._configs.keys())}"
            )

        cls = AGENT_IMPLEMENTATIONS.get(config.agent_type)
        if not cls:
            raise ValueError(
                f"No implementation registered for agent type: {config.agent_type!r}. "
                f"Registered types: {get_registered_types()}. "
                f"Did you forget the @register_agent_type decorator?"
            )

        agent = cls(
            config=config,
            db=db,
            embedder=embedder,
            anthropic_client=anthropic_client,
            **kwargs,
        )
        self._agents[agent_id] = agent
        logger.info(f"Instantiated agent: {agent}")
        return agent

    def instantiate_all(
        self,
        db: Any,
        embedder: Any,
        anthropic_client: Any,
        **kwargs: Any,
    ) -> dict[str, BaseAgent]:
        """Instantiate all discovered agents."""
        for agent_id in self._configs:
            if agent_id not in self._agents:
                try:
                    self.instantiate_agent(
                        agent_id, db, embedder, anthropic_client, **kwargs
                    )
                except ValueError as e:
                    logger.error(f"Skipping agent {agent_id}: {e}")
        return dict(self._agents)

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an instantiated agent by ID."""
        return self._agents.get(agent_id)

    def get_config(self, agent_id: str) -> Optional[AgentInstanceConfig]:
        """Get an agent config by ID."""
        return self._configs.get(agent_id)

    def list_agent_ids(self) -> list[str]:
        """Return all discovered agent IDs."""
        return sorted(self._configs.keys())

    def list_configs(self) -> list[AgentInstanceConfig]:
        """Return all discovered agent configs."""
        return list(self._configs.values())
