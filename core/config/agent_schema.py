"""
Pydantic config models for agent definitions.

Each agent is defined by a YAML file in verticals/{vertical_id}/agents/.
New agent = YAML config + @register_agent_type decorator on implementation class.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentToolConfig(BaseModel):
    """Configuration for a tool available to an agent."""

    name: str
    type: str = Field(
        ...,
        description="Tool type: 'mcp', 'builtin', 'browser', or 'custom'",
    )
    mcp_server: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)


class AgentModelConfig(BaseModel):
    """LLM configuration for an agent."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=100, le=128000)


class HumanGateConfig(BaseModel):
    """Configuration for human-in-the-loop gates."""

    enabled: bool = True
    gate_before: list[str] = Field(
        default_factory=list,
        description="Node names requiring human approval before execution",
    )
    auto_approve_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Auto-approve when confidence exceeds this threshold",
    )


class AgentScheduleConfig(BaseModel):
    """Scheduling configuration for an agent."""

    trigger: str = Field(
        "manual",
        description="Trigger type: 'manual', 'scheduled', or 'event'",
    )
    cron: Optional[str] = None
    event_source: Optional[str] = None


class AgentInstanceConfig(BaseModel):
    """
    Complete configuration for a single agent instance.

    Schema for: verticals/{vertical_id}/agents/{agent_name}.yaml
    """

    agent_id: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique identifier for this agent instance",
    )
    agent_type: str = Field(
        ...,
        description="Agent implementation type (maps to @register_agent_type)",
    )
    vertical_id: str = ""  # Injected by AgentRegistry
    name: str = Field(..., description="Human-readable agent name")
    description: str = ""
    enabled: bool = True

    model: AgentModelConfig = Field(default_factory=AgentModelConfig)
    tools: list[AgentToolConfig] = Field(default_factory=list)
    browser_enabled: bool = False

    human_gates: HumanGateConfig = Field(default_factory=HumanGateConfig)
    schedule: AgentScheduleConfig = Field(default_factory=AgentScheduleConfig)

    system_prompt_path: Optional[str] = None

    # --- Safety & Cost Control ---
    max_consecutive_errors: int = Field(
        5,
        ge=1,
        le=100,
        description="Auto-disable agent after this many consecutive failures (circuit breaker)",
    )
    rag_write_confidence_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to write insights to shared memory",
    )

    # Agent-specific params (passed through to implementation)
    params: dict[str, Any] = Field(default_factory=dict)
