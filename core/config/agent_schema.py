"""
Pydantic config models for agent definitions.

Each agent is defined by a YAML file in verticals/{vertical_id}/agents/.
New agent = YAML config + @register_agent_type decorator on implementation class.

Architecture patterns configured here:
- Neural Router: cheap local models classify intent before waking the LLM
- Refinement Loop: agents critique and refine their own output
- Sandbox Protocol: dangerous tools are intercepted in non-production envs
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AgentToolConfig(BaseModel):
    """
    Configuration for a tool available to an agent.

    In YAML, tools can be specified as either:
    - Simple string: "apollo_search" → AgentToolConfig(name="apollo_search", type="mcp")
    - Full object: {name: "apollo_search", type: "builtin", config: {...}}

    The string shorthand is coerced by AgentInstanceConfig's model_validator.
    """

    name: str
    type: str = Field(
        "mcp",
        description="Tool type: 'mcp', 'builtin', 'browser', or 'custom'",
    )
    mcp_server: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)


class ModelType(str, Enum):
    """Model type for routing decisions."""

    LLM = "llm"
    LOCAL_CLASSIFICATION = "local_classification"


class AgentModelConfig(BaseModel):
    """
    LLM configuration for an agent.

    The "Kill Switch": change `provider` to "ollama" in any agent YAML
    to run fully local if cloud APIs go down. Sovereignty preserved.

    Example YAML:
        model:
          provider: ollama
          model: llama3.1:70b
          base_url: http://localhost:11434
    """

    provider: Literal["anthropic", "openai", "ollama"] = Field(
        "anthropic",
        description=(
            "LLM provider: 'anthropic' (Claude), 'openai' (GPT), "
            "'ollama' (local). Change to 'ollama' for full sovereignty."
        ),
    )
    model: str = "claude-sonnet-4-20250514"
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=100, le=128000)
    base_url: Optional[str] = Field(
        None,
        description=(
            "Custom API base URL. Required for Ollama "
            "(e.g., 'http://localhost:11434'). Optional override for "
            "OpenAI-compatible endpoints."
        ),
    )
    context_window: int = Field(
        128000,
        ge=1024,
        le=2000000,
        description="Model context window size in tokens",
    )
    model_type: ModelType = Field(
        ModelType.LLM,
        description=(
            "Model type: 'llm' for full LLM calls, "
            "'local_classification' for cheap local classifiers (BERT/YOLO)"
        ),
    )


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


class RoutingConfig(BaseModel):
    """
    Neural Router configuration — cheap intent classification before LLM.

    The router intercepts incoming tasks and classifies intent using a
    small, fast model (e.g., BERT). Based on the classified intent, the
    router can short-circuit (skip the expensive LLM entirely) or route
    to a specific handler.

    Example YAML:
        routing:
          enabled: true
          model_type: local_classification
          intent_actions:
            out_of_office: sleep
            spam: discard
            unsubscribe: remove_contact
          fallback_action: proceed
    """

    enabled: bool = Field(
        False,
        description="Enable the neural router (requires a trained classifier)",
    )
    model_type: ModelType = Field(
        ModelType.LOCAL_CLASSIFICATION,
        description="Model type for intent classification",
    )
    intent_actions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of classified intent -> action. "
            "Actions: 'proceed' (wake LLM), 'sleep' (skip), "
            "'discard' (drop task), or a custom handler name"
        ),
    )
    fallback_action: str = Field(
        "proceed",
        description="Action when intent doesn't match any route (default: proceed to LLM)",
    )
    confidence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum classification confidence to trust the router's decision",
    )


class RefinementConfig(BaseModel):
    """
    Self-correction loop — agents critique and refine their own output.

    When enabled, the agent's output passes through a critic step that
    evaluates quality against a rubric. If the output fails the rubric,
    it loops back for refinement up to max_iterations times.

    Example YAML:
        refinement:
          enabled: true
          critic_prompt: "Rate this email draft 1-10 for professionalism,
            relevance, and compliance. If below 7, suggest specific improvements."
          max_iterations: 2
    """

    enabled: bool = Field(
        False,
        description="Enable the self-correction refinement loop",
    )
    critic_prompt: str = Field(
        "",
        description=(
            "Rubric/prompt for the critic to evaluate agent output. "
            "Should describe what 'good' looks like and when to refine."
        ),
    )
    max_iterations: int = Field(
        1,
        ge=1,
        le=5,
        description="Maximum refinement iterations before accepting output",
    )
    quality_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum quality score (0-1) from critic to accept output. "
            "Below this triggers another refinement iteration."
        ),
    )


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

    # --- Neural Router & Refinement Loop ---
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

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

    # --- Shadow Mode (God Mode Lite) ---
    shadow_mode: bool = Field(
        False,
        description=(
            "When True, this agent runs in parallel with its champion "
            "but NEVER triggers external effects (emails, APIs). "
            "Used for safe A/B testing of new strategies on real data."
        ),
    )
    shadow_of: Optional[str] = Field(
        None,
        description=(
            "The agent_id of the 'champion' agent this shadow copies. "
            "When the champion gets a task, the shadow gets a duplicate. "
            "Only used when shadow_mode=True."
        ),
    )

    # Agent-specific params (passed through to implementation)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_tools(cls, values: Any) -> Any:
        """
        Allow tools to be specified as simple strings in YAML.

        Converts:
            tools: ["apollo_search", "send_email"]
        Into:
            tools: [AgentToolConfig(name="apollo_search", type="mcp"),
                     AgentToolConfig(name="send_email", type="mcp")]

        The verbose format still works:
            tools:
              - name: apollo_search
                type: builtin
        """
        if isinstance(values, dict):
            raw_tools = values.get("tools", [])
            if isinstance(raw_tools, list):
                coerced = []
                for tool in raw_tools:
                    if isinstance(tool, str):
                        coerced.append({"name": tool, "type": "mcp"})
                    else:
                        coerced.append(tool)
                values["tools"] = coerced
        return values
