"""
Tool Use / Function Calling — Unified interface across LLM providers.

Provides a provider-agnostic way to define tools (functions) and pass
them to LLMs. Handles schema translation between Anthropic's tool_use
format and OpenAI's function_calling format.

Usage:
    from core.llm.tools import ToolDefinition, ToolRouter, ToolResult

    # Define tools
    search_tool = ToolDefinition(
        name="search_leads",
        description="Search for sales leads matching criteria",
        parameters={
            "title": {"type": "string", "description": "Job title filter"},
            "location": {"type": "string", "description": "Location filter"},
            "limit": {"type": "integer", "description": "Max results", "default": 10},
        },
        required=["title"],
    )

    # Create router with tools
    tool_router = ToolRouter(
        anthropic_client=client,
        tools=[search_tool],
    )

    # Call with tools enabled
    response = await tool_router.route_with_tools(
        intent="reasoning",
        system_prompt="You are a sales agent assistant.",
        user_prompt="Find C-level executives in Austin, TX.",
    )

    if response.tool_calls:
        for call in response.tool_calls:
            print(f"Tool: {call.name}, Args: {call.arguments}")
            result = await execute_tool(call)
            # Continue conversation with tool results...
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.llm.llm_config import (
    LLMConfig,
    LLMIntent,
    ModelProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """
    Provider-agnostic tool/function definition.

    Translates to both Anthropic's tool format and OpenAI's
    function_calling format automatically.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def to_anthropic(self) -> dict[str, Any]:
        """Convert to Anthropic Claude tool format."""
        properties = {}
        for param_name, param_spec in self.parameters.items():
            if isinstance(param_spec, dict):
                properties[param_name] = param_spec
            else:
                properties[param_name] = {"type": "string", "description": str(param_spec)}

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": self.required,
            },
        }

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI function_calling format."""
        properties = {}
        for param_name, param_spec in self.parameters.items():
            if isinstance(param_spec, dict):
                properties[param_name] = param_spec
            else:
                properties[param_name] = {"type": "string", "description": str(param_spec)}

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required,
                },
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate that arguments satisfy required parameters."""
        errors = []
        for req in self.required:
            if req not in arguments:
                errors.append(f"Missing required parameter: {req}")
        return errors


# ---------------------------------------------------------------------------
# Tool Call (LLM wants to use a tool)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""

    id: str                         # Provider's tool_call id
    name: str                       # Tool function name
    arguments: dict[str, Any]       # Parsed arguments
    raw_arguments: str = ""         # Raw JSON string from provider

    def validate_against(self, definition: ToolDefinition) -> list[str]:
        """Check if this call satisfies the tool's required params."""
        return definition.validate_arguments(self.arguments)


# ---------------------------------------------------------------------------
# Tool Result (human/system provides result back)
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of executing a tool, to be fed back to the LLM."""

    tool_call_id: str               # Must match the ToolCall.id
    content: str                    # Stringified result
    is_error: bool = False          # True if the tool execution failed

    def to_anthropic(self) -> dict[str, Any]:
        """Convert to Anthropic tool_result format."""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI tool response format."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


# ---------------------------------------------------------------------------
# Tool Response (LLM response that may contain tool calls)
# ---------------------------------------------------------------------------

@dataclass
class ToolResponse:
    """Response from an LLM call that may include tool calls."""

    text: str                       # Any text content in the response
    provider: str = ""
    model: str = ""
    intent: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    stop_reason: str = ""           # "end_turn", "tool_use", "max_tokens"
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """True if the LLM wants to use tools."""
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Manages tool definitions and provides lookup by name.

    Agents register their tools here, and the ToolRouter
    uses it to validate calls and translate schemas.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool
        logger.debug(
            "tool_registered",
            extra={"tool": tool.name},
        )

    def register_many(self, tools: list[ToolDefinition]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def to_anthropic(self) -> list[dict[str, Any]]:
        """Convert all tools to Anthropic format."""
        return [tool.to_anthropic() for tool in self._tools.values()]

    def to_openai(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai() for tool in self._tools.values()]

    def validate_call(self, call: ToolCall) -> list[str]:
        """Validate a tool call against its definition."""
        tool = self._tools.get(call.name)
        if tool is None:
            return [f"Unknown tool: {call.name}"]
        return call.validate_against(tool)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Tool Router
# ---------------------------------------------------------------------------

class ToolRouter:
    """
    Routes LLM calls with tool/function calling support.

    Wraps ModelRouter's provider logic with tool schema translation
    and response parsing for tool calls.
    """

    def __init__(
        self,
        anthropic_client: Any = None,
        openai_client: Any = None,
        tools: Optional[list[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ):
        self._anthropic = anthropic_client
        self._openai = openai_client
        self._config = config or LLMConfig()

        self.registry = ToolRegistry()
        if tools:
            self.registry.register_many(tools)

        # Cumulative tracking
        self._call_count: int = 0
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_cost(self) -> float:
        return self._total_cost

    # --- Main Tool Routing API ---

    async def route_with_tools(
        self,
        intent: str | LLMIntent,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> ToolResponse:
        """
        Route an LLM call with tools, using intent-based model selection.

        Args:
            intent: Task category for model routing
            system_prompt: System instructions
            user_prompt: User message (ignored if messages provided)
            tools: Override tools for this call (default: registry tools)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            tool_choice: "auto", "any", "none", or specific tool name
            messages: Full message history (for multi-turn tool use)

        Returns:
            ToolResponse which may contain tool_calls
        """
        route = self._config.get_route(intent)
        profile = route.primary
        intent_str = intent if isinstance(intent, str) else intent.value

        active_tools = tools or list(self.registry._tools.values())

        if profile.provider == "anthropic":
            response = await self._anthropic_tool_call(
                profile=profile,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                messages=messages,
            )
        elif profile.provider == "openai":
            response = await self._openai_tool_call(
                profile=profile,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                messages=messages,
            )
        else:
            raise ValueError(
                f"Provider {profile.provider} does not support tool use. "
                f"Use 'anthropic' or 'openai'."
            )

        response.intent = intent_str
        self._track_usage(response)

        logger.info(
            "tool_route_completed",
            extra={
                "intent": intent_str,
                "provider": profile.provider,
                "model": profile.model,
                "tool_calls": len(response.tool_calls),
                "stop_reason": response.stop_reason,
                "tokens": response.total_tokens,
                "cost": f"${response.cost:.4f}",
            },
        )

        return response

    # --- Continue conversation with tool results ---

    async def continue_with_results(
        self,
        intent: str | LLMIntent,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tool_results: list[ToolResult],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolResponse:
        """
        Continue a conversation after providing tool results.

        This is the second step in the tool use loop:
        1. route_with_tools() → LLM returns tool_calls
        2. Execute the tools → get results
        3. continue_with_results() → LLM processes results
        """
        route = self._config.get_route(intent)
        profile = route.primary

        if profile.provider == "anthropic":
            # Add tool results as user message
            result_content = [r.to_anthropic() for r in tool_results]
            full_messages = messages + [{"role": "user", "content": result_content}]
        elif profile.provider == "openai":
            # Add tool results as separate messages
            full_messages = messages.copy()
            for r in tool_results:
                full_messages.append(r.to_openai())
        else:
            raise ValueError(f"Provider {profile.provider} does not support tool use.")

        return await self.route_with_tools(
            intent=intent,
            system_prompt=system_prompt,
            user_prompt="",
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=full_messages,
        )

    # --- Provider Adapters ---

    async def _anthropic_tool_call(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tool_choice: str,
        messages: Optional[list[dict[str, Any]]],
    ) -> ToolResponse:
        """Call Anthropic with tools."""
        if self._anthropic is None:
            raise ValueError("Anthropic client not configured.")

        start = time.monotonic()
        temp = temperature if temperature is not None else profile.temperature
        tokens = max_tokens if max_tokens is not None else profile.max_tokens

        # Build messages
        if messages:
            msg_list = messages
        else:
            msg_list = [{"role": "user", "content": user_prompt}]

        # Build tool_choice
        if tool_choice == "auto":
            tc = {"type": "auto"}
        elif tool_choice == "any":
            tc = {"type": "any"}
        elif tool_choice == "none":
            tc = {"type": "auto"}  # Anthropic doesn't have "none" — pass no tools
            tools = []
        else:
            # Specific tool name
            tc = {"type": "tool", "name": tool_choice}

        # Convert tools to Anthropic format
        anthropic_tools = [t.to_anthropic() for t in tools]

        kwargs: dict[str, Any] = {
            "model": profile.model,
            "max_tokens": tokens,
            "temperature": temp,
            "system": system_prompt,
            "messages": msg_list,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            kwargs["tool_choice"] = tc

        response = self._anthropic.messages.create(**kwargs)
        elapsed = (time.monotonic() - start) * 1000

        # Parse response
        text_parts = []
        tool_calls = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                    raw_arguments=json.dumps(block.input) if block.input else "{}",
                ))

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        cost = (
            input_tokens / 1000 * profile.cost_per_1k_input
            + output_tokens / 1000 * profile.cost_per_1k_output
        )

        return ToolResponse(
            text="\n".join(text_parts),
            provider="anthropic",
            model=profile.model,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
            stop_reason=getattr(response, "stop_reason", ""),
            raw_response=response,
        )

    async def _openai_tool_call(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tool_choice: str,
        messages: Optional[list[dict[str, Any]]],
    ) -> ToolResponse:
        """Call OpenAI with tools."""
        if self._openai is None:
            raise ValueError("OpenAI client not configured.")

        start = time.monotonic()
        temp = temperature if temperature is not None else profile.temperature
        tokens = max_tokens if max_tokens is not None else profile.max_tokens

        # Build messages
        if messages:
            msg_list = [{"role": "system", "content": system_prompt}] + messages
        else:
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        # Convert tools to OpenAI format
        openai_tools = [t.to_openai() for t in tools]

        # Build tool_choice
        if tool_choice == "auto":
            tc = "auto"
        elif tool_choice == "any":
            tc = "required"
        elif tool_choice == "none":
            tc = "none"
        else:
            tc = {"type": "function", "function": {"name": tool_choice}}

        kwargs: dict[str, Any] = {
            "model": profile.model,
            "messages": msg_list,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = tc

        response = self._openai.chat.completions.create(**kwargs)
        elapsed = (time.monotonic() - start) * 1000

        # Parse response
        choice = response.choices[0] if response.choices else None
        text = ""
        tool_calls = []

        if choice and choice.message:
            text = choice.message.content or ""

            if choice.message.tool_calls:
                for tc_obj in choice.message.tool_calls:
                    try:
                        args = json.loads(tc_obj.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        args = {}

                    tool_calls.append(ToolCall(
                        id=tc_obj.id,
                        name=tc_obj.function.name,
                        arguments=args,
                        raw_arguments=tc_obj.function.arguments or "{}",
                    ))

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = (
            input_tokens / 1000 * profile.cost_per_1k_input
            + output_tokens / 1000 * profile.cost_per_1k_output
        )

        stop_reason = ""
        if choice:
            stop_reason = choice.finish_reason or ""

        return ToolResponse(
            text=text,
            provider="openai",
            model=profile.model,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
            stop_reason=stop_reason,
            raw_response=response,
        )

    # --- Usage Tracking ---

    def _track_usage(self, response: ToolResponse) -> None:
        """Track cumulative usage."""
        self._call_count += 1
        self._total_cost += response.cost
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative tool call statistics."""
        return {
            "total_calls": self._call_count,
            "total_cost_usd": round(self._total_cost, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "registered_tools": self.registry.list_tools(),
        }

    def reset_usage(self) -> None:
        """Reset usage counters."""
        self._call_count = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
