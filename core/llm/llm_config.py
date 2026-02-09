"""
LLM Configuration — Model routing rules and provider definitions.

Defines which models handle which intents, with fallback chains
for resilience. Each agent can override these defaults via YAML config.

Usage:
    from core.llm.llm_config import LLMConfig, ModelProfile, DEFAULT_ROUTING

    config = LLMConfig()
    profile = config.get_model_for_intent("creative_writing")
    # → ModelProfile(provider="openai", model="gpt-4o", ...)

    profile = config.get_model_for_intent("classification")
    # → ModelProfile(provider="anthropic", model="claude-3-5-haiku-20241022", ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent Categories
# ---------------------------------------------------------------------------

class LLMIntent(str, Enum):
    """Categories of work that route to different models."""

    REASONING = "reasoning"          # Complex analysis, strategy, planning
    CODING = "coding"                # Code generation, debugging, review
    CREATIVE_WRITING = "creative_writing"  # Marketing copy, tweets, blog posts
    CLASSIFICATION = "classification"  # Intent detection, sentiment, routing
    EXTRACTION = "extraction"        # Structured data extraction from text
    SUMMARIZATION = "summarization"  # Condensing long text
    VISION = "vision"                # Image analysis
    IMAGE_GENERATION = "image_generation"  # Creating images from text
    GENERAL = "general"              # Default / unspecified


# ---------------------------------------------------------------------------
# Model Profile
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """A specific model configuration for an LLM call."""

    provider: str           # "anthropic", "openai", "ollama"
    model: str              # e.g., "claude-sonnet-4-20250514", "gpt-4o"
    temperature: float = 0.5
    max_tokens: int = 4096
    base_url: Optional[str] = None  # Required for Ollama
    cost_per_1k_input: float = 0.0  # USD per 1K input tokens
    cost_per_1k_output: float = 0.0  # USD per 1K output tokens
    supports_vision: bool = False
    supports_tools: bool = True
    context_window: int = 128000

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"


# ---------------------------------------------------------------------------
# Route Definition
# ---------------------------------------------------------------------------

@dataclass
class RouteConfig:
    """
    Routing rule: intent → primary model + optional fallback.

    If the primary provider fails (API error, timeout, rate limit),
    the router automatically retries with the fallback.
    """

    intent: LLMIntent
    primary: ModelProfile
    fallback: Optional[ModelProfile] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Default Model Profiles
# ---------------------------------------------------------------------------

# --- Anthropic ---
CLAUDE_SONNET = ModelProfile(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.5,
    max_tokens=4096,
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
    supports_vision=True,
    context_window=200000,
)

CLAUDE_HAIKU = ModelProfile(
    provider="anthropic",
    model="claude-3-5-haiku-20241022",
    temperature=0.3,
    max_tokens=2048,
    cost_per_1k_input=0.001,
    cost_per_1k_output=0.005,
    supports_vision=True,
    context_window=200000,
)

# --- OpenAI ---
GPT_4O = ModelProfile(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    cost_per_1k_input=0.005,
    cost_per_1k_output=0.015,
    supports_vision=True,
    context_window=128000,
)

GPT_4O_MINI = ModelProfile(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=4096,
    cost_per_1k_input=0.00015,
    cost_per_1k_output=0.0006,
    supports_vision=True,
    context_window=128000,
)

# --- Ollama (local) ---
OLLAMA_LLAMA = ModelProfile(
    provider="ollama",
    model="llama3.1:8b",
    temperature=0.5,
    max_tokens=4096,
    base_url="http://localhost:11434",
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
    supports_vision=False,
    context_window=128000,
)


# ---------------------------------------------------------------------------
# Default Routing Table
# ---------------------------------------------------------------------------

DEFAULT_ROUTING: dict[LLMIntent, RouteConfig] = {
    LLMIntent.REASONING: RouteConfig(
        intent=LLMIntent.REASONING,
        primary=CLAUDE_SONNET,
        fallback=GPT_4O,
        description="Complex reasoning: strategy, analysis, planning",
    ),
    LLMIntent.CODING: RouteConfig(
        intent=LLMIntent.CODING,
        primary=CLAUDE_SONNET,
        fallback=GPT_4O,
        description="Code generation, review, debugging",
    ),
    LLMIntent.CREATIVE_WRITING: RouteConfig(
        intent=LLMIntent.CREATIVE_WRITING,
        primary=GPT_4O,
        fallback=CLAUDE_SONNET,
        description="Marketing copy, tweets, blog posts, creative content",
    ),
    LLMIntent.CLASSIFICATION: RouteConfig(
        intent=LLMIntent.CLASSIFICATION,
        primary=CLAUDE_HAIKU,
        fallback=GPT_4O_MINI,
        description="Intent detection, sentiment, cheap routing decisions",
    ),
    LLMIntent.EXTRACTION: RouteConfig(
        intent=LLMIntent.EXTRACTION,
        primary=CLAUDE_SONNET,
        fallback=GPT_4O,
        description="Structured data extraction from unstructured text",
    ),
    LLMIntent.SUMMARIZATION: RouteConfig(
        intent=LLMIntent.SUMMARIZATION,
        primary=CLAUDE_HAIKU,
        fallback=GPT_4O_MINI,
        description="Condensing long documents",
    ),
    LLMIntent.VISION: RouteConfig(
        intent=LLMIntent.VISION,
        primary=CLAUDE_SONNET,
        fallback=GPT_4O,
        description="Image analysis and understanding",
    ),
    LLMIntent.IMAGE_GENERATION: RouteConfig(
        intent=LLMIntent.IMAGE_GENERATION,
        primary=ModelProfile(
            provider="openai",
            model="dall-e-3",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            supports_vision=False,
            supports_tools=False,
        ),
        description="Image creation from text prompts",
    ),
    LLMIntent.GENERAL: RouteConfig(
        intent=LLMIntent.GENERAL,
        primary=CLAUDE_SONNET,
        fallback=GPT_4O,
        description="Default: general-purpose reasoning",
    ),
}


# ---------------------------------------------------------------------------
# LLM Config Manager
# ---------------------------------------------------------------------------

class LLMConfig:
    """
    Manages model routing configuration.

    Can be customized per-agent by overriding routes.
    Falls back to DEFAULT_ROUTING for unconfigured intents.
    """

    def __init__(
        self,
        routing: Optional[dict[LLMIntent, RouteConfig]] = None,
    ):
        self._routing = routing or dict(DEFAULT_ROUTING)

    def get_route(self, intent: str | LLMIntent) -> RouteConfig:
        """Get the routing config for an intent."""
        if isinstance(intent, str):
            try:
                intent = LLMIntent(intent)
            except ValueError:
                intent = LLMIntent.GENERAL

        return self._routing.get(intent, self._routing[LLMIntent.GENERAL])

    def get_model_for_intent(self, intent: str | LLMIntent) -> ModelProfile:
        """Get the primary model profile for an intent."""
        return self.get_route(intent).primary

    def get_fallback_for_intent(
        self, intent: str | LLMIntent
    ) -> Optional[ModelProfile]:
        """Get the fallback model profile for an intent."""
        return self.get_route(intent).fallback

    def override_route(
        self,
        intent: LLMIntent,
        primary: ModelProfile,
        fallback: Optional[ModelProfile] = None,
    ) -> None:
        """Override routing for a specific intent."""
        self._routing[intent] = RouteConfig(
            intent=intent,
            primary=primary,
            fallback=fallback,
        )

    def set_all_to_provider(self, provider: str, model: str, **kwargs: Any) -> None:
        """
        Override ALL routes to use a single provider/model.

        Useful for testing or local-only mode:
            config.set_all_to_provider("ollama", "llama3.1:8b",
                                        base_url="http://localhost:11434")
        """
        profile = ModelProfile(provider=provider, model=model, **kwargs)
        for intent in LLMIntent:
            self._routing[intent] = RouteConfig(
                intent=intent,
                primary=profile,
            )

    def list_routes(self) -> list[dict[str, Any]]:
        """Return a summary of all configured routes."""
        return [
            {
                "intent": route.intent.value,
                "primary": route.primary.display_name,
                "fallback": route.fallback.display_name if route.fallback else None,
                "description": route.description,
            }
            for route in self._routing.values()
        ]
