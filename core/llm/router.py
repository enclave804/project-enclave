"""
Model Router — Intent-based multi-model routing with automatic fallback.

Routes LLM calls to the optimal provider based on task intent:
- Reasoning/Coding → Claude Sonnet (best accuracy)
- Creative Writing → GPT-4o (best creative output)
- Classification/Fast → Claude Haiku (cheapest, fastest)
- Vision → Claude Sonnet or GPT-4o (both support vision)

Automatic fallback: if the primary provider fails (timeout, rate limit,
API error), the router retries with the configured fallback model.

Usage:
    from core.llm.router import ModelRouter

    router = ModelRouter(
        anthropic_client=anthropic.Anthropic(),
        openai_client=openai.OpenAI(),
    )

    # Route by intent
    response = await router.route(
        intent="creative_writing",
        system_prompt="You are a witty copywriter.",
        user_prompt="Write a tweet about AI.",
    )
    print(response.text)
    print(f"Used: {response.model} (${response.cost:.4f})")

    # Direct call to specific provider
    response = await router.call_anthropic(
        model="claude-sonnet-4-20250514",
        system_prompt="...",
        user_prompt="...",
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.llm.llm_config import (
    LLMConfig,
    LLMIntent,
    ModelProfile,
    DEFAULT_ROUTING,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response Type
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    text: str
    provider: str           # "anthropic", "openai", "ollama"
    model: str              # Actual model used
    intent: str = ""        # Intent that triggered this call
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0       # Estimated USD cost
    latency_ms: float = 0.0
    is_fallback: bool = False  # True if primary failed and fallback was used
    raw_response: Any = None   # Provider-specific response object

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Routes LLM calls to the optimal provider based on intent.

    Supports Anthropic (Claude), OpenAI (GPT), and Ollama (local).
    Automatically falls back to secondary provider on failure.

    Thread-safe: each call is independent, no shared mutable state.
    """

    def __init__(
        self,
        anthropic_client: Any = None,
        openai_client: Any = None,
        ollama_base_url: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        self._anthropic = anthropic_client
        self._openai = openai_client
        self._ollama_base_url = ollama_base_url or "http://localhost:11434"
        self._config = config or LLMConfig()

        # Usage tracking
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

    @property
    def config(self) -> LLMConfig:
        return self._config

    # --- Main Routing API ---

    async def route(
        self,
        intent: str | LLMIntent,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Route an LLM call to the optimal model based on intent.

        Args:
            intent: Task category (e.g., "reasoning", "creative_writing")
            system_prompt: System instructions
            user_prompt: User message / task
            temperature: Override model default temperature
            max_tokens: Override model default max_tokens
            images: Optional list of image dicts for vision tasks
                    [{"type": "base64", "media_type": "image/png", "data": "..."}]

        Returns:
            LLMResponse with text, cost, and metadata
        """
        route = self._config.get_route(intent)
        primary = route.primary
        fallback = route.fallback
        intent_str = intent if isinstance(intent, str) else intent.value

        # Try primary
        try:
            response = await self._call_model(
                profile=primary,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                images=images,
            )
            response.intent = intent_str
            self._track_usage(response)

            logger.info(
                "llm_routed",
                extra={
                    "intent": intent_str,
                    "provider": primary.provider,
                    "model": primary.model,
                    "tokens": response.total_tokens,
                    "cost": f"${response.cost:.4f}",
                    "latency_ms": response.latency_ms,
                },
            )
            return response

        except Exception as primary_error:
            logger.warning(
                "llm_primary_failed",
                extra={
                    "intent": intent_str,
                    "provider": primary.provider,
                    "model": primary.model,
                    "error": str(primary_error)[:200],
                },
            )

            # Try fallback
            if fallback is not None:
                try:
                    response = await self._call_model(
                        profile=fallback,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        images=images,
                    )
                    response.intent = intent_str
                    response.is_fallback = True
                    self._track_usage(response)

                    logger.info(
                        "llm_fallback_used",
                        extra={
                            "intent": intent_str,
                            "provider": fallback.provider,
                            "model": fallback.model,
                            "tokens": response.total_tokens,
                            "primary_error": str(primary_error)[:100],
                        },
                    )
                    return response

                except Exception as fallback_error:
                    logger.error(
                        "llm_fallback_also_failed",
                        extra={
                            "intent": intent_str,
                            "primary_error": str(primary_error)[:100],
                            "fallback_error": str(fallback_error)[:100],
                        },
                    )
                    raise fallback_error from primary_error

            # No fallback configured
            raise

    # --- Direct Provider Calls ---

    async def call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.5,
        max_tokens: int = 4096,
        images: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call Anthropic (Claude) directly, bypassing routing."""
        profile = ModelProfile(
            provider="anthropic",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._call_model(
            profile=profile,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        self._track_usage(response)
        return response

    async def call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        images: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call OpenAI (GPT) directly, bypassing routing."""
        profile = ModelProfile(
            provider="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._call_model(
            profile=profile,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        self._track_usage(response)
        return response

    # --- Provider Adapters ---

    async def _call_model(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Dispatch to the correct provider adapter."""
        temp = temperature if temperature is not None else profile.temperature
        tokens = max_tokens if max_tokens is not None else profile.max_tokens

        if profile.provider == "anthropic":
            return await self._anthropic_call(
                profile, system_prompt, user_prompt, temp, tokens, images
            )
        elif profile.provider == "openai":
            return await self._openai_call(
                profile, system_prompt, user_prompt, temp, tokens, images
            )
        elif profile.provider == "ollama":
            return await self._ollama_call(
                profile, system_prompt, user_prompt, temp, tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {profile.provider}")

    async def _anthropic_call(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        images: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call Anthropic Claude API."""
        if self._anthropic is None:
            raise ValueError(
                "Anthropic client not configured. "
                "Pass anthropic_client to ModelRouter()."
            )

        start = time.monotonic()

        # Build message content (text or multimodal)
        if images:
            content: list[dict[str, Any]] = []
            for img in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": img.get("type", "base64"),
                        "media_type": img.get("media_type", "image/png"),
                        "data": img["data"],
                    },
                })
            content.append({"type": "text", "text": user_prompt})
        else:
            content = user_prompt  # type: ignore[assignment]

        response = self._anthropic.messages.create(
            model=profile.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        elapsed = (time.monotonic() - start) * 1000

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        cost = (
            input_tokens / 1000 * profile.cost_per_1k_input
            + output_tokens / 1000 * profile.cost_per_1k_output
        )

        return LLMResponse(
            text=response.content[0].text if response.content else "",
            provider="anthropic",
            model=profile.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
            raw_response=response,
        )

    async def _openai_call(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        images: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call OpenAI GPT API."""
        if self._openai is None:
            raise ValueError(
                "OpenAI client not configured. "
                "Pass openai_client to ModelRouter()."
            )

        start = time.monotonic()

        messages = [{"role": "system", "content": system_prompt}]

        # Build user message (text or multimodal)
        if images:
            user_content: list[dict[str, Any]] = []
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.get('media_type', 'image/png')};base64,{img['data']}",
                    },
                })
            user_content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        response = self._openai.chat.completions.create(
            model=profile.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = (time.monotonic() - start) * 1000

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = (
            input_tokens / 1000 * profile.cost_per_1k_input
            + output_tokens / 1000 * profile.cost_per_1k_output
        )

        return LLMResponse(
            text=text or "",
            provider="openai",
            model=profile.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
            raw_response=response,
        )

    async def _ollama_call(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Ollama local API via httpx."""
        import httpx

        start = time.monotonic()
        base_url = profile.base_url or self._ollama_base_url

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": profile.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = (time.monotonic() - start) * 1000

        text = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            provider="ollama",
            model=profile.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,  # Local = free
            latency_ms=elapsed,
            raw_response=data,
        )

    # --- Usage Tracking ---

    def _track_usage(self, response: LLMResponse) -> None:
        """Track cumulative usage stats."""
        self._call_count += 1
        self._total_cost += response.cost
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_cost_usd": round(self._total_cost, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def reset_usage(self) -> None:
        """Reset usage counters (e.g., start of a billing period)."""
        self._call_count = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
