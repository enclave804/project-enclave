"""
Streaming LLM Responses — AsyncIterator interface for real-time output.

Streams tokens as they arrive from the LLM provider, enabling:
- Real-time UI updates (dashboard, chat)
- Lower perceived latency (first token in ~200ms vs ~2s for full)
- Progress indicators for long generations

Supports Anthropic, OpenAI, and Ollama streaming protocols.

Usage:
    from core.llm.streaming import StreamingRouter

    router = StreamingRouter(
        anthropic_client=anthropic.Anthropic(),
        openai_client=openai.OpenAI(),
    )

    # Stream by intent
    async for chunk in router.stream(
        intent="creative_writing",
        system_prompt="You are a copywriter.",
        user_prompt="Write a blog post about AI agents.",
    ):
        print(chunk.text, end="", flush=True)

    # Get final stats after streaming
    stats = router.last_stream_stats
    print(f"\\nTokens: {stats['total_tokens']}, Cost: ${stats['cost']:.4f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from core.llm.llm_config import (
    LLMConfig,
    LLMIntent,
    ModelProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream Chunk
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single chunk of streamed text from an LLM provider."""

    text: str                   # The new text fragment
    provider: str = ""          # "anthropic", "openai", "ollama"
    model: str = ""             # Model identifier
    is_final: bool = False      # True for the last chunk (stats available)
    input_tokens: int = 0       # Only populated on final chunk
    output_tokens: int = 0      # Only populated on final chunk
    cost: float = 0.0           # Only populated on final chunk
    latency_ms: float = 0.0     # Only populated on final chunk

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Stream Stats
# ---------------------------------------------------------------------------

@dataclass
class StreamStats:
    """Statistics for a completed stream."""

    provider: str = ""
    model: str = ""
    intent: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    total_text_length: int = 0
    chunk_count: int = 0
    is_fallback: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Streaming Router
# ---------------------------------------------------------------------------

class StreamingRouter:
    """
    Streams LLM responses as async iterators with automatic fallback.

    Extends ModelRouter's intent-based routing with streaming support.
    Falls back to the secondary provider if primary stream fails.

    Thread-safe: each stream is independent, no shared mutable state
    except usage tracking (additive only).
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

        # Last stream stats (for post-stream inspection)
        self._last_stats: Optional[StreamStats] = None

        # Cumulative tracking
        self._stream_count: int = 0
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    @property
    def last_stream_stats(self) -> Optional[dict[str, Any]]:
        """Get stats from the most recent stream (after iteration completes)."""
        if self._last_stats is None:
            return None
        s = self._last_stats
        return {
            "provider": s.provider,
            "model": s.model,
            "intent": s.intent,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "total_tokens": s.total_tokens,
            "cost": s.cost,
            "latency_ms": s.latency_ms,
            "total_text_length": s.total_text_length,
            "chunk_count": s.chunk_count,
            "is_fallback": s.is_fallback,
        }

    @property
    def stream_count(self) -> int:
        return self._stream_count

    @property
    def total_cost(self) -> float:
        return self._total_cost

    # --- Main Streaming API ---

    async def stream(
        self,
        intent: str | LLMIntent,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream an LLM response based on intent, with automatic fallback.

        Yields StreamChunk objects as tokens arrive. The final chunk has
        is_final=True and contains usage statistics.

        Args:
            intent: Task category for routing
            system_prompt: System instructions
            user_prompt: User message / task
            temperature: Override model default temperature
            max_tokens: Override model default max_tokens

        Yields:
            StreamChunk with incremental text fragments
        """
        route = self._config.get_route(intent)
        primary = route.primary
        fallback = route.fallback
        intent_str = intent if isinstance(intent, str) else intent.value

        # Try primary
        try:
            async for chunk in self._stream_model(
                profile=primary,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                intent_str=intent_str,
                is_fallback=False,
            ):
                yield chunk
            return  # Success — done

        except Exception as primary_error:
            logger.warning(
                "stream_primary_failed",
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
                    async for chunk in self._stream_model(
                        profile=fallback,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        intent_str=intent_str,
                        is_fallback=True,
                    ):
                        yield chunk
                    return  # Fallback success

                except Exception as fallback_error:
                    logger.error(
                        "stream_fallback_also_failed",
                        extra={
                            "intent": intent_str,
                            "primary_error": str(primary_error)[:100],
                            "fallback_error": str(fallback_error)[:100],
                        },
                    )
                    raise fallback_error from primary_error

            # No fallback configured
            raise

    # --- Direct Provider Streams ---

    async def stream_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Stream directly from Anthropic, bypassing routing."""
        profile = ModelProfile(
            provider="anthropic",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async for chunk in self._stream_model(
            profile=profile,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            intent_str="direct",
            is_fallback=False,
        ):
            yield chunk

    async def stream_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Stream directly from OpenAI, bypassing routing."""
        profile = ModelProfile(
            provider="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async for chunk in self._stream_model(
            profile=profile,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            intent_str="direct",
            is_fallback=False,
        ):
            yield chunk

    # --- Provider Stream Adapters ---

    async def _stream_model(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        intent_str: str = "",
        is_fallback: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Dispatch to the correct streaming adapter."""
        temp = temperature if temperature is not None else profile.temperature
        tokens = max_tokens if max_tokens is not None else profile.max_tokens

        if profile.provider == "anthropic":
            async for chunk in self._stream_anthropic(
                profile, system_prompt, user_prompt, temp, tokens,
                intent_str, is_fallback,
            ):
                yield chunk
        elif profile.provider == "openai":
            async for chunk in self._stream_openai(
                profile, system_prompt, user_prompt, temp, tokens,
                intent_str, is_fallback,
            ):
                yield chunk
        elif profile.provider == "ollama":
            async for chunk in self._stream_ollama(
                profile, system_prompt, user_prompt, temp, tokens,
                intent_str, is_fallback,
            ):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider for streaming: {profile.provider}")

    async def _stream_anthropic(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        intent_str: str,
        is_fallback: bool,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Anthropic Claude using the Messages API stream."""
        if self._anthropic is None:
            raise ValueError(
                "Anthropic client not configured. "
                "Pass anthropic_client to StreamingRouter()."
            )

        start = time.monotonic()
        collected_text = ""
        chunk_count = 0
        input_tokens = 0
        output_tokens = 0

        # Use Anthropic's streaming context manager
        with self._anthropic.messages.stream(
            model=profile.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text_chunk in stream.text_stream:
                collected_text += text_chunk
                chunk_count += 1
                yield StreamChunk(
                    text=text_chunk,
                    provider="anthropic",
                    model=profile.model,
                )

            # Get final message for usage stats
            final_message = stream.get_final_message()
            if final_message and hasattr(final_message, "usage"):
                input_tokens = getattr(final_message.usage, "input_tokens", 0)
                output_tokens = getattr(final_message.usage, "output_tokens", 0)

        elapsed = (time.monotonic() - start) * 1000
        cost = (
            input_tokens / 1000 * profile.cost_per_1k_input
            + output_tokens / 1000 * profile.cost_per_1k_output
        )

        # Yield final chunk with stats
        yield StreamChunk(
            text="",
            provider="anthropic",
            model=profile.model,
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
        )

        # Record stats
        self._record_stats(
            provider="anthropic",
            model=profile.model,
            intent=intent_str,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
            total_text_length=len(collected_text),
            chunk_count=chunk_count,
            is_fallback=is_fallback,
        )

    async def _stream_openai(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        intent_str: str,
        is_fallback: bool,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from OpenAI GPT using the Chat Completions API."""
        if self._openai is None:
            raise ValueError(
                "OpenAI client not configured. "
                "Pass openai_client to StreamingRouter()."
            )

        start = time.monotonic()
        collected_text = ""
        chunk_count = 0

        response_stream = self._openai.chat.completions.create(
            model=profile.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        input_tokens = 0
        output_tokens = 0

        for chunk in response_stream:
            # Check for usage in the final chunk
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                collected_text += text
                chunk_count += 1
                yield StreamChunk(
                    text=text,
                    provider="openai",
                    model=profile.model,
                )

        elapsed = (time.monotonic() - start) * 1000
        cost = (
            input_tokens / 1000 * profile.cost_per_1k_input
            + output_tokens / 1000 * profile.cost_per_1k_output
        )

        # Yield final chunk with stats
        yield StreamChunk(
            text="",
            provider="openai",
            model=profile.model,
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
        )

        self._record_stats(
            provider="openai",
            model=profile.model,
            intent=intent_str,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=elapsed,
            total_text_length=len(collected_text),
            chunk_count=chunk_count,
            is_fallback=is_fallback,
        )

    async def _stream_ollama(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        intent_str: str,
        is_fallback: bool,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Ollama local API using httpx streaming."""
        import httpx

        start = time.monotonic()
        collected_text = ""
        chunk_count = 0
        base_url = profile.base_url or self._ollama_base_url

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat",
                json={
                    "model": profile.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()
                import json as json_mod

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json_mod.loads(line)

                    if data.get("done", False):
                        # Final message with stats
                        break

                    msg = data.get("message", {})
                    text = msg.get("content", "")
                    if text:
                        collected_text += text
                        chunk_count += 1
                        yield StreamChunk(
                            text=text,
                            provider="ollama",
                            model=profile.model,
                        )

        elapsed = (time.monotonic() - start) * 1000

        # Ollama doesn't provide token counts in streaming
        # Estimate from text length
        estimated_output_tokens = len(collected_text) // 4

        yield StreamChunk(
            text="",
            provider="ollama",
            model=profile.model,
            is_final=True,
            output_tokens=estimated_output_tokens,
            cost=0.0,
            latency_ms=elapsed,
        )

        self._record_stats(
            provider="ollama",
            model=profile.model,
            intent=intent_str,
            input_tokens=0,
            output_tokens=estimated_output_tokens,
            cost=0.0,
            latency_ms=elapsed,
            total_text_length=len(collected_text),
            chunk_count=chunk_count,
            is_fallback=is_fallback,
        )

    # --- Stats & Tracking ---

    def _record_stats(
        self,
        provider: str,
        model: str,
        intent: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
        total_text_length: int,
        chunk_count: int,
        is_fallback: bool,
    ) -> None:
        """Record stream stats and update cumulative counters."""
        self._last_stats = StreamStats(
            provider=provider,
            model=model,
            intent=intent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            total_text_length=total_text_length,
            chunk_count=chunk_count,
            is_fallback=is_fallback,
        )

        self._stream_count += 1
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        logger.info(
            "stream_completed",
            extra={
                "provider": provider,
                "model": model,
                "intent": intent,
                "chunks": chunk_count,
                "text_length": total_text_length,
                "tokens": input_tokens + output_tokens,
                "cost": f"${cost:.4f}",
                "latency_ms": round(latency_ms, 1),
                "is_fallback": is_fallback,
            },
        )

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative streaming usage statistics."""
        return {
            "total_streams": self._stream_count,
            "total_cost_usd": round(self._total_cost, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def reset_usage(self) -> None:
        """Reset usage counters."""
        self._stream_count = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_stats = None


# ---------------------------------------------------------------------------
# Helper: Collect full stream into text
# ---------------------------------------------------------------------------

async def collect_stream(
    stream: AsyncIterator[StreamChunk],
) -> tuple[str, StreamChunk]:
    """
    Consume a full stream and return (full_text, final_chunk).

    Useful when you want streaming internally but need the full
    text for downstream processing:

        chunks = router.stream(intent="reasoning", ...)
        text, final = await collect_stream(chunks)
        print(text)
        print(f"Cost: ${final.cost:.4f}")
    """
    collected = []
    final_chunk = StreamChunk(text="", is_final=True)

    async for chunk in stream:
        if chunk.is_final:
            final_chunk = chunk
        elif chunk.text:
            collected.append(chunk.text)

    return "".join(collected), final_chunk
