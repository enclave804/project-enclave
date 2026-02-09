"""
Tests for Phase 6 Sprint 2: Streaming, Tool Use, and Caching.

Tests cover:
1. StreamingRouter — async iteration, fallback, stats
2. ToolRouter — tool definitions, tool calls, multi-turn
3. ResponseCache — TTL, LRU eviction, hit/miss, temperature guard
4. BaseAgent integration — lazy init, cached routing
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.llm.llm_config import (
    LLMConfig,
    LLMIntent,
    ModelProfile,
    RouteConfig,
    CLAUDE_SONNET,
    GPT_4O,
    CLAUDE_HAIKU,
)


# ===========================================================================
# Mock Factories
# ===========================================================================

def _mock_anthropic_config():
    """Create a minimal AgentInstanceConfig-like mock."""
    config = MagicMock()
    config.agent_id = "test_agent"
    config.agent_type = "test"
    config.vertical_id = "test_vertical"
    config.name = "Test Agent"
    config.routing = MagicMock(enabled=False)
    config.refinement = MagicMock(enabled=False)
    config.max_consecutive_errors = 5
    config.rag_write_confidence_threshold = 0.7
    config.params = {}
    return config


class MockAnthropicStreamMessage:
    """Mock for Anthropic's final stream message."""

    def __init__(self, text: str, input_tokens: int = 50, output_tokens: int = 100):
        self.content = [MagicMock(text=text)]

        class Usage:
            pass

        self.usage = Usage()
        self.usage.input_tokens = input_tokens
        self.usage.output_tokens = output_tokens


class MockAnthropicStream:
    """Mock for Anthropic's streaming context manager."""

    def __init__(self, text_chunks: list[str], input_tokens: int = 50, output_tokens: int = 100):
        self._chunks = text_chunks
        self._final_message = MockAnthropicStreamMessage(
            "".join(text_chunks), input_tokens, output_tokens
        )

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return self._final_message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockOpenAIStreamChunk:
    """Mock for an OpenAI streaming chunk."""

    def __init__(self, text: str = "", usage=None):
        if text:
            delta = MagicMock()
            delta.content = text
            choice = MagicMock()
            choice.delta = delta
            self.choices = [choice]
        else:
            self.choices = []
        self.usage = usage


class MockOpenAIUsage:
    """Mock OpenAI usage stats."""

    def __init__(self, prompt_tokens=50, completion_tokens=100):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


# ===========================================================================
# Test: StreamingRouter
# ===========================================================================

class TestStreamingRouter:
    """Test the StreamingRouter async iterator interface."""

    def _make_router(self, anthropic_client=None, openai_client=None):
        from core.llm.streaming import StreamingRouter
        return StreamingRouter(
            anthropic_client=anthropic_client,
            openai_client=openai_client,
        )

    @pytest.mark.asyncio
    async def test_stream_anthropic(self):
        """Stream from Anthropic produces chunks."""
        mock_client = MagicMock()
        chunks = ["Hello", " world", "!"]
        mock_client.messages.stream.return_value = MockAnthropicStream(
            chunks, input_tokens=10, output_tokens=20
        )

        router = self._make_router(anthropic_client=mock_client)
        # Override to route to Anthropic
        router._config.override_route(
            LLMIntent.GENERAL,
            CLAUDE_SONNET,
        )

        collected = []
        final = None
        async for chunk in router.stream(
            intent="general",
            system_prompt="Test",
            user_prompt="Hello",
        ):
            if chunk.is_final:
                final = chunk
            else:
                collected.append(chunk.text)

        assert collected == ["Hello", " world", "!"]
        assert final is not None
        assert final.is_final
        assert final.input_tokens == 10
        assert final.output_tokens == 20

    @pytest.mark.asyncio
    async def test_stream_openai(self):
        """Stream from OpenAI produces chunks."""
        mock_client = MagicMock()

        # Create stream chunks
        oai_chunks = [
            MockOpenAIStreamChunk("Hi"),
            MockOpenAIStreamChunk(" there"),
            MockOpenAIStreamChunk("", usage=MockOpenAIUsage(30, 40)),
        ]
        mock_client.chat.completions.create.return_value = iter(oai_chunks)

        router = self._make_router(openai_client=mock_client)
        router._config.override_route(LLMIntent.GENERAL, GPT_4O)

        collected = []
        final = None
        async for chunk in router.stream("general", "Test", "Hi"):
            if chunk.is_final:
                final = chunk
            else:
                collected.append(chunk.text)

        assert collected == ["Hi", " there"]
        assert final is not None
        assert final.input_tokens == 30
        assert final.output_tokens == 40

    @pytest.mark.asyncio
    async def test_stream_fallback(self):
        """Fallback used when primary stream fails."""
        mock_anthropic = MagicMock()
        mock_anthropic.messages.stream.side_effect = Exception("API down")

        mock_openai = MagicMock()
        oai_chunks = [
            MockOpenAIStreamChunk("Fallback"),
            MockOpenAIStreamChunk("", usage=MockOpenAIUsage(5, 10)),
        ]
        mock_openai.chat.completions.create.return_value = iter(oai_chunks)

        config = LLMConfig()
        config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET, GPT_4O)

        router = self._make_router(
            anthropic_client=mock_anthropic,
            openai_client=mock_openai,
        )
        router._config = config

        collected = []
        async for chunk in router.stream("general", "Test", "Hi"):
            if not chunk.is_final:
                collected.append(chunk.text)

        assert collected == ["Fallback"]

    @pytest.mark.asyncio
    async def test_stream_both_fail_raises(self):
        """Both providers failing raises the fallback error."""
        mock_anthropic = MagicMock()
        mock_anthropic.messages.stream.side_effect = Exception("Primary down")

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = Exception("Fallback down")

        config = LLMConfig()
        config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET, GPT_4O)

        router = self._make_router(
            anthropic_client=mock_anthropic,
            openai_client=mock_openai,
        )
        router._config = config

        with pytest.raises(Exception, match="Fallback down"):
            async for _ in router.stream("general", "Test", "Hi"):
                pass

    @pytest.mark.asyncio
    async def test_stream_stats(self):
        """Stream stats are recorded after iteration."""
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = MockAnthropicStream(
            ["Token"], input_tokens=100, output_tokens=200
        )

        router = self._make_router(anthropic_client=mock_client)
        router._config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET)

        async for _ in router.stream("general", "Sys", "User"):
            pass

        stats = router.last_stream_stats
        assert stats is not None
        assert stats["provider"] == "anthropic"
        assert stats["input_tokens"] == 100
        assert stats["output_tokens"] == 200
        assert stats["chunk_count"] == 1
        assert stats["total_text_length"] == 5  # "Token"

    @pytest.mark.asyncio
    async def test_stream_usage_tracking(self):
        """Cumulative stream usage is tracked."""
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = MockAnthropicStream(
            ["A"], input_tokens=10, output_tokens=20
        )

        router = self._make_router(anthropic_client=mock_client)
        router._config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET)

        # Stream twice
        async for _ in router.stream("general", "S", "U"):
            pass

        mock_client.messages.stream.return_value = MockAnthropicStream(
            ["B"], input_tokens=30, output_tokens=40
        )

        async for _ in router.stream("general", "S", "U"):
            pass

        assert router.stream_count == 2
        stats = router.get_usage_stats()
        assert stats["total_input_tokens"] == 40
        assert stats["total_output_tokens"] == 60

    @pytest.mark.asyncio
    async def test_stream_no_client_raises(self):
        """Streaming without a client raises ValueError."""
        router = self._make_router()
        router._config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET)

        with pytest.raises(ValueError, match="Anthropic client not configured"):
            async for _ in router.stream("general", "S", "U"):
                pass

    def test_reset_usage(self):
        """Reset clears all streaming counters."""
        router = self._make_router()
        router._stream_count = 5
        router._total_cost = 1.0
        router.reset_usage()
        assert router.stream_count == 0
        assert router.total_cost == 0.0
        assert router.last_stream_stats is None


# ===========================================================================
# Test: collect_stream helper
# ===========================================================================

class TestCollectStream:
    """Test the collect_stream convenience function."""

    @pytest.mark.asyncio
    async def test_collect_stream(self):
        """Collect full stream into text + final stats."""
        from core.llm.streaming import StreamChunk, collect_stream

        async def mock_stream():
            yield StreamChunk(text="Hello", provider="test")
            yield StreamChunk(text=" World", provider="test")
            yield StreamChunk(
                text="", provider="test", is_final=True,
                input_tokens=10, output_tokens=20, cost=0.001,
            )

        text, final = await collect_stream(mock_stream())
        assert text == "Hello World"
        assert final.is_final
        assert final.input_tokens == 10
        assert final.cost == 0.001


# ===========================================================================
# Test: ToolDefinition
# ===========================================================================

class TestToolDefinition:
    """Test tool definition and schema conversion."""

    def test_to_anthropic(self):
        """Convert tool to Anthropic format."""
        from core.llm.tools import ToolDefinition

        tool = ToolDefinition(
            name="search_leads",
            description="Search for leads",
            parameters={
                "title": {"type": "string", "description": "Job title"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            required=["title"],
        )

        schema = tool.to_anthropic()
        assert schema["name"] == "search_leads"
        assert schema["description"] == "Search for leads"
        assert schema["input_schema"]["type"] == "object"
        assert "title" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["title"]

    def test_to_openai(self):
        """Convert tool to OpenAI format."""
        from core.llm.tools import ToolDefinition

        tool = ToolDefinition(
            name="get_weather",
            description="Get current weather",
            parameters={
                "city": {"type": "string", "description": "City name"},
            },
            required=["city"],
        )

        schema = tool.to_openai()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "city" in schema["function"]["parameters"]["properties"]

    def test_validate_arguments_passes(self):
        """Valid arguments pass validation."""
        from core.llm.tools import ToolDefinition

        tool = ToolDefinition(
            name="test",
            description="Test",
            required=["name", "age"],
        )
        errors = tool.validate_arguments({"name": "Alice", "age": 30})
        assert errors == []

    def test_validate_arguments_fails(self):
        """Missing required arguments produce errors."""
        from core.llm.tools import ToolDefinition

        tool = ToolDefinition(
            name="test",
            description="Test",
            required=["name", "age"],
        )
        errors = tool.validate_arguments({"name": "Alice"})
        assert len(errors) == 1
        assert "age" in errors[0]


# ===========================================================================
# Test: ToolRegistry
# ===========================================================================

class TestToolRegistry:
    """Test tool registration and lookup."""

    def test_register_and_lookup(self):
        """Register a tool and look it up by name."""
        from core.llm.tools import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tool = ToolDefinition(name="search", description="Search")
        registry.register(tool)

        assert "search" in registry
        assert registry.get("search") is tool
        assert len(registry) == 1

    def test_register_many(self):
        """Register multiple tools at once."""
        from core.llm.tools import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tools = [
            ToolDefinition(name="a", description="A"),
            ToolDefinition(name="b", description="B"),
            ToolDefinition(name="c", description="C"),
        ]
        registry.register_many(tools)
        assert len(registry) == 3
        assert registry.list_tools() == ["a", "b", "c"]

    def test_to_anthropic_format(self):
        """Convert all tools to Anthropic format."""
        from core.llm.tools import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="test", description="Test tool",
            parameters={"x": {"type": "string"}},
        ))

        schemas = registry.to_anthropic()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test"

    def test_to_openai_format(self):
        """Convert all tools to OpenAI format."""
        from core.llm.tools import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test", description="Test"))

        schemas = registry.to_openai()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"

    def test_validate_call_unknown_tool(self):
        """Validate fails for unknown tool name."""
        from core.llm.tools import ToolCall, ToolRegistry

        registry = ToolRegistry()
        call = ToolCall(id="1", name="unknown", arguments={})
        errors = registry.validate_call(call)
        assert len(errors) == 1
        assert "Unknown tool" in errors[0]


# ===========================================================================
# Test: ToolCall and ToolResult
# ===========================================================================

class TestToolCallResult:
    """Test ToolCall and ToolResult data classes."""

    def test_tool_call_validate(self):
        """ToolCall validates against definition."""
        from core.llm.tools import ToolCall, ToolDefinition

        tool = ToolDefinition(name="search", description="Search", required=["query"])
        call = ToolCall(id="1", name="search", arguments={"query": "test"})

        errors = call.validate_against(tool)
        assert errors == []

    def test_tool_result_to_anthropic(self):
        """ToolResult converts to Anthropic format."""
        from core.llm.tools import ToolResult

        result = ToolResult(
            tool_call_id="tc_123",
            content='{"data": "found 5 leads"}',
        )
        schema = result.to_anthropic()
        assert schema["type"] == "tool_result"
        assert schema["tool_use_id"] == "tc_123"
        assert schema["is_error"] is False

    def test_tool_result_to_openai(self):
        """ToolResult converts to OpenAI format."""
        from core.llm.tools import ToolResult

        result = ToolResult(
            tool_call_id="call_abc",
            content="Success",
        )
        schema = result.to_openai()
        assert schema["role"] == "tool"
        assert schema["tool_call_id"] == "call_abc"

    def test_tool_result_error(self):
        """ToolResult with is_error flag."""
        from core.llm.tools import ToolResult

        result = ToolResult(
            tool_call_id="tc_err",
            content="Connection timeout",
            is_error=True,
        )
        assert result.to_anthropic()["is_error"] is True


# ===========================================================================
# Test: ToolRouter
# ===========================================================================

class TestToolRouter:
    """Test the ToolRouter with mocked providers."""

    def _make_tool_router(self, anthropic_client=None, openai_client=None):
        from core.llm.tools import ToolRouter, ToolDefinition
        tools = [
            ToolDefinition(
                name="search_leads",
                description="Search for sales leads",
                parameters={"query": {"type": "string"}},
                required=["query"],
            ),
        ]
        return ToolRouter(
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_anthropic_tool_call(self):
        """Anthropic tool call parses tool_use blocks."""
        mock_client = MagicMock()

        # Mock response with tool_use block
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.name = "search_leads"
        tool_block.input = {"query": "CEO in Austin"}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'll search for leads."

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.stop_reason = "tool_use"

        mock_client.messages.create.return_value = mock_response

        router = self._make_tool_router(anthropic_client=mock_client)
        router._config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET)

        response = await router.route_with_tools(
            intent="general",
            system_prompt="You help find leads.",
            user_prompt="Find CEOs in Austin.",
        )

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_leads"
        assert response.tool_calls[0].arguments == {"query": "CEO in Austin"}
        assert response.text == "I'll search for leads."
        assert response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_openai_tool_call(self):
        """OpenAI tool call parses function_call format."""
        mock_client = MagicMock()

        # Mock response with tool_calls
        tool_call_obj = MagicMock()
        tool_call_obj.id = "call_abc"
        tool_call_obj.function.name = "search_leads"
        tool_call_obj.function.arguments = '{"query": "CTO"}'

        choice = MagicMock()
        choice.message.content = "Searching..."
        choice.message.tool_calls = [tool_call_obj]
        choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response

        router = self._make_tool_router(openai_client=mock_client)
        router._config.override_route(LLMIntent.GENERAL, GPT_4O)

        response = await router.route_with_tools(
            intent="general",
            system_prompt="Find leads.",
            user_prompt="Find CTOs.",
        )

        assert response.has_tool_calls
        assert response.tool_calls[0].name == "search_leads"
        assert response.tool_calls[0].arguments == {"query": "CTO"}

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        """Response without tool calls has empty list."""
        mock_client = MagicMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I don't need any tools."

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        router = self._make_tool_router(anthropic_client=mock_client)
        router._config.override_route(LLMIntent.GENERAL, CLAUDE_SONNET)

        response = await router.route_with_tools(
            intent="general",
            system_prompt="Help.",
            user_prompt="Hello.",
        )

        assert not response.has_tool_calls
        assert response.text == "I don't need any tools."

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        """Ollama doesn't support tool use — raises ValueError."""
        from core.llm.tools import ToolRouter, ToolDefinition
        from core.llm.llm_config import OLLAMA_LLAMA

        router = ToolRouter()
        router._config.override_route(LLMIntent.GENERAL, OLLAMA_LLAMA)

        with pytest.raises(ValueError, match="does not support tool use"):
            await router.route_with_tools(
                intent="general",
                system_prompt="Test",
                user_prompt="Test",
            )

    def test_usage_tracking(self):
        """Tool router tracks cumulative usage."""
        router = self._make_tool_router()
        assert router.call_count == 0

        stats = router.get_usage_stats()
        assert stats["total_calls"] == 0
        assert "registered_tools" in stats
        assert "search_leads" in stats["registered_tools"]

    def test_reset_usage(self):
        """Reset clears tool router counters."""
        router = self._make_tool_router()
        router._call_count = 10
        router._total_cost = 5.0
        router.reset_usage()
        assert router.call_count == 0
        assert router.total_cost == 0.0


# ===========================================================================
# Test: ToolResponse
# ===========================================================================

class TestToolResponse:
    """Test ToolResponse data class."""

    def test_has_tool_calls(self):
        """has_tool_calls property works correctly."""
        from core.llm.tools import ToolResponse, ToolCall

        empty = ToolResponse(text="Hello")
        assert not empty.has_tool_calls

        with_calls = ToolResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="test", arguments={})],
        )
        assert with_calls.has_tool_calls

    def test_total_tokens(self):
        """total_tokens sums input and output."""
        from core.llm.tools import ToolResponse

        resp = ToolResponse(text="", input_tokens=100, output_tokens=50)
        assert resp.total_tokens == 150


# ===========================================================================
# Test: ResponseCache
# ===========================================================================

class TestResponseCache:
    """Test the TTL-based LLM response cache."""

    def test_put_and_get(self):
        """Store and retrieve a cached response."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(ttl_seconds=300)
        key = ResponseCache.make_key("anthropic", "model", "sys", "user", 0.0, 4096)

        mock_response = {"text": "Hello", "cost": 0.01}
        cache.put(key, mock_response, provider="anthropic", model="model")

        result = cache.get(key)
        assert result == mock_response
        assert cache.size == 1

    def test_cache_miss(self):
        """Missing key returns None."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Expired entries are evicted on access."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(ttl_seconds=0)  # Immediate expiry
        key = ResponseCache.make_key("a", "m", "s", "u", 0.0, 100)
        cache.put(key, "data")

        # Entry should be expired immediately
        import time
        time.sleep(0.01)
        result = cache.get(key)
        assert result is None

    def test_lru_eviction(self):
        """Oldest entry evicted when max_entries reached."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(max_entries=3, ttl_seconds=300)

        for i in range(4):
            key = f"key_{i}"
            cache.put(key, f"value_{i}")

        # key_0 should be evicted (oldest)
        assert cache.get("key_0") is None
        assert cache.get("key_1") is not None
        assert cache.get("key_3") is not None
        assert cache.size == 3

    def test_hit_rate(self):
        """Hit rate calculation is correct."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(ttl_seconds=300)
        key = "test_key"
        cache.put(key, "value")

        # 1 hit
        cache.get(key)
        # 1 miss
        cache.get("missing")

        assert cache.hit_rate == 0.5

    def test_hit_rate_empty(self):
        """Hit rate is 0.0 when no requests made."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        assert cache.hit_rate == 0.0

    def test_should_cache_temperature(self):
        """Temperature guard controls caching."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(max_cacheable_temperature=0.3)
        assert cache.should_cache(0.0) is True
        assert cache.should_cache(0.3) is True
        assert cache.should_cache(0.5) is False
        assert cache.should_cache(1.0) is False

    def test_invalidate_single(self):
        """Invalidate a specific entry."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") is not None

    def test_invalidate_nonexistent(self):
        """Invalidating missing key returns False."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        assert cache.invalidate("nope") is False

    def test_invalidate_by_intent(self):
        """Remove all entries for an intent."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        cache.put("k1", "v1", intent="classification")
        cache.put("k2", "v2", intent="classification")
        cache.put("k3", "v3", intent="reasoning")

        removed = cache.invalidate_by_intent("classification")
        assert removed == 2
        assert cache.size == 1

    def test_clear(self):
        """Clear removes all entries."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        cache.put("a", 1)
        cache.put("b", 2)

        count = cache.clear()
        assert count == 2
        assert cache.size == 0

    def test_cleanup_expired(self):
        """Cleanup removes only expired entries."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(ttl_seconds=0)
        cache.put("expired1", "v1")
        cache.put("expired2", "v2")

        time.sleep(0.01)
        removed = cache.cleanup_expired()
        assert removed == 2
        assert cache.size == 0

    def test_get_stats(self):
        """Stats reflect cache state."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(ttl_seconds=300, max_entries=100)
        cache.put("k", "v")
        cache.get("k")  # hit
        cache.get("miss")  # miss

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["stores"] == 1
        assert stats["max_entries"] == 100
        assert stats["ttl_seconds"] == 300

    def test_reset_stats(self):
        """Reset stats without clearing data."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        cache.put("k", "v")
        cache.get("k")

        cache.reset_stats()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert cache.size == 1  # Data still there

    def test_list_entries(self):
        """List entries returns metadata for debugging."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache()
        cache.put("k1", "v1", provider="anthropic", model="sonnet", intent="reasoning")

        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0]["provider"] == "anthropic"
        assert entries[0]["intent"] == "reasoning"

    def test_make_key_deterministic(self):
        """Same inputs produce the same cache key."""
        from core.llm.cache import ResponseCache

        key1 = ResponseCache.make_key("a", "m", "s", "u", 0.5, 4096)
        key2 = ResponseCache.make_key("a", "m", "s", "u", 0.5, 4096)
        assert key1 == key2

    def test_make_key_different_inputs(self):
        """Different inputs produce different keys."""
        from core.llm.cache import ResponseCache

        key1 = ResponseCache.make_key("a", "m", "s", "prompt_A", 0.5, 4096)
        key2 = ResponseCache.make_key("a", "m", "s", "prompt_B", 0.5, 4096)
        assert key1 != key2

    def test_lru_moves_to_end_on_hit(self):
        """Cache hit moves entry to end (most recently used)."""
        from core.llm.cache import ResponseCache

        cache = ResponseCache(max_entries=3, ttl_seconds=300)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access "a" — should move it to end (MRU)
        cache.get("a")

        # Add a new entry — "b" should be evicted (LRU), not "a"
        cache.put("d", 4)

        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None
        assert cache.get("d") is not None


# ===========================================================================
# Test: cached_route helper
# ===========================================================================

class TestCachedRoute:
    """Test the cached_route convenience function."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Cached route returns cached response on hit."""
        from core.llm.cache import ResponseCache, cached_route
        from core.llm.router import ModelRouter, LLMResponse

        cache = ResponseCache(ttl_seconds=300, max_cacheable_temperature=1.0)

        # Pre-populate cache
        key = ResponseCache.make_key(
            "anthropic", CLAUDE_HAIKU.model, "classify", "good product", 0.3, 2048
        )
        cached_resp = LLMResponse(
            text="positive", provider="anthropic", model=CLAUDE_HAIKU.model,
        )
        cache.put(key, cached_resp)

        # Router should NOT be called
        mock_anthropic = MagicMock()
        router = ModelRouter(anthropic_client=mock_anthropic)

        result = await cached_route(
            cache, router,
            intent="classification",
            system_prompt="classify",
            user_prompt="good product",
        )

        assert result.text == "positive"
        mock_anthropic.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_router(self):
        """Cached route calls router on miss and stores result."""
        from core.llm.cache import ResponseCache, cached_route
        from core.llm.router import ModelRouter, LLMResponse

        cache = ResponseCache(ttl_seconds=300, max_cacheable_temperature=1.0)

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="negative")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_anthropic.messages.create.return_value = mock_response

        router = ModelRouter(anthropic_client=mock_anthropic)

        result = await cached_route(
            cache, router,
            intent="classification",
            system_prompt="classify",
            user_prompt="bad product",
        )

        assert result.text == "negative"
        assert cache.size == 1  # Stored in cache

    @pytest.mark.asyncio
    async def test_high_temperature_skips_cache(self):
        """High-temperature calls bypass cache entirely."""
        from core.llm.cache import ResponseCache, cached_route
        from core.llm.router import ModelRouter

        cache = ResponseCache(max_cacheable_temperature=0.3)

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="creative output")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_anthropic.messages.create.return_value = mock_response

        router = ModelRouter(anthropic_client=mock_anthropic)

        result = await cached_route(
            cache, router,
            intent="creative_writing",
            system_prompt="Be creative",
            user_prompt="Write a poem",
            temperature=0.9,
        )

        assert result.text == "creative output"
        assert cache.size == 0  # Not cached


# ===========================================================================
# Test: BaseAgent Sprint 2 Integration
# ===========================================================================

class TestBaseAgentSprint2:
    """Test BaseAgent integration with streaming, tools, and cache."""

    def _make_agent(self):
        """Create a concrete test agent."""
        from core.agents.base import BaseAgent
        from core.agents.state import BaseAgentState

        class TestAgent(BaseAgent):
            agent_type = "test_sprint2"

            def build_graph(self):
                return MagicMock()

            def get_tools(self):
                return []

            def get_state_class(self):
                return BaseAgentState

        mock_db = MagicMock()
        mock_db.reset_agent_errors = MagicMock()
        mock_db.record_agent_error = MagicMock(return_value=None)

        return TestAgent(
            config=_mock_anthropic_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_streaming_lazy_init(self):
        """streaming property creates StreamingRouter on first access."""
        agent = self._make_agent()
        assert agent._streaming_router is None

        streamer = agent.streaming
        assert streamer is not None

        from core.llm.streaming import StreamingRouter
        assert isinstance(streamer, StreamingRouter)

        # Same instance on second access
        assert agent.streaming is streamer

    def test_tool_router_lazy_init(self):
        """tool_router property creates ToolRouter on first access."""
        agent = self._make_agent()
        assert agent._tool_router is None

        tr = agent.tool_router
        assert tr is not None

        from core.llm.tools import ToolRouter
        assert isinstance(tr, ToolRouter)

    def test_cache_lazy_init(self):
        """cache property creates ResponseCache on first access."""
        agent = self._make_agent()
        assert agent._cache is None

        c = agent.cache
        assert c is not None

        from core.llm.cache import ResponseCache
        assert isinstance(c, ResponseCache)

    @pytest.mark.asyncio
    async def test_route_llm_cached(self):
        """route_llm_cached uses cache wrapper."""
        agent = self._make_agent()

        # Mock the router's route method
        from core.llm.router import LLMResponse
        mock_response = LLMResponse(
            text="cached result",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=5,
        )
        agent.router.route = AsyncMock(return_value=mock_response)

        # First call — cache miss
        result1 = await agent.route_llm_cached(
            intent="classification",
            system_prompt="classify",
            user_prompt="test input",
        )
        assert result1.text == "cached result"

        # Second call — cache hit (router not called again)
        agent.router.route.reset_mock()
        result2 = await agent.route_llm_cached(
            intent="classification",
            system_prompt="classify",
            user_prompt="test input",
        )
        assert result2.text == "cached result"
        # Router should not be called on cache hit
        agent.router.route.assert_not_called()


# ===========================================================================
# Test: StreamChunk & StreamStats data classes
# ===========================================================================

class TestDataClasses:
    """Test data class properties and defaults."""

    def test_stream_chunk_defaults(self):
        """StreamChunk has sensible defaults."""
        from core.llm.streaming import StreamChunk

        chunk = StreamChunk(text="hello")
        assert chunk.text == "hello"
        assert chunk.is_final is False
        assert chunk.total_tokens == 0

    def test_stream_chunk_final(self):
        """Final StreamChunk carries stats."""
        from core.llm.streaming import StreamChunk

        chunk = StreamChunk(
            text="", is_final=True,
            input_tokens=100, output_tokens=200,
            cost=0.05, latency_ms=1234.5,
        )
        assert chunk.total_tokens == 300
        assert chunk.cost == 0.05

    def test_stream_stats_total_tokens(self):
        """StreamStats total_tokens property."""
        from core.llm.streaming import StreamStats

        stats = StreamStats(input_tokens=50, output_tokens=150)
        assert stats.total_tokens == 200

    def test_tool_response_total_tokens(self):
        """ToolResponse total_tokens property."""
        from core.llm.tools import ToolResponse

        resp = ToolResponse(text="", input_tokens=75, output_tokens=25)
        assert resp.total_tokens == 100
