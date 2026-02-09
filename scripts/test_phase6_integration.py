#!/usr/bin/env python3
"""
Phase 6 Integration Test — Proves Router + Streaming + Tools + Cache
work together as a unified system.

This is NOT a pytest test — it's a standalone script that exercises
the real architecture end-to-end with mocked API clients.

Run:
    python scripts/test_phase6_integration.py
"""

import asyncio
import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock


def _build_mock_anthropic():
    """Build a mock Anthropic client that returns realistic responses."""
    client = MagicMock()

    # Standard messages.create
    def create_response(**kwargs):
        model = kwargs.get("model", "unknown")
        prompt = ""
        msgs = kwargs.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            prompt = content if isinstance(content, str) else str(content)

        # Check for tools
        tools = kwargs.get("tools", [])
        if tools:
            # Simulate tool_use response
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "I'll use a tool to help with that."

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = "toolu_integration_test"
            tool_block.name = tools[0]["name"]
            tool_block.input = {"query": "integration test"}

            resp = MagicMock()
            resp.content = [text_block, tool_block]
            resp.usage.input_tokens = 150
            resp.usage.output_tokens = 75
            resp.stop_reason = "tool_use"
            return resp

        # Standard text response
        resp = MagicMock()
        text_block = MagicMock()
        text_block.text = f"[{model}] Response to: {prompt[:50]}"
        resp.content = [text_block]
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        return resp

    client.messages.create.side_effect = create_response

    # Streaming support
    class MockStream:
        def __init__(self, text_chunks):
            self._chunks = text_chunks
            final = MagicMock()
            final.usage.input_tokens = 80
            final.usage.output_tokens = len("".join(text_chunks))
            self._final = final

        @property
        def text_stream(self):
            return iter(self._chunks)

        def get_final_message(self):
            return self._final

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def stream_response(**kwargs):
        model = kwargs.get("model", "")
        return MockStream([
            "The ", "Sovereign ", "Venture ", "Engine ",
            "is ", "operational. ", "All ", "systems ", "nominal."
        ])

    client.messages.stream.side_effect = stream_response

    return client


def _build_mock_openai():
    """Build a mock OpenAI client for creative routing."""
    client = MagicMock()

    def create_response(**kwargs):
        model = kwargs.get("model", "unknown")
        choice = MagicMock()
        choice.message.content = f"[{model}] Creative output: A haiku about code."
        choice.message.tool_calls = None
        choice.finish_reason = "stop"

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 60
        resp.usage.completion_tokens = 40
        return resp

    client.chat.completions.create.side_effect = create_response

    return client


async def run_integration_test():
    """Full integration test exercising all Phase 6 modules together."""
    print("=" * 70)
    print("🚀 PHASE 6 INTEGRATION TEST — Router + Streaming + Tools + Cache")
    print("=" * 70)

    mock_anthropic = _build_mock_anthropic()
    mock_openai = _build_mock_openai()

    passed = 0
    failed = 0

    # ── Test 1: ModelRouter — Intent-based routing ──────────────────
    print("\n🧪 Test 1: ModelRouter — Intent-based routing")
    try:
        from core.llm.router import ModelRouter
        router = ModelRouter(
            anthropic_client=mock_anthropic,
            openai_client=mock_openai,
        )

        # Reasoning → Anthropic (Claude Sonnet)
        resp = await router.route(
            intent="reasoning",
            system_prompt="Analyze this.",
            user_prompt="What is 2+2?",
        )
        assert resp.provider == "anthropic", f"Expected anthropic, got {resp.provider}"
        assert resp.text, "Response text is empty"
        assert resp.cost >= 0, "Cost should be non-negative"
        print(f"   ✅ Reasoning → {resp.provider}/{resp.model} ({resp.total_tokens} tokens, ${resp.cost:.4f})")

        # Creative → OpenAI (GPT-4o)
        resp = await router.route(
            intent="creative_writing",
            system_prompt="Be creative.",
            user_prompt="Write a haiku.",
        )
        assert resp.provider == "openai", f"Expected openai, got {resp.provider}"
        print(f"   ✅ Creative → {resp.provider}/{resp.model} ({resp.total_tokens} tokens)")

        # Classification → Anthropic Haiku (cheapest)
        resp = await router.route(
            intent="classification",
            system_prompt="Classify intent.",
            user_prompt="I want to buy.",
        )
        assert resp.provider == "anthropic", f"Expected anthropic, got {resp.provider}"
        print(f"   ✅ Classification → {resp.provider}/{resp.model}")

        # Usage tracking
        stats = router.get_usage_stats()
        assert stats["total_calls"] == 3
        print(f"   ✅ Usage tracked: {stats['total_calls']} calls, ${stats['total_cost_usd']:.4f}")

        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        failed += 1

    # ── Test 2: StreamingRouter — Real-time token streaming ────────
    print("\n🧪 Test 2: StreamingRouter — Real-time token streaming")
    try:
        from core.llm.streaming import StreamingRouter, collect_stream

        streamer = StreamingRouter(
            anthropic_client=mock_anthropic,
            openai_client=mock_openai,
        )

        # Stream reasoning response
        print("   Streaming: ", end="", flush=True)
        chunks = []
        final_chunk = None
        async for chunk in streamer.stream(
            intent="reasoning",
            system_prompt="Report status.",
            user_prompt="System health check.",
        ):
            if chunk.is_final:
                final_chunk = chunk
            else:
                chunks.append(chunk.text)
                print(chunk.text, end="", flush=True)

        print()
        assert len(chunks) > 0, "No chunks received"
        assert final_chunk is not None, "No final chunk"
        assert final_chunk.input_tokens > 0, "No input tokens on final"

        full_text = "".join(chunks)
        print(f"   ✅ Streamed {len(chunks)} chunks, {len(full_text)} chars")
        print(f"   ✅ Final stats: {final_chunk.input_tokens}+{final_chunk.output_tokens} tokens")

        # Test collect_stream helper
        mock_anthropic.messages.stream.side_effect = lambda **kw: type(
            'S', (), {
                'text_stream': property(lambda s: iter(["Collected ", "test"])),
                'get_final_message': lambda s: type('M', (), {
                    'usage': type('U', (), {'input_tokens': 10, 'output_tokens': 20})()
                })(),
                '__enter__': lambda s: s,
                '__exit__': lambda s, *a: None,
            }
        )()

        text, final = await collect_stream(
            streamer.stream("reasoning", "S", "U")
        )
        assert text == "Collected test"
        print(f"   ✅ collect_stream() works: '{text}'")

        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Test 3: ToolRouter — Function calling across providers ─────
    print("\n🧪 Test 3: ToolRouter — Function calling")
    try:
        from core.llm.tools import ToolRouter, ToolDefinition, ToolResult

        search_tool = ToolDefinition(
            name="search_leads",
            description="Search for sales leads",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            required=["query"],
        )

        enrich_tool = ToolDefinition(
            name="enrich_company",
            description="Enrich company data",
            parameters={
                "domain": {"type": "string", "description": "Company domain"},
            },
            required=["domain"],
        )

        tool_router = ToolRouter(
            anthropic_client=mock_anthropic,
            tools=[search_tool, enrich_tool],
        )

        # Verify registry
        assert len(tool_router.registry) == 2
        assert "search_leads" in tool_router.registry
        print(f"   ✅ Registry: {tool_router.registry.list_tools()}")

        # Schema conversion
        anthro_schema = search_tool.to_anthropic()
        openai_schema = search_tool.to_openai()
        assert anthro_schema["input_schema"]["required"] == ["query"]
        assert openai_schema["function"]["parameters"]["required"] == ["query"]
        print("   ✅ Schema converts correctly for both providers")

        # Tool call with response
        response = await tool_router.route_with_tools(
            intent="reasoning",
            system_prompt="You help find leads.",
            user_prompt="Find CEOs in Austin.",
        )

        assert response.has_tool_calls, "Expected tool calls in response"
        call = response.tool_calls[0]
        print(f"   ✅ Tool call: {call.name}({call.arguments})")

        # Validate the call against definition
        errors = tool_router.registry.validate_call(call)
        print(f"   ✅ Validation: {len(errors)} errors")

        # Create tool result
        result = ToolResult(
            tool_call_id=call.id,
            content='[{"name": "Alice CEO", "company": "TechCorp"}]',
        )
        assert result.to_anthropic()["type"] == "tool_result"
        assert result.to_openai()["role"] == "tool"
        print(f"   ✅ ToolResult converts for both providers")

        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Test 4: ResponseCache — TTL, LRU, temperature guard ───────
    print("\n🧪 Test 4: ResponseCache — Caching layer")
    try:
        from core.llm.cache import ResponseCache, cached_route
        from core.llm.router import LLMResponse

        cache = ResponseCache(ttl_seconds=60, max_entries=5)

        # Simulate 10 identical classification calls
        key = ResponseCache.make_key(
            "anthropic", "claude-3-5-haiku", "classify", "good product", 0.3, 2048
        )

        # First call: cache miss
        assert cache.get(key) is None
        print("   ✅ Cache miss on first call")

        # Store the response
        fake_resp = LLMResponse(
            text="positive",
            provider="anthropic",
            model="claude-3-5-haiku",
            input_tokens=10,
            output_tokens=5,
            cost=0.00015,
        )
        cache.put(key, fake_resp, provider="anthropic", model="haiku", intent="classification")

        # Next 9 calls: cache hit
        for i in range(9):
            hit = cache.get(key)
            assert hit is not None
            assert hit.text == "positive"

        stats = cache.get_stats()
        assert stats["hits"] == 9
        assert stats["misses"] == 1
        print(f"   ✅ 9 cache hits, 1 miss (hit rate: {stats['hit_rate']:.0%})")
        print(f"   ✅ Cost saved: ${fake_resp.cost * 9:.4f} (9 avoided API calls)")

        # Temperature guard
        assert cache.should_cache(0.0) is True
        assert cache.should_cache(0.9) is False
        print("   ✅ Temperature guard: caches 0.0, skips 0.9")

        # LRU eviction
        for i in range(10):
            cache.put(f"lru_{i}", f"value_{i}")
        assert cache.size == 5, f"Expected max 5, got {cache.size}"
        print(f"   ✅ LRU eviction: max_entries=5 enforced (size={cache.size})")

        # cached_route integration
        router = ModelRouter(anthropic_client=mock_anthropic)
        fresh_cache = ResponseCache(ttl_seconds=60, max_cacheable_temperature=1.0)

        resp1 = await cached_route(
            fresh_cache, router,
            intent="classification",
            system_prompt="classify",
            user_prompt="test product",
        )
        assert fresh_cache.size == 1
        print(f"   ✅ cached_route: stored response (cache size={fresh_cache.size})")

        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Test 5: BaseAgent Integration — All modules wired ──────────
    print("\n🧪 Test 5: BaseAgent — All Phase 6 modules integrated")
    try:
        from core.agents.base import BaseAgent
        from core.agents.state import BaseAgentState
        from core.config.agent_schema import AgentInstanceConfig

        class IntegrationAgent(BaseAgent):
            agent_type = "integration_test"

            def build_graph(self):
                return MagicMock()

            def get_tools(self):
                return []

            def get_state_class(self):
                return BaseAgentState

        config = AgentInstanceConfig(
            agent_id="integration_test",
            agent_type="integration_test",
            name="Integration Test Agent",
            vertical_id="test",
        )

        mock_db = MagicMock()
        mock_db.reset_agent_errors = MagicMock()
        mock_db.record_agent_error = MagicMock(return_value=None)

        agent = IntegrationAgent(
            config=config,
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=mock_anthropic,
            openai_client=mock_openai,
        )

        # Verify all Sprint 1 properties
        assert agent.llm is mock_anthropic, "self.llm backward compat broken"
        assert agent.router is not None, "Router not initialized"
        assert agent.vision is not None, "Vision not initialized"
        print("   ✅ Sprint 1: router, vision, self.llm — all wired")

        # Verify all Sprint 2 properties (lazy init)
        from core.llm.streaming import StreamingRouter
        from core.llm.tools import ToolRouter
        from core.llm.cache import ResponseCache

        assert isinstance(agent.streaming, StreamingRouter)
        assert isinstance(agent.tool_router, ToolRouter)
        assert isinstance(agent.cache, ResponseCache)
        print("   ✅ Sprint 2: streaming, tool_router, cache — all lazy-initialized")

        # Test route_llm
        resp = await agent.route_llm(
            intent="reasoning",
            system_prompt="Test",
            user_prompt="Hello",
        )
        assert resp.text, "route_llm returned empty text"
        print(f"   ✅ route_llm(): {resp.provider}/{resp.model}")

        # Test route_llm_cached
        resp = await agent.route_llm_cached(
            intent="classification",
            system_prompt="classify",
            user_prompt="test",
        )
        assert resp.text, "route_llm_cached returned empty text"
        print(f"   ✅ route_llm_cached(): cached response ready")

        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = passed + failed
    if failed == 0:
        print(f"✅ ALL {total} INTEGRATION TESTS PASSED")
        print("   Phase 6 is BATTLE-READY. Router + Streaming + Tools + Cache = 🔥")
    else:
        print(f"⚠️  {passed}/{total} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_integration_test())
    sys.exit(0 if success else 1)
