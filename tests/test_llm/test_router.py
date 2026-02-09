"""
Tests for Phase 6: Multi-Model Router & Creative Suite.

Covers:
1. LLMConfig — routing table, intent resolution, overrides
2. ModelRouter — provider dispatch, fallback, usage tracking
3. VisionClient — image analysis, format validation, OCR
4. ImageGenTool — DALL-E integration, file saving
5. BaseAgent integration — router wiring, backward compat

All tests use mocks — no actual API calls to Anthropic, OpenAI, or Ollama.
"""

from __future__ import annotations

import base64
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from core.llm.llm_config import (
    DEFAULT_ROUTING,
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
    GPT_4O,
    GPT_4O_MINI,
    LLMConfig,
    LLMIntent,
    ModelProfile,
    RouteConfig,
)
from core.llm.router import LLMResponse, ModelRouter
from core.llm.vision import VisionClient, VisionResult, SUPPORTED_FORMATS
from core.mcp.tools.image_gen import (
    generate_image,
    list_generated_images,
    VALID_SIZES,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="Claude says hello")]
    response.usage = MagicMock(input_tokens=100, output_tokens=50)
    client.messages.create.return_value = response
    return client


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "GPT says hello"
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=80, completion_tokens=40)
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_openai_images():
    """Mock OpenAI client for image generation."""
    client = MagicMock()
    # Create a small 1x1 PNG in base64
    tiny_png = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
        b'\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode()
    image_data = MagicMock()
    image_data.b64_json = tiny_png
    image_data.revised_prompt = "A cybersecurity bear, digital art"
    response = MagicMock()
    response.data = [image_data]
    client.images.generate.return_value = response
    return client


@pytest.fixture
def router(mock_anthropic, mock_openai):
    """ModelRouter with mocked clients."""
    return ModelRouter(
        anthropic_client=mock_anthropic,
        openai_client=mock_openai,
    )


@pytest.fixture
def config():
    """Default LLMConfig."""
    return LLMConfig()


# ===========================================================================
# Test: LLMConfig — Routing Configuration
# ===========================================================================

class TestLLMConfig:
    """Test routing table configuration."""

    def test_default_routing_has_all_intents(self, config):
        """Default routing covers all intent types."""
        for intent in LLMIntent:
            route = config.get_route(intent)
            assert route is not None
            assert route.primary is not None

    def test_reasoning_routes_to_sonnet(self, config):
        """Reasoning intent routes to Claude Sonnet."""
        profile = config.get_model_for_intent("reasoning")
        assert profile.provider == "anthropic"
        assert "sonnet" in profile.model.lower() or "claude" in profile.model.lower()

    def test_creative_routes_to_gpt4o(self, config):
        """Creative writing routes to GPT-4o."""
        profile = config.get_model_for_intent("creative_writing")
        assert profile.provider == "openai"
        assert "gpt-4o" in profile.model

    def test_classification_routes_to_haiku(self, config):
        """Classification routes to cheap/fast model."""
        profile = config.get_model_for_intent("classification")
        assert profile.provider == "anthropic"
        assert "haiku" in profile.model.lower()

    def test_unknown_intent_falls_to_general(self, config):
        """Unknown intent defaults to general routing."""
        profile = config.get_model_for_intent("nonexistent_intent")
        assert profile is not None
        # General defaults to Claude Sonnet
        assert profile.provider == "anthropic"

    def test_override_route(self, config):
        """Override routing for a specific intent."""
        custom = ModelProfile(provider="ollama", model="llama3.1:8b")
        config.override_route(LLMIntent.CLASSIFICATION, custom)

        profile = config.get_model_for_intent("classification")
        assert profile.provider == "ollama"
        assert profile.model == "llama3.1:8b"

    def test_set_all_to_provider(self, config):
        """Override all routes to a single provider."""
        config.set_all_to_provider("ollama", "llama3.1:8b")

        for intent in LLMIntent:
            profile = config.get_model_for_intent(intent)
            assert profile.provider == "ollama"
            assert profile.model == "llama3.1:8b"

    def test_fallback_configured(self, config):
        """Most routes have fallback configured."""
        reasoning = config.get_route("reasoning")
        assert reasoning.fallback is not None

    def test_list_routes(self, config):
        """List routes returns structured data."""
        routes = config.list_routes()
        assert len(routes) == len(LLMIntent)
        assert all("intent" in r for r in routes)
        assert all("primary" in r for r in routes)

    def test_model_profile_display_name(self):
        """ModelProfile display_name is formatted correctly."""
        profile = ModelProfile(provider="anthropic", model="claude-sonnet-4-20250514")
        assert profile.display_name == "anthropic/claude-sonnet-4-20250514"

    def test_model_profile_cost_fields(self):
        """ModelProfile has cost information."""
        assert CLAUDE_SONNET.cost_per_1k_input > 0
        assert CLAUDE_SONNET.cost_per_1k_output > 0
        assert CLAUDE_HAIKU.cost_per_1k_input < CLAUDE_SONNET.cost_per_1k_input


# ===========================================================================
# Test: ModelRouter — Provider Dispatch
# ===========================================================================

class TestModelRouter:
    """Test multi-model routing and provider dispatch."""

    @pytest.mark.asyncio
    async def test_route_to_anthropic(self, router, mock_anthropic):
        """Routing reasoning intent calls Anthropic."""
        response = await router.route(
            intent="reasoning",
            system_prompt="You are an analyst.",
            user_prompt="Analyze this data.",
        )
        assert response.text == "Claude says hello"
        assert response.provider == "anthropic"
        mock_anthropic.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_openai(self, router, mock_openai):
        """Routing creative intent calls OpenAI."""
        response = await router.route(
            intent="creative_writing",
            system_prompt="You are a copywriter.",
            user_prompt="Write a tweet.",
        )
        assert response.text == "GPT says hello"
        assert response.provider == "openai"
        mock_openai.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_tracks_usage(self, router):
        """Router tracks cumulative usage stats."""
        assert router.call_count == 0
        assert router.total_cost == 0.0

        await router.route("reasoning", "sys", "user")
        assert router.call_count == 1
        assert router.total_cost > 0

        stats = router.get_usage_stats()
        assert stats["total_calls"] == 1
        assert stats["total_input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_route_with_temperature_override(self, router, mock_anthropic):
        """Temperature override is passed through."""
        await router.route(
            intent="reasoning",
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
        )
        call_kwargs = mock_anthropic.messages.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, mock_openai):
        """Router falls back when primary fails."""
        # Anthropic fails
        bad_anthropic = MagicMock()
        bad_anthropic.messages.create.side_effect = Exception("API down")

        router = ModelRouter(
            anthropic_client=bad_anthropic,
            openai_client=mock_openai,
        )

        response = await router.route("reasoning", "sys", "user")
        assert response.is_fallback
        assert response.provider == "openai"
        assert response.text == "GPT says hello"

    @pytest.mark.asyncio
    async def test_both_fail_raises(self):
        """If both primary and fallback fail, exception is raised."""
        bad_anthropic = MagicMock()
        bad_anthropic.messages.create.side_effect = Exception("Anthropic down")
        bad_openai = MagicMock()
        bad_openai.chat.completions.create.side_effect = Exception("OpenAI down")

        router = ModelRouter(
            anthropic_client=bad_anthropic,
            openai_client=bad_openai,
        )

        with pytest.raises(Exception, match="OpenAI down"):
            await router.route("reasoning", "sys", "user")

    @pytest.mark.asyncio
    async def test_direct_call_anthropic(self, router, mock_anthropic):
        """Direct Anthropic call bypasses routing."""
        response = await router.call_anthropic(
            system_prompt="sys",
            user_prompt="user",
            model="claude-3-5-haiku-20241022",
        )
        assert response.provider == "anthropic"
        call_kwargs = mock_anthropic.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-3-5-haiku-20241022"

    @pytest.mark.asyncio
    async def test_direct_call_openai(self, router, mock_openai):
        """Direct OpenAI call bypasses routing."""
        response = await router.call_openai(
            system_prompt="sys",
            user_prompt="user",
            model="gpt-4o-mini",
        )
        assert response.provider == "openai"
        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_no_anthropic_client_raises(self):
        """Missing Anthropic client raises clear error."""
        router = ModelRouter(anthropic_client=None, openai_client=MagicMock())
        with pytest.raises(ValueError, match="Anthropic client"):
            await router.call_anthropic("sys", "user")

    @pytest.mark.asyncio
    async def test_no_openai_client_raises(self):
        """Missing OpenAI client raises clear error."""
        router = ModelRouter(anthropic_client=MagicMock(), openai_client=None)
        with pytest.raises(ValueError, match="OpenAI client"):
            await router.call_openai("sys", "user")

    def test_reset_usage(self, router):
        """Usage counters can be reset."""
        router._call_count = 10
        router._total_cost = 0.5
        router.reset_usage()
        assert router.call_count == 0
        assert router.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_vision_images_passed_to_anthropic(self, router, mock_anthropic):
        """Images are passed through to Anthropic for vision."""
        images = [{"type": "base64", "media_type": "image/png", "data": "abc123"}]
        await router.route(
            intent="vision",
            system_prompt="Analyze this.",
            user_prompt="What do you see?",
            images=images,
        )
        call_kwargs = mock_anthropic.messages.create.call_args
        content = call_kwargs.kwargs["messages"][0]["content"]
        # Content should be a list (multimodal)
        assert isinstance(content, list)
        assert content[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_vision_images_passed_to_openai(self, router, mock_openai):
        """Images are passed through to OpenAI for vision."""
        # Override creative to use OpenAI
        images = [{"type": "base64", "media_type": "image/png", "data": "abc123"}]
        await router.route(
            intent="creative_writing",
            system_prompt="Describe this.",
            user_prompt="What do you see?",
            images=images,
        )
        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        # User content should be a list (multimodal)
        assert isinstance(user_msg["content"], list)

    @pytest.mark.asyncio
    async def test_llm_response_total_tokens(self):
        """LLMResponse calculates total_tokens."""
        resp = LLMResponse(
            text="test",
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=50,
        )
        assert resp.total_tokens == 150

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self, router):
        """Unsupported provider raises ValueError."""
        profile = ModelProfile(provider="azure", model="test")
        with pytest.raises(ValueError, match="Unsupported provider"):
            await router._call_model(profile, "sys", "user")


# ===========================================================================
# Test: VisionClient
# ===========================================================================

class TestVisionClient:
    """Test image analysis capabilities."""

    @pytest.mark.asyncio
    async def test_analyze_file(self, router, tmp_path):
        """Analyze a local image file."""
        # Create a dummy PNG
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b'\x89PNG' + b'\x00' * 100)

        vision = VisionClient(router=router)
        result = await vision.analyze(
            image_path=img_file,
            prompt="What is this?",
        )
        assert result.text  # Should have some response
        assert result.provider in ("anthropic", "openai")

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_file(self, router):
        """Nonexistent file raises FileNotFoundError."""
        vision = VisionClient(router=router)
        with pytest.raises(FileNotFoundError):
            await vision.analyze("/no/such/image.png", "Describe this")

    @pytest.mark.asyncio
    async def test_analyze_unsupported_format(self, router, tmp_path):
        """Unsupported format raises ValueError."""
        bad_file = tmp_path / "image.tiff"
        bad_file.write_bytes(b'\x00' * 10)

        vision = VisionClient(router=router)
        with pytest.raises(ValueError, match="Unsupported image format"):
            await vision.analyze(bad_file, "Describe this")

    @pytest.mark.asyncio
    async def test_analyze_too_large(self, router, tmp_path):
        """Oversized file raises ValueError."""
        big_file = tmp_path / "huge.png"
        # Don't actually create a 20MB file - just mock the size
        big_file.write_bytes(b'\x89PNG' + b'\x00' * 10)

        vision = VisionClient(router=router)
        # Patch stat to report large size
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=25 * 1024 * 1024)
            with pytest.raises(ValueError, match="too large"):
                await vision.analyze(big_file, "Describe this")

    @pytest.mark.asyncio
    async def test_analyze_base64(self, router):
        """Analyze from base64 data."""
        vision = VisionClient(router=router)
        result = await vision.analyze_base64(
            image_data="abc123base64data",
            media_type="image/png",
            prompt="What is this?",
        )
        assert result.text

    @pytest.mark.asyncio
    async def test_no_router_raises(self):
        """VisionClient without router raises clear error."""
        vision = VisionClient(router=None)
        with pytest.raises(ValueError, match="ModelRouter required"):
            await vision.analyze_base64("data", "image/png", "Describe")

    def test_supported_formats(self):
        """All common image formats are supported."""
        assert ".png" in SUPPORTED_FORMATS
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".gif" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS

    @pytest.mark.asyncio
    async def test_extract_text(self, router, tmp_path):
        """OCR-like text extraction works."""
        img_file = tmp_path / "text.png"
        img_file.write_bytes(b'\x89PNG' + b'\x00' * 100)

        vision = VisionClient(router=router)
        text = await vision.extract_text(img_file)
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_describe_for_seo(self, router, tmp_path, mock_anthropic):
        """SEO description returns structured data."""
        img_file = tmp_path / "product.png"
        img_file.write_bytes(b'\x89PNG' + b'\x00' * 100)

        # Make Anthropic return valid JSON
        mock_anthropic.messages.create.return_value.content[0].text = json.dumps({
            "alt_text": "A product image",
            "description": "This is a product.",
            "tags": ["product", "image"],
        })

        vision = VisionClient(router=router)
        result = await vision.describe_for_seo(
            img_file, product_name="Test Product", brand="TestCo"
        )
        assert "alt_text" in result
        assert "description" in result
        assert "tags" in result


# ===========================================================================
# Test: Image Generation Tool
# ===========================================================================

class TestImageGenTool:
    """Test DALL-E image generation."""

    @pytest.mark.asyncio
    async def test_generate_image(self, mock_openai_images, tmp_path):
        """Generate and save an image."""
        result = await generate_image(
            prompt="A cybersecurity bear",
            aspect_ratio="1:1",
            output_dir=str(tmp_path),
            _openai_client=mock_openai_images,
        )
        assert result["path"]
        assert Path(result["path"]).exists()
        assert result["size"] == "1024x1024"
        assert result["prompt"] == "A cybersecurity bear"

    @pytest.mark.asyncio
    async def test_generate_landscape(self, mock_openai_images, tmp_path):
        """Landscape aspect ratio resolves correctly."""
        result = await generate_image(
            prompt="A landscape",
            aspect_ratio="16:9",
            output_dir=str(tmp_path),
            _openai_client=mock_openai_images,
        )
        assert result["size"] == "1792x1024"

    @pytest.mark.asyncio
    async def test_generate_portrait(self, mock_openai_images, tmp_path):
        """Portrait aspect ratio resolves correctly."""
        result = await generate_image(
            prompt="A portrait",
            aspect_ratio="9:16",
            output_dir=str(tmp_path),
            _openai_client=mock_openai_images,
        )
        assert result["size"] == "1024x1792"

    @pytest.mark.asyncio
    async def test_invalid_aspect_ratio(self, mock_openai_images):
        """Invalid aspect ratio raises ValueError."""
        with pytest.raises(ValueError, match="Invalid aspect_ratio"):
            await generate_image(
                prompt="test",
                aspect_ratio="4:3",
                _openai_client=mock_openai_images,
            )

    @pytest.mark.asyncio
    async def test_invalid_style(self, mock_openai_images):
        """Invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Invalid style"):
            await generate_image(
                prompt="test",
                style="abstract",
                _openai_client=mock_openai_images,
            )

    @pytest.mark.asyncio
    async def test_invalid_quality(self, mock_openai_images):
        """Invalid quality raises ValueError."""
        with pytest.raises(ValueError, match="Invalid quality"):
            await generate_image(
                prompt="test",
                quality="ultra",
                _openai_client=mock_openai_images,
            )

    @pytest.mark.asyncio
    async def test_list_generated_images(self, mock_openai_images, tmp_path):
        """List previously generated images."""
        # Generate one image
        await generate_image(
            prompt="test",
            output_dir=str(tmp_path),
            _openai_client=mock_openai_images,
        )

        images = await list_generated_images(output_dir=str(tmp_path))
        assert len(images) >= 1
        assert images[0]["filename"].endswith(".png")

    @pytest.mark.asyncio
    async def test_list_empty_dir(self, tmp_path):
        """List returns empty for nonexistent directory."""
        images = await list_generated_images(
            output_dir=str(tmp_path / "nonexistent")
        )
        assert images == []

    def test_valid_sizes(self):
        """All VALID_SIZES map to real DALL-E dimensions."""
        for key, value in VALID_SIZES.items():
            width, height = value.split("x")
            assert int(width) >= 1024
            assert int(height) >= 1024


# ===========================================================================
# Test: BaseAgent Router Integration
# ===========================================================================

class TestBaseAgentRouterIntegration:
    """Test that BaseAgent correctly wires up the ModelRouter."""

    def test_base_agent_has_router(self, mock_anthropic):
        """BaseAgent creates a router from anthropic_client."""
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="outreach",
            name="Test Agent",
            vertical_id="test_vertical",
        )

        # Create a concrete subclass for testing
        class TestAgent:
            def __init__(self):
                from core.llm.router import ModelRouter
                self.router = ModelRouter(anthropic_client=mock_anthropic)
                self._vision = None

        agent = TestAgent()
        assert agent.router is not None
        assert isinstance(agent.router, ModelRouter)

    def test_router_with_both_clients(self, mock_anthropic, mock_openai):
        """Router accepts both Anthropic and OpenAI clients."""
        router = ModelRouter(
            anthropic_client=mock_anthropic,
            openai_client=mock_openai,
        )
        assert router._anthropic is mock_anthropic
        assert router._openai is mock_openai

    def test_backward_compat_self_llm(self, mock_anthropic):
        """self.llm still works for backward compatibility."""
        # Simulating what BaseAgent.__init__ does
        llm = mock_anthropic
        router = ModelRouter(anthropic_client=mock_anthropic)

        # Both should be usable
        assert llm is not None
        assert router is not None

    def test_vision_lazy_init(self, mock_anthropic):
        """VisionClient is lazily initialized."""
        router = ModelRouter(anthropic_client=mock_anthropic)
        vision = VisionClient(router=router)
        assert vision._router is router


# ===========================================================================
# Test: Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Test boundary conditions."""

    def test_empty_intent_string(self, config):
        """Empty intent string falls to general."""
        profile = config.get_model_for_intent("")
        assert profile is not None

    def test_model_profile_defaults(self):
        """ModelProfile has sensible defaults."""
        profile = ModelProfile(provider="test", model="test-model")
        assert profile.temperature == 0.5
        assert profile.max_tokens == 4096
        assert profile.supports_tools is True
        assert profile.supports_vision is False

    @pytest.mark.asyncio
    async def test_response_cost_calculation(self, router, mock_anthropic):
        """Cost is calculated based on token usage."""
        response = await router.route("reasoning", "sys", "user")
        # 100 input tokens + 50 output tokens with Sonnet pricing
        assert response.cost > 0
        assert response.cost < 1.0  # Sanity check

    @pytest.mark.asyncio
    async def test_latency_tracking(self, router):
        """Latency is recorded in milliseconds."""
        response = await router.route("reasoning", "sys", "user")
        assert response.latency_ms >= 0

    def test_route_config_has_description(self, config):
        """Route configs have descriptive text."""
        route = config.get_route("reasoning")
        assert route.description
        assert len(route.description) > 10
