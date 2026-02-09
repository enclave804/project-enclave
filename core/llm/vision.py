"""
Vision Module — Image analysis via LLM vision models.

Provides a clean interface for sending images to vision-capable
LLMs (Claude Sonnet, GPT-4o) for analysis, description, and
structured data extraction.

Use cases:
- Analyze competitor landing pages
- Extract text from screenshots (OCR)
- Describe product images for SEO
- Verify visual brand compliance

Usage:
    from core.llm.vision import VisionClient

    vision = VisionClient(router=model_router)

    # Analyze a local image
    result = await vision.analyze(
        image_path="/path/to/screenshot.png",
        prompt="Describe the layout and CTA of this landing page.",
    )
    print(result.text)

    # Analyze from base64
    result = await vision.analyze_base64(
        image_data=base64_string,
        media_type="image/png",
        prompt="Extract all text from this image.",
    )

    # Analyze from URL
    result = await vision.analyze_url(
        image_url="https://example.com/page.png",
        prompt="What products are shown?",
    )
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Max file size for vision (20MB)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024


@dataclass
class VisionResult:
    """Result from an image analysis."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0


class VisionClient:
    """
    Image analysis client using vision-capable LLM models.

    Wraps the ModelRouter to send images with prompts to
    Claude Sonnet or GPT-4o for visual understanding.
    """

    def __init__(
        self,
        router: Any = None,
        default_provider: str = "anthropic",
    ):
        """
        Args:
            router: ModelRouter instance for dispatching calls.
                    If None, vision calls will raise ValueError.
            default_provider: Which provider to prefer for vision tasks.
        """
        self._router = router
        self._default_provider = default_provider

    async def analyze(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        system_prompt: str = "You are a visual analysis expert. Describe what you see accurately and thoroughly.",
        max_tokens: int = 2048,
    ) -> VisionResult:
        """
        Analyze a local image file.

        Args:
            image_path: Path to the image file.
            prompt: What to analyze / extract from the image.
            system_prompt: System instructions for the vision model.
            max_tokens: Maximum output length.

        Returns:
            VisionResult with analysis text and metadata.

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If file format is unsupported or file too large.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}"
            )

        file_size = path.stat().st_size
        if file_size > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.0f}MB)"
            )

        # Read and encode
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
        media_type = SUPPORTED_FORMATS[suffix]

        logger.info(
            "vision_analyze_file",
            extra={
                "file": str(path),
                "format": media_type,
                "size_kb": file_size // 1024,
            },
        )

        return await self.analyze_base64(
            image_data=image_data,
            media_type=media_type,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    async def analyze_base64(
        self,
        image_data: str,
        media_type: str,
        prompt: str,
        *,
        system_prompt: str = "You are a visual analysis expert. Describe what you see accurately and thoroughly.",
        max_tokens: int = 2048,
    ) -> VisionResult:
        """
        Analyze an image from base64-encoded data.

        Args:
            image_data: Base64-encoded image string.
            media_type: MIME type (e.g., "image/png").
            prompt: What to analyze / extract.
            system_prompt: System instructions.
            max_tokens: Maximum output length.

        Returns:
            VisionResult with analysis text and metadata.
        """
        if self._router is None:
            raise ValueError(
                "ModelRouter required for vision analysis. "
                "Pass router to VisionClient()."
            )

        images = [{
            "type": "base64",
            "media_type": media_type,
            "data": image_data,
        }]

        response = await self._router.route(
            intent="vision",
            system_prompt=system_prompt,
            user_prompt=prompt,
            images=images,
            max_tokens=max_tokens,
        )

        logger.info(
            "vision_analysis_complete",
            extra={
                "provider": response.provider,
                "model": response.model,
                "tokens": response.total_tokens,
                "cost": f"${response.cost:.4f}",
            },
        )

        return VisionResult(
            text=response.text,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
        )

    async def analyze_url(
        self,
        image_url: str,
        prompt: str,
        *,
        system_prompt: str = "You are a visual analysis expert. Describe what you see accurately and thoroughly.",
        max_tokens: int = 2048,
    ) -> VisionResult:
        """
        Analyze an image from a URL (downloads first).

        Args:
            image_url: URL of the image to analyze.
            prompt: What to analyze / extract.
            system_prompt: System instructions.
            max_tokens: Maximum output length.

        Returns:
            VisionResult with analysis text and metadata.
        """
        import httpx

        logger.info("vision_downloading_image", extra={"url": image_url[:100]})

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()

        # Detect media type from content-type header
        content_type = resp.headers.get("content-type", "image/png")
        media_type = content_type.split(";")[0].strip()

        image_data = base64.b64encode(resp.content).decode("utf-8")

        return await self.analyze_base64(
            image_data=image_data,
            media_type=media_type,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    async def extract_text(
        self,
        image_path: str | Path,
        *,
        language: str = "English",
    ) -> str:
        """
        Extract all text from an image (OCR-like).

        Args:
            image_path: Path to the image.
            language: Expected text language.

        Returns:
            Extracted text as a string.
        """
        result = await self.analyze(
            image_path=image_path,
            prompt=(
                f"Extract ALL text visible in this image, preserving layout "
                f"as much as possible. The text is in {language}. "
                f"Return ONLY the extracted text, no commentary."
            ),
            system_prompt="You are an OCR specialist. Extract text exactly as shown.",
        )
        return result.text

    async def describe_for_seo(
        self,
        image_path: str | Path,
        *,
        product_name: str = "",
        brand: str = "",
    ) -> dict[str, str]:
        """
        Generate SEO-optimized description for a product image.

        Returns dict with 'alt_text', 'description', and 'tags'.
        """
        context = ""
        if product_name:
            context += f"Product: {product_name}. "
        if brand:
            context += f"Brand: {brand}. "

        result = await self.analyze(
            image_path=image_path,
            prompt=(
                f"{context}"
                f"Generate SEO-optimized metadata for this image. "
                f"Return a JSON object with:\n"
                f'  "alt_text": concise alt text (under 125 chars),\n'
                f'  "description": detailed description (2-3 sentences),\n'
                f'  "tags": list of 5-10 relevant keywords\n'
                f"Return ONLY the JSON."
            ),
            system_prompt="You are an SEO specialist analyzing product images.",
        )

        # Try to parse JSON, fall back to raw text
        import json
        try:
            return json.loads(result.text)
        except json.JSONDecodeError:
            return {
                "alt_text": result.text[:125],
                "description": result.text,
                "tags": [],
            }
