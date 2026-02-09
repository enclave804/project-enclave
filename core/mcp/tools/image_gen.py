"""
Image Generation MCP Tool — Create images from text prompts.

Uses OpenAI DALL-E 3 as the primary backend, with a provider-agnostic
interface so we can swap to Flux/Midjourney/Replicate later.

Generated images are saved to storage/images/{uuid}.png and the
local path is returned for use by other agents.

Usage:
    result = await generate_image(
        prompt="A cybersecurity bear mascot in armor, digital art style",
        aspect_ratio="1:1",
        style="vivid",
    )
    print(result["path"])  # storage/images/abc123.png

Safety:
    - Prompt is logged for audit
    - Images saved locally (no external URLs in agent state)
    - Size limit enforced
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default storage directory for generated images
DEFAULT_IMAGE_DIR = Path("storage/images")

# Valid sizes for DALL-E 3
VALID_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "square": "1024x1024",
    "landscape": "1792x1024",
    "portrait": "1024x1792",
}


async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    style: str = "vivid",
    quality: str = "standard",
    *,
    output_dir: Optional[str] = None,
    _openai_client: Any = None,
) -> dict[str, Any]:
    """
    Generate an image from a text prompt using DALL-E 3.

    Args:
        prompt: Description of the image to generate.
        aspect_ratio: Image proportions — "1:1", "16:9", "9:16",
                      "square", "landscape", or "portrait".
        style: "vivid" (dramatic) or "natural" (realistic).
        quality: "standard" or "hd" (higher detail, 2x cost).
        output_dir: Override default image storage directory.
        _openai_client: Injected OpenAI client (for testing/DI).

    Returns:
        Dict with:
            - path: Local file path to the saved image
            - prompt: The prompt used (may be revised by DALL-E)
            - size: Image dimensions
            - style: Style used
            - revised_prompt: DALL-E's interpretation of your prompt

    Note: This is a billable API call. DALL-E 3 pricing:
          Standard: $0.04/image (1024x1024), $0.08/image (1792x1024)
          HD: $0.08/image (1024x1024), $0.12/image (1792x1024)
    """
    if _openai_client is None:
        try:
            import openai
            _openai_client = openai.OpenAI()
        except ImportError:
            raise ImportError(
                "openai package required for image generation. "
                "Install with: pip install openai"
            )

    # Resolve size
    size = VALID_SIZES.get(aspect_ratio)
    if size is None:
        raise ValueError(
            f"Invalid aspect_ratio: {aspect_ratio}. "
            f"Valid options: {list(VALID_SIZES.keys())}"
        )

    # Validate style
    if style not in ("vivid", "natural"):
        raise ValueError(f"Invalid style: {style}. Use 'vivid' or 'natural'.")

    # Validate quality
    if quality not in ("standard", "hd"):
        raise ValueError(f"Invalid quality: {quality}. Use 'standard' or 'hd'.")

    logger.info(
        "image_generation_started",
        extra={
            "prompt_preview": prompt[:80],
            "size": size,
            "style": style,
            "quality": quality,
        },
    )

    # Call DALL-E 3
    response = _openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        style=style,
        quality=quality,
        n=1,
        response_format="b64_json",
    )

    # Extract image data
    image_data = response.data[0]
    b64_data = image_data.b64_json
    revised_prompt = getattr(image_data, "revised_prompt", prompt)

    # Save to disk
    save_dir = Path(output_dir) if output_dir else DEFAULT_IMAGE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    image_id = str(uuid.uuid4())[:12]
    filename = f"{image_id}.png"
    file_path = save_dir / filename

    image_bytes = base64.b64decode(b64_data)
    file_path.write_bytes(image_bytes)

    file_size_kb = len(image_bytes) // 1024

    logger.info(
        "image_generation_complete",
        extra={
            "path": str(file_path),
            "size_kb": file_size_kb,
            "revised_prompt": revised_prompt[:80],
        },
    )

    return {
        "path": str(file_path),
        "filename": filename,
        "prompt": prompt,
        "revised_prompt": revised_prompt,
        "size": size,
        "style": style,
        "quality": quality,
        "file_size_kb": file_size_kb,
    }


async def list_generated_images(
    output_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    List all previously generated images.

    Returns list of dicts with path, filename, and size info.
    """
    save_dir = Path(output_dir) if output_dir else DEFAULT_IMAGE_DIR
    if not save_dir.exists():
        return []

    images = []
    for img_path in sorted(save_dir.glob("*.png")):
        images.append({
            "path": str(img_path),
            "filename": img_path.name,
            "size_kb": img_path.stat().st_size // 1024,
        })
    return images


def get_image_gen_tool_names() -> list[str]:
    """Return all tool names for this integration."""
    return ["generate_image", "list_generated_images"]
