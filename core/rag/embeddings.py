"""
Embedding generation for Project Enclave RAG system.

Uses Anthropic's Voyage AI embeddings or OpenAI's text-embedding-3-small
for generating vector representations of text chunks.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx


class EmbeddingEngine:
    """
    Generates text embeddings for RAG storage and retrieval.

    Supports multiple providers. Default is OpenAI text-embedding-3-small
    (1536 dimensions) which is cost-effective and performant for this scale.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ):
        self.provider = provider

        if provider == "openai":
            self.model = model or "text-embedding-3-small"
            self.api_key = os.environ.get(api_key_env or "OPENAI_API_KEY", "")
            self.base_url = "https://api.openai.com/v1/embeddings"
            self.dimensions = 1536
        elif provider == "voyage":
            self.model = model or "voyage-3-lite"
            self.api_key = os.environ.get(api_key_env or "VOYAGE_API_KEY", "")
            self.base_url = "https://api.voyageai.com/v1/embeddings"
            self.dimensions = 1024
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        if not self.api_key:
            raise EnvironmentError(
                f"API key not found for embedding provider '{provider}'. "
                f"Set {api_key_env or 'OPENAI_API_KEY'} environment variable."
            )

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed. Max recommended batch: 100.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        # Truncate overly long texts (most models have a token limit)
        processed = [t[:8000] for t in texts]

        async with httpx.AsyncClient(timeout=30.0) as client:
            if self.provider == "openai":
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": processed,
                        "model": self.model,
                    },
                )
            elif self.provider == "voyage":
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": processed,
                        "model": self.model,
                    },
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            response.raise_for_status()
            data = response.json()

            # Both OpenAI and Voyage return the same format
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

    def get_dimensions(self) -> int:
        """Return the embedding dimension for the current model."""
        return self.dimensions
