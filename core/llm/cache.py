"""
Prompt Cache — In-memory LLM response caching with TTL.

Caches LLM responses to avoid redundant API calls. Useful for:
- Repeated classification/extraction tasks on similar data
- System prompts that produce deterministic outputs
- Development/testing (avoid burning API credits)

Cache keys are computed from (provider, model, system_prompt, user_prompt,
temperature, max_tokens) — so identical prompts to the same model hit cache.

Safety: Only caches when temperature < 0.3 (near-deterministic) by default.
Higher-temperature calls (creative) are never cached because they should
produce varied output.

Usage:
    from core.llm.cache import ResponseCache, cached_route

    cache = ResponseCache(ttl_seconds=300, max_entries=500)

    # Manual: check/store
    key = cache.make_key("anthropic", "claude-sonnet-4-20250514", system, user, 0.0, 4096)
    cached = cache.get(key)
    if cached:
        return cached
    result = await router.route(...)
    cache.put(key, result)

    # Automatic: wrap the router
    response = await cached_route(
        cache, router,
        intent="classification",
        system_prompt="Classify intent: positive/negative/neutral",
        user_prompt="Great product, love it!",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache Entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A cached LLM response with expiration metadata."""

    key: str
    response: Any                 # LLMResponse object
    created_at: float             # time.monotonic()
    expires_at: float             # time.monotonic() + ttl
    hit_count: int = 0            # Times this entry was served
    provider: str = ""
    model: str = ""
    intent: str = ""

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


# ---------------------------------------------------------------------------
# Response Cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    In-memory LRU cache for LLM responses with TTL expiration.

    Thread-safe for single-threaded async (asyncio). For multi-threaded
    environments, wrap access with a lock.

    Features:
    - TTL expiration (default 5 minutes)
    - LRU eviction (oldest entries removed when max_entries reached)
    - Determinism guard (only caches low-temperature calls by default)
    - Cache key includes all prompt parameters for correctness
    - Stats tracking (hits, misses, evictions)
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
        max_cacheable_temperature: float = 0.3,
    ):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._max_temp = max_cacheable_temperature

        # LRU ordered dict: newest at end
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stats
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._stores: int = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    # --- Key Generation ---

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Create a deterministic cache key from call parameters.

        Uses SHA-256 hash to keep keys short and fixed-length.
        """
        raw = json.dumps({
            "provider": provider,
            "model": model,
            "system": system_prompt,
            "user": user_prompt,
            "temp": temperature,
            "max_tokens": max_tokens,
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    # --- Core Operations ---

    def get(self, key: str) -> Optional[Any]:
        """
        Look up a cached response.

        Returns the cached LLMResponse if found and not expired,
        otherwise None. Automatically evicts expired entries.
        """
        entry = self._entries.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            # Expired — evict
            del self._entries[key]
            self._misses += 1
            self._evictions += 1
            return None

        # Cache hit — move to end (most recently used)
        self._entries.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1

        logger.debug(
            "cache_hit",
            extra={
                "key": key[:16],
                "provider": entry.provider,
                "model": entry.model,
                "hit_count": entry.hit_count,
                "age_seconds": round(entry.age_seconds, 1),
            },
        )

        return entry.response

    def put(
        self,
        key: str,
        response: Any,
        *,
        provider: str = "",
        model: str = "",
        intent: str = "",
    ) -> None:
        """
        Store a response in the cache.

        Evicts the oldest entry if max_entries is reached.
        """
        now = time.monotonic()

        # Remove existing entry for this key (to refresh TTL)
        if key in self._entries:
            del self._entries[key]

        # Evict oldest if at capacity
        while len(self._entries) >= self._max_entries:
            evicted_key, _ = self._entries.popitem(last=False)
            self._evictions += 1
            logger.debug(
                "cache_eviction",
                extra={"key": evicted_key[:16]},
            )

        self._entries[key] = CacheEntry(
            key=key,
            response=response,
            created_at=now,
            expires_at=now + self._ttl,
            provider=provider,
            model=model,
            intent=intent,
        )
        self._stores += 1

    def should_cache(self, temperature: float) -> bool:
        """Check if a call with this temperature should be cached."""
        return temperature <= self._max_temp

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry from cache. Returns True if found."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def invalidate_by_intent(self, intent: str) -> int:
        """Remove all cached entries for a specific intent."""
        keys_to_remove = [
            k for k, v in self._entries.items()
            if v.intent == intent
        ]
        for key in keys_to_remove:
            del self._entries[key]
        return len(keys_to_remove)

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries cleared."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns number removed."""
        now = time.monotonic()
        expired_keys = [
            k for k, v in self._entries.items()
            if now > v.expires_at
        ]
        for key in expired_keys:
            del self._entries[key]
            self._evictions += 1
        return len(expired_keys)

    # --- Stats ---

    def get_stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        return {
            "size": self.size,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "evictions": self._evictions,
            "stores": self._stores,
            "max_cacheable_temperature": self._max_temp,
        }

    def reset_stats(self) -> None:
        """Reset performance counters without clearing cached data."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._stores = 0

    # --- Inspection ---

    def list_entries(self) -> list[dict[str, Any]]:
        """Return metadata for all cached entries (for debugging)."""
        return [
            {
                "key": entry.key[:16] + "...",
                "provider": entry.provider,
                "model": entry.model,
                "intent": entry.intent,
                "age_seconds": round(entry.age_seconds, 1),
                "hit_count": entry.hit_count,
                "is_expired": entry.is_expired,
            }
            for entry in self._entries.values()
        ]


# ---------------------------------------------------------------------------
# Convenience: cached_route
# ---------------------------------------------------------------------------

async def cached_route(
    cache: ResponseCache,
    router: Any,  # ModelRouter
    intent: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Route an LLM call through the cache.

    Checks cache first, calls router on miss, stores result.
    Respects temperature guard: high-temperature calls skip cache.

    Args:
        cache: ResponseCache instance
        router: ModelRouter instance
        intent: LLM intent for routing
        system_prompt: System instructions
        user_prompt: User prompt
        temperature: Override temperature (also affects caching)
        max_tokens: Override max_tokens

    Returns:
        LLMResponse (from cache or fresh from router)
    """
    # Resolve effective temperature
    route = router.config.get_route(intent)
    effective_temp = temperature if temperature is not None else route.primary.temperature
    effective_tokens = max_tokens if max_tokens is not None else route.primary.max_tokens

    # Check if this call should be cached
    if cache.should_cache(effective_temp):
        key = ResponseCache.make_key(
            provider=route.primary.provider,
            model=route.primary.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=effective_temp,
            max_tokens=effective_tokens,
        )

        # Try cache
        cached = cache.get(key)
        if cached is not None:
            return cached

        # Cache miss — call router
        response = await router.route(
            intent=intent,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Store in cache
        cache.put(
            key,
            response,
            provider=response.provider,
            model=response.model,
            intent=intent,
        )
        return response

    # Temperature too high — skip cache entirely
    return await router.route(
        intent=intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
