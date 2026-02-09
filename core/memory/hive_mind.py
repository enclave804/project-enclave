"""
The Hive Mind — Cross-Agent Shared Intelligence.

The Synapse of the Sovereign Venture Engine. Enables real-time knowledge
transfer between agents. When the OutreachAgent learns that "Subject Line A"
works best for Fintech CEOs, the SocialAgent, AdsAgent, and ProposalAgent
all benefit immediately.

Architecture:
    - Built ON TOP of existing shared_insights table (pgvector)
    - Adds topic-based routing, insight aggregation, and usage tracking
    - Provides a high-level API that agents use instead of raw DB calls
    - Maintains a cross-reference graph: which agents produced/consumed which insights

Usage:
    hive = HiveMind(db=db, embedder=embedder)

    # Publish a learning
    hive.publish(
        source_agent="outreach",
        topic="email_subject_lines",
        content="Short subject lines (<50 chars) get 2x open rate for Fintech CEOs",
        confidence=0.85,
        evidence={"sample_size": 150, "metric": "open_rate", "improvement": "2x"},
    )

    # Query collective wisdom
    insights = hive.query(
        topic="email_subject_lines",
        consumer_agent="social",
        min_confidence=0.7,
        limit=5,
    )

    # Get cross-agent summary
    digest = hive.get_digest(vertical_id="enclave_guard")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Insight Categories ────────────────────────────────────────

INSIGHT_CATEGORIES = {
    # Outreach learnings
    "email_performance": "What email strategies work (subject lines, timing, tone)",
    "audience_response": "How different audiences react (industry, title, company size)",
    "objection_handling": "Successful rebuttals and objection patterns",

    # Content & marketing learnings
    "content_performance": "What content topics/formats perform best",
    "keyword_trends": "Search terms and trending topics",
    "social_engagement": "What social content gets traction",
    "ad_performance": "Which ad campaigns/creatives convert",

    # Sales learnings
    "deal_patterns": "What deal structures close (pricing, scope, timeline)",
    "prospect_signals": "Buying signals and disqualification patterns",
    "meeting_insights": "What discovery call approaches work",

    # Operations learnings
    "client_health": "Client satisfaction and churn signals",
    "payment_patterns": "Invoice payment behavior and timing",
    "financial_metrics": "Revenue trends and cost patterns",

    # Meta learnings
    "market_signal": "External market trends and opportunities",
    "competitive_intel": "Competitor positioning and activity",
}


# ── The Hive Mind ─────────────────────────────────────────────

class HiveMind:
    """
    Cross-agent shared intelligence layer.

    Sits on top of the existing shared_insights table (pgvector)
    and adds topic routing, usage tracking, insight aggregation,
    and a high-level query API for agents.
    """

    def __init__(
        self,
        db: Any,
        embedder: Any,
        vertical_id: str = "enclave_guard",
    ):
        self.db = db
        self.embedder = embedder
        self.vertical_id = vertical_id

        # In-memory cache for hot insights (refreshed on query)
        self._cache: dict[str, list[dict]] = {}
        self._cache_ts: dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Track cross-agent knowledge flow
        self._flow_log: list[dict] = []

    # ── Publish ───────────────────────────────────────────────

    def publish(
        self,
        source_agent: str,
        topic: str,
        content: str,
        confidence: float = 0.7,
        title: str = "",
        evidence: Optional[dict[str, Any]] = None,
        related_entity_id: Optional[str] = None,
        related_entity_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """
        Publish an insight to the Hive Mind.

        Unlike raw store_insight(), this method:
        1. Enriches metadata with evidence and tags
        2. Generates embeddings for semantic search
        3. Validates topic against known categories
        4. Logs the knowledge flow

        Args:
            source_agent: Agent ID that produced this insight.
            topic: Topic category (e.g., "email_performance").
            content: The actual learning (plain English).
            confidence: How confident (0.0-1.0).
            title: Short summary title.
            evidence: Supporting data (sample_size, metrics, etc.).
            related_entity_id: Optional link to a specific entity.
            related_entity_type: Type of entity (lead, campaign, etc.).
            tags: Additional classification tags.

        Returns:
            Stored insight record, or None on failure.
        """
        if confidence < 0.0 or confidence > 1.0:
            logger.warning(f"Confidence {confidence} out of range, clamping to [0, 1]")
            confidence = max(0.0, min(1.0, confidence))

        # Build enriched metadata
        metadata = {
            "topic": topic,
            "evidence": evidence or {},
            "tags": tags or [],
            "published_at": datetime.now(timezone.utc).isoformat(),
        }

        # Generate embedding for semantic search
        embedding = None
        try:
            search_text = f"{title} {content}" if title else content
            embedding = self.embedder.embed_query(search_text)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")

        # Map topic to insight_type
        insight_type = topic if topic in INSIGHT_CATEGORIES else "market_signal"

        try:
            result = self.db.store_insight(
                source_agent_id=source_agent,
                insight_type=insight_type,
                title=title or content[:100],
                content=content,
                confidence_score=confidence,
                related_entity_id=related_entity_id,
                related_entity_type=related_entity_type,
                metadata=metadata,
                vertical_id=self.vertical_id,
                embedding=embedding,
            )

            # Invalidate cache entries related to this topic
            stale_keys = [k for k in self._cache if k.startswith(f"{topic}:")]
            for k in stale_keys:
                self._cache.pop(k, None)

            # Log the knowledge flow
            self._flow_log.append({
                "direction": "publish",
                "agent": source_agent,
                "topic": topic,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            logger.info(
                "hive_mind_publish",
                extra={
                    "source_agent": source_agent,
                    "topic": topic,
                    "confidence": confidence,
                    "has_evidence": bool(evidence),
                },
            )

            return result

        except Exception as e:
            logger.error(f"HiveMind publish failed: {e}")
            return None

    # ── Query ─────────────────────────────────────────────────

    def query(
        self,
        topic: str,
        consumer_agent: str = "",
        min_confidence: float = 0.7,
        limit: int = 5,
        max_age_days: Optional[int] = None,
        source_agents: Optional[list[str]] = None,
        exclude_own: bool = False,
    ) -> list[dict]:
        """
        Query collective wisdom from the Hive Mind.

        Uses semantic search over shared_insights with topic filtering.
        Automatically tracks usage for knowledge flow analysis.

        Args:
            topic: What you want to know about (natural language or category).
            consumer_agent: Agent requesting the insights.
            min_confidence: Minimum confidence threshold.
            limit: Maximum insights to return.
            max_age_days: Only return insights newer than N days.
            source_agents: Only from these specific agents.
            exclude_own: Exclude insights from the consumer agent itself.

        Returns:
            List of insight dicts sorted by relevance.
        """
        # Check cache first
        cache_key = f"{topic}:{min_confidence}:{limit}"
        if cache_key in self._cache:
            cache_age = datetime.now(timezone.utc) - self._cache_ts.get(
                cache_key, datetime.min.replace(tzinfo=timezone.utc)
            )
            if cache_age < self._cache_ttl:
                insights = self._cache[cache_key]
                self._log_consumption(consumer_agent, topic, len(insights))
                return insights

        # Generate query embedding
        try:
            query_embedding = self.embedder.embed_query(topic)
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # Determine source filter
        source_filter = None
        if source_agents and len(source_agents) == 1:
            source_filter = source_agents[0]

        try:
            results = self.db.search_insights(
                query_embedding=query_embedding,
                insight_type=None,  # Topic routing via semantic similarity
                source_agent_id=source_filter,
                limit=limit * 2,  # Over-fetch for post-filtering
                similarity_threshold=min_confidence * 0.5,  # Looser similarity, filter by confidence
            )
        except Exception as e:
            logger.warning(f"HiveMind query failed: {e}")
            return []

        # Post-filter
        filtered = []
        for insight in results:
            # Confidence filter
            if insight.get("confidence_score", 0) < min_confidence:
                continue

            # Exclude own insights
            if exclude_own and consumer_agent and insight.get("source_agent_id") == consumer_agent:
                continue

            # Source agent filter (multi-agent)
            if source_agents and len(source_agents) > 1:
                if insight.get("source_agent_id") not in source_agents:
                    continue

            filtered.append(insight)

        # Trim to limit
        filtered = filtered[:limit]

        # Update cache
        self._cache[cache_key] = filtered
        self._cache_ts[cache_key] = datetime.now(timezone.utc)

        # Track consumption
        self._log_consumption(consumer_agent, topic, len(filtered))

        # Increment usage counts (best-effort)
        self._increment_usage(filtered)

        return filtered

    def query_by_text(
        self,
        question: str,
        consumer_agent: str = "",
        min_confidence: float = 0.5,
        limit: int = 5,
    ) -> list[dict]:
        """
        Natural language query — ask the Hive Mind a question.

        Example:
            hive.query_by_text("What pricing model works for startups?")
        """
        return self.query(
            topic=question,
            consumer_agent=consumer_agent,
            min_confidence=min_confidence,
            limit=limit,
        )

    # ── Aggregation ───────────────────────────────────────────

    def get_digest(
        self,
        days: int = 7,
        min_confidence: float = 0.7,
        limit_per_topic: int = 3,
    ) -> dict[str, Any]:
        """
        Get a weekly digest of top insights across all agents.

        Returns a structured summary organized by topic category.
        """
        digest: dict[str, list[dict]] = defaultdict(list)
        topics_covered = set()

        # Query each known topic category
        for topic in INSIGHT_CATEGORIES:
            try:
                insights = self.query(
                    topic=topic,
                    min_confidence=min_confidence,
                    limit=limit_per_topic,
                )
                if insights:
                    digest[topic] = insights
                    topics_covered.add(topic)
            except Exception:
                continue

        # Compute knowledge flow stats
        agent_contributions: dict[str, int] = defaultdict(int)
        for topic_insights in digest.values():
            for insight in topic_insights:
                agent = insight.get("source_agent_id", "unknown")
                agent_contributions[agent] += 1

        return {
            "period_days": days,
            "topics_covered": len(topics_covered),
            "total_insights": sum(len(v) for v in digest.values()),
            "insights_by_topic": dict(digest),
            "agent_contributions": dict(agent_contributions),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_knowledge_flow(self) -> list[dict]:
        """
        Get the knowledge flow log — who published/consumed what.

        Useful for visualizing the cross-agent intelligence network.
        """
        return list(self._flow_log)

    def get_agent_connections(self) -> dict[str, Any]:
        """
        Analyze cross-agent knowledge connections.

        Returns a graph of which agents produce/consume from each other.
        """
        publishers: dict[str, int] = defaultdict(int)
        consumers: dict[str, int] = defaultdict(int)
        topics_by_agent: dict[str, set] = defaultdict(set)

        for entry in self._flow_log:
            agent = entry["agent"]
            topic = entry["topic"]
            if entry["direction"] == "publish":
                publishers[agent] += 1
            else:
                consumers[agent] += 1
            topics_by_agent[agent].add(topic)

        return {
            "publishers": dict(publishers),
            "consumers": dict(consumers),
            "topics_by_agent": {k: list(v) for k, v in topics_by_agent.items()},
            "total_flows": len(self._flow_log),
        }

    # ── Reinforcement ─────────────────────────────────────────

    def boost_insight(self, insight_id: str, amount: float = 0.05) -> None:
        """
        Boost an insight's confidence when it proves useful.

        Called when an agent uses an insight and gets a positive outcome.
        """
        try:
            self.db.client.rpc(
                "boost_insight_confidence",
                {"p_insight_id": str(insight_id), "p_boost_amount": amount},
            ).execute()
        except Exception as e:
            # Silently fail — this is enhancement, not critical
            logger.debug(f"Insight boost failed (non-critical): {e}")

    def decay_insight(self, insight_id: str, amount: float = 0.02) -> None:
        """
        Decay an insight's confidence when it proves wrong.

        Called when an agent uses an insight and gets a negative outcome.
        """
        try:
            self.db.client.rpc(
                "decay_insight_confidence",
                {"p_insight_id": str(insight_id), "p_decay_amount": amount},
            ).execute()
        except Exception as e:
            logger.debug(f"Insight decay failed (non-critical): {e}")

    # ── Internal Helpers ──────────────────────────────────────

    def _log_consumption(self, consumer: str, topic: str, count: int) -> None:
        """Log when an agent consumes insights."""
        if consumer:
            self._flow_log.append({
                "direction": "consume",
                "agent": consumer,
                "topic": topic,
                "count": count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def _increment_usage(self, insights: list[dict]) -> None:
        """Increment usage_count for consumed insights (best-effort)."""
        for insight in insights:
            iid = insight.get("id")
            if iid:
                try:
                    self.db.client.table("shared_insights").update(
                        {"usage_count": insight.get("usage_count", 0) + 1}
                    ).eq("id", str(iid)).execute()
                except Exception:
                    pass  # Non-critical

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._cache_ts.clear()

    def __repr__(self) -> str:
        return (
            f"HiveMind(vertical_id={self.vertical_id!r}, "
            f"cache_size={len(self._cache)}, "
            f"flow_entries={len(self._flow_log)})"
        )
