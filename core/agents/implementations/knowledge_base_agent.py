"""
Knowledge Base Agent — The Librarian.

Identifies FAQ and documentation gaps by analyzing resolved support
tickets and Hive Mind insights, clusters topics into themes, and
generates draft KB articles to fill those gaps. Continuously improves
the self-service knowledge base so common questions stop reaching
human support. Works across all verticals.

Architecture (LangGraph State Machine):
    scan_sources → analyze_gaps → generate_articles →
    human_review → report → END

Trigger Events:
    - scheduled: Weekly KB gap analysis
    - event: support_ticket_resolved (checks if cluster threshold met)
    - manual: On-demand article generation

Shared Brain Integration:
    - Reads: resolved support tickets, Hive Mind shared insights
    - Writes: knowledge base quality metrics, coverage gaps, article metadata

Safety:
    - All generated articles require human review before publishing
    - Ticket data is anonymized before LLM processing
    - Content is scoped to the vertical

Usage:
    agent = KnowledgeBaseAgent(config, db, embedder, llm)
    result = await agent.run({
        "lookback_days": 30,
        "min_cluster_size": 3,
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import KnowledgeBaseAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

ARTICLE_CATEGORIES = [
    "getting_started",
    "troubleshooting",
    "billing",
    "integration",
    "security",
    "general",
    "best_practices",
]

MIN_TICKET_CLUSTER_SIZE = 3

GAP_ANALYSIS_PROMPT = """\
You are a knowledge base curator analyzing resolved support tickets \
to identify documentation gaps.

Resolved Tickets (last {lookback_days} days):
{tickets_json}

Existing Knowledge Base Categories: {categories}

Hive Mind Insights:
{hive_insights_json}

Analyze the tickets and identify:
1. Recurring topics/questions that appear {min_cluster_size}+ times
2. Topics NOT covered by existing KB articles
3. Topics where existing articles may be outdated or insufficient

Return a JSON array of topic clusters:
[
    {{
        "topic": "Clear topic name",
        "category": "one of: {categories}",
        "ticket_count": 0,
        "sample_questions": ["question 1", "question 2"],
        "gap_description": "What's missing from the KB",
        "priority": "high|medium|low",
        "suggested_article_title": "Suggested KB article title"
    }}
]

Focus on actionable gaps that would reduce repeat support tickets. \
Return ONLY the JSON array, no markdown code fences.
"""

ARTICLE_GENERATION_PROMPT = """\
You are a technical writer creating a knowledge base article \
to address a recurring support topic.

Topic: {topic}
Category: {category}
Gap Description: {gap_description}

Sample Customer Questions:
{sample_questions_json}

Relevant Resolved Ticket Summaries:
{ticket_summaries_json}

Write a comprehensive, helpful KB article that:
1. Has a clear, search-friendly title
2. Starts with a brief problem statement
3. Provides step-by-step instructions where applicable
4. Includes troubleshooting tips
5. Ends with related resources or next steps

Return as a JSON object:
{{
    "title": "Article title",
    "category": "{category}",
    "summary": "One-sentence summary for search results",
    "content": "Full article content in markdown format",
    "tags": ["tag1", "tag2"],
    "related_topics": ["related_topic_1"]
}}

Write for a non-technical audience unless the category is \
'integration' or 'security'. Return ONLY the JSON, no markdown fences.
"""


@register_agent_type("knowledge_base")
class KnowledgeBaseAgent(BaseAgent):
    """
    Knowledge base gap analysis and article generation agent.

    Nodes:
        1. scan_sources      -- Query resolved support_tickets + Hive Mind insights
        2. analyze_gaps       -- LLM clusters topics, identifies KB gaps
        3. generate_articles  -- LLM drafts FAQ articles from ticket clusters
        4. human_review       -- Gate: approve articles before publishing
        5. report             -- Save to knowledge_articles table + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Knowledge Base Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(KnowledgeBaseAgentState)

        workflow.add_node("scan_sources", self._node_scan_sources)
        workflow.add_node("analyze_gaps", self._node_analyze_gaps)
        workflow.add_node("generate_articles", self._node_generate_articles)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_sources")

        workflow.add_edge("scan_sources", "analyze_gaps")
        workflow.add_edge("analyze_gaps", "generate_articles")
        workflow.add_edge("generate_articles", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "report",
                "rejected": "report",
            },
        )
        workflow.add_edge("report", END)

        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        return self.mcp_tools or []

    @classmethod
    def get_state_class(cls) -> Type[KnowledgeBaseAgentState]:
        return KnowledgeBaseAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "resolved_tickets": [],
            "hive_insights": [],
            "total_sources_scanned": 0,
            "topic_clusters": [],
            "identified_gaps": [],
            "total_gaps": 0,
            "draft_articles": [],
            "articles_generated": 0,
            "articles_saved": 0,
            "articles_published": 0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Sources ────────────────────────────────────────

    async def _node_scan_sources(
        self, state: KnowledgeBaseAgentState
    ) -> dict[str, Any]:
        """Node 1: Query resolved support tickets and Hive Mind insights."""
        task = state.get("task_input", {})

        logger.info(
            "knowledge_base_scan_started",
            extra={"agent_id": self.agent_id},
        )

        lookback_days = task.get("lookback_days", self.config.params.get("lookback_days", 30))
        cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).isoformat()

        resolved_tickets: list[dict[str, Any]] = []
        hive_insights: list[dict[str, Any]] = []
        total_scanned = 0

        # Query resolved support tickets
        try:
            result = (
                self.db.client.table("support_tickets")
                .select("id, subject, category, resolution_summary, created_at")
                .eq("vertical_id", self.vertical_id)
                .eq("status", "resolved")
                .gte("created_at", cutoff_date)
                .order("created_at", desc=True)
                .limit(100)
                .execute()
            )
            if result.data:
                for ticket in result.data:
                    resolved_tickets.append({
                        "ticket_id": ticket.get("id", ""),
                        "subject": ticket.get("subject", ""),
                        "category": ticket.get("category", "general"),
                        "resolution_summary": ticket.get("resolution_summary", ""),
                        "created_at": ticket.get("created_at", ""),
                    })
                total_scanned += len(result.data)
                logger.info(
                    "knowledge_base_tickets_loaded",
                    extra={"count": len(resolved_tickets)},
                )
        except Exception as e:
            logger.warning(
                "knowledge_base_tickets_error",
                extra={"error": str(e)[:200]},
            )

        # Query Hive Mind for relevant shared insights
        try:
            insights = self.consult_hive(
                "What are the most common customer questions and support patterns?",
                min_confidence=0.6,
                limit=10,
            )
            for insight in insights:
                hive_insights.append({
                    "topic": insight.get("topic", ""),
                    "content": insight.get("content", ""),
                    "confidence": insight.get("confidence", 0),
                    "source_agent": insight.get("source_agent", ""),
                })
            total_scanned += len(insights)
            logger.info(
                "knowledge_base_hive_insights_loaded",
                extra={"count": len(hive_insights)},
            )
        except Exception as e:
            logger.warning(
                "knowledge_base_hive_error",
                extra={"error": str(e)[:200]},
            )

        # Add any task-provided sources
        task_tickets = task.get("resolved_tickets", [])
        resolved_tickets.extend(task_tickets)
        total_scanned += len(task_tickets)

        logger.info(
            "knowledge_base_scan_complete",
            extra={
                "tickets": len(resolved_tickets),
                "hive_insights": len(hive_insights),
                "total_scanned": total_scanned,
            },
        )

        return {
            "current_node": "scan_sources",
            "resolved_tickets": resolved_tickets,
            "hive_insights": hive_insights,
            "total_sources_scanned": total_scanned,
        }

    # ─── Node 2: Analyze Gaps ────────────────────────────────────────

    async def _node_analyze_gaps(
        self, state: KnowledgeBaseAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM clusters topics and identifies KB gaps."""
        resolved_tickets = state.get("resolved_tickets", [])
        hive_insights = state.get("hive_insights", [])
        task = state.get("task_input", {})

        min_cluster_size = task.get(
            "min_cluster_size",
            self.config.params.get("min_cluster_size", MIN_TICKET_CLUSTER_SIZE),
        )
        lookback_days = task.get("lookback_days", 30)

        logger.info(
            "knowledge_base_gap_analysis_started",
            extra={
                "tickets": len(resolved_tickets),
                "min_cluster": min_cluster_size,
            },
        )

        topic_clusters: list[dict[str, Any]] = []
        identified_gaps: list[dict[str, Any]] = []

        if not resolved_tickets and not hive_insights:
            logger.info("knowledge_base_no_sources_to_analyze")
            return {
                "current_node": "analyze_gaps",
                "topic_clusters": [],
                "identified_gaps": [],
                "total_gaps": 0,
            }

        try:
            # Anonymize ticket data before sending to LLM
            anonymized_tickets = []
            for ticket in resolved_tickets[:50]:
                anonymized_tickets.append({
                    "subject": ticket.get("subject", "")[:200],
                    "category": ticket.get("category", "general"),
                    "resolution_summary": ticket.get("resolution_summary", "")[:300],
                })

            anonymized_insights = []
            for insight in hive_insights[:10]:
                anonymized_insights.append({
                    "topic": insight.get("topic", ""),
                    "content": insight.get("content", "")[:300],
                })

            categories_str = ", ".join(ARTICLE_CATEGORIES)

            prompt = GAP_ANALYSIS_PROMPT.format(
                lookback_days=lookback_days,
                tickets_json=json.dumps(anonymized_tickets, indent=2),
                categories=categories_str,
                hive_insights_json=json.dumps(anonymized_insights, indent=2),
                min_cluster_size=min_cluster_size,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a knowledge base curator and content strategist.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                clusters_data = json.loads(llm_text)
                if isinstance(clusters_data, list):
                    topic_clusters = clusters_data

                    # Filter by minimum cluster size
                    identified_gaps = [
                        c for c in topic_clusters
                        if c.get("ticket_count", 0) >= min_cluster_size
                        or c.get("priority") == "high"
                    ]
            except (json.JSONDecodeError, KeyError):
                logger.warning("knowledge_base_gap_parse_error")

        except Exception as e:
            logger.error(
                "knowledge_base_gap_llm_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "knowledge_base_gap_analysis_complete",
            extra={
                "clusters": len(topic_clusters),
                "gaps": len(identified_gaps),
            },
        )

        return {
            "current_node": "analyze_gaps",
            "topic_clusters": topic_clusters,
            "identified_gaps": identified_gaps,
            "total_gaps": len(identified_gaps),
        }

    # ─── Node 3: Generate Articles ───────────────────────────────────

    async def _node_generate_articles(
        self, state: KnowledgeBaseAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM drafts FAQ articles from identified gaps."""
        identified_gaps = state.get("identified_gaps", [])
        resolved_tickets = state.get("resolved_tickets", [])

        max_articles = self.config.params.get("max_articles_per_run", 5)

        logger.info(
            "knowledge_base_article_generation_started",
            extra={
                "gaps": len(identified_gaps),
                "max_articles": max_articles,
            },
        )

        draft_articles: list[dict[str, Any]] = []

        # Sort gaps by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_gaps = sorted(
            identified_gaps,
            key=lambda g: priority_order.get(g.get("priority", "low"), 2),
        )

        for gap in sorted_gaps[:max_articles]:
            try:
                topic = gap.get("topic", "")
                category = gap.get("category", "general")
                gap_description = gap.get("gap_description", "")
                sample_questions = gap.get("sample_questions", [])

                # Gather relevant ticket summaries for this topic
                relevant_summaries = []
                for ticket in resolved_tickets:
                    subject_lower = ticket.get("subject", "").lower()
                    if any(
                        word.lower() in subject_lower
                        for word in topic.split()
                        if len(word) > 3
                    ):
                        relevant_summaries.append({
                            "subject": ticket.get("subject", "")[:200],
                            "resolution": ticket.get("resolution_summary", "")[:300],
                        })
                        if len(relevant_summaries) >= 5:
                            break

                prompt = ARTICLE_GENERATION_PROMPT.format(
                    topic=topic,
                    category=category,
                    gap_description=gap_description,
                    sample_questions_json=json.dumps(sample_questions[:5], indent=2),
                    ticket_summaries_json=json.dumps(relevant_summaries[:5], indent=2),
                )

                llm_response = self.llm.messages.create(
                    model="claude-haiku-4-5-20250514",
                    system="You are a technical writer creating clear, helpful KB articles.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )

                llm_text = llm_response.content[0].text.strip()
                article_data: dict[str, Any] = {}

                try:
                    article_data = json.loads(llm_text)
                except (json.JSONDecodeError, KeyError):
                    logger.debug("knowledge_base_article_parse_error")
                    article_data = {
                        "title": gap.get("suggested_article_title", topic),
                        "category": category,
                        "summary": gap_description[:200],
                        "content": llm_text[:3000],
                        "tags": [],
                    }

                draft_articles.append({
                    "title": article_data.get("title", topic),
                    "category": article_data.get("category", category),
                    "summary": article_data.get("summary", ""),
                    "content": article_data.get("content", ""),
                    "tags": article_data.get("tags", []),
                    "related_topics": article_data.get("related_topics", []),
                    "source_gap": topic,
                    "source_ticket_count": gap.get("ticket_count", 0),
                    "priority": gap.get("priority", "medium"),
                })

                logger.info(
                    "knowledge_base_article_drafted",
                    extra={
                        "title": article_data.get("title", "")[:60],
                        "category": category,
                    },
                )

            except Exception as e:
                logger.warning(
                    "knowledge_base_article_error",
                    extra={
                        "topic": gap.get("topic", ""),
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "knowledge_base_articles_generated",
            extra={"articles": len(draft_articles)},
        )

        return {
            "current_node": "generate_articles",
            "draft_articles": draft_articles,
            "articles_generated": len(draft_articles),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: KnowledgeBaseAgentState
    ) -> dict[str, Any]:
        """Node 4: Present draft articles for human approval."""
        articles = state.get("draft_articles", [])
        gaps = state.get("total_gaps", 0)

        logger.info(
            "knowledge_base_human_review_pending",
            extra={
                "articles": len(articles),
                "gaps": gaps,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: KnowledgeBaseAgentState
    ) -> dict[str, Any]:
        """Node 5: Save articles to knowledge_articles table and generate insights."""
        now = datetime.now(timezone.utc).isoformat()
        draft_articles = state.get("draft_articles", [])
        topic_clusters = state.get("topic_clusters", [])
        identified_gaps = state.get("identified_gaps", [])
        total_sources = state.get("total_sources_scanned", 0)
        approval_status = state.get("human_approval_status", "approved")

        articles_saved = 0
        articles_published = 0

        # Save approved articles to database
        if approval_status == "approved":
            for article in draft_articles:
                try:
                    record = {
                        "vertical_id": self.vertical_id,
                        "agent_id": self.agent_id,
                        "title": article.get("title", ""),
                        "category": article.get("category", "general"),
                        "summary": article.get("summary", "")[:500],
                        "content": article.get("content", "")[:10000],
                        "tags": json.dumps(article.get("tags", [])),
                        "related_topics": json.dumps(article.get("related_topics", [])),
                        "source_gap": article.get("source_gap", ""),
                        "source_ticket_count": article.get("source_ticket_count", 0),
                        "status": "draft",
                        "created_at": now,
                    }
                    self.db.client.table("knowledge_articles").insert(record).execute()
                    articles_saved += 1
                except Exception as e:
                    logger.warning(
                        "knowledge_base_save_error",
                        extra={
                            "title": article.get("title", "")[:60],
                            "error": str(e)[:200],
                        },
                    )

        # Build category distribution
        category_counts: dict[str, int] = {}
        for article in draft_articles:
            cat = article.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Build report
        sections = [
            "# Knowledge Base Report",
            f"*Generated: {now}*\n",
            f"## Summary",
            f"- **Sources Scanned:** {total_sources}",
            f"- **Topic Clusters Found:** {len(topic_clusters)}",
            f"- **Gaps Identified:** {len(identified_gaps)}",
            f"- **Articles Generated:** {len(draft_articles)}",
            f"- **Articles Saved:** {articles_saved}",
            f"- **Approval Status:** {approval_status}",
        ]

        if category_counts:
            sections.append("\n## Articles by Category")
            for cat, count in sorted(category_counts.items()):
                sections.append(f"- **{cat.replace('_', ' ').title()}:** {count}")

        if identified_gaps:
            sections.append("\n## Identified Gaps")
            for i, gap in enumerate(identified_gaps[:10], 1):
                sections.append(
                    f"{i}. **{gap.get('topic', 'Unknown')}** "
                    f"({gap.get('priority', 'medium').upper()}) — "
                    f"{gap.get('gap_description', 'N/A')[:80]}"
                )

        if draft_articles:
            sections.append("\n## Draft Articles")
            for i, art in enumerate(draft_articles[:10], 1):
                sections.append(
                    f"{i}. **{art.get('title', 'Untitled')}** "
                    f"[{art.get('category', 'general')}] — "
                    f"Based on {art.get('source_ticket_count', 0)} tickets"
                )

        report = "\n".join(sections)

        # Store insight
        if identified_gaps:
            self.store_insight(InsightData(
                insight_type="knowledge_base_gaps",
                title=f"KB Gaps: {len(identified_gaps)} gaps, {len(draft_articles)} articles drafted",
                content=(
                    f"Scanned {total_sources} sources and identified "
                    f"{len(identified_gaps)} knowledge base gaps across "
                    f"{len(topic_clusters)} topic clusters. "
                    f"Generated {len(draft_articles)} draft articles. "
                    f"Categories: {json.dumps(category_counts)}."
                ),
                confidence=0.75,
                metadata={
                    "total_sources": total_sources,
                    "clusters": len(topic_clusters),
                    "gaps": len(identified_gaps),
                    "articles_generated": len(draft_articles),
                    "articles_saved": articles_saved,
                    "category_distribution": category_counts,
                },
            ))

        logger.info(
            "knowledge_base_report_generated",
            extra={
                "gaps": len(identified_gaps),
                "articles": len(draft_articles),
                "saved": articles_saved,
            },
        )

        return {
            "current_node": "report",
            "articles_saved": articles_saved,
            "articles_published": articles_published,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: KnowledgeBaseAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<KnowledgeBaseAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
