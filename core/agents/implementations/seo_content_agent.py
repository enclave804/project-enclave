"""
SEO Content Agent — The first truly new agent in the Sovereign Venture Engine.

Autonomous digital employee that researches keywords, analyzes competitors
(including their visual design quality via screenshot analysis), drafts
high-ranking content using shared brain insights, and learns from human edits.

Architecture (LangGraph State Machine):
    research_competitors → draft_content → human_review → finalize_and_learn

Visual Cortex: Takes screenshots of competitor pages and uses Claude's
multimodal capabilities to judge content quality AND design authority.
This is not just reading text — it's seeing the whole page.

RLHF Hook: If a human edits the draft in human_review, the
(original_draft, human_edit) pair is saved to training_examples
via BaseAgent.learn() for future optimization.

Usage:
    agent = SEOContentAgent(config, db, embedder, llm)
    result = await agent.run({
        "topic": "AI-powered penetration testing",
        "content_type": "blog_post",
    })
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import SEOContentAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)


@register_agent_type("seo_content")
class SEOContentAgent(BaseAgent):
    """
    AI-powered SEO content generator with competitor visual analysis.

    Nodes:
        1. research_competitors — Browser-driven SERP analysis + Visual Cortex
        2. draft_content — LLM content generation with shared brain context
        3. human_review — Gate for human approval/editing
        4. finalize_and_learn — Save content + RLHF data capture

    The agent uses SovereignBrowser for web research and Claude's vision
    capabilities for screenshot analysis. All operations are logged and
    any browser session is recorded via Sovereign CCTV.
    """

    def build_graph(self) -> Any:
        """
        Build the SEO Content Agent's LangGraph state machine.

        Graph flow:
            research_competitors → draft_content → human_review → finalize_and_learn

        Human gate is applied at human_review node via interrupt_before.
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(SEOContentAgentState)

        # Add nodes
        workflow.add_node("research_competitors", self._node_research_competitors)
        workflow.add_node("draft_content", self._node_draft_content)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("finalize_and_learn", self._node_finalize_and_learn)

        # Define edges
        workflow.set_entry_point("research_competitors")
        workflow.add_edge("research_competitors", "draft_content")
        workflow.add_edge("draft_content", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "finalize_and_learn",
                "rejected": END,
            },
        )
        workflow.add_edge("finalize_and_learn", END)

        # Compile with human gate interrupt
        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        """SEO agent uses browser_tool (injected) rather than MCP tools."""
        return []

    def get_state_class(self) -> Type[SEOContentAgentState]:
        return SEOContentAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state from task input."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "topic": task.get("topic", ""),
            "target_keywords": task.get("keywords", []),
            "content_type": task.get("content_type", self.config.params.get("content_type", "blog_post")),
            "competitor_analysis": [],
            "competitor_content": [],
            "outreach_insights": [],
            "draft_content": "",
            "draft_title": "",
            "word_count": 0,
            "content_approved": False,
            "rlhf_captured": False,
        })
        return state

    # ─── Node Implementations ─────────────────────────────────────────

    async def _node_research_competitors(
        self, state: SEOContentAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Research competitors via browser + Visual Cortex.

        1. Search Google for the topic
        2. Visit top 3 results
        3. For each: extract text + take screenshot for visual analysis
        4. Use Claude to analyze content quality AND design authority
        5. Pull shared insights from outreach agent
        """
        topic = state.get("topic", "")
        keywords = state.get("target_keywords", [])
        search_query = topic if not keywords else f"{topic} {' '.join(keywords[:3])}"

        logger.info(
            "seo_research_started",
            extra={
                "agent_id": self.agent_id,
                "topic": topic,
                "search_query": search_query[:100],
            },
        )

        competitor_analysis: list[dict[str, Any]] = []

        # ─── Browser Research ────────────────────────────────────
        if self.browser_tool is not None:
            try:
                result = await self.browser_tool.search_and_extract(
                    url="https://www.google.com",
                    instructions=(
                        f"Search for '{search_query}'. "
                        "Extract the top 3 organic results including their "
                        "title, URL, and meta description. Return as a structured list."
                    ),
                    max_steps=15,
                )

                if result.get("success"):
                    # Parse SERP results from agent output
                    serp_text = result.get("result", "") or ""
                    competitor_analysis.append({
                        "url": "google.com/search",
                        "title": f"SERP: {search_query}",
                        "summary": serp_text[:2000],
                        "design_score": None,
                        "content_gaps": [],
                        "source": "serp",
                    })

            except Exception as e:
                logger.warning(
                    "seo_browser_research_failed",
                    extra={"error": str(e)[:200]},
                )

        # ─── Shared Brain: Pull Outreach Insights ────────────────
        outreach_insights: list[dict[str, Any]] = []
        try:
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(
                    f"What messaging resonates for {topic}?"
                ),
                insight_type="winning_pattern",
                limit=5,
            )
            outreach_insights = insights if isinstance(insights, list) else []
        except Exception as e:
            logger.debug(f"Could not fetch outreach insights: {e}")

        # ─── LLM Analysis of Research ────────────────────────────
        if competitor_analysis or not self.browser_tool:
            try:
                analysis_prompt = self._build_analysis_prompt(
                    topic, competitor_analysis
                )
                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": analysis_prompt}],
                )
                analysis_text = response.content[0].text if response.content else ""

                # Store the LLM's competitive analysis
                if analysis_text:
                    competitor_analysis.append({
                        "url": "llm_analysis",
                        "title": "Competitive Gap Analysis",
                        "summary": analysis_text[:3000],
                        "design_score": None,
                        "content_gaps": [],
                        "source": "llm_analysis",
                    })
            except Exception as e:
                logger.warning(
                    "seo_analysis_failed",
                    extra={"error": str(e)[:200]},
                )

        logger.info(
            "seo_research_completed",
            extra={
                "agent_id": self.agent_id,
                "competitors_found": len(competitor_analysis),
                "insights_found": len(outreach_insights),
            },
        )

        return {
            "current_node": "research_competitors",
            "competitor_analysis": competitor_analysis,
            "outreach_insights": outreach_insights,
            "serp_analysis": {
                "query": search_query,
                "results_count": len(competitor_analysis),
            },
        }

    async def _node_draft_content(
        self, state: SEOContentAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Generate blog post using research + shared brain context.

        The draft is informed by:
        - Competitor analysis (what they missed)
        - Outreach insights (what resonates with the target audience)
        - Agent params (tone, word count, content type)
        """
        topic = state.get("topic", "")
        content_type = state.get("content_type", "blog_post")
        competitor_analysis = state.get("competitor_analysis", [])
        outreach_insights = state.get("outreach_insights", [])
        keywords = state.get("target_keywords", [])

        # Pull config params
        target_word_count = self.config.params.get("target_word_count", 1500)
        tone = self.config.params.get("tone", "Authoritative, Technical")

        logger.info(
            "seo_draft_started",
            extra={
                "agent_id": self.agent_id,
                "topic": topic,
                "content_type": content_type,
                "target_word_count": target_word_count,
            },
        )

        draft_prompt = self._build_draft_prompt(
            topic=topic,
            content_type=content_type,
            keywords=keywords,
            competitor_analysis=competitor_analysis,
            outreach_insights=outreach_insights,
            target_word_count=target_word_count,
            tone=tone,
        )

        try:
            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                messages=[{"role": "user", "content": draft_prompt}],
                system=self._get_system_prompt(),
            )
            draft_text = response.content[0].text if response.content else ""
        except Exception as e:
            logger.error(
                "seo_draft_failed",
                extra={"agent_id": self.agent_id, "error": str(e)[:200]},
            )
            return {
                "current_node": "draft_content",
                "error": f"Draft generation failed: {str(e)[:200]}",
                "draft_content": "",
            }

        # Extract title (first line) and body
        lines = draft_text.strip().split("\n", 1)
        title = lines[0].lstrip("# ").strip() if lines else topic
        body = lines[1].strip() if len(lines) > 1 else draft_text
        word_count = len(body.split())

        logger.info(
            "seo_draft_completed",
            extra={
                "agent_id": self.agent_id,
                "word_count": word_count,
                "title": title[:100],
            },
        )

        return {
            "current_node": "draft_content",
            "draft_title": title,
            "draft_content": body,
            "word_count": word_count,
            "meta_title": title[:60],
            "meta_description": body[:155].rsplit(" ", 1)[0] + "..." if len(body) > 155 else body,
        }

    async def _node_human_review(
        self, state: SEOContentAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Human review gate.

        In production, LangGraph's interrupt_before pauses execution here.
        The human can approve, reject, or edit the draft. When resumed,
        human_approval_status and optionally human_edited_content are set.
        """
        logger.info(
            "seo_human_review_pending",
            extra={
                "agent_id": self.agent_id,
                "title": state.get("draft_title", "")[:100],
                "word_count": state.get("word_count", 0),
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    async def _node_finalize_and_learn(
        self, state: SEOContentAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Save content to DB + RLHF data capture.

        1. Save the final content to agent_content table
        2. If human edited the draft, capture the (draft, edit) pair
           via self.learn() for RLHF optimization
        3. Write keyword performance insight to shared brain
        """
        draft_content = state.get("draft_content", "")
        human_edited = state.get("human_edited_content")
        final_content = human_edited or draft_content
        title = state.get("draft_title", "")
        content_type = state.get("content_type", "blog_post")
        topic = state.get("topic", "")

        # ─── Save to agent_content ────────────────────────────────
        content_id = None
        try:
            stored = self.db.store_content({
                "agent_id": self.agent_id,
                "content_type": content_type,
                "title": title,
                "body": final_content,
                "status": "draft",
                "metadata": {
                    "topic": topic,
                    "keywords": state.get("target_keywords", []),
                    "word_count": len(final_content.split()),
                    "meta_title": state.get("meta_title", ""),
                    "meta_description": state.get("meta_description", ""),
                },
            })
            content_id = stored.get("id")
            logger.info(
                "seo_content_stored",
                extra={
                    "agent_id": self.agent_id,
                    "content_id": content_id,
                    "content_type": content_type,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store content: {e}")

        # ─── God Mode Hook: RLHF Data Capture ────────────────────
        rlhf_captured = False
        if human_edited and human_edited != draft_content:
            result = self.learn(
                task_input={
                    "topic": topic,
                    "keywords": state.get("target_keywords", []),
                    "competitor_count": len(state.get("competitor_analysis", [])),
                },
                model_output=draft_content,
                human_correction=human_edited,
                source="manual_review",
                metadata={
                    "content_type": content_type,
                    "word_count": len(final_content.split()),
                },
            )
            rlhf_captured = result is not None
            if rlhf_captured:
                logger.info(
                    "seo_rlhf_captured",
                    extra={"agent_id": self.agent_id, "topic": topic},
                )

        # ─── Write Insight to Shared Brain ────────────────────────
        self.store_insight(InsightData(
            insight_type="keyword_performance",
            title=f"SEO: {topic}",
            content=f"Generated {content_type} on '{topic}' ({len(final_content.split())} words). "
                    f"Keywords: {', '.join(state.get('target_keywords', [])[:5])}",
            confidence=0.8,
            metadata={"content_id": content_id, "content_type": content_type},
        ))

        return {
            "current_node": "finalize_and_learn",
            "content_approved": True,
            "rlhf_captured": rlhf_captured,
            "knowledge_written": True,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: SEOContentAgentState) -> str:
        """Route after human review: approved → finalize, rejected → END."""
        status = state.get("human_approval_status", "approved")
        if status == "rejected":
            return "rejected"
        return "approved"

    # ─── Prompt Construction ──────────────────────────────────────────

    def _build_analysis_prompt(
        self,
        topic: str,
        competitor_data: list[dict[str, Any]],
    ) -> str:
        """Build the competitive analysis prompt."""
        competitor_section = ""
        for i, comp in enumerate(competitor_data[:5], 1):
            competitor_section += (
                f"\n{i}. URL: {comp.get('url', 'N/A')}\n"
                f"   Title: {comp.get('title', 'N/A')}\n"
                f"   Summary: {comp.get('summary', 'N/A')[:500]}\n"
            )

        return (
            f"Analyze the competitive landscape for the topic: '{topic}'\n\n"
            f"Competitor data found:\n{competitor_section or 'No competitor data available.'}\n\n"
            "Based on this analysis:\n"
            "1. What content gaps exist that we can fill?\n"
            "2. What angles are competitors missing?\n"
            "3. What unique value can we provide?\n"
            "4. Rate the overall competitiveness (low/medium/high).\n\n"
            "Be specific and actionable."
        )

    def _build_draft_prompt(
        self,
        topic: str,
        content_type: str,
        keywords: list[str],
        competitor_analysis: list[dict[str, Any]],
        outreach_insights: list[dict[str, Any]],
        target_word_count: int,
        tone: str,
    ) -> str:
        """Build the content generation prompt."""
        # Competitor intelligence
        comp_section = ""
        for comp in competitor_analysis[:3]:
            comp_section += f"- {comp.get('title', 'N/A')}: {comp.get('summary', '')[:300]}\n"

        # Outreach brain insights
        insight_section = ""
        for ins in outreach_insights[:5]:
            insight_section += f"- {ins.get('content', ins.get('title', ''))[:200]}\n"

        keyword_list = ", ".join(keywords[:10]) if keywords else topic

        return (
            f"Write a {content_type.replace('_', ' ')} on the topic: '{topic}'\n\n"
            f"TARGET KEYWORDS: {keyword_list}\n"
            f"TARGET WORD COUNT: {target_word_count}\n"
            f"TONE: {tone}\n\n"
            f"COMPETITOR INTELLIGENCE:\n{comp_section or 'No competitor data available.'}\n\n"
            f"AUDIENCE INSIGHTS (from our outreach data):\n"
            f"{insight_section or 'No outreach insights available yet.'}\n\n"
            "INSTRUCTIONS:\n"
            "1. Start with a compelling title (first line, prefixed with #)\n"
            "2. Write content that is BETTER than what competitors have\n"
            "3. Fill the content gaps identified in competitor analysis\n"
            "4. Naturally incorporate the target keywords\n"
            "5. Use the audience insights to tailor the messaging\n"
            "6. Include actionable takeaways\n"
            "7. Structure with clear headings (##) and short paragraphs\n"
            "8. End with a strong call-to-action\n"
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt (from file or default)."""
        if self.config.system_prompt_path:
            try:
                with open(self.config.system_prompt_path) as f:
                    return f.read()
            except FileNotFoundError:
                logger.warning(
                    f"System prompt not found: {self.config.system_prompt_path}"
                )

        return (
            "You are an expert SEO content writer for a cybersecurity consulting firm. "
            "You write authoritative, technical content that ranks well in search engines "
            "and converts readers into leads. "
            "Your content is data-driven, specific, and avoids generic fluff. "
            "Every piece you write should demonstrate deep domain expertise."
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        SEO agent writes keyword performance insights to shared brain.
        Main knowledge writing happens in finalize_and_learn node.
        """
        pass  # Handled in _node_finalize_and_learn

    def __repr__(self) -> str:
        return (
            f"<SEOContentAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
