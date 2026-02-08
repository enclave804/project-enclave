"""
LangGraph workflow engine for Project Enclave.

Builds the sales pipeline graph from a vertical configuration.
This is the central orchestration component that wires together
all nodes, edges, and checkpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import Anthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from core.config.schema import VerticalConfig
from core.graph.edges import (
    route_after_compliance,
    route_after_duplicate_check,
    route_after_human_review,
    route_after_qualification,
)
from core.graph.nodes import PipelineNodes
from core.graph.state import LeadState, create_initial_lead_state
from datetime import datetime, timezone
from core.integrations.supabase_client import EnclaveDB
from core.rag.embeddings import EmbeddingEngine
from core.rag.ingestion import KnowledgeIngester
from core.rag.retrieval import KnowledgeRetriever

logger = logging.getLogger(__name__)


def build_pipeline_graph(
    config: VerticalConfig,
    db: EnclaveDB,
    apollo: Any,
    embedder: EmbeddingEngine,
    anthropic_client: Anthropic,
    checkpointer: Any = None,
    test_mode: bool = False,
) -> StateGraph:
    """
    Build the LangGraph sales pipeline from a vertical configuration.

    The graph topology:

    [START]
      → check_duplicate
        → (if duplicate) → write_to_rag → [END]
        → (not duplicate) → enrich_company
      → enrich_contact
      → qualify_lead
        → (disqualified) → write_to_rag → [END]
        → (qualified) → select_strategy
      → draft_outreach
      → compliance_check
        → (failed) → write_to_rag → [END]
        → (passed) → human_review [INTERRUPT]
      → (approved) → send_outreach → write_to_rag → [END]
      → (rejected) → draft_outreach (loop back)
      → (skipped) → write_to_rag → [END]

    Args:
        config: Vertical configuration.
        db: Supabase database client.
        apollo: Apollo.io API client.
        embedder: Embedding engine for RAG.
        anthropic_client: Anthropic API client.
        checkpointer: LangGraph checkpointer for persistence.
                      Defaults to MemorySaver for development.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Initialize dependencies
    retriever = KnowledgeRetriever(db=db, embedder=embedder)
    ingester = KnowledgeIngester(db=db, embedder=embedder)

    nodes = PipelineNodes(
        config=config,
        db=db,
        apollo=apollo,
        retriever=retriever,
        ingester=ingester,
        anthropic_client=anthropic_client,
    )

    # Build the graph
    graph = StateGraph(LeadState)

    # Add nodes
    graph.add_node("check_duplicate", nodes.check_duplicate)
    graph.add_node("enrich_company", nodes.enrich_company)
    graph.add_node("enrich_contact", nodes.enrich_contact)
    graph.add_node("qualify_lead", nodes.qualify_lead)
    graph.add_node("select_strategy", nodes.select_strategy)
    graph.add_node("draft_outreach", nodes.draft_outreach)
    graph.add_node("compliance_check", nodes.compliance_check)

    if test_mode:
        graph.add_node("human_review", _test_auto_approve)
        graph.add_node("send_outreach", _mock_send_outreach)
    else:
        graph.add_node("human_review", _human_review_interrupt)
        graph.add_node("send_outreach", nodes.send_outreach)

    graph.add_node("write_to_rag", nodes.write_to_rag)

    # Set entry point
    graph.set_entry_point("check_duplicate")

    # Add edges
    graph.add_conditional_edges(
        "check_duplicate",
        route_after_duplicate_check,
        {
            "enrich_company": "enrich_company",
            "write_to_rag": "write_to_rag",
        },
    )

    graph.add_edge("enrich_company", "enrich_contact")
    graph.add_edge("enrich_contact", "qualify_lead")

    graph.add_conditional_edges(
        "qualify_lead",
        route_after_qualification,
        {
            "select_strategy": "select_strategy",
            "write_to_rag": "write_to_rag",
        },
    )

    graph.add_edge("select_strategy", "draft_outreach")
    graph.add_edge("draft_outreach", "compliance_check")

    graph.add_conditional_edges(
        "compliance_check",
        route_after_compliance,
        {
            "human_review": "human_review",
            "write_to_rag": "write_to_rag",
        },
    )

    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "send_outreach": "send_outreach",
            "draft_outreach": "draft_outreach",
            "write_to_rag": "write_to_rag",
        },
    )

    graph.add_edge("send_outreach", "write_to_rag")
    graph.add_edge("write_to_rag", END)

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[] if test_mode else ["human_review"],
    )

    return compiled


def _human_review_interrupt(state: LeadState) -> dict:
    """
    Human review node. This node is interrupted BEFORE execution
    by the LangGraph checkpoint mechanism.

    When resumed, the state should contain:
    - human_review_status: "approved", "rejected", "edited", or "skipped"
    - human_feedback: feedback text if rejected
    - edited_subject: if edited
    - edited_body: if edited

    These values are injected by the review interface (CLI or Streamlit)
    before resuming the graph.
    """
    updates: dict[str, Any] = {"current_node": "human_review"}

    # Increment review attempts
    attempts = state.get("review_attempts", 0) + 1
    updates["review_attempts"] = attempts

    # The actual review data is injected before this node runs
    # via graph.update_state() in the review interface
    return updates


def _test_auto_approve(state: LeadState) -> dict:
    """
    Test-mode human review: auto-approves the draft.

    Used when test_mode=True to skip the interrupt and continue
    the pipeline through send_outreach.
    """
    logger.info("[TEST] Auto-approving draft (test mode)")
    return {
        "current_node": "human_review",
        "review_attempts": state.get("review_attempts", 0) + 1,
        "human_review_status": "approved",
    }


def _mock_send_outreach(state: LeadState) -> dict:
    """
    Test-mode send: logs what would be sent without side effects.

    Does NOT write to the database or call any email provider.
    """
    email = state.get("contact_email", "unknown")
    subject = state.get("draft_email_subject", "")
    logger.info(f"[TEST] Would send email to {email}: {subject}")
    return {
        "current_node": "send_outreach",
        "email_sent": True,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "sending_provider_id": "TEST_MODE_NO_SEND",
    }


async def process_lead(
    graph: StateGraph,
    lead_data: dict[str, Any],
    vertical_id: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    Process a single lead through the pipeline.

    Args:
        graph: Compiled LangGraph.
        lead_data: Raw lead data from Apollo ({"contact": {...}, "company": {...}}).
        vertical_id: Which vertical this lead belongs to.
        thread_id: Optional thread ID for checkpointing.

    Returns:
        Final state after processing (or partial state if interrupted).
    """
    import uuid

    initial_state = create_initial_lead_state(lead_data, vertical_id)

    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    result = await graph.ainvoke(initial_state, config=config)
    return result


async def process_batch(
    graph: StateGraph,
    leads: list[dict[str, Any]],
    vertical_id: str,
) -> dict[str, Any]:
    """
    Process a batch of leads through the pipeline.

    Processes leads sequentially (to respect rate limits).
    Returns summary statistics.
    """
    results = {
        "total": len(leads),
        "processed": 0,
        "sent": 0,
        "skipped": 0,
        "errors": 0,
        "interrupted": 0,  # waiting for human review
        "details": [],
    }

    for lead in leads:
        try:
            result = await process_lead(graph, lead, vertical_id)

            detail = {
                "email": lead.get("contact", {}).get("email"),
                "company": lead.get("company", {}).get("domain"),
            }

            if result.get("email_sent"):
                results["sent"] += 1
                detail["status"] = "sent"
            elif result.get("skip_reason"):
                results["skipped"] += 1
                detail["status"] = f"skipped: {result['skip_reason']}"
            elif result.get("current_node") == "human_review":
                results["interrupted"] += 1
                detail["status"] = "awaiting_review"
            else:
                detail["status"] = "unknown"

            results["processed"] += 1
            results["details"].append(detail)

        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "email": lead.get("contact", {}).get("email"),
                "status": f"error: {str(e)}",
            })
            logger.error(f"Error processing lead: {e}")

    return results
