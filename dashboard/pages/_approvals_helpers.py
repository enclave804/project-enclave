"""
Extracted helpers for the Approval Queue page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from typing import Any, Optional


def capture_rlhf(
    db: Any,
    agent_id: str,
    vertical_id: str,
    original: str,
    edited: str,
    item: dict[str, Any],
) -> None:
    """
    Save an RLHF training example when a human edits content.

    This is the core data flywheel: every human correction becomes
    a (bad_draft, good_rewrite) pair for fine-tuning.

    Best-effort: never crashes the dashboard on failure.
    """
    try:
        db.store_training_example(
            agent_id=agent_id,
            vertical_id=vertical_id,
            task_input={
                "content_type": item.get("content_type", ""),
                "title": item.get("title", ""),
                "meta_title": item.get("meta_title", ""),
            },
            model_output=original,
            human_correction=edited,
            score=None,
            source="manual_review",
            metadata={
                "content_id": str(item.get("id", "")),
                "dashboard_edit": True,
            },
        )
    except Exception:
        pass  # Best-effort — never crash the dashboard
