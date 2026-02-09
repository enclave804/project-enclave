"""
Sovereign Cockpit — Approval Queue

Human-in-the-loop gate for agent outputs. Displays items requiring
human review (blog drafts, email drafts, task approvals) and allows
Approve / Reject / Edit actions.

When a human edits content, the (original, edited) pair is saved
as an RLHF training example via the agent's learn() method.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


st.set_page_config(
    page_title="Approvals — Sovereign Cockpit",
    page_icon="✅",
    layout="wide",
)


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db(vid: str):
    from core.integrations.supabase_client import EnclaveDB
    return EnclaveDB(vid)


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

from dashboard.sidebar import render_sidebar

vertical_id = render_sidebar(title="✅ Approval Queue", show_version=False)

queue_filter = st.sidebar.radio(
    "Show",
    ["Pending Review", "All Content", "Task Queue"],
)


# ---------------------------------------------------------------------------
# Content Approval Queue
# ---------------------------------------------------------------------------

def show_content_approvals(status_filter: str | None = None):
    """Show agent-generated content awaiting review."""

    db = get_db(vertical_id)

    if status_filter:
        items = _safe_call(
            lambda: db.list_content(status=status_filter, limit=50), []
        )
    else:
        items = _safe_call(lambda: db.list_content(limit=100), [])

    if not items:
        st.info("No content items found. The SEO agent will populate this queue.")
        return

    st.markdown(f"**{len(items)} items**")

    for idx, item in enumerate(items):
        content_id = item.get("id", "")
        title = item.get("title", "Untitled")
        content_type = item.get("content_type", "unknown")
        status = item.get("status", "draft")
        agent_id = item.get("agent_id", "?")
        body = item.get("body", "")
        seo_score = item.get("seo_score")
        created = (item.get("created_at") or "")[:19]

        # Status badge
        status_colors = {
            "draft": "🟡",
            "review": "🟠",
            "approved": "🟢",
            "published": "🔵",
            "archived": "⚪",
        }
        badge = status_colors.get(status, "❓")

        with st.expander(
            f"{badge} [{content_type}] {title} — by `{agent_id}` ({created})",
            expanded=(status in ("draft", "review")),
        ):
            # Metadata row
            meta_cols = st.columns(4)
            with meta_cols[0]:
                st.caption(f"**Type:** {content_type}")
            with meta_cols[1]:
                st.caption(f"**Agent:** {agent_id}")
            with meta_cols[2]:
                if seo_score is not None:
                    st.caption(f"**SEO Score:** {seo_score:.0f}/100")
            with meta_cols[3]:
                st.caption(f"**Status:** {status}")

            # Meta title & description
            meta_title = item.get("meta_title", "")
            meta_desc = item.get("meta_description", "")
            if meta_title:
                st.markdown(f"**Meta Title:** {meta_title}")
            if meta_desc:
                st.markdown(f"**Meta Description:** {meta_desc}")

            st.markdown("---")

            # Content body (editable)
            edited_body = st.text_area(
                "Content",
                value=body or "",
                height=300,
                key=f"content_{content_id}_{idx}",
            )

            # Action buttons
            btn_cols = st.columns(4)

            with btn_cols[0]:
                if st.button("✅ Approve", key=f"approve_{content_id}_{idx}"):
                    updates = {"status": "approved"}

                    # Check if human edited the content
                    if edited_body and edited_body != body:
                        updates["body"] = edited_body
                        # Capture RLHF training example
                        _capture_rlhf(
                            db, agent_id, vertical_id,
                            original=body or "",
                            edited=edited_body,
                            item=item,
                        )

                    _safe_call(lambda: db.update_content(content_id, updates))
                    st.success(f"Approved: {title}")
                    st.rerun()

            with btn_cols[1]:
                if st.button("❌ Reject", key=f"reject_{content_id}_{idx}"):
                    _safe_call(
                        lambda: db.update_content(content_id, {"status": "archived"})
                    )
                    st.warning(f"Rejected: {title}")
                    st.rerun()

            with btn_cols[2]:
                if st.button("📝 Save Edit", key=f"edit_{content_id}_{idx}"):
                    if edited_body and edited_body != body:
                        _safe_call(
                            lambda: db.update_content(
                                content_id, {"body": edited_body, "status": "review"}
                            )
                        )
                        _capture_rlhf(
                            db, agent_id, vertical_id,
                            original=body or "",
                            edited=edited_body,
                            item=item,
                        )
                        st.success(f"Saved edits for: {title}")
                        st.rerun()
                    else:
                        st.info("No changes detected.")

            with btn_cols[3]:
                if st.button("🔄 Request Rewrite", key=f"rewrite_{content_id}_{idx}"):
                    _safe_call(
                        lambda: db.update_content(content_id, {"status": "draft"})
                    )
                    st.info(f"Sent back for rewrite: {title}")
                    st.rerun()


def _capture_rlhf(db, agent_id, vertical_id, original, edited, item):
    """Save an RLHF training example when human edits content."""
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
            score=None,  # Human can rate later
            source="manual_review",
            metadata={
                "content_id": str(item.get("id", "")),
                "dashboard_edit": True,
            },
        )
    except Exception:
        pass  # Best-effort RLHF capture


# ---------------------------------------------------------------------------
# Task Queue View
# ---------------------------------------------------------------------------

def show_task_queue():
    """Show the inter-agent task queue."""
    db = get_db(vertical_id)

    status_filter = st.selectbox(
        "Filter by status",
        ["All", "pending", "claimed", "running", "completed", "failed"],
    )

    if status_filter == "All":
        tasks = _safe_call(lambda: db.list_tasks(limit=100), [])
    else:
        tasks = _safe_call(lambda: db.list_tasks(status=status_filter, limit=100), [])

    if not tasks:
        st.info("No tasks in the queue.")
        return

    st.markdown(f"**{len(tasks)} tasks**")

    table_data = []
    for t in tasks:
        status = t.get("status", "?")
        status_emoji = {
            "pending": "⏳",
            "claimed": "🔒",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(status, "❓")

        table_data.append(
            {
                "Status": f"{status_emoji} {status}",
                "Target Agent": t.get("target_agent_id", ""),
                "Source": t.get("source_agent_id", "manual"),
                "Type": t.get("task_type", ""),
                "Priority": t.get("priority", 5),
                "Retries": f"{t.get('retry_count', 0)}/{t.get('max_retries', 3)}",
                "Error": (t.get("error_message") or "")[:60],
                "Created": (t.get("created_at") or "")[:19],
            }
        )

    st.dataframe(table_data, use_container_width=True)


# ---------------------------------------------------------------------------
# Page Router
# ---------------------------------------------------------------------------

st.title("✅ Approval Queue")

if queue_filter == "Pending Review":
    show_content_approvals(status_filter="review")
elif queue_filter == "All Content":
    show_content_approvals(status_filter=None)
elif queue_filter == "Task Queue":
    show_task_queue()
