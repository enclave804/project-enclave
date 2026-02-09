"""
Sovereign Cockpit — Approval Queue

Human-in-the-loop gate for agent outputs. Features:
- Kanban board view (Drafts -> In Review -> Approved -> Published)
- Individual content cards with metadata
- Edit-in-place with RLHF capture
- Approve / Reject / Edit / Request Rewrite actions
- Task Queue monitor for inter-agent coordination
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)


st.set_page_config(
    page_title="Approvals — Sovereign Cockpit",
    page_icon="◆",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Auth + Theme + Sidebar
# ---------------------------------------------------------------------------

from dashboard.auth import require_auth

require_auth()

from dashboard.theme import (
    COLORS, inject_theme_css, page_header, section_header,
    render_empty_state, render_divider,
)

inject_theme_css()

from dashboard.sidebar import render_sidebar

vertical_id = render_sidebar()


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
# Sidebar Filters
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    f'<div style="height: 1px; background: {COLORS["border_subtle"]}; margin: 12px 0;"></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f'<div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; '
    f'color: {COLORS["text_secondary"]}; font-weight: 600; margin-bottom: 8px;">View Mode</div>',
    unsafe_allow_html=True,
)

view_mode = st.sidebar.radio(
    "View",
    ["Kanban Board", "Content List", "Task Queue"],
    label_visibility="collapsed",
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

page_header(
    "Approval Queue",
    "Review, edit, and approve agent outputs before they go live",
)

# Human-in-the-loop reminder
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px 16px;
                 background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2);
                 border-radius: 8px; margin-bottom: 16px;">
        <span style="font-size: 1.1rem;">🛡</span>
        <span style="font-size: 0.78rem; color: {COLORS['text_secondary']};">
            <strong style="color: {COLORS['text_accent']};">Human-in-the-Loop active</strong> —
            Nothing reaches customers without your explicit approval.
            Every edit trains the AI to do better next time.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Content: Kanban Board
# ---------------------------------------------------------------------------

def show_kanban_board():
    """Show content in a Kanban-style board layout."""
    db = get_db(vertical_id)

    # Fetch all content
    all_content = _safe_call(lambda: db.list_content(limit=200), [])

    if not all_content:
        st.markdown(
            render_empty_state(
                "◌",
                "No content in the pipeline",
                "Content will appear here when agents generate blog posts, "
                "proposals, or email drafts. Each piece requires human approval.",
            ),
            unsafe_allow_html=True,
        )
        return

    # Group by status
    columns = {
        "draft": {"label": "DRAFTS", "color": COLORS["status_yellow"], "items": []},
        "review": {"label": "IN REVIEW", "color": COLORS["status_blue"], "items": []},
        "approved": {"label": "APPROVED", "color": COLORS["status_green"], "items": []},
        "published": {"label": "PUBLISHED", "color": COLORS["status_purple"], "items": []},
    }

    for item in all_content:
        status = item.get("status", "draft")
        if status in columns:
            columns[status]["items"].append(item)

    # Render Kanban columns
    cols = st.columns(4)

    for i, (status_key, col_data) in enumerate(columns.items()):
        with cols[i]:
            count = len(col_data["items"])
            st.markdown(
                f"""
                <div class="sov-kanban-col">
                    <div class="sov-kanban-header">
                        <span style="color: {col_data['color']};">{col_data['label']}</span>
                        <span class="sov-kanban-count">{count}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not col_data["items"]:
                st.markdown(
                    f'<div style="text-align: center; padding: 20px; color: {COLORS["text_tertiary"]}; '
                    f'font-size: 0.75rem;">Empty</div>',
                    unsafe_allow_html=True,
                )
            else:
                for item in col_data["items"][:10]:  # Limit per column
                    _render_kanban_card(db, item, status_key)


def _render_kanban_card(db, item: dict, status_key: str):
    """Render a single Kanban card."""
    content_id = item.get("id", "")
    title = item.get("title", "Untitled")
    content_type = item.get("content_type", "")
    agent_id = item.get("agent_id", "?")
    body = item.get("body", "")
    seo_score = item.get("seo_score")
    created = (item.get("created_at") or "")[:10]

    # Type icon
    type_icons = {
        "blog_post": "📝",
        "landing_page": "🌐",
        "case_study": "📊",
        "email_draft": "📧",
        "proposal": "📋",
        "ad_copy": "📢",
    }
    icon = type_icons.get(content_type, "📄")

    with st.expander(f"{icon} {title[:40]}", expanded=False):
        # Metadata
        st.markdown(
            f"""
            <div style="font-size: 0.7rem; color: {COLORS['text_tertiary']}; margin-bottom: 8px;">
                <span>{content_type}</span> · <span>{agent_id}</span> · <span>{created}</span>
                {f' · <span style="color: {COLORS["status_green"]};">SEO: {seo_score:.0f}</span>' if seo_score is not None else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Content preview / edit
        edited_body = st.text_area(
            "Content",
            value=body or "",
            height=150,
            key=f"kb_{content_id}",
            label_visibility="collapsed",
        )

        # Actions
        action_cols = st.columns(4)

        with action_cols[0]:
            if status_key in ("draft", "review"):
                if st.button("✓ Approve", key=f"kba_{content_id}", use_container_width=True):
                    updates = {"status": "approved"}
                    if edited_body and edited_body != body:
                        updates["body"] = edited_body
                        _capture_rlhf(db, agent_id, vertical_id, body or "", edited_body, item)
                    _safe_call(lambda: db.update_content(content_id, updates))
                    st.rerun()

        with action_cols[1]:
            if status_key in ("draft", "review"):
                if st.button("✕ Reject", key=f"kbr_{content_id}", use_container_width=True):
                    _safe_call(lambda: db.update_content(content_id, {"status": "archived"}))
                    st.rerun()

        with action_cols[2]:
            if edited_body and edited_body != body:
                if st.button("Save", key=f"kbs_{content_id}", use_container_width=True):
                    _safe_call(lambda: db.update_content(content_id, {"body": edited_body, "status": "review"}))
                    _capture_rlhf(db, agent_id, vertical_id, body or "", edited_body, item)
                    st.rerun()

        with action_cols[3]:
            if status_key in ("review", "approved"):
                if st.button("↩ Draft", key=f"kbd_{content_id}", use_container_width=True):
                    _safe_call(lambda: db.update_content(content_id, {"status": "draft"}))
                    st.rerun()


# ---------------------------------------------------------------------------
# Content: List View
# ---------------------------------------------------------------------------

def show_content_list():
    """Show content in a list view with full editing."""
    db = get_db(vertical_id)

    # Filter
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "draft", "review", "approved", "published", "archived"],
        key="content_status_filter",
    )

    if status_filter == "All":
        items = _safe_call(lambda: db.list_content(limit=100), [])
    else:
        items = _safe_call(lambda: db.list_content(status=status_filter, limit=100), [])

    if not items:
        st.markdown(
            render_empty_state(
                "◌",
                "No content found",
                "Try adjusting your filter or wait for agents to generate content.",
            ),
            unsafe_allow_html=True,
        )
        return

    section_header("CONTENT PIPELINE", f"{len(items)} items")

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
            "draft": ("sov-badge-yellow", "DRAFT"),
            "review": ("sov-badge-blue", "REVIEW"),
            "approved": ("sov-badge-green", "APPROVED"),
            "published": ("sov-badge-purple", "PUBLISHED"),
            "archived": ("sov-badge-gray", "ARCHIVED"),
        }
        badge_class, badge_label = status_colors.get(status, ("sov-badge-gray", status.upper()))

        with st.expander(
            f"[{content_type}] {title} — {agent_id} ({created})",
            expanded=(status in ("draft", "review")),
        ):
            # Header
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <span class="sov-badge {badge_class}">{badge_label}</span>
                    <span style="font-size: 0.72rem; color: {COLORS['text_tertiary']};">
                        {content_type} · {agent_id}
                        {f' · SEO: {seo_score:.0f}/100' if seo_score is not None else ''}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Meta
            meta_title = item.get("meta_title", "")
            meta_desc = item.get("meta_description", "")
            if meta_title:
                st.markdown(f"**Meta Title:** {meta_title}")
            if meta_desc:
                st.markdown(f"**Meta Description:** {meta_desc}")

            # Editable body
            edited_body = st.text_area(
                "Content",
                value=body or "",
                height=300,
                key=f"cl_{content_id}_{idx}",
                label_visibility="collapsed",
            )

            # Actions
            btn_cols = st.columns(4)

            with btn_cols[0]:
                if st.button("✓ Approve", key=f"cla_{content_id}_{idx}", use_container_width=True):
                    updates = {"status": "approved"}
                    if edited_body and edited_body != body:
                        updates["body"] = edited_body
                        _capture_rlhf(db, agent_id, vertical_id, body or "", edited_body, item)
                    _safe_call(lambda: db.update_content(content_id, updates))
                    st.rerun()

            with btn_cols[1]:
                if st.button("✕ Reject", key=f"clr_{content_id}_{idx}", use_container_width=True):
                    _safe_call(lambda: db.update_content(content_id, {"status": "archived"}))
                    st.rerun()

            with btn_cols[2]:
                if st.button("Save Edit", key=f"cls_{content_id}_{idx}", use_container_width=True):
                    if edited_body and edited_body != body:
                        _safe_call(lambda: db.update_content(content_id, {"body": edited_body, "status": "review"}))
                        _capture_rlhf(db, agent_id, vertical_id, body or "", edited_body, item)
                        st.rerun()
                    else:
                        st.info("No changes detected.")

            with btn_cols[3]:
                if st.button("↩ Rewrite", key=f"clw_{content_id}_{idx}", use_container_width=True):
                    _safe_call(lambda: db.update_content(content_id, {"status": "draft"}))
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
            score=None,
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
        key="task_status_filter",
    )

    if status_filter == "All":
        tasks = _safe_call(lambda: db.list_tasks(limit=100), [])
    else:
        tasks = _safe_call(lambda: db.list_tasks(status=status_filter, limit=100), [])

    if not tasks:
        st.markdown(
            render_empty_state(
                "◌",
                "No tasks in the queue",
                "Inter-agent tasks will appear here when agents coordinate work.",
            ),
            unsafe_allow_html=True,
        )
        return

    section_header("TASK QUEUE", f"{len(tasks)} tasks")

    table_data = []
    for t in tasks:
        status = t.get("status", "?")
        status_emoji = {
            "pending": "◌",
            "claimed": "◐",
            "running": "◉",
            "completed": "✓",
            "failed": "✕",
        }.get(status, "?")

        table_data.append(
            {
                "Status": f"{status_emoji} {status}",
                "Target": t.get("target_agent_id", ""),
                "Source": t.get("source_agent_id", "manual"),
                "Type": t.get("task_type", ""),
                "Priority": t.get("priority", 5),
                "Retries": f"{t.get('retry_count', 0)}/{t.get('max_retries', 3)}",
                "Error": (t.get("error_message") or "—")[:60],
                "Created": (t.get("created_at") or "")[:19],
            }
        )

    st.dataframe(table_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page Router
# ---------------------------------------------------------------------------

if view_mode == "Kanban Board":
    show_kanban_board()
elif view_mode == "Content List":
    show_content_list()
elif view_mode == "Task Queue":
    show_task_queue()
