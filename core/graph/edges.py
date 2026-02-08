"""
LangGraph conditional edge functions for Project Enclave.

These functions determine the routing between nodes based on
the current state of a lead being processed.
"""

from __future__ import annotations

from typing import Literal

from core.graph.state import LeadState


def route_after_duplicate_check(
    state: LeadState,
) -> Literal["enrich_company", "write_to_rag"]:
    """
    After checking for duplicates, decide whether to proceed or skip.

    Skip if:
    - Contact was reached within the cooldown period
    """
    if state.get("is_duplicate"):
        return "write_to_rag"  # log the skip and end
    return "enrich_company"


def route_after_qualification(
    state: LeadState,
) -> Literal["select_strategy", "write_to_rag"]:
    """
    After qualifying, decide whether to proceed with outreach or skip.

    Skip if:
    - Lead is disqualified (low score or disqualifiers triggered)
    """
    if not state.get("qualified"):
        return "write_to_rag"  # log disqualification and end
    return "select_strategy"


def route_after_compliance(
    state: LeadState,
) -> Literal["human_review", "write_to_rag"]:
    """
    After compliance check, decide whether to proceed to review or skip.

    Skip if:
    - Compliance check failed (suppressed, excluded country, etc.)
    """
    if not state.get("compliance_passed"):
        return "write_to_rag"  # log compliance failure and end
    return "human_review"


def route_after_human_review(
    state: LeadState,
) -> Literal["send_outreach", "draft_outreach", "write_to_rag"]:
    """
    After human review, decide next action.

    - approved/edited → send the email
    - rejected → loop back to draft with feedback
    - skipped → log and end
    """
    status = state.get("human_review_status")

    if status == "approved" or status == "edited":
        return "send_outreach"
    elif status == "rejected":
        # Check if we've already retried too many times
        attempts = state.get("review_attempts", 0)
        if attempts >= 3:
            return "write_to_rag"  # give up after 3 rejections
        return "draft_outreach"  # re-draft with feedback
    else:
        # skipped or unknown
        return "write_to_rag"
