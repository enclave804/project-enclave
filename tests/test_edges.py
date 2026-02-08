"""
Unit tests for LangGraph conditional edge routing functions.

These are pure functions that take state and return a node name — no
APIs, no DB, no I/O. Perfect for fast unit testing.
"""

import pytest

from core.graph.edges import (
    route_after_compliance,
    route_after_duplicate_check,
    route_after_human_review,
    route_after_qualification,
)


# ------------------------------------------------------------------
# route_after_duplicate_check
# ------------------------------------------------------------------

class TestRouteAfterDuplicateCheck:
    def test_duplicate_routes_to_rag(self):
        state = {"is_duplicate": True}
        assert route_after_duplicate_check(state) == "write_to_rag"

    def test_not_duplicate_routes_to_enrich(self):
        state = {"is_duplicate": False}
        assert route_after_duplicate_check(state) == "enrich_company"

    def test_missing_key_routes_to_enrich(self):
        """Missing is_duplicate should be falsy → enrich."""
        state = {}
        assert route_after_duplicate_check(state) == "enrich_company"


# ------------------------------------------------------------------
# route_after_qualification
# ------------------------------------------------------------------

class TestRouteAfterQualification:
    def test_qualified_routes_to_strategy(self):
        state = {"qualified": True}
        assert route_after_qualification(state) == "select_strategy"

    def test_disqualified_routes_to_rag(self):
        state = {"qualified": False}
        assert route_after_qualification(state) == "write_to_rag"

    def test_missing_key_routes_to_rag(self):
        """Missing qualified should be falsy → write_to_rag."""
        state = {}
        assert route_after_qualification(state) == "write_to_rag"


# ------------------------------------------------------------------
# route_after_compliance
# ------------------------------------------------------------------

class TestRouteAfterCompliance:
    def test_passed_routes_to_review(self):
        state = {"compliance_passed": True}
        assert route_after_compliance(state) == "human_review"

    def test_failed_routes_to_rag(self):
        state = {"compliance_passed": False}
        assert route_after_compliance(state) == "write_to_rag"

    def test_missing_key_routes_to_rag(self):
        state = {}
        assert route_after_compliance(state) == "write_to_rag"


# ------------------------------------------------------------------
# route_after_human_review
# ------------------------------------------------------------------

class TestRouteAfterHumanReview:
    def test_approved_routes_to_send(self):
        state = {"human_review_status": "approved"}
        assert route_after_human_review(state) == "send_outreach"

    def test_edited_routes_to_send(self):
        state = {"human_review_status": "edited"}
        assert route_after_human_review(state) == "send_outreach"

    def test_rejected_routes_to_draft(self):
        state = {"human_review_status": "rejected", "review_attempts": 1}
        assert route_after_human_review(state) == "draft_outreach"

    def test_rejected_after_3_attempts_routes_to_rag(self):
        state = {"human_review_status": "rejected", "review_attempts": 3}
        assert route_after_human_review(state) == "write_to_rag"

    def test_rejected_after_many_attempts_routes_to_rag(self):
        state = {"human_review_status": "rejected", "review_attempts": 5}
        assert route_after_human_review(state) == "write_to_rag"

    def test_skipped_routes_to_rag(self):
        state = {"human_review_status": "skipped"}
        assert route_after_human_review(state) == "write_to_rag"

    def test_none_status_routes_to_rag(self):
        state = {"human_review_status": None}
        assert route_after_human_review(state) == "write_to_rag"

    def test_missing_status_routes_to_rag(self):
        state = {}
        assert route_after_human_review(state) == "write_to_rag"

    def test_rejected_zero_attempts_routes_to_draft(self):
        """First rejection should always loop back to draft."""
        state = {"human_review_status": "rejected", "review_attempts": 0}
        assert route_after_human_review(state) == "draft_outreach"
