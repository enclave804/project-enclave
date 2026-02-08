"""
Unit tests for LeadState creation and initial state factory.

Validates that create_initial_lead_state correctly maps raw Apollo
data to the pipeline state schema.
"""

import pytest

from core.graph.state import LeadState, create_initial_lead_state


class TestCreateInitialLeadState:
    """Tests for the create_initial_lead_state factory function."""

    @pytest.fixture
    def mock_lead_data(self):
        return {
            "contact": {
                "name": "Marcus Chen",
                "email": "mchen@example.com",
                "title": "CTO",
                "apollo_id": "apollo_contact_123",
                "seniority": "c_suite",
            },
            "company": {
                "name": "TestCorp",
                "domain": "testcorp.example.com",
                "industry": "fintech",
                "employee_count": 45,
                "apollo_id": "apollo_org_456",
                "tech_stack": {"WordPress": "5.8", "PHP": "7.4"},
            },
        }

    def test_basic_fields_populated(self, mock_lead_data):
        state = create_initial_lead_state(mock_lead_data, "enclave_guard")

        assert state["contact_name"] == "Marcus Chen"
        assert state["contact_email"] == "mchen@example.com"
        assert state["contact_title"] == "CTO"
        assert state["company_name"] == "TestCorp"
        assert state["company_domain"] == "testcorp.example.com"
        assert state["company_industry"] == "fintech"
        assert state["company_size"] == 45
        assert state["vertical_id"] == "enclave_guard"

    def test_lead_id_generated(self, mock_lead_data):
        state = create_initial_lead_state(mock_lead_data, "enclave_guard")
        assert state["lead_id"]
        assert len(state["lead_id"]) == 36  # UUID format

    def test_unique_lead_ids(self, mock_lead_data):
        state1 = create_initial_lead_state(mock_lead_data, "enclave_guard")
        state2 = create_initial_lead_state(mock_lead_data, "enclave_guard")
        assert state1["lead_id"] != state2["lead_id"]

    def test_tech_stack_preserved(self, mock_lead_data):
        state = create_initial_lead_state(mock_lead_data, "enclave_guard")
        assert state["tech_stack"] == {"WordPress": "5.8", "PHP": "7.4"}

    def test_defaults_for_empty_data(self):
        state = create_initial_lead_state({"contact": {}, "company": {}}, "test")

        assert state["contact_name"] == ""
        assert state["contact_email"] == ""
        assert state["company_name"] == ""
        assert state["company_domain"] == ""
        assert state["company_size"] == 0
        assert state["tech_stack"] == {}
        assert state["vulnerabilities"] == []
        assert state["is_duplicate"] is False
        assert state["qualified"] is False
        assert state["compliance_passed"] is False
        assert state["email_sent"] is False
        assert state["knowledge_written"] is False
        assert state["review_attempts"] == 0

    def test_raw_apollo_data_preserved(self, mock_lead_data):
        state = create_initial_lead_state(mock_lead_data, "enclave_guard")
        assert state["raw_apollo_data"] == mock_lead_data

    def test_enrichment_sources_default(self, mock_lead_data):
        state = create_initial_lead_state(mock_lead_data, "enclave_guard")
        assert state["enrichment_sources"] == ["apollo"]

    def test_none_employee_count_becomes_zero(self):
        """When Apollo returns None for employee_count, it should become 0."""
        data = {
            "contact": {"name": "Test"},
            "company": {"name": "Corp", "employee_count": None},
        }
        state = create_initial_lead_state(data, "test")
        assert state["company_size"] == 0

    def test_missing_company_key(self):
        """Completely missing 'company' key should not crash."""
        state = create_initial_lead_state({"contact": {"name": "Solo"}}, "test")
        assert state["company_name"] == ""
        assert state["company_domain"] == ""

    def test_missing_contact_key(self):
        """Completely missing 'contact' key should not crash."""
        state = create_initial_lead_state({"company": {"name": "Corp"}}, "test")
        assert state["contact_name"] == ""
        assert state["contact_email"] == ""
