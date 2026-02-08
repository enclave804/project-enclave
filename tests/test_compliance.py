"""
Unit tests for the ComplianceResult class and ComplianceChecker content checks.

Tests the pure-logic parts of compliance that don't need a DB connection.
"""

import pytest

from core.outreach.compliance import ComplianceResult


class TestComplianceResult:
    """Tests for ComplianceResult data class."""

    def test_passed_result(self):
        result = ComplianceResult(passed=True, issues=[])
        assert result.passed is True
        assert bool(result) is True
        assert result.blocking_issues == []
        assert result.warnings == []
        assert not result.has_blocking_issues

    def test_failed_with_blocking(self):
        issues = ["BLOCKED: Email is on suppression list"]
        result = ComplianceResult(passed=False, issues=issues)
        assert result.passed is False
        assert bool(result) is False
        assert len(result.blocking_issues) == 1
        assert result.has_blocking_issues
        assert result.warnings == []

    def test_warning_only(self):
        issues = ["WARNING: Subject line is all caps"]
        result = ComplianceResult(passed=False, issues=issues)
        assert len(result.warnings) == 1
        assert result.blocking_issues == []
        assert not result.has_blocking_issues

    def test_mixed_issues(self):
        issues = [
            "BLOCKED: Suppressed email",
            "WARNING: Subject too long",
            "WARNING: Spam trigger word",
        ]
        result = ComplianceResult(passed=False, issues=issues)
        assert len(result.blocking_issues) == 1
        assert len(result.warnings) == 2
        assert result.has_blocking_issues

    def test_repr(self):
        result = ComplianceResult(passed=True, issues=[])
        assert "PASS" in repr(result)

        result = ComplianceResult(passed=False, issues=["BLOCKED: test"])
        assert "FAIL" in repr(result)

    def test_empty_issues_is_pass(self):
        result = ComplianceResult(passed=True, issues=[])
        assert result.passed
        assert len(result.issues) == 0
