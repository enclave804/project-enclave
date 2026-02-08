"""
Unit tests for TemplateEngine — path traversal defense and rendering logic.

Tests focus on the security-critical path validation and edge cases.
"""

import os
import tempfile

import pytest

from core.outreach.template_engine import TemplateEngine, build_template_context


class TestTemplateEnginePathTraversal:
    """Verify that path traversal attacks are blocked."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a TemplateEngine pointing at a real temp directory."""
        template = tmp_path / "good_template.md"
        template.write_text("Subject: Hello\n---\nBody here")
        return TemplateEngine(tmp_path)

    def test_normal_template_loads(self, engine):
        content = engine.get_template_content("good_template.md")
        assert "Subject: Hello" in content

    def test_path_traversal_blocked(self, engine):
        with pytest.raises(ValueError, match="path traversal"):
            engine.get_template_content("../../etc/passwd")

    def test_path_traversal_with_dots(self, engine):
        with pytest.raises((ValueError, FileNotFoundError)):
            engine.get_template_content("../../../etc/shadow")

    def test_absolute_path_blocked(self, engine):
        """An absolute path that escapes the template dir should be blocked."""
        with pytest.raises((ValueError, FileNotFoundError)):
            engine.get_template_content("/etc/passwd")

    def test_nonexistent_template_raises(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.get_template_content("nonexistent.md")


class TestTemplateEngineNoneEnv:
    """Test behavior when template directory doesn't exist."""

    def test_missing_dir_sets_env_none(self, tmp_path):
        engine = TemplateEngine(tmp_path / "nonexistent")
        assert engine.env is None

    def test_render_with_none_env_raises(self, tmp_path):
        engine = TemplateEngine(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            engine.render("anything.md", {})

    def test_list_templates_with_none_env_returns_empty(self, tmp_path):
        engine = TemplateEngine(tmp_path / "nonexistent")
        assert engine.list_templates() == []


class TestTemplateEngineRendering:
    """Test the Jinja2 rendering pipeline."""

    @pytest.fixture
    def engine(self, tmp_path):
        template = tmp_path / "test.md"
        template.write_text(
            "Subject: Hello {{ contact_name }}\n"
            "---\n"
            "Hi {{ contact_first_name }},\n"
            "Your company {{ company_name }} uses {{ tech_stack_list|join(', ') }}."
        )
        return TemplateEngine(tmp_path)

    def test_render_parses_subject_and_body(self, engine):
        result = engine.render("test.md", {
            "contact_name": "Marcus Chen",
            "contact_first_name": "Marcus",
            "company_name": "TestCorp",
            "tech_stack_list": ["WordPress", "PHP"],
        })
        assert result["subject"] == "Hello Marcus Chen"
        assert "Hi Marcus" in result["body"]
        assert "WordPress, PHP" in result["body"]

    def test_render_empty_context(self, engine):
        result = engine.render("test.md", {
            "contact_name": "",
            "contact_first_name": "",
            "company_name": "",
            "tech_stack_list": [],
        })
        assert result["subject"] == "Hello"
        assert "Hi ," in result["body"]


class TestBuildTemplateContext:
    """Test the context builder helper."""

    def test_basic_context_fields(self):
        state = {
            "contact_name": "Jane Doe",
            "contact_title": "CTO",
            "contact_email": "jane@example.com",
            "company_name": "TestCorp",
            "company_domain": "testcorp.com",
            "company_industry": "fintech",
            "company_size": 45,
            "tech_stack": {"WordPress": "5.8"},
            "vulnerabilities": [{"type": "ssl", "description": "expired cert"}],
        }
        ctx = build_template_context(state)

        assert ctx["contact_name"] == "Jane Doe"
        assert ctx["contact_first_name"] == "Jane"
        assert ctx["company_name"] == "TestCorp"
        assert ctx["tech_stack_list"] == ["WordPress"]
        assert ctx["vulnerability_count"] == 1

    def test_empty_name_first_name(self):
        state = {"contact_name": ""}
        ctx = build_template_context(state)
        assert ctx["contact_first_name"] == ""

    def test_none_name_first_name(self):
        state = {"contact_name": None}
        ctx = build_template_context(state)
        assert ctx["contact_first_name"] == ""

    def test_config_extras_merged(self):
        state = {"contact_name": "Test"}
        ctx = build_template_context(state, config_extras={"custom_field": "value"})
        assert ctx["custom_field"] == "value"

    def test_missing_state_keys_default(self):
        ctx = build_template_context({})
        assert ctx["contact_name"] == ""
        assert ctx["company_size"] == 0
        assert ctx["tech_stack"] == {}
        assert ctx["vulnerabilities"] == []
        assert ctx["vulnerability_count"] == 0
