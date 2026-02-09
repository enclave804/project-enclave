"""
Tests for dashboard/theme.py — Design System & Theme Components.

Tests the color palette, status config, and HTML-rendering helper functions
WITHOUT importing streamlit. These are pure-function tests.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------


class TestColorPalette:
    """Verify the color palette has all required entries."""

    def test_colors_dict_exists(self):
        from dashboard.theme import COLORS
        assert isinstance(COLORS, dict)
        assert len(COLORS) > 20  # Should have many entries

    def test_background_colors(self):
        from dashboard.theme import COLORS
        for key in ("bg_primary", "bg_secondary", "bg_card", "bg_elevated", "bg_input"):
            assert key in COLORS, f"Missing background color: {key}"
            assert COLORS[key].startswith("#") or COLORS[key].startswith("rgba")

    def test_accent_colors(self):
        from dashboard.theme import COLORS
        for key in ("accent_primary", "accent_secondary", "accent_tertiary"):
            assert key in COLORS, f"Missing accent color: {key}"

    def test_status_colors(self):
        from dashboard.theme import COLORS
        for key in ("status_green", "status_yellow", "status_red",
                     "status_blue", "status_purple", "status_gray"):
            assert key in COLORS, f"Missing status color: {key}"

    def test_text_colors(self):
        from dashboard.theme import COLORS
        for key in ("text_primary", "text_secondary", "text_tertiary", "text_accent"):
            assert key in COLORS, f"Missing text color: {key}"

    def test_border_colors(self):
        from dashboard.theme import COLORS
        for key in ("border_subtle", "border_default", "border_focus"):
            assert key in COLORS, f"Missing border color: {key}"


# ---------------------------------------------------------------------------
# Status Config
# ---------------------------------------------------------------------------


class TestStatusConfig:
    """Verify status configuration is complete."""

    def test_status_config_dict_exists(self):
        from dashboard.theme import STATUS_CONFIG
        assert isinstance(STATUS_CONFIG, dict)

    def test_all_statuses_present(self):
        from dashboard.theme import STATUS_CONFIG
        required = ["active", "paused", "shadow", "circuit_breaker",
                     "idle", "running", "completed", "failed", "pending"]
        for status in required:
            assert status in STATUS_CONFIG, f"Missing status: {status}"

    def test_status_has_required_fields(self):
        from dashboard.theme import STATUS_CONFIG
        for status, config in STATUS_CONFIG.items():
            assert "icon" in config, f"{status} missing icon"
            assert "color" in config, f"{status} missing color"
            assert "label" in config, f"{status} missing label"
            assert "glow" in config, f"{status} missing glow"

    def test_active_status_is_green(self):
        from dashboard.theme import STATUS_CONFIG, COLORS
        assert STATUS_CONFIG["active"]["color"] == COLORS["status_green"]

    def test_paused_status_is_red(self):
        from dashboard.theme import STATUS_CONFIG, COLORS
        assert STATUS_CONFIG["paused"]["color"] == COLORS["status_red"]

    def test_shadow_status_is_purple(self):
        from dashboard.theme import STATUS_CONFIG, COLORS
        assert STATUS_CONFIG["shadow"]["color"] == COLORS["status_purple"]

    def test_failed_status_is_red(self):
        from dashboard.theme import STATUS_CONFIG, COLORS
        assert STATUS_CONFIG["failed"]["color"] == COLORS["status_red"]


# ---------------------------------------------------------------------------
# HTML Component Functions (no streamlit dependency)
# ---------------------------------------------------------------------------


class TestStatusBadge:
    """Test the status_badge HTML generator."""

    def test_returns_html_string(self):
        from dashboard.theme import status_badge
        html = status_badge("active")
        assert isinstance(html, str)
        assert "sov-badge" in html

    def test_active_badge_has_green_class(self):
        from dashboard.theme import status_badge
        html = status_badge("active")
        assert "sov-badge-green" in html

    def test_paused_badge_has_red_class(self):
        from dashboard.theme import status_badge
        html = status_badge("paused")
        assert "sov-badge-red" in html

    def test_shadow_badge_has_purple_class(self):
        from dashboard.theme import status_badge
        html = status_badge("shadow")
        assert "sov-badge-purple" in html

    def test_unknown_status_falls_back_to_gray(self):
        from dashboard.theme import status_badge
        html = status_badge("nonexistent_status")
        assert "sov-badge-gray" in html

    def test_badge_contains_label(self):
        from dashboard.theme import status_badge
        html = status_badge("active")
        assert "ACTIVE" in html

    def test_badge_contains_icon(self):
        from dashboard.theme import status_badge
        html = status_badge("active")
        assert "●" in html  # active icon


class TestStatusDot:
    """Test the status_dot HTML generator."""

    def test_returns_html_string(self):
        from dashboard.theme import status_dot
        html = status_dot("active")
        assert isinstance(html, str)
        assert "sov-status-dot" in html

    def test_active_dot_has_glow(self):
        from dashboard.theme import status_dot
        html = status_dot("active")
        assert "glow" in html

    def test_paused_dot_has_no_glow(self):
        from dashboard.theme import status_dot
        html = status_dot("paused")
        # paused has glow=False
        assert 'class="sov-status-dot "' in html

    def test_custom_size(self):
        from dashboard.theme import status_dot
        html = status_dot("active", size=20)
        assert "width:20px" in html
        assert "height:20px" in html


class TestFeedItem:
    """Test the feed_item HTML generator."""

    def test_returns_html_string(self):
        from dashboard.theme import feed_item
        html = feed_item("12:30", "outreach", "Completed run", "completed")
        assert isinstance(html, str)
        assert "sov-feed-item" in html

    def test_contains_agent_name(self):
        from dashboard.theme import feed_item
        html = feed_item("12:30", "my_agent", "Some text", "active")
        assert "my_agent" in html

    def test_contains_time(self):
        from dashboard.theme import feed_item
        html = feed_item("14:22:05", "agent1", "text", "active")
        assert "14:22:05" in html

    def test_contains_text(self):
        from dashboard.theme import feed_item
        html = feed_item("12:30", "agent1", "Completed in 450ms", "completed")
        assert "Completed in 450ms" in html


class TestKpiCard:
    """Test the kpi_card HTML generator."""

    def test_returns_html_string(self):
        from dashboard.theme import kpi_card
        html = kpi_card("Total Leads", "1,234")
        assert isinstance(html, str)
        assert "sov-kpi-card" in html

    def test_contains_label(self):
        from dashboard.theme import kpi_card
        html = kpi_card("Pipeline Value", "$50,000")
        assert "Pipeline Value" in html

    def test_contains_value(self):
        from dashboard.theme import kpi_card
        html = kpi_card("Leads", "999")
        assert "999" in html

    def test_delta_included_when_provided(self):
        from dashboard.theme import kpi_card
        html = kpi_card("Leads", "999", delta="+15%")
        assert "+15%" in html

    def test_delta_color_applied(self):
        from dashboard.theme import kpi_card, COLORS
        html = kpi_card("Leads", "999", delta="-5%", delta_color=COLORS["status_red"])
        assert COLORS["status_red"] in html

    def test_no_delta_when_empty(self):
        from dashboard.theme import kpi_card
        html = kpi_card("Leads", "999")
        assert "sov-kpi-delta" not in html


class TestHealthIndicator:
    """Test the render_health_indicator HTML generator."""

    def test_returns_html_string(self):
        from dashboard.theme import render_health_indicator
        html = render_health_indicator(5, 2, 1)
        assert isinstance(html, str)

    def test_all_healthy(self):
        from dashboard.theme import render_health_indicator, COLORS
        html = render_health_indicator(10, 0, 0)
        assert COLORS["status_green"] in html
        assert "width: 100.0%" in html

    def test_mixed_health(self):
        from dashboard.theme import render_health_indicator
        html = render_health_indicator(5, 3, 2)
        assert "width: 50.0%" in html  # 5/10 = 50%

    def test_zero_total_no_crash(self):
        from dashboard.theme import render_health_indicator
        html = render_health_indicator(0, 0, 0)
        assert isinstance(html, str)
