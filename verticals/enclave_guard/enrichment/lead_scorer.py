"""
Enclave Guard - Lead Scoring Model.

Cybersecurity-specific lead scoring that combines ICP fit,
security posture signals, and engagement history.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config.schema import VerticalConfig

logger = logging.getLogger(__name__)


class CybersecurityLeadScorer:
    """
    Scores leads based on cybersecurity-specific criteria.

    Scoring dimensions:
    1. ICP Fit (company size, industry, persona match)
    2. Security Posture (vulnerabilities found, tech stack risk)
    3. Timing Signals (breach news, compliance deadlines, hiring)
    4. Engagement History (past opens, replies, meetings)
    """

    def __init__(self, config: VerticalConfig):
        self.config = config

    def score(
        self,
        company_size: int,
        industry: str,
        contact_title: str,
        tech_stack: dict[str, str],
        vulnerabilities: list[dict[str, Any]],
        previous_outreach: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate a composite lead score.

        Returns:
            Dict with:
            - total_score: float 0.0-1.0
            - breakdown: per-dimension scores
            - priority: 'high', 'medium', or 'low'
            - signals: list of positive signals found
            - recommendations: suggested approach
        """
        breakdown: dict[str, float] = {}
        signals: list[str] = []

        # Dimension 1: ICP Fit (weight: 0.30)
        icp_score = self._score_icp_fit(
            company_size, industry, contact_title, signals
        )
        breakdown["icp_fit"] = icp_score

        # Dimension 2: Security Posture (weight: 0.35)
        security_score = self._score_security_posture(
            tech_stack, vulnerabilities, signals
        )
        breakdown["security_posture"] = security_score

        # Dimension 3: Timing (weight: 0.15)
        timing_score = self._score_timing(tech_stack, signals)
        breakdown["timing"] = timing_score

        # Dimension 4: Engagement History (weight: 0.20)
        engagement_score = self._score_engagement(
            previous_outreach or [], signals
        )
        breakdown["engagement"] = engagement_score

        # Weighted total
        total = (
            icp_score * 0.30
            + security_score * 0.35
            + timing_score * 0.15
            + engagement_score * 0.20
        )

        # Priority classification
        if total >= 0.7:
            priority = "high"
        elif total >= 0.4:
            priority = "medium"
        else:
            priority = "low"

        # Approach recommendation
        recommendation = self._recommend_approach(
            breakdown, vulnerabilities, contact_title
        )

        return {
            "total_score": round(total, 3),
            "breakdown": {k: round(v, 3) for k, v in breakdown.items()},
            "priority": priority,
            "signals": signals,
            "recommendation": recommendation,
        }

    def _score_icp_fit(
        self,
        company_size: int,
        industry: str,
        contact_title: str,
        signals: list[str],
    ) -> float:
        """Score ICP fit based on config criteria."""
        score = 0.0
        icp = self.config.targeting.ideal_customer_profile

        # Company size (0.4 of this dimension)
        if icp.company_size[0] <= company_size <= icp.company_size[1]:
            # Sweet spot is 50-200 employees for cybersecurity consulting
            if 50 <= company_size <= 200:
                score += 0.4
                signals.append("ideal_company_size")
            else:
                score += 0.2
                signals.append("acceptable_company_size")

        # Industry match (0.4 of this dimension)
        industry_lower = industry.lower()
        high_value_industries = {"healthcare", "fintech", "financial", "legal"}
        good_industries = {"saas", "ecommerce", "insurance", "technology"}

        if any(ind in industry_lower for ind in high_value_industries):
            score += 0.4
            signals.append(f"high_value_industry_{industry_lower}")
        elif any(ind in industry_lower for ind in good_industries):
            score += 0.25
            signals.append(f"good_industry_{industry_lower}")

        # Persona match (0.2 of this dimension)
        title_lower = contact_title.lower()
        high_value_titles = {"ciso", "cto", "vp engineering", "head of security"}
        decision_makers = {"ceo", "founder", "owner", "managing director"}

        if any(t in title_lower for t in high_value_titles):
            score += 0.2
            signals.append("high_value_persona")
        elif any(t in title_lower for t in decision_makers):
            score += 0.15
            signals.append("decision_maker_persona")

        return min(score, 1.0)

    def _score_security_posture(
        self,
        tech_stack: dict[str, str],
        vulnerabilities: list[dict[str, Any]],
        signals: list[str],
    ) -> float:
        """Score based on detected security findings."""
        score = 0.0

        # Critical vulnerabilities are high-value signals
        critical_count = sum(
            1 for v in vulnerabilities
            if v.get("severity") == "critical"
        )
        high_count = sum(
            1 for v in vulnerabilities
            if v.get("severity") == "high"
        )
        medium_count = sum(
            1 for v in vulnerabilities
            if v.get("severity") == "medium"
        )

        if critical_count > 0:
            score += 0.5
            signals.append(f"critical_vulns_{critical_count}")
        if high_count > 0:
            score += min(high_count * 0.15, 0.3)
            signals.append(f"high_vulns_{high_count}")
        if medium_count > 0:
            score += min(medium_count * 0.05, 0.15)

        # Risky tech stack signals
        risky_tech = {
            "wordpress": 0.1,
            "drupal": 0.05,
            "php": 0.05,
            "apache": 0.03,
            "ftp": 0.1,
            "telnet": 0.15,
        }
        for tech_name in tech_stack:
            for risky, value in risky_tech.items():
                if risky in tech_name.lower():
                    score += value
                    signals.append(f"risky_tech_{tech_name}")

        # Missing security headers
        for vuln in vulnerabilities:
            if vuln.get("type") == "missing_security_headers":
                score += 0.1
                signals.append("missing_security_headers")

        return min(score, 1.0)

    def _score_timing(
        self,
        tech_stack: dict[str, str],
        signals: list[str],
    ) -> float:
        """
        Score timing relevance.

        Good timing signals:
        - Company using outdated software versions
        - Compliance deadline approaching (SOC2, HIPAA, PCI)
        - Recent hiring for security roles (indicates awareness)
        """
        score = 0.3  # baseline — cold outreach always has moderate timing

        # Outdated tech detection using known EOL/outdated version thresholds
        outdated_thresholds: dict[str, list[str]] = {
            "php": ["5.", "7.0", "7.1", "7.2", "7.3", "7.4"],
            "wordpress": ["4.", "5.0", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "5.9"],
            "apache": ["2.2", "2.0", "1."],
            "nginx": ["1.18", "1.16", "1.14"],
            "node.js": ["12", "14", "16"],
            "mysql": ["5.5", "5.6", "5.7"],
            "postgresql": ["11", "12", "13"],
            "java": ["8", "11"],
            "python": ["2.", "3.6", "3.7", "3.8"],
            "windows server": ["2012", "2016"],
            "jquery": ["1.", "2."],
            "react": ["16.", "15.", "14."],
            "angular": ["8", "9", "10", "11", "12"],
            "openssl": ["1.0", "1.1.0"],
            "iis": ["7.", "8.", "8.5"],
            "ftp": [],  # Any FTP usage is a signal (should be SFTP)
        }

        outdated_count = 0
        for tech, version in tech_stack.items():
            tech_lower = tech.lower()
            version_str = str(version).lower().strip()

            # FTP presence alone is a signal
            if tech_lower == "ftp":
                outdated_count += 1
                signals.append(f"outdated_{tech_lower}")
                continue

            for known_tech, old_versions in outdated_thresholds.items():
                if known_tech in tech_lower and version_str:
                    if any(version_str.startswith(old) for old in old_versions):
                        outdated_count += 1
                        signals.append(f"outdated_{known_tech}")
                        break

        # Score boost based on outdated tech count
        if outdated_count >= 3:
            score += 0.5  # Multiple outdated technologies — strong timing signal
        elif outdated_count >= 1:
            score += 0.3  # Some outdated tech

        return min(score, 1.0)

    def _score_engagement(
        self,
        previous_outreach: list[dict[str, Any]],
        signals: list[str],
    ) -> float:
        """Score based on prior engagement history."""
        if not previous_outreach:
            return 0.5  # neutral — no prior engagement

        score = 0.3  # base for having history

        for event in previous_outreach:
            metadata = event.get("metadata", {})
            outcome = metadata.get("outcome", "")

            if outcome == "replied_positive":
                score += 0.3
                signals.append("prior_positive_reply")
            elif outcome == "replied_negative":
                score -= 0.2
            elif outcome == "meeting_request":
                score += 0.5
                signals.append("prior_meeting_request")

        return max(min(score, 1.0), 0.0)

    def _recommend_approach(
        self,
        breakdown: dict[str, float],
        vulnerabilities: list[dict[str, Any]],
        contact_title: str,
    ) -> str:
        """Recommend the best outreach approach based on scores."""
        title_lower = contact_title.lower()

        # If we have critical/high vulns, lead with vulnerability approach
        has_critical = any(
            v.get("severity") in ("critical", "high")
            for v in vulnerabilities
        )
        if has_critical and breakdown.get("security_posture", 0) > 0.5:
            return "vulnerability_alert"

        # For C-suite, use business risk framing
        if any(t in title_lower for t in ["ceo", "founder", "owner", "president"]):
            return "business_risk"

        # For CISOs/security directors, use compliance angle
        if any(t in title_lower for t in ["ciso", "security", "compliance"]):
            return "compliance_gap"

        # Default to vulnerability alert (our strongest approach)
        return "vulnerability_alert"
