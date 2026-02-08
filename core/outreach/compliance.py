"""
Email compliance engine for Project Enclave.

Ensures all outreach complies with CAN-SPAM, GDPR, and other
email marketing regulations. This is the legal safety net.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional

from core.config.schema import ComplianceConfig, Jurisdiction
from core.integrations.supabase_client import EnclaveDB

logger = logging.getLogger(__name__)

# EU member state TLDs (for basic GDPR detection)
EU_TLDS = {
    "at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr",
    "de", "gr", "hu", "ie", "it", "lv", "lt", "lu", "mt", "nl",
    "pl", "pt", "ro", "sk", "si", "es", "se", "eu",
}

# UK TLD
UK_TLDS = {"uk", "gb"}


class ComplianceChecker:
    """
    Checks outreach emails for legal compliance.

    Validates against:
    - CAN-SPAM Act (US)
    - GDPR (EU)
    - Suppression list (do-not-contact)
    - Country exclusions
    - Content requirements
    """

    def __init__(
        self,
        config: ComplianceConfig,
        db: EnclaveDB,
    ):
        self.config = config
        self.db = db
        self._file_suppress_list: set[str] = set()
        self._load_file_suppress_list()

    def _load_file_suppress_list(self) -> None:
        """Load suppression list from CSV file if configured."""
        if not self.config.suppress_list_path:
            return

        path = Path(self.config.suppress_list_path)
        if not path.exists():
            logger.info(
                f"Suppression list file not found: {path} (will use DB only)"
            )
            return

        try:
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        self._file_suppress_list.add(row[0].strip().lower())
            logger.info(
                f"Loaded {len(self._file_suppress_list)} entries "
                f"from suppression list file"
            )
        except Exception as e:
            logger.error(f"Failed to load suppression list: {e}")

    def check(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
    ) -> ComplianceResult:
        """
        Run all compliance checks on an outreach email.

        Args:
            to_email: Recipient email address.
            subject: Email subject line.
            body: Email body text.
            from_email: Sender email address.

        Returns:
            ComplianceResult with pass/fail status and issues list.
        """
        issues: list[str] = []

        # 1. Suppression list check (DB + file)
        if self._is_suppressed(to_email):
            issues.append(
                f"BLOCKED: {to_email} is on the suppression list"
            )

        # 2. Country/jurisdiction exclusion
        country_issue = self._check_country_exclusion(to_email)
        if country_issue:
            issues.append(country_issue)

        # 3. CAN-SPAM requirements
        if Jurisdiction.US_CAN_SPAM in [
            j.value if hasattr(j, 'value') else j
            for j in self.config.jurisdictions
        ]:
            can_spam_issues = self._check_can_spam(subject, body)
            issues.extend(can_spam_issues)

        # 4. Content checks
        content_issues = self._check_content(subject, body)
        issues.extend(content_issues)

        passed = len(issues) == 0
        return ComplianceResult(passed=passed, issues=issues)

    def _is_suppressed(self, email: str) -> bool:
        """Check if email is on any suppression list."""
        email_lower = email.lower()

        # Check file-based list
        if email_lower in self._file_suppress_list:
            return True

        # Check database list
        return self.db.is_suppressed(email_lower)

    def _check_country_exclusion(self, email: str) -> Optional[str]:
        """Check if recipient is in an excluded country based on email TLD."""
        if not email or "@" not in email:
            return "Invalid email address"

        domain = email.rsplit("@", 1)[1].lower()
        tld = domain.rsplit(".", 1)[-1]

        excluded = {c.lower() for c in self.config.exclude_countries}

        # Check country-code TLDs
        if tld in EU_TLDS and tld.upper() in self.config.exclude_countries:
            return f"BLOCKED: Recipient appears to be in EU ({tld} TLD)"

        if tld in UK_TLDS and tld.upper() in self.config.exclude_countries:
            return f"BLOCKED: Recipient appears to be in UK ({tld} TLD)"

        # Check .eu specifically
        if tld == "eu" and any(c in excluded for c in EU_TLDS):
            return "BLOCKED: Recipient uses .eu domain (GDPR applies)"

        return None

    def _check_can_spam(self, subject: str, body: str) -> list[str]:
        """
        Check CAN-SPAM Act requirements.

        Required elements:
        1. Accurate "From" and "Subject" (we validate subject isn't deceptive)
        2. Physical postal address included
        3. Opt-out/unsubscribe mechanism
        4. Not misleading header information
        """
        issues = []

        # Physical address check
        if self.config.physical_address:
            if self.config.physical_address not in body:
                issues.append(
                    "WARNING: Physical address not found in email body "
                    "(required by CAN-SPAM). Will be added by email engine."
                )

        # Subject line checks
        if not subject:
            issues.append("BLOCKED: Empty subject line")
        elif subject.upper() == subject and len(subject) > 10:
            issues.append("WARNING: Subject line is all caps (spam trigger)")
        elif any(word in subject.lower() for word in [
            "free", "act now", "urgent", "limited time", "guaranteed",
            "no obligation", "winner", "congratulations",
        ]):
            issues.append(
                "WARNING: Subject contains spam trigger words "
                f"('{subject}')"
            )

        return issues

    def _check_content(self, subject: str, body: str) -> list[str]:
        """Check email content for issues."""
        issues = []

        if not body:
            issues.append("BLOCKED: Empty email body")
            return issues

        if len(body) > 5000:
            issues.append(
                f"WARNING: Email body is very long ({len(body)} chars). "
                f"Consider shortening to under 500 chars."
            )

        if len(subject) > 80:
            issues.append(
                f"WARNING: Subject line is {len(subject)} chars "
                f"(recommended max: 80)"
            )

        # Check for potentially problematic claims
        body_lower = body.lower()
        if "we hacked" in body_lower or "we breached" in body_lower:
            issues.append(
                "BLOCKED: Email implies unauthorized access. "
                "Use 'we identified publicly visible' instead."
            )

        if "your data is at risk" in body_lower:
            issues.append(
                "WARNING: Fear-based language may reduce trust. "
                "Consider factual, professional tone."
            )

        return issues

    def add_to_suppression(
        self, email: str, reason: str = "manual"
    ) -> None:
        """Add an email to the suppression list."""
        self.db.add_to_suppression(email.lower(), reason)
        self._file_suppress_list.add(email.lower())

    def process_unsubscribe(self, email: str) -> None:
        """Process an unsubscribe request."""
        self.add_to_suppression(email, reason="unsubscribed")
        logger.info(f"Unsubscribe processed: {email}")

    def process_bounce(self, email: str) -> None:
        """Process a bounced email."""
        self.add_to_suppression(email, reason="bounced")
        logger.info(f"Bounce processed: {email}")


class ComplianceResult:
    """Result of a compliance check."""

    def __init__(self, passed: bool, issues: list[str]):
        self.passed = passed
        self.issues = issues
        self.blocking_issues = [i for i in issues if i.startswith("BLOCKED")]
        self.warnings = [i for i in issues if i.startswith("WARNING")]

    @property
    def has_blocking_issues(self) -> bool:
        return len(self.blocking_issues) > 0

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"ComplianceResult({status}, issues={len(self.issues)})"
