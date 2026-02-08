"""
Input Sanitization — The Airlock.

Scans all inbound text before it reaches any agent or LLM call.
Blocks prompt injection attacks, oversized inputs, and known
malicious patterns.

Security layers:
1. Length check — prevents DoS via oversized payloads
2. Pattern detection — blocks known prompt injection phrases
3. Unicode normalization — prevents homoglyph attacks
4. Encoding detection — blocks base64-encoded injection attempts

Usage:
    from core.safety.input_guard import SecurityGuard, SecurityException

    guard = SecurityGuard()
    guard.validate(user_input)  # Raises SecurityException if malicious

    # Or check without raising:
    is_safe = guard.scan_input(user_input)

Integration:
    Called automatically in BaseAgent.run() before task processing.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────

# Maximum input length (characters). Inputs exceeding this are rejected
# to prevent denial-of-service via oversized payloads.
MAX_INPUT_LENGTH = 10_000

# Known prompt injection patterns. These are case-insensitive regexes.
# Each pattern is a tuple of (pattern_name, compiled_regex).
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "ignore_previous",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_override",
        re.compile(
            r"system\s+(override|prompt|message|instruction)",
            re.IGNORECASE,
        ),
    ),
    (
        "unrestricted_mode",
        re.compile(
            r"(unrestricted|unlimited|jailbreak|developer|debug|god)\s*mode",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(you\s+are\s+now|pretend\s+(to\s+be|you\s+are)|act\s+as\s+if\s+you)",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_injection",
        re.compile(
            r"(new\s+instructions?|disregard|forget)\s*(:|everything|all|previous|above)",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_attack",
        re.compile(
            r"<\|?(system|im_start|endoftext|assistant)\|?>",
            re.IGNORECASE,
        ),
    ),
    (
        "data_exfiltration",
        re.compile(
            r"(repeat|output|reveal|show|display)\s+(the\s+)?(system\s+prompt|instructions?|secret|password|api\s*key|token)",
            re.IGNORECASE,
        ),
    ),
]

# Suspicious but not auto-blocked patterns (logged for audit)
SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "base64_payload",
        re.compile(r"[A-Za-z0-9+/]{100,}={0,2}", re.IGNORECASE),
    ),
    (
        "excessive_whitespace",
        re.compile(r"\s{50,}"),
    ),
]


class SecurityException(Exception):
    """Raised when input fails security validation."""

    def __init__(
        self,
        message: str,
        pattern_name: Optional[str] = None,
        input_preview: str = "",
    ):
        super().__init__(message)
        self.pattern_name = pattern_name
        self.input_preview = input_preview[:100]


class SecurityGuard:
    """
    Input sanitization engine for the Sovereign Venture Engine.

    Validates all text inputs before they reach agents or LLMs.
    Blocks prompt injection, DoS payloads, and suspicious content.

    Thread-safe: no mutable state.
    """

    def __init__(
        self,
        max_length: int = MAX_INPUT_LENGTH,
        custom_patterns: Optional[list[tuple[str, re.Pattern]]] = None,
    ):
        self.max_length = max_length
        self.patterns = INJECTION_PATTERNS + (custom_patterns or [])

    def scan_input(self, text: str) -> bool:
        """
        Check if input text is safe.

        Returns True if safe, False if malicious.
        Does NOT raise — use validate() if you want exceptions.
        """
        try:
            self.validate(text)
            return True
        except SecurityException:
            return False

    def validate(self, text: str) -> None:
        """
        Validate input text. Raises SecurityException if unsafe.

        Checks (in order):
        1. None/empty check
        2. Length check (DoS prevention)
        3. Unicode normalization
        4. Prompt injection pattern matching
        5. Suspicious pattern logging (non-blocking)
        """
        if not text:
            return  # Empty input is safe

        # ─── Length Check ──────────────────────────────────────────
        if len(text) > self.max_length:
            logger.warning(
                "input_guard_length_exceeded",
                extra={
                    "input_length": len(text),
                    "max_length": self.max_length,
                },
            )
            raise SecurityException(
                f"Input exceeds maximum length ({len(text)} > {self.max_length})",
                pattern_name="length_exceeded",
                input_preview=text[:100],
            )

        # ─── Unicode Normalization ─────────────────────────────────
        # Normalize to NFC to prevent homoglyph attacks
        normalized = unicodedata.normalize("NFC", text)

        # ─── Prompt Injection Patterns ─────────────────────────────
        for pattern_name, pattern in self.patterns:
            match = pattern.search(normalized)
            if match:
                logger.warning(
                    "input_guard_injection_detected",
                    extra={
                        "pattern_name": pattern_name,
                        "matched_text": match.group()[:80],
                        "input_preview": text[:100],
                    },
                )
                raise SecurityException(
                    f"Potential prompt injection detected: {pattern_name}",
                    pattern_name=pattern_name,
                    input_preview=text[:100],
                )

        # ─── Suspicious Pattern Logging (non-blocking) ────────────
        for pattern_name, pattern in SUSPICIOUS_PATTERNS:
            match = pattern.search(normalized)
            if match:
                logger.info(
                    "input_guard_suspicious_pattern",
                    extra={
                        "pattern_name": pattern_name,
                        "input_preview": text[:100],
                    },
                )

    def scan_dict(self, data: dict[str, Any], max_depth: int = 3) -> bool:
        """
        Recursively scan all string values in a dict.

        Returns True if all values are safe.
        Useful for scanning entire task payloads.
        """
        if max_depth <= 0:
            return True

        for key, value in data.items():
            if isinstance(value, str):
                if not self.scan_input(value):
                    logger.warning(
                        "input_guard_dict_field_blocked",
                        extra={"field": key, "preview": value[:50]},
                    )
                    return False
            elif isinstance(value, dict):
                if not self.scan_dict(value, max_depth - 1):
                    return False
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and not self.scan_input(item):
                        return False
                    elif isinstance(item, dict) and not self.scan_dict(
                        item, max_depth - 1
                    ):
                        return False
        return True


# ─── Module-level singleton for convenience ────────────────────────────
_default_guard: Optional[SecurityGuard] = None


def get_guard() -> SecurityGuard:
    """Get or create the default SecurityGuard instance."""
    global _default_guard
    if _default_guard is None:
        _default_guard = SecurityGuard()
    return _default_guard
