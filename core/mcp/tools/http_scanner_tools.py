"""
HTTP security scanner MCP tools for the Sovereign Venture Engine.

Provides tools for analyzing web application security headers, CORS
policies, cookie configurations, and Content Security Policy. Used
by the AppSec Reviewer agent for non-intrusive external assessment.

Tools:
    - scan_security_headers: Check for presence and strength of security headers
    - check_cors_policy: Analyze CORS configuration for over-permissiveness
    - analyze_cookie_security: Audit cookie flags (Secure, HttpOnly, SameSite)
    - check_csp_policy: Parse and grade Content Security Policy

TODO: Replace mock implementations with real HTTP scanning:
    - httpx for async HTTP requests with header inspection
    - Custom CSP parser for directive-level analysis
    - Cookie jar inspection via response headers
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Expected security headers and their purpose
SECURITY_HEADERS = {
    "Strict-Transport-Security": {
        "description": "Enforces HTTPS connections",
        "recommended": "max-age=31536000; includeSubDomains; preload",
    },
    "Content-Security-Policy": {
        "description": "Controls resource loading to prevent XSS",
        "recommended": "default-src 'self'",
    },
    "X-Content-Type-Options": {
        "description": "Prevents MIME type sniffing",
        "recommended": "nosniff",
    },
    "X-Frame-Options": {
        "description": "Prevents clickjacking via iframes",
        "recommended": "DENY",
    },
    "Referrer-Policy": {
        "description": "Controls referrer information leakage",
        "recommended": "strict-origin-when-cross-origin",
    },
    "Permissions-Policy": {
        "description": "Controls browser feature access",
        "recommended": "camera=(), microphone=(), geolocation=()",
    },
    "X-XSS-Protection": {
        "description": "Legacy XSS filter (deprecated but still checked)",
        "recommended": "0",
    },
}


async def scan_security_headers(
    url: str,
    follow_redirects: bool = True,
) -> str:
    """
    Check for the presence and correctness of HTTP security headers.

    Scans a URL for standard security headers (HSTS, CSP, X-Frame-Options,
    X-Content-Type-Options, Referrer-Policy, Permissions-Policy) and grades
    each one as present/missing/misconfigured.

    Args:
        url: The URL to scan (e.g. "https://example.com").
        follow_redirects: Whether to follow redirects (default: True).

    Returns:
        JSON string with header analysis and overall score.
    """
    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "scan_security_headers",
            "url": url,
        },
    )

    # TODO: Replace with real HTTP request via httpx
    # Real implementation would:
    # 1. Send GET request to URL
    # 2. Extract all response headers
    # 3. Compare against SECURITY_HEADERS expectations
    # 4. Score each header (present + correct = pass)

    mock_present = {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    mock_missing = ["Content-Security-Policy", "Permissions-Policy"]

    headers_analysis = {}
    for header, meta in SECURITY_HEADERS.items():
        if header in mock_present:
            headers_analysis[header] = {
                "present": True,
                "value": mock_present[header],
                "recommended": meta["recommended"],
                "status": "pass",
                "description": meta["description"],
            }
        else:
            headers_analysis[header] = {
                "present": False,
                "value": None,
                "recommended": meta["recommended"],
                "status": "fail",
                "description": meta["description"],
            }

    present_count = sum(1 for h in headers_analysis.values() if h["present"])
    total_count = len(headers_analysis)
    score = round((present_count / total_count) * 100)

    result = {
        "status": "success",
        "url": url,
        "headers": headers_analysis,
        "summary": {
            "present": present_count,
            "missing": total_count - present_count,
            "total": total_count,
            "score": score,
            "grade": "B" if score >= 70 else "C" if score >= 50 else "F",
        },
        "missing_headers": mock_missing,
        "findings": [
            {
                "severity": "medium",
                "title": f"Missing {h} header",
                "detail": SECURITY_HEADERS[h]["description"],
            }
            for h in mock_missing
        ],
    }

    return json.dumps(result, indent=2)


async def check_cors_policy(
    url: str,
    test_origins: Optional[list[str]] = None,
) -> str:
    """
    Analyze CORS configuration for over-permissive or dangerous settings.

    Sends preflight-style requests to check Access-Control-Allow-Origin,
    credentials handling, and allowed methods/headers.

    Args:
        url: The URL to check (e.g. "https://api.example.com").
        test_origins: List of origins to test against (default: common test set).

    Returns:
        JSON string with CORS policy analysis and risk assessment.
    """
    if test_origins is None:
        test_origins = [
            "https://evil.com",
            "https://attacker.example.com",
            "null",
        ]

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "check_cors_policy",
            "url": url,
            "test_origin_count": len(test_origins),
        },
    )

    # TODO: Replace with real CORS probing via httpx OPTIONS requests
    # Real implementation would:
    # 1. Send OPTIONS request with each test origin
    # 2. Check Access-Control-Allow-Origin response
    # 3. Check Access-Control-Allow-Credentials
    # 4. Check Access-Control-Allow-Methods breadth

    result = {
        "status": "success",
        "url": url,
        "cors_enabled": True,
        "policy": {
            "allow_origin": "https://app.example.com",
            "allow_credentials": False,
            "allow_methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "max_age": 3600,
        },
        "wildcard_origin": False,
        "reflects_origin": False,
        "null_origin_allowed": False,
        "credentials_with_wildcard": False,
        "test_results": [
            {"origin": origin, "reflected": False, "allowed": False}
            for origin in test_origins
        ],
        "risk_level": "low",
        "findings": [],
    }

    return json.dumps(result, indent=2)


async def analyze_cookie_security(
    url: str,
) -> str:
    """
    Audit cookie security flags for a URL's response cookies.

    Checks each cookie for Secure, HttpOnly, SameSite attributes and
    identifies cookies that may be vulnerable to theft or manipulation.

    Args:
        url: The URL to analyze (e.g. "https://example.com").

    Returns:
        JSON string with per-cookie analysis and risk findings.
    """
    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "analyze_cookie_security",
            "url": url,
        },
    )

    # TODO: Replace with real cookie extraction via httpx
    # Real implementation would parse Set-Cookie headers from the response

    mock_cookies = [
        {
            "name": "session_id",
            "secure": True,
            "httponly": True,
            "samesite": "Strict",
            "path": "/",
            "domain": None,
            "expires": "session",
            "issues": [],
        },
        {
            "name": "_analytics",
            "secure": True,
            "httponly": False,
            "samesite": "Lax",
            "path": "/",
            "domain": None,
            "expires": "365 days",
            "issues": ["Missing HttpOnly flag — accessible via JavaScript"],
        },
    ]

    total = len(mock_cookies)
    secure_count = sum(1 for c in mock_cookies if c["secure"])
    httponly_count = sum(1 for c in mock_cookies if c["httponly"])
    samesite_count = sum(1 for c in mock_cookies if c["samesite"])

    result = {
        "status": "success",
        "url": url,
        "cookies": mock_cookies,
        "summary": {
            "total_cookies": total,
            "secure_flag": f"{secure_count}/{total}",
            "httponly_flag": f"{httponly_count}/{total}",
            "samesite_flag": f"{samesite_count}/{total}",
        },
        "findings": [
            {
                "severity": "low",
                "title": f"Cookie '{c['name']}' — {issue}",
                "detail": issue,
            }
            for c in mock_cookies
            for issue in c["issues"]
        ],
    }

    return json.dumps(result, indent=2)


async def check_csp_policy(
    url: str,
) -> str:
    """
    Parse and grade the Content Security Policy for a URL.

    Analyzes CSP directives for completeness, unsafe-inline usage,
    wildcard sources, and overall policy strength.

    Args:
        url: The URL to check (e.g. "https://example.com").

    Returns:
        JSON string with directive-level analysis and grade.
    """
    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "check_csp_policy",
            "url": url,
        },
    )

    # TODO: Replace with real CSP header extraction and parsing
    # Real implementation would:
    # 1. Fetch the page and extract CSP header (or meta tag)
    # 2. Parse each directive
    # 3. Flag unsafe-inline, unsafe-eval, wildcard sources
    # 4. Check for missing directives (default-src, script-src, etc.)

    mock_csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'"

    result = {
        "status": "success",
        "url": url,
        "csp_present": True,
        "raw_policy": mock_csp,
        "directives": {
            "default-src": {"value": "'self'", "issues": []},
            "script-src": {
                "value": "'self' 'unsafe-inline'",
                "issues": ["'unsafe-inline' allows inline scripts — XSS risk"],
            },
            "style-src": {
                "value": "'self' 'unsafe-inline'",
                "issues": ["'unsafe-inline' allows inline styles"],
            },
            "img-src": {"value": "'self' data:", "issues": []},
            "font-src": {"value": "'self'", "issues": []},
        },
        "missing_directives": [
            "frame-ancestors",
            "base-uri",
            "form-action",
            "object-src",
        ],
        "uses_unsafe_inline": True,
        "uses_unsafe_eval": False,
        "uses_wildcard": False,
        "grade": "C",
        "findings": [
            {
                "severity": "medium",
                "title": "CSP allows 'unsafe-inline' in script-src",
                "detail": "Inline scripts are permitted, reducing XSS protection.",
            },
            {
                "severity": "low",
                "title": "Missing frame-ancestors directive",
                "detail": "Consider adding frame-ancestors to prevent clickjacking.",
            },
        ],
    }

    return json.dumps(result, indent=2)
