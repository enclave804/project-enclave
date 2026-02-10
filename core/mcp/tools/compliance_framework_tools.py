"""
Compliance framework MCP tools for the Sovereign Venture Engine.

Provides tools for mapping security findings to regulatory compliance
controls and calculating compliance scores. Used by the Compliance
Mapper agent to generate audit-ready gap analysis reports.

Tools:
    - get_framework_requirements: Retrieve control requirements for a framework
    - map_finding_to_controls: Map a security finding to relevant controls
    - get_compliance_score: Calculate compliance percentage across findings

Supported frameworks:
    - SOC 2 Type II (Trust Services Criteria 2017)
    - HIPAA Security Rule (2013)
    - PCI DSS v4.0
    - ISO 27001:2022

TODO: Replace mock implementations with:
    - Structured framework database (JSON/YAML or Supabase table)
    - LLM-assisted control mapping with confidence scores
    - Integration with GRC platforms (Vanta, Drata, etc.)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Abbreviated control catalogs for mock responses
# In production, these would come from a database or structured files
_FRAMEWORK_CONTROLS = {
    "SOC2": {
        "name": "SOC 2 Type II",
        "version": "2017",
        "categories": {
            "CC6 - Logical and Physical Access Controls": [
                {"id": "CC6.1", "title": "Logical access security", "description": "The entity implements logical access security measures to protect against unauthorized access."},
                {"id": "CC6.2", "title": "User authentication", "description": "Prior to issuing system credentials, the entity registers and authorizes new users."},
                {"id": "CC6.3", "title": "Role-based access", "description": "The entity authorizes, modifies, or removes access based on roles and responsibilities."},
                {"id": "CC6.6", "title": "System boundaries", "description": "The entity implements logical access security measures to manage access at system boundaries."},
                {"id": "CC6.8", "title": "Malicious software prevention", "description": "The entity implements controls to prevent or detect and act upon malicious software."},
            ],
            "CC7 - System Operations": [
                {"id": "CC7.1", "title": "Monitoring", "description": "The entity uses detection and monitoring procedures to identify changes to configurations."},
                {"id": "CC7.2", "title": "Anomaly detection", "description": "The entity monitors system components for anomalies that are indicative of malicious acts."},
                {"id": "CC7.3", "title": "Incident evaluation", "description": "The entity evaluates events to determine whether they constitute security incidents."},
                {"id": "CC7.4", "title": "Incident response", "description": "The entity responds to identified security incidents by executing a defined incident response program."},
            ],
            "CC8 - Change Management": [
                {"id": "CC8.1", "title": "Change authorization", "description": "The entity authorizes, designs, develops, configures, documents, tests, approves, and implements changes."},
            ],
        },
    },
    "HIPAA": {
        "name": "HIPAA Security Rule",
        "version": "2013",
        "categories": {
            "Administrative Safeguards (164.308)": [
                {"id": "164.308(a)(1)", "title": "Security management process", "description": "Implement policies and procedures to prevent, detect, contain, and correct security violations."},
                {"id": "164.308(a)(3)", "title": "Workforce security", "description": "Implement policies and procedures to ensure appropriate access to ePHI by workforce members."},
                {"id": "164.308(a)(5)", "title": "Security awareness training", "description": "Implement a security awareness and training program for all workforce members."},
                {"id": "164.308(a)(6)", "title": "Security incident procedures", "description": "Implement policies and procedures to address security incidents."},
                {"id": "164.308(a)(7)", "title": "Contingency plan", "description": "Establish policies and procedures for responding to emergency or disaster."},
            ],
            "Technical Safeguards (164.312)": [
                {"id": "164.312(a)(1)", "title": "Access control", "description": "Implement technical policies for electronic systems that maintain ePHI."},
                {"id": "164.312(b)", "title": "Audit controls", "description": "Implement hardware, software, and procedural mechanisms to record and examine activity."},
                {"id": "164.312(c)(1)", "title": "Integrity", "description": "Implement policies to protect ePHI from improper alteration or destruction."},
                {"id": "164.312(d)", "title": "Person authentication", "description": "Implement procedures to verify the identity of persons seeking access to ePHI."},
                {"id": "164.312(e)(1)", "title": "Transmission security", "description": "Implement technical measures to guard against unauthorized access to ePHI during transmission."},
            ],
        },
    },
    "PCI": {
        "name": "PCI DSS",
        "version": "4.0",
        "categories": {
            "Requirement 1 - Network Security Controls": [
                {"id": "1.2.1", "title": "Network segmentation", "description": "Configuration standards for network security controls are defined and maintained."},
            ],
            "Requirement 6 - Secure Systems and Software": [
                {"id": "6.2.4", "title": "Secure coding", "description": "Software engineering techniques prevent common coding vulnerabilities."},
                {"id": "6.4.1", "title": "Web application protection", "description": "Public-facing web applications are protected against attacks."},
            ],
            "Requirement 8 - Strong Authentication": [
                {"id": "8.3.6", "title": "Password complexity", "description": "Authentication credentials meet minimum complexity requirements."},
                {"id": "8.4.2", "title": "MFA for remote access", "description": "MFA is implemented for all remote network access."},
            ],
        },
    },
    "ISO27001": {
        "name": "ISO 27001",
        "version": "2022",
        "categories": {
            "A.5 - Organizational Controls": [
                {"id": "A.5.1", "title": "Information security policies", "description": "A set of policies for information security shall be defined and approved."},
            ],
            "A.8 - Technological Controls": [
                {"id": "A.8.1", "title": "User endpoint devices", "description": "Information stored on, processed by or accessible via user endpoint devices shall be protected."},
                {"id": "A.8.5", "title": "Secure authentication", "description": "Secure authentication technologies and procedures shall be established."},
                {"id": "A.8.9", "title": "Configuration management", "description": "Configurations, including security configurations, shall be established and managed."},
                {"id": "A.8.15", "title": "Logging", "description": "Logs that record activities, exceptions, faults and other relevant events shall be produced and stored."},
                {"id": "A.8.24", "title": "Use of cryptography", "description": "Rules for effective use of cryptography shall be defined and implemented."},
            ],
        },
    },
}


async def get_framework_requirements(
    framework: str,
    category: Optional[str] = None,
) -> str:
    """
    Retrieve control requirements for a compliance framework.

    Returns the full control catalog (or a specific category) for
    supported frameworks: SOC2, HIPAA, PCI, ISO27001.

    Args:
        framework: Framework identifier (SOC2, HIPAA, PCI, ISO27001).
        category: Optional category filter to return only one section.

    Returns:
        JSON string with framework controls and metadata.
    """
    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "get_framework_requirements",
            "framework": framework,
            "category": category,
        },
    )

    fw_key = framework.upper().replace(" ", "").replace("-", "")
    fw_data = _FRAMEWORK_CONTROLS.get(fw_key)

    if fw_data is None:
        return json.dumps({
            "status": "error",
            "error": f"Unknown framework: {framework}",
            "supported_frameworks": list(_FRAMEWORK_CONTROLS.keys()),
        })

    categories = fw_data["categories"]
    if category:
        categories = {
            k: v for k, v in categories.items()
            if category.lower() in k.lower()
        }

    total_controls = sum(len(v) for v in categories.values())

    result = {
        "status": "success",
        "framework": fw_data["name"],
        "version": fw_data["version"],
        "categories": categories,
        "total_controls": total_controls,
        "category_count": len(categories),
    }

    return json.dumps(result, indent=2)


async def map_finding_to_controls(
    finding: dict[str, Any],
    framework: str,
) -> str:
    """
    Map a security finding to relevant compliance framework controls.

    Uses finding metadata (type, severity, affected component) to
    identify which framework controls are impacted.

    Args:
        finding: A security finding dict with keys:
            - title (str): Finding title
            - severity (str): critical/high/medium/low
            - category (str): e.g. "ssl", "headers", "iam", "network"
            - detail (str): Finding description
        framework: Framework to map against (SOC2, HIPAA, PCI, ISO27001).

    Returns:
        JSON string with mapped controls and gap classification.
    """
    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "map_finding_to_controls",
            "finding_title": finding.get("title", "")[:80],
            "framework": framework,
        },
    )

    # TODO: Replace with LLM-assisted or rules-based control mapping
    # Real implementation would:
    # 1. Analyze finding category and keywords
    # 2. Match against control descriptions using semantic similarity
    # 3. Return controls with confidence scores

    fw_key = framework.upper().replace(" ", "").replace("-", "")
    fw_data = _FRAMEWORK_CONTROLS.get(fw_key)

    if fw_data is None:
        return json.dumps({
            "status": "error",
            "error": f"Unknown framework: {framework}",
        })

    # Simple category-based mock mapping
    category = finding.get("category", "").lower()
    severity = finding.get("severity", "medium")

    # Map finding categories to likely control areas
    category_mapping = {
        "ssl": ["access control", "cryptography", "transmission"],
        "headers": ["web application", "secure systems", "configuration"],
        "iam": ["access control", "authentication", "workforce"],
        "network": ["network", "segmentation", "boundaries"],
        "encryption": ["cryptography", "integrity", "transmission"],
        "logging": ["audit", "monitoring", "logging"],
        "incident": ["incident", "contingency", "response"],
    }

    keywords = category_mapping.get(category, [category])

    mapped_controls = []
    for cat_name, controls in fw_data["categories"].items():
        for control in controls:
            desc_lower = (control["description"] + " " + control["title"]).lower()
            if any(kw in desc_lower for kw in keywords):
                mapped_controls.append({
                    "control_id": control["id"],
                    "control_title": control["title"],
                    "category": cat_name,
                    "gap_status": "non_compliant" if severity in ("critical", "high") else "partial",
                    "confidence": 0.85,
                })

    result = {
        "status": "success",
        "finding": {
            "title": finding.get("title", ""),
            "severity": severity,
            "category": category,
        },
        "framework": fw_data["name"],
        "mapped_controls": mapped_controls,
        "controls_affected": len(mapped_controls),
    }

    return json.dumps(result, indent=2)


async def get_compliance_score(
    findings: list[dict[str, Any]],
    framework: str,
) -> str:
    """
    Calculate overall compliance percentage for a framework based on findings.

    Aggregates all findings, maps them to controls, and calculates
    the percentage of controls that are compliant, partially compliant,
    or non-compliant.

    Args:
        findings: List of finding dicts (same format as map_finding_to_controls).
        framework: Framework to score against (SOC2, HIPAA, PCI, ISO27001).

    Returns:
        JSON string with compliance score, breakdown, and status.
    """
    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "get_compliance_score",
            "framework": framework,
            "finding_count": len(findings),
        },
    )

    fw_key = framework.upper().replace(" ", "").replace("-", "")
    fw_data = _FRAMEWORK_CONTROLS.get(fw_key)

    if fw_data is None:
        return json.dumps({
            "status": "error",
            "error": f"Unknown framework: {framework}",
        })

    # TODO: Replace with real scoring based on actual control mapping
    # Real implementation would:
    # 1. Map each finding to controls (call map_finding_to_controls)
    # 2. Track unique controls affected
    # 3. Calculate compliant vs non-compliant vs partial

    total_controls = sum(
        len(controls) for controls in fw_data["categories"].values()
    )

    # Mock scoring based on finding severity distribution
    critical_count = sum(1 for f in findings if f.get("severity") == "critical")
    high_count = sum(1 for f in findings if f.get("severity") == "high")
    medium_count = sum(1 for f in findings if f.get("severity") == "medium")
    low_count = sum(1 for f in findings if f.get("severity") == "low")

    # Estimate controls affected (rough heuristic for mock)
    non_compliant = min(critical_count + high_count, total_controls)
    partial = min(medium_count, total_controls - non_compliant)
    compliant = max(total_controls - non_compliant - partial, 0)

    score = round((compliant / total_controls) * 100) if total_controls > 0 else 0

    if score >= 80:
        status = "passing"
    elif score >= 60:
        status = "warning"
    else:
        status = "failing"

    result = {
        "status": "success",
        "framework": fw_data["name"],
        "version": fw_data["version"],
        "score": {
            "percentage": score,
            "status": status,
            "compliant": compliant,
            "partial": partial,
            "non_compliant": non_compliant,
            "not_assessed": 0,
            "total_controls": total_controls,
        },
        "finding_summary": {
            "total": len(findings),
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
        },
        "recommendation": (
            "Critical and high-severity findings must be addressed before "
            "achieving compliance. Focus on remediation of access control "
            "and encryption gaps first."
            if status == "failing"
            else "Good progress toward compliance. Address remaining partial "
            "controls to strengthen posture."
        ),
    }

    return json.dumps(result, indent=2)
