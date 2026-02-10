"""
Application Security Reviewer Agent — The Code Shield.

Evaluates the application-layer security posture of a target domain
by checking HTTP security headers, CSP, cookies, CORS, TLS config,
and mapping findings against the OWASP Top 10.

Architecture (LangGraph State Machine):
    discover_endpoints -> scan_headers -> check_owasp ->
    human_review -> save_findings -> report -> END

Trigger Events:
    - new_assessment: Triggered for each prospect/client domain
    - scheduled (weekly): Periodic application security monitoring
    - manual: On-demand AppSec review

Shared Brain Integration:
    - Reads: previous AppSec findings, industry header baselines
    - Writes: AppSec scores, header compliance data, OWASP findings

Safety:
    - NEVER performs active exploitation or fuzzing
    - All findings require human_review before persistence
    - Only analyzes publicly accessible HTTP responses
    - Does not send crafted payloads to endpoints

Usage:
    agent = AppSecReviewerAgent(config, db, embedder, llm)
    result = await agent.run({"company_domain": "example.com"})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import AppSecReviewerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

SECURITY_HEADERS_SPEC = {
    "Strict-Transport-Security": {
        "required": True,
        "recommended": "max-age=31536000; includeSubDomains; preload",
        "severity": "high",
    },
    "Content-Security-Policy": {
        "required": True,
        "recommended": "default-src 'self'",
        "severity": "high",
    },
    "X-Content-Type-Options": {
        "required": True,
        "recommended": "nosniff",
        "severity": "medium",
    },
    "X-Frame-Options": {
        "required": True,
        "recommended": "DENY",
        "severity": "medium",
    },
    "Referrer-Policy": {
        "required": True,
        "recommended": "strict-origin-when-cross-origin",
        "severity": "medium",
    },
    "Permissions-Policy": {
        "required": False,
        "recommended": "camera=(), microphone=(), geolocation=()",
        "severity": "low",
    },
    "X-XSS-Protection": {
        "required": False,
        "recommended": "0",
        "severity": "low",
    },
}

OWASP_TOP_10 = [
    {"id": "A01", "name": "Broken Access Control", "severity": "critical"},
    {"id": "A02", "name": "Cryptographic Failures", "severity": "critical"},
    {"id": "A03", "name": "Injection", "severity": "critical"},
    {"id": "A04", "name": "Insecure Design", "severity": "high"},
    {"id": "A05", "name": "Security Misconfiguration", "severity": "high"},
    {"id": "A06", "name": "Vulnerable Components", "severity": "high"},
    {"id": "A07", "name": "Auth Failures", "severity": "critical"},
    {"id": "A08", "name": "Data Integrity Failures", "severity": "high"},
    {"id": "A09", "name": "Logging & Monitoring Failures", "severity": "medium"},
    {"id": "A10", "name": "Server-Side Request Forgery", "severity": "high"},
]

APPSEC_SYSTEM_PROMPT = """\
You are an expert application security reviewer for {company_name}. \
You analyze HTTP headers, TLS configuration, and web application \
security practices against OWASP Top 10 standards.

Given the scan data below for {domain}, produce a JSON object with:
{{
    "appsec_score": 0.0-10.0,
    "owasp_findings": [
        {{
            "category": "A01-A10 OWASP category ID",
            "severity": "critical|high|medium|low",
            "title": "Short finding title",
            "description": "What was found",
            "remediation": "How to fix"
        }}
    ]
}}

Rules:
- Missing HSTS or CSP is always high severity
- Evaluate cookie security (Secure, HttpOnly, SameSite flags)
- TLS < 1.2 is a critical finding
- Consider header interactions (e.g., CSP makes XSS-Protection redundant)
- Focus on actionable findings, not theoretical risks

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("appsec_reviewer")
class AppSecReviewerAgent(BaseAgent):
    """
    Application-layer security assessment agent.

    Nodes:
        1. discover_endpoints  -- Identify web endpoints and technologies
        2. scan_headers        -- Evaluate HTTP security headers and cookies
        3. check_owasp         -- Map findings to OWASP Top 10 categories
        4. human_review        -- Gate: review findings before persistence
        5. save_findings       -- Write to security_assessments + security_findings
        6. report              -- Summary + write insights to Hive Mind
    """

    def build_graph(self) -> Any:
        """Build the AppSec Reviewer's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(AppSecReviewerAgentState)

        workflow.add_node("discover_endpoints", self._node_discover_endpoints)
        workflow.add_node("scan_headers", self._node_scan_headers)
        workflow.add_node("check_owasp", self._node_check_owasp)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("save_findings", self._node_save_findings)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("discover_endpoints")

        workflow.add_edge("discover_endpoints", "scan_headers")
        workflow.add_edge("scan_headers", "check_owasp")
        workflow.add_edge("check_owasp", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "save_findings",
                "rejected": "report",
            },
        )
        workflow.add_edge("save_findings", "report")
        workflow.add_edge("report", END)

        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        return self.mcp_tools or []

    @classmethod
    def get_state_class(cls) -> Type[AppSecReviewerAgentState]:
        return AppSecReviewerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "company_domain": task.get("company_domain", ""),
            "company_name": task.get("company_name", ""),
            "endpoints": [],
            "technologies_detected": [],
            "security_headers": {},
            "csp_analysis": {},
            "cors_analysis": {},
            "cookie_analysis": [],
            "owasp_findings": [],
            "tls_config": {},
            "findings": [],
            "appsec_score": 0.0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "assessment_id": "",
            "findings_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Discover Endpoints ─────────────────────────────────

    async def _node_discover_endpoints(
        self, state: AppSecReviewerAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Identify web endpoints and detect technologies.

        Queries cached HTTP response data to discover accessible
        endpoints and server/framework fingerprints.
        """
        domain = state.get("company_domain", "")

        logger.info("appsec_discover: scanning endpoints for %s", domain)

        endpoints: list[dict[str, Any]] = []
        technologies: list[str] = []

        try:
            result = (
                self.db.client.table("tech_stack_cache")
                .select("headers, technologies, status_code")
                .eq("domain", domain)
                .limit(1)
                .execute()
            )
            if result.data:
                row = result.data[0]
                headers = row.get("headers", {})
                technologies = row.get("technologies", []) or []

                # Primary endpoint
                endpoints.append({
                    "url": f"https://{domain}",
                    "method": "GET",
                    "status_code": row.get("status_code", 200),
                    "content_type": headers.get("Content-Type", ""),
                })

                # Detect server from headers
                server = headers.get("Server", headers.get("server", ""))
                if server and server not in technologies:
                    technologies.append(server)

                powered_by = headers.get("X-Powered-By", headers.get("x-powered-by", ""))
                if powered_by and powered_by not in technologies:
                    technologies.append(powered_by)
        except Exception as e:
            logger.debug(f"Endpoint discovery failed for {domain}: {e}")

        if not endpoints:
            endpoints.append({
                "url": f"https://{domain}",
                "method": "GET",
                "status_code": 0,
                "content_type": "unknown",
            })

        logger.info(
            "appsec_discover_complete: %d endpoints, %d technologies",
            len(endpoints), len(technologies),
        )

        return {
            "current_node": "discover_endpoints",
            "endpoints": endpoints,
            "technologies_detected": technologies,
        }

    # ─── Node 2: Scan Headers ──────────────────────────────────────

    async def _node_scan_headers(
        self, state: AppSecReviewerAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Evaluate HTTP security headers, cookies, CSP, and CORS.

        Checks each required security header against best practices,
        analyzes Content-Security-Policy directives, and evaluates
        cookie security attributes.
        """
        domain = state.get("company_domain", "")

        logger.info("appsec_headers: scanning headers for %s", domain)

        security_headers: dict[str, Any] = {}
        csp_analysis: dict[str, Any] = {}
        cors_analysis: dict[str, Any] = {}
        cookie_analysis: list[dict[str, Any]] = []
        tls_config: dict[str, Any] = {}

        raw_headers: dict[str, str] = {}
        try:
            result = (
                self.db.client.table("tech_stack_cache")
                .select("headers, ssl_data, cookies")
                .eq("domain", domain)
                .limit(1)
                .execute()
            )
            if result.data:
                row = result.data[0]
                raw_headers = row.get("headers", {})

                # TLS config
                ssl_data = row.get("ssl_data", {})
                tls_config = {
                    "protocol_version": ssl_data.get("protocol_version", "unknown"),
                    "cipher_suite": ssl_data.get("cipher_suite", "unknown"),
                    "cert_issuer": ssl_data.get("issuer", "unknown"),
                    "cert_expiry": ssl_data.get("expires", ""),
                    "supports_tls_1_0": ssl_data.get("supports_tls_1_0", False),
                    "supports_tls_1_1": ssl_data.get("supports_tls_1_1", False),
                }

                # Cookie analysis
                cookies = row.get("cookies", []) or []
                for cookie in cookies:
                    cookie_analysis.append({
                        "name": cookie.get("name", ""),
                        "secure": cookie.get("secure", False),
                        "httponly": cookie.get("httponly", False),
                        "samesite": cookie.get("samesite", "None"),
                    })
        except Exception as e:
            logger.debug(f"Header scan failed for {domain}: {e}")

        # Normalize header keys for case-insensitive lookup
        lower_headers = {k.lower(): v for k, v in raw_headers.items()}

        # Check each security header
        for header_name, spec in SECURITY_HEADERS_SPEC.items():
            present = header_name.lower() in lower_headers
            value = lower_headers.get(header_name.lower(), "")
            compliant = present  # Simplified compliance check

            if present and header_name == "X-Content-Type-Options":
                compliant = value.lower() == "nosniff"
            elif present and header_name == "X-Frame-Options":
                compliant = value.upper() in ("DENY", "SAMEORIGIN")

            security_headers[header_name] = {
                "present": present,
                "value": value,
                "compliant": compliant,
                "required": spec["required"],
                "severity": spec["severity"],
            }

        # CSP analysis
        csp_value = lower_headers.get("content-security-policy", "")
        if csp_value:
            has_unsafe_inline = "'unsafe-inline'" in csp_value
            has_unsafe_eval = "'unsafe-eval'" in csp_value
            csp_analysis = {
                "present": True,
                "raw_value": csp_value[:500],
                "has_default_src": "default-src" in csp_value,
                "has_script_src": "script-src" in csp_value,
                "allows_unsafe_inline": has_unsafe_inline,
                "allows_unsafe_eval": has_unsafe_eval,
                "grade": "weak" if (has_unsafe_inline or has_unsafe_eval) else "strong",
            }
        else:
            csp_analysis = {"present": False, "grade": "missing"}

        # CORS analysis
        cors_origin = lower_headers.get("access-control-allow-origin", "")
        cors_analysis = {
            "present": bool(cors_origin),
            "allow_origin": cors_origin,
            "allows_any_origin": cors_origin == "*",
            "allows_credentials": lower_headers.get(
                "access-control-allow-credentials", ""
            ).lower() == "true",
        }

        logger.info(
            "appsec_headers_complete: %d headers checked, csp=%s, cors=%s",
            len(security_headers),
            csp_analysis.get("grade", "n/a"),
            "open" if cors_analysis.get("allows_any_origin") else "restricted",
        )

        return {
            "current_node": "scan_headers",
            "security_headers": security_headers,
            "csp_analysis": csp_analysis,
            "cors_analysis": cors_analysis,
            "cookie_analysis": cookie_analysis,
            "tls_config": tls_config,
        }

    # ─── Node 3: Check OWASP ──────────────────────────────────────

    async def _node_check_owasp(
        self, state: AppSecReviewerAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Map findings to OWASP Top 10 categories.

        Uses rule-based checks plus LLM analysis to evaluate the
        application against OWASP Top 10 vulnerability categories.
        """
        domain = state.get("company_domain", "")
        security_headers = state.get("security_headers", {})
        csp = state.get("csp_analysis", {})
        cors = state.get("cors_analysis", {})
        cookies = state.get("cookie_analysis", [])
        tls = state.get("tls_config", {})

        logger.info("appsec_owasp: checking OWASP Top 10 for %s", domain)

        all_findings: list[dict[str, Any]] = []

        # --- Rule-based header findings ---
        for header_name, info in security_headers.items():
            if info.get("required") and not info.get("present"):
                all_findings.append({
                    "category": "A05",
                    "severity": info.get("severity", "medium"),
                    "title": f"Missing {header_name} Header",
                    "description": f"The {header_name} header is not set.",
                    "remediation": f"Add {header_name}: {SECURITY_HEADERS_SPEC.get(header_name, {}).get('recommended', '')}",
                })

        # --- CSP findings ---
        if not csp.get("present"):
            all_findings.append({
                "category": "A05",
                "severity": "high",
                "title": "No Content Security Policy",
                "description": "No CSP header found, increasing risk of XSS attacks.",
                "remediation": "Implement a Content-Security-Policy header with restrictive defaults.",
            })
        elif csp.get("allows_unsafe_inline"):
            all_findings.append({
                "category": "A03",
                "severity": "medium",
                "title": "CSP Allows unsafe-inline",
                "description": "CSP permits inline scripts, weakening XSS protection.",
                "remediation": "Remove 'unsafe-inline' from CSP and use nonces or hashes.",
            })

        # --- CORS findings ---
        if cors.get("allows_any_origin") and cors.get("allows_credentials"):
            all_findings.append({
                "category": "A01",
                "severity": "critical",
                "title": "Dangerous CORS Configuration",
                "description": "CORS allows any origin with credentials, enabling cross-origin attacks.",
                "remediation": "Restrict Access-Control-Allow-Origin to trusted domains.",
            })

        # --- Cookie findings ---
        for cookie in cookies:
            issues = []
            if not cookie.get("secure"):
                issues.append("missing Secure flag")
            if not cookie.get("httponly"):
                issues.append("missing HttpOnly flag")
            if cookie.get("samesite", "").lower() == "none":
                issues.append("SameSite=None")
            if issues:
                all_findings.append({
                    "category": "A02",
                    "severity": "medium",
                    "title": f"Insecure Cookie: {cookie.get('name', 'unnamed')}",
                    "description": f"Cookie has: {', '.join(issues)}.",
                    "remediation": "Set Secure, HttpOnly, and SameSite=Strict on all cookies.",
                })

        # --- TLS findings ---
        if tls.get("supports_tls_1_0"):
            all_findings.append({
                "category": "A02",
                "severity": "high",
                "title": "TLS 1.0 Supported",
                "description": "Server supports TLS 1.0 which has known vulnerabilities.",
                "remediation": "Disable TLS 1.0 and 1.1, enforce TLS 1.2 minimum.",
            })

        # --- LLM OWASP analysis ---
        company_name = state.get("company_name", "") or self.config.params.get("company_name", "the target")
        try:
            scan_data = json.dumps({
                "headers": {k: v for k, v in security_headers.items()},
                "csp": csp,
                "cors": cors,
                "tls": tls,
                "cookie_count": len(cookies),
                "technologies": state.get("technologies_detected", []),
            }, indent=2)

            prompt = (
                f"Application security data for {domain}:\n{scan_data}\n\n"
                f"Rule-based findings so far: {len(all_findings)}\n\n"
                f"Generate the AppSec assessment JSON."
            )
            system = APPSEC_SYSTEM_PROMPT.format(
                company_name=company_name, domain=domain,
            )

            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
                system=system,
            )
            text = response.content[0].text.strip() if response.content else ""
            if text:
                try:
                    parsed = json.loads(text)
                    owasp_llm = parsed.get("owasp_findings", [])
                    # Merge LLM findings, avoiding duplicates by title
                    existing_titles = {f.get("title", "").lower() for f in all_findings}
                    for f in owasp_llm:
                        if f.get("title", "").lower() not in existing_titles:
                            all_findings.append(f)
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception as e:
            logger.debug(f"LLM OWASP analysis failed: {e}")

        # Compute AppSec score
        severity_weights = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.3}
        penalty = sum(
            severity_weights.get(f.get("severity", "medium"), 1.0)
            for f in all_findings
        )
        appsec_score = max(0.0, min(10.0, round(penalty, 1)))

        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in all_findings:
            sev = f.get("severity", "medium")
            if sev in counts:
                counts[sev] += 1

        logger.info(
            "appsec_owasp_complete: %d findings, score=%.1f",
            len(all_findings), appsec_score,
        )

        return {
            "current_node": "check_owasp",
            "owasp_findings": [f for f in all_findings if f.get("category", "").startswith("A")],
            "findings": all_findings,
            "appsec_score": appsec_score,
            "critical_count": counts["critical"],
            "high_count": counts["high"],
            "medium_count": counts["medium"],
            "low_count": counts["low"],
        }

    # ─── Node 4: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: AppSecReviewerAgentState
    ) -> dict[str, Any]:
        """Node 4: Present AppSec findings for human approval."""
        logger.info(
            "appsec_human_review_pending: %d findings, score=%.1f",
            len(state.get("findings", [])),
            state.get("appsec_score", 0.0),
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Save Findings ────────────────────────────────────

    async def _node_save_findings(
        self, state: AppSecReviewerAgentState
    ) -> dict[str, Any]:
        """Node 5: Persist approved AppSec findings to the database."""
        domain = state.get("company_domain", "")
        findings = state.get("findings", [])

        logger.info("appsec_save: saving %d findings for %s", len(findings), domain)

        assessment_id = ""
        now = datetime.now(timezone.utc).isoformat()

        try:
            result = self.db.client.table("security_assessments").insert({
                "vertical_id": self.vertical_id,
                "domain": domain,
                "assessment_type": "appsec_review",
                "risk_score": state.get("appsec_score", 0.0),
                "executive_summary": f"AppSec review: {len(findings)} findings.",
                "finding_count": len(findings),
                "critical_count": state.get("critical_count", 0),
                "high_count": state.get("high_count", 0),
                "medium_count": state.get("medium_count", 0),
                "low_count": state.get("low_count", 0),
                "status": "completed",
                "created_at": now,
            }).execute()
            if result.data:
                assessment_id = result.data[0].get("id", "")
        except Exception as e:
            logger.debug(f"Failed to create AppSec assessment: {e}")

        if assessment_id and findings:
            for finding in findings:
                try:
                    self.db.client.table("security_findings").insert({
                        "assessment_id": assessment_id,
                        "vertical_id": self.vertical_id,
                        "domain": domain,
                        "severity": finding.get("severity", "medium"),
                        "title": finding.get("title", ""),
                        "description": finding.get("description", ""),
                        "remediation": finding.get("remediation", ""),
                        "owasp_category": finding.get("category", ""),
                        "status": "open",
                        "created_at": now,
                    }).execute()
                except Exception as e:
                    logger.debug(f"Failed to save AppSec finding: {e}")

        if findings:
            self.store_insight(InsightData(
                insight_type="appsec_assessment",
                title=f"AppSec: {domain} — Score {state.get('appsec_score', 0.0):.1f}/10",
                content=(
                    f"AppSec review of {domain}: {len(findings)} findings "
                    f"({state.get('critical_count', 0)} critical, "
                    f"{state.get('high_count', 0)} high)."
                ),
                confidence=0.85,
                metadata={
                    "domain": domain,
                    "appsec_score": state.get("appsec_score", 0.0),
                    "assessment_id": assessment_id,
                },
            ))

        return {
            "current_node": "save_findings",
            "assessment_id": assessment_id,
            "findings_saved": bool(assessment_id),
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: AppSecReviewerAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate AppSec review summary report."""
        now = datetime.now(timezone.utc).isoformat()
        domain = state.get("company_domain", "")

        sections = [
            "# Application Security Review Report",
            f"**Domain:** {domain}",
            f"*Generated: {now}*\n",
            "## AppSec Score",
            f"**{state.get('appsec_score', 0.0):.1f} / 10.0**\n",
            "## Findings Summary",
            f"- **Critical:** {state.get('critical_count', 0)}",
            f"- **High:** {state.get('high_count', 0)}",
            f"- **Medium:** {state.get('medium_count', 0)}",
            f"- **Low:** {state.get('low_count', 0)}",
            f"- **Total:** {len(state.get('findings', []))}",
        ]

        csp = state.get("csp_analysis", {})
        cors = state.get("cors_analysis", {})
        sections.append("\n## Security Posture")
        sections.append(f"- **CSP:** {'Present' if csp.get('present') else 'Missing'} ({csp.get('grade', 'n/a')})")
        sections.append(f"- **CORS:** {'Open (*)' if cors.get('allows_any_origin') else 'Restricted'}")
        sections.append(f"- **Technologies:** {', '.join(state.get('technologies_detected', [])) or 'N/A'}")

        if state.get("findings_saved"):
            sections.append(f"\n## Persistence\n- Assessment ID: `{state.get('assessment_id', '')}`")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: AppSecReviewerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── Knowledge ───────────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<AppSecReviewerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
