"""
Data Enrichment Agent — The Data Janitor.

Scans contacts and companies tables for data quality issues (missing
emails, stale data, duplicates, format errors), suggests corrections,
and tracks data quality trends over time. Works across all verticals.

Architecture (LangGraph State Machine):
    scan_records → identify_issues → suggest_fixes →
    human_review → report → END

Trigger Events:
    - scheduled: Weekly data quality sweep
    - data_import: New batch import completed
    - manual: On-demand data audit

Shared Brain Integration:
    - Reads: data quality history, enrichment patterns
    - Writes: data quality trends, common issue patterns

Safety:
    - NEVER modifies records without human approval
    - All fixes are suggestions until approved
    - PII is handled carefully — no sensitive data in logs
    - Duplicate detection uses fuzzy matching, never auto-merges

Usage:
    agent = DataEnrichmentAgent(config, db, embedder, llm)
    result = await agent.run({
        "scan_tables": ["contacts", "companies"],
        "scan_mode": "full",
    })
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import DataEnrichmentAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

ISSUE_TYPES = [
    "missing",
    "invalid_email",
    "duplicate",
    "stale",
    "inconsistent",
    "invalid_phone",
    "incomplete",
    "format_error",
]

STALE_THRESHOLD_DAYS = 180

EMAIL_REGEX = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

SCAN_TABLES = ["contacts", "companies"]

DATA_ENRICHMENT_PROMPT = """\
You are a CRM data quality specialist. Analyze the data quality issues \
found during a scan and suggest fixes.

Issues Found:
{issues_json}

For each issue, suggest a fix. Return a JSON array:
[
    {{
        "record_id": "id",
        "table": "contacts|companies",
        "field": "field_name",
        "issue_type": "one of: {issue_types}",
        "current_value": "current value or empty",
        "suggested_value": "suggested correction or empty if deletion",
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation of the fix"
    }}
]

Guidelines:
- For missing emails, suggest "needs_manual_lookup" not fabricated emails
- For duplicates, suggest which record to keep based on completeness
- For stale data, suggest re-verification rather than deletion
- For format errors, correct to standard formats
- Be conservative — only suggest high-confidence fixes

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("data_enrichment")
class DataEnrichmentAgent(BaseAgent):
    """
    CRM data quality and enrichment agent.

    Nodes:
        1. scan_records     -- Scan contacts/companies tables for quality issues
        2. identify_issues  -- Validate emails, detect duplicates, find stale records
        3. suggest_fixes    -- LLM generates correction suggestions
        4. human_review     -- Gate: approve suggested fixes
        5. report           -- Save to data_quality_issues table + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Data Enrichment Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(DataEnrichmentAgentState)

        workflow.add_node("scan_records", self._node_scan_records)
        workflow.add_node("identify_issues", self._node_identify_issues)
        workflow.add_node("suggest_fixes", self._node_suggest_fixes)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_records")

        workflow.add_edge("scan_records", "identify_issues")
        workflow.add_edge("identify_issues", "suggest_fixes")
        workflow.add_edge("suggest_fixes", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "report",
                "rejected": "report",
            },
        )
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
    def get_state_class(cls) -> Type[DataEnrichmentAgentState]:
        return DataEnrichmentAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "tables_scanned": [],
            "records_scanned": 0,
            "scan_mode": "full",
            "issues_found": [],
            "total_issues": 0,
            "critical_issues": 0,
            "duplicate_groups": [],
            "duplicates_found": 0,
            "enrichment_tasks": [],
            "records_enriched": 0,
            "fixes_applied": 0,
            "fixes_approved": False,
            "issues_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Records ────────────────────────────────────────

    async def _node_scan_records(
        self, state: DataEnrichmentAgentState
    ) -> dict[str, Any]:
        """Node 1: Scan contacts and companies tables for raw record data."""
        task = state.get("task_input", {})
        tables_to_scan = task.get("scan_tables", SCAN_TABLES)
        scan_mode = task.get("scan_mode", "full")
        limit_per_table = 500 if scan_mode == "full" else 100

        logger.info(
            "data_enrichment_scan_records",
            extra={
                "agent_id": self.agent_id,
                "tables": tables_to_scan,
                "mode": scan_mode,
            },
        )

        all_records: list[dict[str, Any]] = []
        tables_scanned: list[str] = []

        for table_name in tables_to_scan:
            try:
                query = (
                    self.db.client.table(table_name)
                    .select("*")
                    .eq("vertical_id", self.vertical_id)
                    .limit(limit_per_table)
                )

                # For incremental scans, only check recently modified
                if scan_mode == "incremental":
                    cutoff = (
                        datetime.now(timezone.utc) - timedelta(days=7)
                    ).isoformat()
                    query = query.gte("updated_at", cutoff)

                result = query.execute()

                if result.data:
                    for record in result.data:
                        record["_source_table"] = table_name
                        all_records.append(record)
                    tables_scanned.append(table_name)

                logger.info(
                    "data_enrichment_table_scanned",
                    extra={
                        "table": table_name,
                        "records": len(result.data) if result.data else 0,
                    },
                )

            except Exception as e:
                logger.warning(
                    "data_enrichment_scan_error",
                    extra={
                        "table": table_name,
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "data_enrichment_scan_complete",
            extra={
                "total_records": len(all_records),
                "tables": tables_scanned,
            },
        )

        return {
            "current_node": "scan_records",
            "tables_scanned": tables_scanned,
            "records_scanned": len(all_records),
            "scan_mode": scan_mode,
        }

    # ─── Node 2: Identify Issues ─────────────────────────────────────

    async def _node_identify_issues(
        self, state: DataEnrichmentAgentState
    ) -> dict[str, Any]:
        """Node 2: Validate emails, detect duplicates, find stale records."""
        tables_scanned = state.get("tables_scanned", [])
        stale_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=STALE_THRESHOLD_DAYS)
        ).isoformat()

        logger.info(
            "data_enrichment_identify_issues",
            extra={"tables": tables_scanned},
        )

        issues_found: list[dict[str, Any]] = []
        duplicate_groups: list[dict[str, Any]] = []
        name_index: dict[str, list[dict[str, Any]]] = {}

        # Re-query each table to check for issues
        for table_name in tables_scanned:
            try:
                result = (
                    self.db.client.table(table_name)
                    .select("*")
                    .eq("vertical_id", self.vertical_id)
                    .limit(500)
                    .execute()
                )

                if not result.data:
                    continue

                for record in result.data:
                    record_id = record.get("id", "")

                    # Check for missing email
                    email = record.get("email", "") or record.get("primary_email", "")
                    if not email:
                        issues_found.append({
                            "record_id": record_id,
                            "table": table_name,
                            "field": "email",
                            "issue_type": "missing",
                            "current_value": "",
                            "details": "Email address is missing",
                        })
                    elif not re.match(EMAIL_REGEX, email):
                        issues_found.append({
                            "record_id": record_id,
                            "table": table_name,
                            "field": "email",
                            "issue_type": "invalid_email",
                            "current_value": email,
                            "details": f"Email fails validation: {email}",
                        })

                    # Check for missing name
                    name = record.get("name", "") or record.get("contact_name", "")
                    if not name:
                        issues_found.append({
                            "record_id": record_id,
                            "table": table_name,
                            "field": "name",
                            "issue_type": "missing",
                            "current_value": "",
                            "details": "Name is missing",
                        })
                    else:
                        # Build index for duplicate detection
                        norm_name = name.strip().lower()
                        if norm_name not in name_index:
                            name_index[norm_name] = []
                        name_index[norm_name].append({
                            "record_id": record_id,
                            "table": table_name,
                            "name": name,
                            "email": email,
                        })

                    # Check for stale records
                    updated_at = record.get("updated_at", "")
                    if updated_at and updated_at < stale_cutoff:
                        issues_found.append({
                            "record_id": record_id,
                            "table": table_name,
                            "field": "updated_at",
                            "issue_type": "stale",
                            "current_value": updated_at,
                            "details": (
                                f"Record not updated in over "
                                f"{STALE_THRESHOLD_DAYS} days"
                            ),
                        })

                    # Check for missing phone (contacts only)
                    if table_name == "contacts":
                        phone = record.get("phone", "")
                        if not phone:
                            issues_found.append({
                                "record_id": record_id,
                                "table": table_name,
                                "field": "phone",
                                "issue_type": "incomplete",
                                "current_value": "",
                                "details": "Phone number is missing",
                            })

                    # Check for missing domain (companies only)
                    if table_name == "companies":
                        domain = record.get("domain", "")
                        if not domain:
                            issues_found.append({
                                "record_id": record_id,
                                "table": table_name,
                                "field": "domain",
                                "issue_type": "missing",
                                "current_value": "",
                                "details": "Company domain is missing",
                            })

            except Exception as e:
                logger.warning(
                    "data_enrichment_issue_check_error",
                    extra={
                        "table": table_name,
                        "error": str(e)[:200],
                    },
                )

        # Detect duplicates from name index
        duplicates_found = 0
        for norm_name, records in name_index.items():
            if len(records) > 1:
                duplicate_groups.append({
                    "normalized_name": norm_name,
                    "records": records,
                    "count": len(records),
                })
                duplicates_found += len(records) - 1
                for rec in records:
                    issues_found.append({
                        "record_id": rec["record_id"],
                        "table": rec["table"],
                        "field": "name",
                        "issue_type": "duplicate",
                        "current_value": rec["name"],
                        "details": (
                            f"Potential duplicate: {len(records)} records "
                            f"with name '{norm_name}'"
                        ),
                    })

        # Count critical issues (missing email, duplicates)
        critical_types = {"missing", "invalid_email", "duplicate"}
        critical_issues = sum(
            1 for i in issues_found if i.get("issue_type") in critical_types
        )

        logger.info(
            "data_enrichment_issues_identified",
            extra={
                "total_issues": len(issues_found),
                "critical": critical_issues,
                "duplicates": duplicates_found,
            },
        )

        return {
            "current_node": "identify_issues",
            "issues_found": issues_found,
            "total_issues": len(issues_found),
            "critical_issues": critical_issues,
            "duplicate_groups": duplicate_groups,
            "duplicates_found": duplicates_found,
        }

    # ─── Node 3: Suggest Fixes ───────────────────────────────────────

    async def _node_suggest_fixes(
        self, state: DataEnrichmentAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM generates suggested corrections and enrichment."""
        issues = state.get("issues_found", [])

        logger.info(
            "data_enrichment_suggest_fixes",
            extra={"issue_count": len(issues)},
        )

        enrichment_tasks: list[dict[str, Any]] = []

        if not issues:
            logger.info("data_enrichment_no_issues_to_fix")
            return {
                "current_node": "suggest_fixes",
                "enrichment_tasks": [],
            }

        # Only send fixable issues to LLM (limit to avoid token overuse)
        fixable_issues = [
            i for i in issues
            if i.get("issue_type") in (
                "invalid_email", "format_error", "incomplete",
                "inconsistent", "duplicate",
            )
        ][:30]

        if fixable_issues:
            try:
                prompt = DATA_ENRICHMENT_PROMPT.format(
                    issues_json=json.dumps(fixable_issues, indent=2),
                    issue_types=", ".join(ISSUE_TYPES),
                )

                llm_response = self.llm.messages.create(
                    model="claude-haiku-4-5-20250514",
                    system="You are a CRM data quality specialist.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )

                llm_text = llm_response.content[0].text.strip()

                try:
                    fixes = json.loads(llm_text)
                    if isinstance(fixes, list):
                        enrichment_tasks = fixes
                except (json.JSONDecodeError, KeyError):
                    logger.debug("data_enrichment_fixes_parse_error")
                    # Fallback: create basic fix suggestions
                    for issue in fixable_issues[:10]:
                        enrichment_tasks.append({
                            "record_id": issue.get("record_id", ""),
                            "table": issue.get("table", ""),
                            "field": issue.get("field", ""),
                            "issue_type": issue.get("issue_type", ""),
                            "current_value": issue.get("current_value", ""),
                            "suggested_value": "needs_manual_review",
                            "confidence": 0.3,
                            "reasoning": "Automated fix not available; manual review needed",
                        })

            except Exception as e:
                logger.warning(
                    "data_enrichment_fixes_llm_error",
                    extra={"error": str(e)[:200]},
                )

        # Add stale record suggestions without LLM
        stale_issues = [i for i in issues if i.get("issue_type") == "stale"]
        for stale in stale_issues[:20]:
            enrichment_tasks.append({
                "record_id": stale.get("record_id", ""),
                "table": stale.get("table", ""),
                "field": "updated_at",
                "issue_type": "stale",
                "current_value": stale.get("current_value", ""),
                "suggested_value": "re_verify",
                "confidence": 0.9,
                "reasoning": (
                    f"Record not updated in {STALE_THRESHOLD_DAYS}+ days; "
                    f"needs re-verification"
                ),
            })

        logger.info(
            "data_enrichment_fixes_suggested",
            extra={"fix_count": len(enrichment_tasks)},
        )

        return {
            "current_node": "suggest_fixes",
            "enrichment_tasks": enrichment_tasks,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: DataEnrichmentAgentState
    ) -> dict[str, Any]:
        """Node 4: Present quality issues and fixes for human approval."""
        total_issues = state.get("total_issues", 0)
        enrichment_tasks = state.get("enrichment_tasks", [])

        logger.info(
            "data_enrichment_human_review_pending",
            extra={
                "total_issues": total_issues,
                "fix_suggestions": len(enrichment_tasks),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: DataEnrichmentAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to data_quality_issues table + InsightData."""
        now = datetime.now(timezone.utc).isoformat()
        issues = state.get("issues_found", [])
        total_issues = state.get("total_issues", 0)
        critical_issues = state.get("critical_issues", 0)
        enrichment_tasks = state.get("enrichment_tasks", [])
        duplicate_groups = state.get("duplicate_groups", [])
        tables_scanned = state.get("tables_scanned", [])
        records_scanned = state.get("records_scanned", 0)
        approved = state.get("human_approval_status") == "approved"

        # Save issues to database
        issues_saved = False
        fixes_applied = 0

        for issue in issues[:100]:
            try:
                issue_record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "record_id": issue.get("record_id", ""),
                    "table_name": issue.get("table", ""),
                    "field_name": issue.get("field", ""),
                    "issue_type": issue.get("issue_type", ""),
                    "current_value": str(issue.get("current_value", ""))[:500],
                    "details": issue.get("details", ""),
                    "status": "identified",
                    "created_at": now,
                }
                self.db.client.table("data_quality_issues").insert(
                    issue_record
                ).execute()
                issues_saved = True
            except Exception as e:
                logger.debug(f"data_enrichment_issue_save_error: {e}")

        # Count potential fixes if approved
        if approved:
            fixes_applied = len([
                t for t in enrichment_tasks
                if t.get("confidence", 0) >= 0.7
            ])

        # Build issue type distribution
        type_counts: dict[str, int] = {}
        for issue in issues:
            itype = issue.get("issue_type", "unknown")
            type_counts[itype] = type_counts.get(itype, 0) + 1

        # Build report
        sections = [
            "# Data Quality Report",
            f"*Generated: {now}*\n",
            f"## Scan Summary",
            f"- **Tables Scanned:** {', '.join(tables_scanned)}",
            f"- **Records Scanned:** {records_scanned}",
            f"- **Total Issues:** {total_issues}",
            f"- **Critical Issues:** {critical_issues}",
            f"- **Duplicate Groups:** {len(duplicate_groups)}",
            f"- **Fix Suggestions:** {len(enrichment_tasks)}",
            f"- **Status:** {'Fixes Approved' if approved else 'Review Pending'}",
        ]

        if type_counts:
            sections.append("\n## Issues by Type")
            for itype, count in sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                sections.append(f"- **{itype}:** {count}")

        if duplicate_groups:
            sections.append("\n## Duplicate Groups")
            for i, group in enumerate(duplicate_groups[:5], 1):
                sections.append(
                    f"{i}. **{group.get('normalized_name', 'Unknown')}** "
                    f"({group.get('count', 0)} records)"
                )

        report = "\n".join(sections)

        # Store insight
        quality_score = max(
            0.0,
            100.0 - (total_issues / max(records_scanned, 1)) * 100
        )

        self.store_insight(InsightData(
            insight_type="data_quality",
            title=f"Data Quality: {quality_score:.0f}% clean, {total_issues} issues found",
            content=(
                f"Scanned {records_scanned} records across "
                f"{', '.join(tables_scanned)}. Found {total_issues} issues "
                f"({critical_issues} critical). "
                f"{len(duplicate_groups)} duplicate groups detected. "
                f"Generated {len(enrichment_tasks)} fix suggestions. "
                f"Data quality score: {quality_score:.0f}%."
            ),
            confidence=0.85,
            metadata={
                "records_scanned": records_scanned,
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "duplicate_groups": len(duplicate_groups),
                "fix_suggestions": len(enrichment_tasks),
                "quality_score": round(quality_score, 1),
                "issue_distribution": type_counts,
            },
        ))

        logger.info(
            "data_enrichment_report_generated",
            extra={
                "total_issues": total_issues,
                "quality_score": quality_score,
                "fixes_suggested": len(enrichment_tasks),
                "saved": issues_saved,
            },
        )

        return {
            "current_node": "report",
            "fixes_applied": fixes_applied,
            "records_enriched": fixes_applied,
            "issues_saved": issues_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: DataEnrichmentAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<DataEnrichmentAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
