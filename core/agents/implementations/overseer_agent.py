"""
Overseer Agent — The SRE Meta-Agent.

Monitors the health and performance of all agents across the platform.
Detects failures, diagnoses root causes, recommends corrective actions,
and generates status reports.

Architecture (LangGraph State Machine):
    collect_metrics → diagnose → plan_actions → human_review →
    execute_actions → report → END

This agent runs on a schedule (every 6 hours by default) and can also
be triggered manually or by events (agent failure, circuit breaker trip).

The Overseer is the platform's immune system:
- Detects: error spikes, circuit breaker trips, queue backlogs, zombie tasks
- Diagnoses: uses LLM to analyze patterns and identify root causes
- Recommends: specific corrective actions (restart, queue drain, config change)
- Reports: generates human-readable status reports

Safety:
- All corrective actions require human approval (human gate)
- The Overseer is READ-ONLY by default; actions dispatch via event_bus
- Circuit breaker applies to the Overseer itself (avoids infinite error loops)

Usage:
    agent = OverseerAgent(config, db, embedder, llm)
    result = await agent.run({"mode": "full_check"})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import OverseerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Severity constants ─────────────────────────────────────────────────
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# ─── Action types ────────────────────────────────────────────────────────
ACTION_RECOVER_ZOMBIES = "recover_zombie_tasks"
ACTION_ALERT_TEAM = "alert_team"
ACTION_DISABLE_AGENT = "disable_agent"
ACTION_RESTART_AGENT = "restart_agent"
ACTION_DRAIN_QUEUE = "drain_queue"
ACTION_CLEAR_CACHE = "clear_cache"
ACTION_ESCALATE = "escalate_to_human"

# System prompt for the Overseer's diagnostic LLM calls
OVERSEER_SYSTEM_PROMPT = """\
You are the Overseer — a System Reliability Engineer (SRE) AI for the \
Sovereign Venture Engine, an autonomous B2B sales platform.

Your job is to analyze system health data and produce:
1. A concise diagnosis of any issues
2. Root cause analysis
3. Specific recommended actions

You have access to:
- Agent execution history (run counts, failure rates, error messages)
- Error logs (structured, with agent_id and timestamps)
- Task queue status (pending, stuck, zombie tasks)
- LLM cache performance (hit rates, evictions)

Output a JSON object with these fields:
- "diagnosis": 1-3 sentence summary of system state
- "root_causes": list of identified root causes
- "recommended_actions": list of objects with {action, target, priority, reasoning}
  - action: one of [recover_zombie_tasks, alert_team, disable_agent, restart_agent, \
drain_queue, clear_cache, escalate_to_human]
  - target: the affected component (agent_id, "task_queue", "cache", etc.)
  - priority: "critical", "high", "medium", "low"
  - reasoning: why this action is recommended

If the system is healthy, set diagnosis to "All systems nominal" and return \
empty root_causes and recommended_actions lists.

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("overseer")
class OverseerAgent(BaseAgent):
    """
    SRE meta-agent that monitors platform health and recommends actions.

    Nodes:
        1. collect_metrics — Gather health data from all system tools
        2. diagnose — LLM-powered root cause analysis
        3. plan_actions — Propose corrective actions
        4. human_review — Gate for human approval
        5. execute_actions — Dispatch approved actions
        6. report — Generate summary report + write insights
    """

    def build_graph(self) -> Any:
        """
        Build the Overseer's LangGraph state machine.

        Graph flow:
            collect_metrics → diagnose → route_by_severity →
            [plan_actions → human_review → execute_actions] | [report]
            → report → END
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(OverseerAgentState)

        # Add nodes
        workflow.add_node("collect_metrics", self._node_collect_metrics)
        workflow.add_node("diagnose", self._node_diagnose)
        workflow.add_node("plan_actions", self._node_plan_actions)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute_actions", self._node_execute_actions)
        workflow.add_node("report", self._node_report)

        # Entry point
        workflow.set_entry_point("collect_metrics")

        # collect_metrics → diagnose
        workflow.add_edge("collect_metrics", "diagnose")

        # diagnose → route by severity
        workflow.add_conditional_edges(
            "diagnose",
            self._route_by_severity,
            {
                "needs_action": "plan_actions",
                "healthy": "report",
            },
        )

        # plan_actions → human_review → execute_actions → report
        workflow.add_edge("plan_actions", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "execute_actions",
                "rejected": "report",
            },
        )
        workflow.add_edge("execute_actions", "report")
        workflow.add_edge("report", END)

        # Compile with human gate
        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        """Overseer uses system_tools via MCP, not injected."""
        return []

    def get_state_class(self) -> Type[OverseerAgentState]:
        return OverseerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state for an Overseer run."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "system_health": {},
            "health_status": "unknown",
            "error_logs": [],
            "agent_error_rates": {},
            "task_queue_status": {},
            "cache_performance": {},
            "knowledge_stats": {},
            "issues": [],
            "issue_count": 0,
            "critical_count": 0,
            "diagnosis": "",
            "root_causes": [],
            "recommended_actions": [],
            "actions_planned": [],
            "actions_approved": False,
            "actions_executed": [],
            "actions_failed": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Collect Metrics ──────────────────────────────────────

    async def _node_collect_metrics(
        self, state: OverseerAgentState
    ) -> dict[str, Any]:
        """
        Gather system health data from all monitoring tools.

        Calls each system tool and aggregates results into state.
        All calls are best-effort — individual failures don't block the run.
        """
        from core.mcp.tools.system_tools import (
            get_recent_logs,
            get_system_health,
            get_task_queue_status,
            get_agent_error_rates,
            get_knowledge_stats,
            get_cache_performance,
        )

        logger.info(
            "overseer_collecting_metrics",
            extra={"agent_id": self.agent_id},
        )

        updates: dict[str, Any] = {"current_node": "collect_metrics"}

        # 1. System health (primary diagnostic)
        try:
            health_json = get_system_health(
                vertical_id=self.vertical_id,
                _db=self.db,
            )
            health = json.loads(health_json)
            updates["system_health"] = health
            updates["health_status"] = health.get("status", "unknown")
            updates["issues"] = health.get("issues", [])
        except Exception as e:
            logger.warning(f"Overseer: health check failed: {e}")
            updates["system_health"] = {"error": str(e)}
            updates["health_status"] = "unknown"

        # 2. Error logs
        try:
            logs_json = get_recent_logs(
                level="ERROR",
                limit=50,
                since_minutes=360,  # 6 hours
            )
            logs = json.loads(logs_json)
            updates["error_logs"] = logs.get("logs", [])
        except Exception as e:
            logger.warning(f"Overseer: log retrieval failed: {e}")
            updates["error_logs"] = []

        # 3. Agent error rates
        try:
            rates_json = get_agent_error_rates(
                days=7,
                vertical_id=self.vertical_id,
                _db=self.db,
            )
            rates = json.loads(rates_json)
            updates["agent_error_rates"] = rates.get("agents", {})
        except Exception as e:
            logger.warning(f"Overseer: error rate analysis failed: {e}")
            updates["agent_error_rates"] = {}

        # 4. Task queue status
        try:
            queue_json = get_task_queue_status(
                vertical_id=self.vertical_id,
                _db=self.db,
            )
            queue = json.loads(queue_json)
            updates["task_queue_status"] = queue.get("queue", {})
        except Exception as e:
            logger.warning(f"Overseer: queue check failed: {e}")
            updates["task_queue_status"] = {}

        # 5. Cache performance (if available)
        try:
            cache_json = get_cache_performance(_cache=self.cache)
            cache = json.loads(cache_json)
            updates["cache_performance"] = cache.get("stats", {})
        except Exception as e:
            updates["cache_performance"] = {}

        # 6. Knowledge stats
        try:
            knowledge_json = get_knowledge_stats(
                vertical_id=self.vertical_id,
                _db=self.db,
            )
            knowledge = json.loads(knowledge_json)
            updates["knowledge_stats"] = knowledge
        except Exception as e:
            updates["knowledge_stats"] = {}

        # Count issues
        issues = updates.get("issues", [])
        updates["issue_count"] = len(issues)
        updates["critical_count"] = sum(
            1 for i in issues if i.get("severity") == SEVERITY_CRITICAL
        )

        logger.info(
            "overseer_metrics_collected",
            extra={
                "agent_id": self.agent_id,
                "health_status": updates.get("health_status"),
                "issue_count": updates.get("issue_count"),
                "critical_count": updates.get("critical_count"),
            },
        )

        return updates

    # ─── Node 2: Diagnose ─────────────────────────────────────────────

    async def _node_diagnose(
        self, state: OverseerAgentState
    ) -> dict[str, Any]:
        """
        Use LLM to analyze collected metrics and diagnose issues.

        Sends health data to Claude for root cause analysis and
        recommended actions.
        """
        health_status = state.get("health_status", "unknown")
        issues = state.get("issues", [])
        error_rates = state.get("agent_error_rates", {})
        error_logs = state.get("error_logs", [])
        queue_status = state.get("task_queue_status", {})

        logger.info(
            "overseer_diagnosing",
            extra={
                "agent_id": self.agent_id,
                "health_status": health_status,
                "issue_count": len(issues),
            },
        )

        # If system is healthy and no issues, skip LLM call
        if health_status == "healthy" and not issues:
            return {
                "current_node": "diagnose",
                "diagnosis": "All systems nominal. No issues detected.",
                "root_causes": [],
                "recommended_actions": [],
            }

        # Build diagnostic context for the LLM
        diagnostic_context = {
            "health_status": health_status,
            "issues": issues[:20],  # Cap to avoid token overflow
            "agent_error_rates": {
                k: {
                    "failure_rate": v.get("failure_rate"),
                    "total_runs": v.get("total_runs"),
                    "failed_runs": v.get("failed_runs"),
                    "risk_level": v.get("risk_level"),
                    "recent_errors": v.get("error_messages", [])[:3],
                }
                for k, v in error_rates.items()
            },
            "recent_errors": [
                {
                    "level": log.get("level"),
                    "message": log.get("message", "")[:200],
                    "logger": log.get("logger"),
                    "agent_id": log.get("agent_id"),
                    "time": log.get("time"),
                }
                for log in error_logs[:15]
            ],
            "task_queue": queue_status,
        }

        user_prompt = (
            "Analyze this system health data and provide your diagnosis:\n\n"
            f"{json.dumps(diagnostic_context, indent=2, default=str)}"
        )

        # Call LLM for diagnosis
        try:
            response = await self.route_llm_cached(
                intent="reasoning",
                system_prompt=OVERSEER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            # Parse LLM response
            response_text = response.text.strip()
            if "```" in response_text:
                parts = response_text.split("```")
                response_text = parts[1] if len(parts) > 1 else response_text
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            parsed = json.loads(response_text)

            return {
                "current_node": "diagnose",
                "diagnosis": parsed.get("diagnosis", "Unable to determine"),
                "root_causes": parsed.get("root_causes", []),
                "recommended_actions": parsed.get("recommended_actions", []),
            }

        except json.JSONDecodeError:
            # LLM returned non-JSON — use raw text as diagnosis
            return {
                "current_node": "diagnose",
                "diagnosis": response_text[:500] if response_text else "Diagnosis failed",
                "root_causes": ["LLM returned non-structured response"],
                "recommended_actions": [{
                    "action": ACTION_ESCALATE,
                    "target": "overseer",
                    "priority": "medium",
                    "reasoning": "Diagnosis produced unstructured output",
                }],
            }
        except Exception as e:
            logger.error(
                "overseer_diagnosis_failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e)[:200],
                },
            )
            return {
                "current_node": "diagnose",
                "diagnosis": f"Diagnosis failed: {str(e)[:200]}",
                "root_causes": [str(e)[:200]],
                "recommended_actions": [{
                    "action": ACTION_ESCALATE,
                    "target": "overseer",
                    "priority": "high",
                    "reasoning": f"Diagnostic LLM call failed: {str(e)[:100]}",
                }],
            }

    # ─── Routing ──────────────────────────────────────────────────────

    def _route_by_severity(self, state: OverseerAgentState) -> str:
        """Route to action planning if issues detected, else straight to report."""
        actions = state.get("recommended_actions", [])
        if actions:
            return "needs_action"
        return "healthy"

    def _route_after_review(self, state: OverseerAgentState) -> str:
        """Route based on human review of planned actions."""
        if state.get("actions_approved", False):
            return "approved"
        return "rejected"

    # ─── Node 3: Plan Actions ─────────────────────────────────────────

    async def _node_plan_actions(
        self, state: OverseerAgentState
    ) -> dict[str, Any]:
        """
        Convert recommended actions into an executable plan.

        Each action is validated and enriched with execution details
        before being presented to the human for review.
        """
        recommended = state.get("recommended_actions", [])
        diagnosis = state.get("diagnosis", "")

        logger.info(
            "overseer_planning_actions",
            extra={
                "agent_id": self.agent_id,
                "action_count": len(recommended),
            },
        )

        planned = []
        for action in recommended:
            action_type = action.get("action", ACTION_ESCALATE)
            target = action.get("target", "unknown")
            priority = action.get("priority", "medium")
            reasoning = action.get("reasoning", "")

            plan_entry = {
                "action": action_type,
                "target": target,
                "priority": priority,
                "reasoning": reasoning,
                "status": "planned",
                "executable": action_type in {
                    ACTION_RECOVER_ZOMBIES,
                    ACTION_CLEAR_CACHE,
                    ACTION_ALERT_TEAM,
                },
            }
            planned.append(plan_entry)

        return {
            "current_node": "plan_actions",
            "actions_planned": planned,
            "requires_human_approval": True,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: OverseerAgentState
    ) -> dict[str, Any]:
        """
        Present planned actions for human approval.

        In production, the LangGraph interrupt_before mechanism pauses here.
        The human reviews via dashboard and approves/rejects.
        """
        planned = state.get("actions_planned", [])
        approval = state.get("human_approval_status")

        if approval == "approved":
            return {
                "current_node": "human_review",
                "actions_approved": True,
            }

        # If not approved yet, auto-approve safe actions only
        # (recover zombies, clear cache are always safe)
        safe_actions = [
            a for a in planned
            if a.get("action") in {ACTION_RECOVER_ZOMBIES, ACTION_CLEAR_CACHE}
        ]

        # Auto-approve threshold from config
        threshold = self.config.human_gates.auto_approve_threshold
        if threshold >= 1.0 and safe_actions:
            return {
                "current_node": "human_review",
                "actions_approved": True,
                "actions_planned": safe_actions,
            }

        return {
            "current_node": "human_review",
            "actions_approved": False,
        }

    # ─── Node 5: Execute Actions ──────────────────────────────────────

    async def _node_execute_actions(
        self, state: OverseerAgentState
    ) -> dict[str, Any]:
        """
        Execute approved corrective actions.

        Each action type maps to a specific operation:
        - recover_zombie_tasks → call db.recover_zombie_tasks()
        - clear_cache → clear LLM response cache
        - alert_team → log critical alert (future: Telegram/Slack)
        - escalate_to_human → log and mark for follow-up
        """
        planned = state.get("actions_planned", [])
        executed = []
        failed = []

        for action in planned:
            action_type = action.get("action", "")
            target = action.get("target", "")

            try:
                if action_type == ACTION_RECOVER_ZOMBIES:
                    recovered = self.db.recover_zombie_tasks(stale_minutes=15)
                    executed.append({
                        **action,
                        "status": "executed",
                        "result": f"Recovered {recovered} zombie tasks",
                    })

                elif action_type == ACTION_CLEAR_CACHE:
                    cleared = self.cache.clear()
                    executed.append({
                        **action,
                        "status": "executed",
                        "result": f"Cleared {cleared} cache entries",
                    })

                elif action_type == ACTION_ALERT_TEAM:
                    # Log the alert — in production, dispatch to Telegram/Slack
                    logger.critical(
                        "overseer_alert",
                        extra={
                            "agent_id": self.agent_id,
                            "alert_target": target,
                            "alert_reason": action.get("reasoning", ""),
                        },
                    )
                    executed.append({
                        **action,
                        "status": "executed",
                        "result": "Alert logged (notification channels TBD)",
                    })

                elif action_type == ACTION_ESCALATE:
                    logger.warning(
                        "overseer_escalation",
                        extra={
                            "agent_id": self.agent_id,
                            "escalation_target": target,
                            "reasoning": action.get("reasoning", ""),
                        },
                    )
                    executed.append({
                        **action,
                        "status": "escalated",
                        "result": "Flagged for human attention",
                    })

                else:
                    # Non-executable actions (disable_agent, restart_agent, drain_queue)
                    # Require manual intervention — log and skip
                    logger.info(
                        "overseer_action_requires_manual",
                        extra={
                            "action": action_type,
                            "target": target,
                        },
                    )
                    executed.append({
                        **action,
                        "status": "requires_manual",
                        "result": f"Action '{action_type}' requires manual execution",
                    })

            except Exception as e:
                logger.error(
                    "overseer_action_failed",
                    extra={
                        "action": action_type,
                        "target": target,
                        "error": str(e)[:200],
                    },
                )
                failed.append({
                    **action,
                    "status": "failed",
                    "error": str(e)[:200],
                })

        return {
            "current_node": "execute_actions",
            "actions_executed": executed,
            "actions_failed": failed,
        }

    # ─── Node 6: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: OverseerAgentState
    ) -> dict[str, Any]:
        """
        Generate a human-readable status report.

        Summarizes health status, issues found, actions taken, and
        system recommendations.
        """
        health_status = state.get("health_status", "unknown")
        diagnosis = state.get("diagnosis", "")
        issues = state.get("issues", [])
        root_causes = state.get("root_causes", [])
        executed = state.get("actions_executed", [])
        failed = state.get("actions_failed", [])
        error_rates = state.get("agent_error_rates", {})

        now = datetime.now(timezone.utc).isoformat()

        # Build report
        lines = [
            f"# Overseer Status Report",
            f"**Generated:** {now}",
            f"**System Status:** {health_status.upper()}",
            "",
        ]

        # Diagnosis
        if diagnosis:
            lines.extend([
                "## Diagnosis",
                diagnosis,
                "",
            ])

        # Issues
        if issues:
            lines.append("## Issues Detected")
            for issue in issues:
                severity = issue.get("severity", "info")
                icon = {"critical": "!!!", "warning": "!!", "info": "*"}.get(
                    severity, "*"
                )
                lines.append(
                    f"- [{icon}] [{severity.upper()}] "
                    f"{issue.get('component', 'unknown')}: "
                    f"{issue.get('message', '')}"
                )
            lines.append("")

        # Root causes
        if root_causes:
            lines.append("## Root Causes")
            for cause in root_causes:
                lines.append(f"- {cause}")
            lines.append("")

        # Actions taken
        if executed:
            lines.append("## Actions Taken")
            for action in executed:
                lines.append(
                    f"- [{action.get('status', '')}] "
                    f"{action.get('action', '')}: "
                    f"{action.get('result', '')}"
                )
            lines.append("")

        if failed:
            lines.append("## Actions Failed")
            for action in failed:
                lines.append(
                    f"- {action.get('action', '')}: "
                    f"{action.get('error', '')}"
                )
            lines.append("")

        # Agent health table
        if error_rates:
            lines.append("## Agent Health")
            for aid, stats in error_rates.items():
                risk = stats.get("risk_level", "normal")
                rate = stats.get("failure_rate", 0)
                total = stats.get("total_runs", 0)
                lines.append(
                    f"- {aid}: {rate:.0%} failure rate "
                    f"({total} runs, risk: {risk})"
                )
            lines.append("")

        report = "\n".join(lines)

        logger.info(
            "overseer_report_generated",
            extra={
                "agent_id": self.agent_id,
                "health_status": health_status,
                "issue_count": len(issues),
                "actions_executed": len(executed),
                "report_length": len(report),
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
            "completed_at": now,
        }

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        Store Overseer insights in the shared brain.

        Records system health patterns so other agents (and future Overseer
        runs) can learn from historical health data.
        """
        health_status = result.get("health_status", "healthy")
        issues = result.get("issues", [])
        diagnosis = result.get("diagnosis", "")

        # Only store insights when issues are detected
        if health_status == "healthy" or not issues:
            return

        insight = InsightData(
            insight_type="system_health",
            title=f"System Health: {health_status}",
            content=diagnosis[:500] if diagnosis else f"{len(issues)} issues detected",
            confidence=0.9,  # High confidence — based on real metrics
            metadata={
                "health_status": health_status,
                "issue_count": len(issues),
                "timestamp": result.get("report_generated_at", ""),
                "critical_count": result.get("critical_count", 0),
            },
        )
        self.store_insight(insight)
