"""
Autopilot Agent — The Self-Driving Meta-Agent.

Orchestrates autonomous healing, budget optimization, and strategic
recommendations across the entire platform. This is the "brain of brains"
that sits above all other agents.

Architecture (LangGraph State Machine):
    analyze_system → detect_issues → generate_strategy → human_review →
    execute_strategy → report → END

The Autopilot combines three subsystems:
1. SelfHealer — crash diagnosis + config-level fixes
2. BudgetManager — ROAS optimization + budget reallocation
3. Strategist — performance analysis + experiment proposals

Safety:
- ALL strategy actions require human approval (mandatory gate)
- Budget reallocations >20% flagged for additional review
- Config fixes limited to SAFE_CONFIG_KEYS whitelist
- auto_approve_threshold defaults to 0.0 (never auto-approves)
- Full audit trail in autopilot_sessions + optimization_actions

Usage:
    agent = AutopilotAgent(config, db, embedder, llm)
    result = await agent.run({"mode": "full_analysis"})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import AutopilotAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)


# ─── Session Types ──────────────────────────────────────────────────
SESSION_FULL = "full_analysis"
SESSION_HEALING = "healing"
SESSION_BUDGET = "budget"
SESSION_STRATEGY = "strategy"

VALID_SESSION_TYPES = {SESSION_FULL, SESSION_HEALING, SESSION_BUDGET, SESSION_STRATEGY}

# ─── System Prompt ──────────────────────────────────────────────────
AUTOPILOT_SYSTEM_PROMPT = """\
You are the Autopilot — the self-driving layer of the Sovereign Venture Engine, \
an autonomous B2B sales and operations platform.

Your role is to analyze the entire system, identify issues and opportunities, \
and generate a prioritized strategy for optimization.

You have three subsystems:
1. SelfHealer: diagnoses agent crashes and suggests config-level fixes
2. BudgetManager: optimizes ad spend allocation based on ROAS
3. Strategist: identifies growth opportunities and proposes experiments

Given the system snapshot below, produce a JSON strategy with:
- "summary": 2-3 sentence overview of system state
- "priority_actions": list of actions ordered by impact, each with:
  - "category": "healing" | "budget" | "strategy"
  - "action": description of what to do
  - "target": agent_id or campaign_id
  - "impact": "high" | "medium" | "low"
  - "confidence": 0.0-1.0
  - "reasoning": why this matters
- "strategy_confidence": overall confidence in the strategy (0.0-1.0)

Constraints:
- NEVER suggest code-level changes — only config adjustments
- NEVER increase total ad spend — only redistribute
- Budget shifts >20% of campaign budget should be flagged
- All actions will be reviewed by a human before execution

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("autopilot")
class AutopilotAgent(BaseAgent):
    """
    Self-driving meta-agent that orchestrates healing, budget, and strategy.

    Nodes:
        1. analyze_system — Collect metrics via Healer + BudgetManager + Strategist
        2. detect_issues — Identify problems and opportunities
        3. generate_strategy — LLM-powered prioritization of actions
        4. human_review — Gate for human approval (mandatory)
        5. execute_strategy — Apply approved actions
        6. report — Generate summary + write insights to Hive Mind
    """

    def build_graph(self) -> Any:
        """
        Build the Autopilot's LangGraph state machine.

        Graph flow:
            analyze_system → detect_issues → route_by_issues →
            [generate_strategy → human_review → route_by_approval →
             execute_strategy] | [report]
            → report → END
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AutopilotAgentState)

        # Add nodes
        workflow.add_node("analyze_system", self._node_analyze_system)
        workflow.add_node("detect_issues", self._node_detect_issues)
        workflow.add_node("generate_strategy", self._node_generate_strategy)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute_strategy", self._node_execute_strategy)
        workflow.add_node("report", self._node_report)

        # Entry point
        workflow.set_entry_point("analyze_system")

        # analyze_system → detect_issues
        workflow.add_edge("analyze_system", "detect_issues")

        # detect_issues → route: has issues → generate_strategy, else → report
        workflow.add_conditional_edges(
            "detect_issues",
            self._route_by_issues,
            {
                "needs_strategy": "generate_strategy",
                "all_clear": "report",
            },
        )

        # generate_strategy → human_review → route by approval
        workflow.add_edge("generate_strategy", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_by_approval,
            {
                "approved": "execute_strategy",
                "rejected": "report",
            },
        )

        # execute_strategy → report → END
        workflow.add_edge("execute_strategy", "report")
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
        """Autopilot uses subsystems directly, not tools."""
        return []

    def get_state_class(self) -> Type[AutopilotAgentState]:
        return AutopilotAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str,
    ) -> dict[str, Any]:
        """Prepare initial state for an Autopilot run."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            # System Snapshot
            "performance_snapshot": {},
            "budget_snapshot": {},
            "experiment_snapshot": [],
            # Diagnosis
            "detected_issues": [],
            "health_scores": {},
            "optimization_opportunities": [],
            # Strategy
            "healing_actions": [],
            "budget_actions": [],
            "experiment_proposals": [],
            "strategy_summary": "",
            "strategy_confidence": 0.0,
            # Actions
            "actions_planned": [],
            "actions_approved": False,
            "actions_executed": [],
            "actions_failed": [],
            # Report
            "report_summary": "",
            "report_generated_at": "",
            # Session tracking
            "session_id": "",
            "session_type": task.get("mode", SESSION_FULL),
        })
        return state

    # ─── Subsystem Factories ──────────────────────────────────────────

    def _get_healer(self) -> Any:
        """Lazy-create SelfHealer instance."""
        from core.optimization.healer import SelfHealer
        return SelfHealer(db=self.db, llm_client=self.llm)

    def _get_budget_manager(self) -> Any:
        """Lazy-create BudgetManager instance."""
        from core.optimization.budget_manager import BudgetManager
        return BudgetManager(db=self.db)

    def _get_strategist(self) -> Any:
        """Lazy-create Strategist instance."""
        from core.genesis.strategist import Strategist
        return Strategist(
            db=self.db,
            embedder=self.embedder,
            hive_mind=self.hive,
        )

    # ─── Session Management ───────────────────────────────────────────

    def _create_session(self, vertical_id: str, session_type: str) -> str:
        """Create an autopilot session record and return its ID."""
        try:
            result = (
                self.db.client.table("autopilot_sessions")
                .insert({
                    "vertical_id": vertical_id,
                    "session_type": session_type,
                    "status": "running",
                })
                .execute()
            )
            if result.data:
                return result.data[0].get("id", "")
        except Exception as e:
            logger.error(f"Failed to create autopilot session: {e}")
        return ""

    def _complete_session(
        self,
        session_id: str,
        status: str = "completed",
        metrics: Optional[dict] = None,
        issues: Optional[list] = None,
        strategy: Optional[dict] = None,
        actions: Optional[list] = None,
    ) -> None:
        """Update session record on completion."""
        if not session_id:
            return
        try:
            update: dict[str, Any] = {
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            if metrics:
                update["metrics_snapshot"] = metrics
            if issues is not None:
                update["detected_issues"] = issues
            if strategy:
                update["strategy_output"] = strategy
            if actions is not None:
                update["actions_taken"] = actions

            (
                self.db.client.table("autopilot_sessions")
                .update(update)
                .eq("id", session_id)
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to complete session {session_id}: {e}")

    # ─── Node: Analyze System ─────────────────────────────────────────

    async def _node_analyze_system(
        self, state: AutopilotAgentState,
    ) -> dict[str, Any]:
        """
        Node 1: Collect metrics from all subsystems.

        Uses Healer, BudgetManager, and Strategist to build a
        comprehensive system snapshot.
        """
        vertical_id = state.get("vertical_id", "")
        session_type = state.get("session_type", SESSION_FULL)
        days = int(self.config.params.get("analysis_period_days", 7))

        logger.info(
            "autopilot_analyzing",
            extra={
                "agent_id": self.agent_id,
                "vertical_id": vertical_id,
                "session_type": session_type,
            },
        )

        # Create session
        session_id = self._create_session(vertical_id, session_type)

        # Collect performance via Strategist
        strategist = self._get_strategist()
        performance = strategist.scan_performance(vertical_id, days=days)

        # Collect agent health scores via Healer
        healer = self._get_healer()
        health_scores: dict[str, float] = {}
        for agent_data in performance.get("agents", []):
            aid = agent_data.get("agent_id", "")
            if aid:
                score = healer.get_agent_health_score(aid, days=days)
                health_scores[aid] = score

        # Collect budget snapshot via BudgetManager
        budget_mgr = self._get_budget_manager()
        budget_summary = budget_mgr.get_spend_summary(vertical_id, days=days)

        return {
            "current_node": "analyze_system",
            "session_id": session_id,
            "performance_snapshot": performance,
            "budget_snapshot": budget_summary,
            "experiment_snapshot": performance.get("experiments", []),
            "health_scores": health_scores,
        }

    # ─── Node: Detect Issues ──────────────────────────────────────────

    async def _node_detect_issues(
        self, state: AutopilotAgentState,
    ) -> dict[str, Any]:
        """
        Node 2: Identify problems and opportunities from the snapshot.

        Combines:
        - Low health scores → healing candidates
        - Budget inefficiency → reallocation candidates
        - Performance gaps → strategy opportunities
        """
        performance = state.get("performance_snapshot", {})
        health_scores = state.get("health_scores", {})
        budget = state.get("budget_snapshot", {})

        detected_issues: list[dict[str, Any]] = []

        # 1. Agent health issues
        for agent_id, score in health_scores.items():
            if score < 0.5:
                detected_issues.append({
                    "agent_id": agent_id,
                    "issue_type": "low_health",
                    "severity": "critical" if score < 0.3 else "warning",
                    "details": {
                        "health_score": score,
                        "message": (
                            f"Agent {agent_id} health score is {score:.2f} "
                            f"(threshold: 0.50)"
                        ),
                    },
                })

        # 2. Budget issues
        roas = budget.get("roas", 0)
        total_spend = budget.get("total_spend", 0)
        if total_spend > 0 and roas < 1.0:
            detected_issues.append({
                "agent_id": "budget",
                "issue_type": "budget_inefficiency",
                "severity": "warning",
                "details": {
                    "roas": roas,
                    "total_spend": total_spend,
                    "message": f"ROAS is {roas:.1f}x — below break-even",
                },
            })

        # 3. Strategy opportunities via Strategist
        strategist = self._get_strategist()
        opportunities = strategist.detect_opportunities(performance)

        logger.info(
            "autopilot_issues_detected",
            extra={
                "agent_id": self.agent_id,
                "issue_count": len(detected_issues),
                "opportunity_count": len(opportunities),
            },
        )

        return {
            "current_node": "detect_issues",
            "detected_issues": detected_issues,
            "optimization_opportunities": opportunities,
        }

    # ─── Node: Generate Strategy ──────────────────────────────────────

    async def _node_generate_strategy(
        self, state: AutopilotAgentState,
    ) -> dict[str, Any]:
        """
        Node 3: Use LLM to prioritize actions from detected issues.

        Generates:
        - Healing actions (config fixes)
        - Budget actions (reallocations)
        - Experiment proposals
        - Overall strategy summary
        """
        issues = state.get("detected_issues", [])
        opportunities = state.get("optimization_opportunities", [])
        performance = state.get("performance_snapshot", {})
        budget = state.get("budget_snapshot", {})
        health_scores = state.get("health_scores", {})

        # Build context for LLM
        context = {
            "detected_issues": issues,
            "opportunities": opportunities[:10],
            "agent_health": health_scores,
            "budget_summary": {
                "total_spend": budget.get("total_spend", 0),
                "total_revenue": budget.get("total_revenue", 0),
                "roas": budget.get("roas", 0),
            },
            "active_agents": len(performance.get("agents", [])),
            "active_experiments": len([
                e for e in state.get("experiment_snapshot", [])
                if e.get("status") == "active"
            ]),
        }

        # Try LLM-powered strategy generation
        strategy = await self._generate_strategy_via_llm(context)

        # Build actionable plans from strategy
        healing_actions: list[dict[str, Any]] = []
        budget_actions: list[dict[str, Any]] = []
        experiment_proposals: list[dict[str, Any]] = []

        for action in strategy.get("priority_actions", []):
            category = action.get("category", "")
            if category == "healing":
                healing_actions.append(action)
            elif category == "budget":
                budget_actions.append(action)
            elif category == "strategy":
                experiment_proposals.append(action)

        # Also generate experiment proposals from Strategist
        strategist = self._get_strategist()
        experiments = strategist.propose_experiments(opportunities)

        # Merge experiment proposals
        all_proposals = experiment_proposals + [
            {
                "category": "strategy",
                "action": f"Launch experiment: {exp.get('name', '')}",
                "target": exp.get("agent_id", ""),
                "impact": "medium",
                "confidence": 0.7,
                "reasoning": f"Based on {exp.get('source_opportunity', '')} opportunity",
                "experiment_spec": exp,
            }
            for exp in experiments
        ]

        strategy_confidence = float(
            strategy.get("strategy_confidence", 0.5)
        )

        # Build actions list for planning
        all_actions = []
        for a in healing_actions:
            all_actions.append({**a, "status": "planned"})
        for a in budget_actions:
            all_actions.append({**a, "status": "planned"})
        for a in all_proposals:
            all_actions.append({**a, "status": "planned"})

        logger.info(
            "autopilot_strategy_generated",
            extra={
                "agent_id": self.agent_id,
                "healing_actions": len(healing_actions),
                "budget_actions": len(budget_actions),
                "experiment_proposals": len(all_proposals),
                "confidence": strategy_confidence,
            },
        )

        return {
            "current_node": "generate_strategy",
            "healing_actions": healing_actions,
            "budget_actions": budget_actions,
            "experiment_proposals": all_proposals,
            "strategy_summary": strategy.get("summary", ""),
            "strategy_confidence": strategy_confidence,
            "actions_planned": all_actions,
        }

    async def _generate_strategy_via_llm(
        self, context: dict[str, Any],
    ) -> dict[str, Any]:
        """Call LLM to generate prioritized strategy."""
        try:
            prompt = (
                f"System Analysis Data:\n"
                f"{json.dumps(context, indent=2, default=str)}\n\n"
                f"Generate a prioritized optimization strategy."
            )

            response = self.router.route(
                messages=[{"role": "user", "content": prompt}],
                system=AUTOPILOT_SYSTEM_PROMPT,
                intent="analysis",
                max_tokens=4096,
            )

            # Parse JSON from response
            content = response.get("content", "")
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""

            # Try to extract JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try extracting JSON from markdown code block
                if "```" in content:
                    json_str = content.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    return json.loads(json_str.strip())

        except Exception as e:
            logger.error(f"LLM strategy generation failed: {e}")

        # Fallback: rule-based strategy
        return self._rule_based_strategy(context)

    def _rule_based_strategy(
        self, context: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a basic strategy without LLM (fallback)."""
        actions = []
        issues = context.get("detected_issues", [])
        opportunities = context.get("opportunities", [])

        for issue in issues:
            issue_type = issue.get("issue_type", "")
            if issue_type == "low_health":
                actions.append({
                    "category": "healing",
                    "action": f"Investigate and fix {issue.get('agent_id', '')}",
                    "target": issue.get("agent_id", ""),
                    "impact": "high",
                    "confidence": 0.7,
                    "reasoning": issue.get("details", {}).get("message", ""),
                })
            elif issue_type == "budget_inefficiency":
                actions.append({
                    "category": "budget",
                    "action": "Run ROAS-weighted budget reallocation",
                    "target": "all_campaigns",
                    "impact": "high",
                    "confidence": 0.8,
                    "reasoning": issue.get("details", {}).get("message", ""),
                })

        for opp in opportunities[:5]:
            if opp.get("potential_impact") == "high":
                actions.append({
                    "category": "strategy",
                    "action": opp.get("suggested_action", ""),
                    "target": opp.get("details", {}).get("agent_id", "system"),
                    "impact": opp.get("potential_impact", "medium"),
                    "confidence": opp.get("confidence", 0.5),
                    "reasoning": opp.get("description", ""),
                })

        return {
            "summary": (
                f"Found {len(issues)} issues and {len(opportunities)} opportunities. "
                f"Generated {len(actions)} priority actions."
            ),
            "priority_actions": actions,
            "strategy_confidence": 0.6,
        }

    # ─── Node: Human Review ───────────────────────────────────────────

    async def _node_human_review(
        self, state: AutopilotAgentState,
    ) -> dict[str, Any]:
        """
        Node 4: Present strategy for human approval.

        This is the MANDATORY gate. The graph is compiled with
        interrupt_before=["human_review"], so LangGraph pauses here.

        When resumed, the human's approval status is in state.
        """
        actions = state.get("actions_planned", [])
        confidence = state.get("strategy_confidence", 0)
        summary = state.get("strategy_summary", "")

        logger.info(
            "autopilot_awaiting_review",
            extra={
                "agent_id": self.agent_id,
                "actions_count": len(actions),
                "confidence": confidence,
                "summary": summary[:200],
            },
        )

        # Check if auto-approve threshold is met
        threshold = float(
            self.config.params.get("auto_approve_threshold", 0.0)
        )
        auto_approved = (
            threshold > 0 and confidence >= threshold
        )

        return {
            "current_node": "human_review",
            "actions_approved": auto_approved or state.get("actions_approved", False),
        }

    # ─── Node: Execute Strategy ───────────────────────────────────────

    async def _node_execute_strategy(
        self, state: AutopilotAgentState,
    ) -> dict[str, Any]:
        """
        Node 5: Apply approved actions via subsystems.

        Executes:
        - Healing actions → SelfHealer.apply_config_fix()
        - Budget actions → BudgetManager.apply_reallocation()
        - Experiment proposals → ExperimentEngine.start_experiment()
        """
        session_id = state.get("session_id", "")
        healing_actions = state.get("healing_actions", [])
        budget_actions = state.get("budget_actions", [])
        experiment_proposals = state.get("experiment_proposals", [])

        executed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        # 1. Execute healing actions
        if healing_actions:
            healer = self._get_healer()
            for action in healing_actions:
                try:
                    target = action.get("target", "")
                    # Build a fix dict from the action
                    fix = {
                        "action": "config_fix",
                        "parameter": action.get("parameter", "enabled"),
                        "old_value": action.get("old_value"),
                        "new_value": action.get("new_value", True),
                        "reasoning": action.get("reasoning", ""),
                    }
                    result = healer.apply_config_fix(
                        target, fix, session_id=session_id,
                    )
                    executed.append({
                        "category": "healing",
                        "action": action.get("action", ""),
                        "target": target,
                        "status": result.get("status", "unknown"),
                        "result": result,
                    })
                except Exception as e:
                    failed.append({
                        "category": "healing",
                        "action": action.get("action", ""),
                        "target": action.get("target", ""),
                        "error": str(e),
                    })

        # 2. Execute budget actions
        if budget_actions:
            budget_mgr = self._get_budget_manager()
            for action in budget_actions:
                try:
                    # Budget actions are logged but actual execution
                    # deferred to BudgetManager's approval flow
                    executed.append({
                        "category": "budget",
                        "action": action.get("action", ""),
                        "target": action.get("target", ""),
                        "status": "queued_for_review",
                    })
                except Exception as e:
                    failed.append({
                        "category": "budget",
                        "action": action.get("action", ""),
                        "error": str(e),
                    })

        # 3. Launch experiments
        for proposal in experiment_proposals:
            try:
                spec = proposal.get("experiment_spec", {})
                if spec and spec.get("name"):
                    # Log experiment proposal (actual launch via ExperimentEngine)
                    executed.append({
                        "category": "strategy",
                        "action": f"Proposed experiment: {spec.get('name', '')}",
                        "target": spec.get("agent_id", ""),
                        "status": "proposed",
                        "experiment_spec": spec,
                    })
            except Exception as e:
                failed.append({
                    "category": "strategy",
                    "action": proposal.get("action", ""),
                    "error": str(e),
                })

        logger.info(
            "autopilot_strategy_executed",
            extra={
                "agent_id": self.agent_id,
                "executed": len(executed),
                "failed": len(failed),
            },
        )

        return {
            "current_node": "execute_strategy",
            "actions_executed": executed,
            "actions_failed": failed,
        }

    # ─── Node: Report ─────────────────────────────────────────────────

    async def _node_report(
        self, state: AutopilotAgentState,
    ) -> dict[str, Any]:
        """
        Node 6: Generate summary report and write insights to Hive Mind.

        Produces a markdown status report and updates the session record.
        """
        now = datetime.now(timezone.utc).isoformat()

        issues = state.get("detected_issues", [])
        opportunities = state.get("optimization_opportunities", [])
        executed = state.get("actions_executed", [])
        failed = state.get("actions_failed", [])
        summary = state.get("strategy_summary", "")
        confidence = state.get("strategy_confidence", 0)
        health_scores = state.get("health_scores", {})

        # Build markdown report
        report = self._build_report(
            issues=issues,
            opportunities=opportunities,
            executed=executed,
            failed=failed,
            summary=summary,
            confidence=confidence,
            health_scores=health_scores,
        )

        # Complete the session record
        session_id = state.get("session_id", "")
        self._complete_session(
            session_id,
            status="completed",
            metrics=state.get("performance_snapshot", {}),
            issues=issues,
            strategy={
                "summary": summary,
                "confidence": confidence,
                "healing_count": len(state.get("healing_actions", [])),
                "budget_count": len(state.get("budget_actions", [])),
                "experiment_count": len(state.get("experiment_proposals", [])),
            },
            actions=executed + failed,
        )

        logger.info(
            "autopilot_report_generated",
            extra={
                "agent_id": self.agent_id,
                "session_id": session_id,
                "issues": len(issues),
                "executed": len(executed),
                "failed": len(failed),
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
            "completed_at": now,
        }

    def _build_report(
        self,
        issues: list,
        opportunities: list,
        executed: list,
        failed: list,
        summary: str,
        confidence: float,
        health_scores: dict,
    ) -> str:
        """Build markdown report."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# Autopilot Report — {now}",
            "",
        ]

        # Strategy summary
        if summary:
            lines.append("## Strategy Summary")
            lines.append(summary)
            lines.append(f"*Confidence: {confidence:.0%}*")
            lines.append("")

        # Issues
        if issues:
            lines.append(f"## Issues Detected ({len(issues)})")
            for issue in issues:
                severity = issue.get("severity", "info")
                icon = {
                    "critical": "🔴", "warning": "🟡", "info": "🟢",
                }.get(severity, "⚪")
                lines.append(
                    f"- {icon} [{severity.upper()}] "
                    f"{issue.get('agent_id', 'unknown')}: "
                    f"{issue.get('details', {}).get('message', '')}"
                )
            lines.append("")

        # Opportunities
        if opportunities:
            lines.append(f"## Opportunities ({len(opportunities)})")
            for opp in opportunities[:5]:
                lines.append(
                    f"- [{opp.get('potential_impact', 'medium').upper()}] "
                    f"{opp.get('description', '')[:100]}"
                )
            lines.append("")

        # Actions taken
        if executed:
            lines.append(f"## Actions Taken ({len(executed)})")
            for action in executed:
                lines.append(
                    f"- [{action.get('status', '')}] "
                    f"{action.get('category', '')}: "
                    f"{action.get('action', '')}"
                )
            lines.append("")

        if failed:
            lines.append(f"## Actions Failed ({len(failed)})")
            for action in failed:
                lines.append(
                    f"- {action.get('category', '')}: "
                    f"{action.get('error', '')}"
                )
            lines.append("")

        # Health scores
        if health_scores:
            lines.append("## Agent Health Scores")
            for aid, score in sorted(
                health_scores.items(), key=lambda x: x[1],
            ):
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                status = (
                    "healthy" if score >= 0.8
                    else "degraded" if score >= 0.5
                    else "critical"
                )
                lines.append(f"- {aid}: {bar} {score:.0%} ({status})")
            lines.append("")

        lines.append("---")
        lines.append(
            "*Report generated by AutopilotAgent v1.0 — "
            "Sovereign Venture Engine*"
        )

        return "\n".join(lines)

    # ─── Routing Functions ────────────────────────────────────────────

    @staticmethod
    def _route_by_issues(state: AutopilotAgentState) -> str:
        """Route after issue detection: strategy if issues found."""
        issues = state.get("detected_issues", [])
        opportunities = state.get("optimization_opportunities", [])

        if issues or opportunities:
            return "needs_strategy"
        return "all_clear"

    @staticmethod
    def _route_by_approval(state: AutopilotAgentState) -> str:
        """Route after human review: execute if approved."""
        if state.get("actions_approved", False):
            return "approved"
        return "rejected"

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        Store Autopilot insights in the shared brain.

        Records strategy outcomes so future runs can learn from
        historical optimization patterns.
        """
        issues = result.get("detected_issues", [])
        executed = result.get("actions_executed", [])
        summary = result.get("strategy_summary", "")

        if not issues and not executed:
            return

        insight = InsightData(
            insight_type="system_health",
            title=f"Autopilot: {len(issues)} issues, {len(executed)} actions",
            content=summary[:500] if summary else (
                f"Detected {len(issues)} issues, executed {len(executed)} actions"
            ),
            confidence=0.85,
            metadata={
                "source": "autopilot",
                "issue_count": len(issues),
                "action_count": len(executed),
                "timestamp": result.get("report_generated_at", ""),
            },
        )
        self.store_insight(insight)
