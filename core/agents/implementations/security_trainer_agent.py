"""
Security Trainer Agent — The Awareness Coach.

Generates security awareness training modules, phishing simulation
scenarios, and human-layer risk scoring. Assesses the organization's
human risk surface and delivers targeted training content.

Architecture (LangGraph State Machine):
    assess_human_risk → generate_training → create_phishing_sims →
    human_review → deliver → report → END

Trigger Events:
    - security_assessment: Full human-risk evaluation + training plan
    - training_request: Generate specific training modules
    - manual: On-demand training content generation

Shared Brain Integration:
    - Reads: IAM findings, incident readiness gaps, compliance requirements
    - Writes: phishing susceptibility patterns, training effectiveness data

Safety:
    - NEVER sends actual phishing emails (simulation scenarios only)
    - All training content requires human_review before delivery
    - Phishing simulations are templates, not live attacks
    - Training is educational, not punitive

Usage:
    agent = SecurityTrainerAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_name": "Acme Corp",
        "employee_count": 150,
        "questionnaire": {...},
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import SecurityTrainerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

TRAINING_TOPICS = {
    "phishing_awareness": {
        "title": "Phishing & Social Engineering Awareness",
        "difficulty": "beginner",
        "duration_minutes": 20,
        "priority": 1,
    },
    "password_hygiene": {
        "title": "Password Security & Credential Management",
        "difficulty": "beginner",
        "duration_minutes": 15,
        "priority": 2,
    },
    "data_handling": {
        "title": "Secure Data Handling & Classification",
        "difficulty": "intermediate",
        "duration_minutes": 25,
        "priority": 3,
    },
    "remote_work": {
        "title": "Remote Work & BYOD Security",
        "difficulty": "beginner",
        "duration_minutes": 15,
        "priority": 4,
    },
    "incident_reporting": {
        "title": "Incident Recognition & Reporting",
        "difficulty": "beginner",
        "duration_minutes": 10,
        "priority": 5,
    },
    "insider_threats": {
        "title": "Insider Threat Awareness",
        "difficulty": "intermediate",
        "duration_minutes": 20,
        "priority": 6,
    },
    "cloud_security_basics": {
        "title": "Cloud Security Best Practices",
        "difficulty": "intermediate",
        "duration_minutes": 25,
        "priority": 7,
    },
    "executive_security": {
        "title": "Executive & VIP Security Awareness",
        "difficulty": "advanced",
        "duration_minutes": 30,
        "priority": 8,
    },
}

PHISHING_TEMPLATES = {
    "credential_harvest": {
        "name": "Credential Harvesting — Fake Login Page",
        "difficulty": "easy",
        "description": "Simulated email with link to fake SSO login page",
    },
    "invoice_scam": {
        "name": "Invoice/Payment Scam",
        "difficulty": "medium",
        "description": "Fake invoice with urgency language requesting payment",
    },
    "ceo_fraud": {
        "name": "CEO Fraud / Business Email Compromise",
        "difficulty": "hard",
        "description": "Spoofed executive email requesting wire transfer or data",
    },
    "attachment_malware": {
        "name": "Malicious Attachment",
        "difficulty": "easy",
        "description": "Email with suspicious attachment (macro-enabled doc)",
    },
    "tech_support_scam": {
        "name": "IT Support Impersonation",
        "difficulty": "medium",
        "description": "Fake IT department requesting password reset or remote access",
    },
    "spear_phish": {
        "name": "Targeted Spear Phishing",
        "difficulty": "hard",
        "description": "Personalized attack using public info about the target",
    },
}

RISK_CATEGORIES = {
    "critical": {"min_score": 80, "label": "Critical", "action": "Immediate intervention"},
    "high": {"min_score": 60, "label": "High", "action": "Quarterly training mandatory"},
    "medium": {"min_score": 40, "label": "Medium", "action": "Annual training recommended"},
    "low": {"min_score": 0, "label": "Low", "action": "Maintenance training"},
}

TRAINER_SYSTEM_PROMPT = """\
You are a cybersecurity awareness training specialist. Based on the \
organization profile and risk assessment data below, generate a \
targeted training plan.

Produce a JSON object with:
{{
    "training_modules": [
        {{"topic": "...", "title": "...", "content_outline": "...", \
"difficulty": "beginner|intermediate|advanced", \
"target_audience": "...", "key_takeaways": ["..."]}}
    ],
    "phishing_scenarios": [
        {{"name": "...", "difficulty": "easy|medium|hard", \
"target_department": "...", "scenario_description": "...", \
"indicators_to_spot": ["..."]}}
    ],
    "summary": "Brief training plan summary"
}}

Company: {company_name}
Employee Count: {employee_count}
Industry: {industry}
Current Phishing Click Rate: {click_rate}
Previous Training Completion: {training_rate}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("security_trainer")
class SecurityTrainerAgent(BaseAgent):
    """
    Security awareness training agent for Enclave Guard engagements.

    Nodes:
        1. assess_human_risk     -- Score human-layer risk from questionnaire
        2. generate_training     -- Create tailored training modules
        3. create_phishing_sims  -- Generate phishing simulation scenarios
        4. human_review          -- Gate: approve content before delivery
        5. deliver               -- Schedule training and simulation delivery
        6. report                -- Generate training plan report
    """

    def build_graph(self) -> Any:
        """Build the Security Trainer Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(SecurityTrainerAgentState)

        workflow.add_node("assess_human_risk", self._node_assess_human_risk)
        workflow.add_node("generate_training", self._node_generate_training)
        workflow.add_node("create_phishing_sims", self._node_create_phishing_sims)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("deliver", self._node_deliver)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("assess_human_risk")

        workflow.add_edge("assess_human_risk", "generate_training")
        workflow.add_edge("generate_training", "create_phishing_sims")
        workflow.add_edge("create_phishing_sims", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "deliver",
                "rejected": "report",
            },
        )
        workflow.add_edge("deliver", "report")
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
    def get_state_class(cls) -> Type[SecurityTrainerAgentState]:
        return SecurityTrainerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "employee_count": (task or {}).get("employee_count", 0),
            "department_risks": [],
            "previous_incidents": [],
            "phishing_click_rate": 0.0,
            "training_completion_rate": 0.0,
            "human_risk_score": 0.0,
            "training_modules": [],
            "training_topics_covered": [],
            "module_count": 0,
            "phishing_scenarios": [],
            "simulation_count": 0,
            "phishing_difficulty_mix": {},
            "delivery_schedule": [],
            "modules_delivered": 0,
            "simulations_launched": 0,
            "content_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Assess Human Risk ──────────────────────────────────

    async def _node_assess_human_risk(
        self, state: SecurityTrainerAgentState
    ) -> dict[str, Any]:
        """Node 1: Score human-layer risk from questionnaire and incident data."""
        task = state.get("task_input", {})
        responses = task.get("questionnaire", {})
        company = task.get("company_name", "Unknown")
        employee_count = task.get("employee_count", responses.get("employee_count", 0))

        logger.info("trainer_assess_risk", extra={"company": company})

        click_rate = float(responses.get("phishing_click_rate", 0.30))
        training_rate = float(responses.get("training_completion_rate", 0.0))
        previous_incidents = responses.get("previous_security_incidents", [])
        if isinstance(previous_incidents, str):
            previous_incidents = [{"description": previous_incidents}]

        # Score departments
        departments = responses.get("departments", [
            "executive", "finance", "hr", "engineering", "sales", "operations",
        ])
        department_risks: list[dict[str, Any]] = []

        high_risk_depts = {"executive", "finance", "hr"}
        for dept in departments:
            dept_name = dept if isinstance(dept, str) else dept.get("name", "unknown")
            if dept_name.lower() in high_risk_depts:
                risk_level = "high"
                reasoning = f"{dept_name} handles sensitive data and is a prime phishing target"
            else:
                risk_level = "medium"
                reasoning = f"{dept_name} has standard security exposure"

            department_risks.append({
                "department": dept_name,
                "risk_level": risk_level,
                "reasoning": reasoning,
            })

        # Calculate overall human risk score (0-100)
        risk_score = 0.0
        risk_score += click_rate * 40  # Click rate heavily weighted
        risk_score += (1.0 - training_rate) * 25  # Low training = high risk
        risk_score += min(len(previous_incidents) * 10, 20)  # Incident history
        if not responses.get("security_awareness_program", False):
            risk_score += 15  # No formal program

        risk_score = min(100.0, max(0.0, risk_score))

        logger.info(
            "trainer_risk_assessed",
            extra={
                "risk_score": risk_score,
                "click_rate": click_rate,
                "training_rate": training_rate,
            },
        )

        return {
            "current_node": "assess_human_risk",
            "employee_count": employee_count,
            "department_risks": department_risks,
            "previous_incidents": previous_incidents,
            "phishing_click_rate": click_rate,
            "training_completion_rate": training_rate,
            "human_risk_score": round(risk_score, 1),
        }

    # ─── Node 2: Generate Training ──────────────────────────────────

    async def _node_generate_training(
        self, state: SecurityTrainerAgentState
    ) -> dict[str, Any]:
        """Node 2: Create tailored training modules based on risk assessment."""
        risk_score = state.get("human_risk_score", 50)
        click_rate = state.get("phishing_click_rate", 0.3)
        dept_risks = state.get("department_risks", [])

        logger.info(
            "trainer_generate_modules",
            extra={"risk_score": risk_score},
        )

        modules: list[dict[str, Any]] = []
        topics_covered: list[str] = []

        # Always include phishing awareness (top priority)
        modules.append({
            "topic": "phishing_awareness",
            "title": TRAINING_TOPICS["phishing_awareness"]["title"],
            "difficulty": "beginner",
            "duration_minutes": 20,
            "target_audience": "all_employees",
            "key_takeaways": [
                "Identify common phishing indicators",
                "Verify sender identity before clicking links",
                "Report suspicious emails to security team",
            ],
        })
        topics_covered.append("phishing_awareness")

        # Add modules based on risk score
        if risk_score >= 60:
            # High risk: comprehensive training
            for topic_id in ["password_hygiene", "data_handling", "incident_reporting"]:
                info = TRAINING_TOPICS[topic_id]
                modules.append({
                    "topic": topic_id,
                    "title": info["title"],
                    "difficulty": info["difficulty"],
                    "duration_minutes": info["duration_minutes"],
                    "target_audience": "all_employees",
                    "key_takeaways": [
                        f"Core {info['title'].lower()} principles",
                        "Organizational policies and procedures",
                        "Practical exercises and scenarios",
                    ],
                })
                topics_covered.append(topic_id)

        if risk_score >= 40:
            # Medium+ risk: add remote work if applicable
            modules.append({
                "topic": "remote_work",
                "title": TRAINING_TOPICS["remote_work"]["title"],
                "difficulty": "beginner",
                "duration_minutes": 15,
                "target_audience": "remote_workers",
                "key_takeaways": [
                    "Secure home network configuration",
                    "VPN and encrypted communication usage",
                    "Physical security of devices",
                ],
            })
            topics_covered.append("remote_work")

        # Executive-specific training for high-risk departments
        high_risk_depts = [d for d in dept_risks if d.get("risk_level") == "high"]
        if high_risk_depts:
            modules.append({
                "topic": "executive_security",
                "title": TRAINING_TOPICS["executive_security"]["title"],
                "difficulty": "advanced",
                "duration_minutes": 30,
                "target_audience": ", ".join(d["department"] for d in high_risk_depts),
                "key_takeaways": [
                    "Targeted attack patterns (whaling, BEC)",
                    "Secure communication practices",
                    "Travel security and physical awareness",
                ],
            })
            topics_covered.append("executive_security")

        return {
            "current_node": "generate_training",
            "training_modules": modules,
            "training_topics_covered": topics_covered,
            "module_count": len(modules),
        }

    # ─── Node 3: Create Phishing Sims ────────────────────────────────

    async def _node_create_phishing_sims(
        self, state: SecurityTrainerAgentState
    ) -> dict[str, Any]:
        """Node 3: Generate phishing simulation scenarios."""
        risk_score = state.get("human_risk_score", 50)
        dept_risks = state.get("department_risks", [])

        logger.info("trainer_create_phishing", extra={"risk_score": risk_score})

        scenarios: list[dict[str, Any]] = []
        difficulty_mix: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}

        # Always start with easy scenarios
        for tmpl_id in ["credential_harvest", "attachment_malware"]:
            tmpl = PHISHING_TEMPLATES[tmpl_id]
            scenarios.append({
                "name": tmpl["name"],
                "template_id": tmpl_id,
                "difficulty": tmpl["difficulty"],
                "target_department": "all",
                "scenario_description": tmpl["description"],
                "indicators_to_spot": [
                    "Suspicious sender address",
                    "Urgency language",
                    "Mismatched URL",
                ],
            })
            difficulty_mix[tmpl["difficulty"]] += 1

        # Add medium difficulty for moderate+ risk
        if risk_score >= 40:
            for tmpl_id in ["invoice_scam", "tech_support_scam"]:
                tmpl = PHISHING_TEMPLATES[tmpl_id]
                target = "finance" if tmpl_id == "invoice_scam" else "all"
                scenarios.append({
                    "name": tmpl["name"],
                    "template_id": tmpl_id,
                    "difficulty": tmpl["difficulty"],
                    "target_department": target,
                    "scenario_description": tmpl["description"],
                    "indicators_to_spot": [
                        "Unexpected request pattern",
                        "Pressure to act quickly",
                        "Request for sensitive information",
                    ],
                })
                difficulty_mix[tmpl["difficulty"]] += 1

        # Add hard scenarios for high-risk orgs
        if risk_score >= 60:
            for tmpl_id in ["ceo_fraud", "spear_phish"]:
                tmpl = PHISHING_TEMPLATES[tmpl_id]
                scenarios.append({
                    "name": tmpl["name"],
                    "template_id": tmpl_id,
                    "difficulty": tmpl["difficulty"],
                    "target_department": "executive,finance",
                    "scenario_description": tmpl["description"],
                    "indicators_to_spot": [
                        "Verify through secondary channel",
                        "Check email headers for spoofing",
                        "Validate unusual requests with manager",
                    ],
                })
                difficulty_mix[tmpl["difficulty"]] += 1

        return {
            "current_node": "create_phishing_sims",
            "phishing_scenarios": scenarios,
            "simulation_count": len(scenarios),
            "phishing_difficulty_mix": difficulty_mix,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: SecurityTrainerAgentState
    ) -> dict[str, Any]:
        """Node 4: Present training content for human approval."""
        modules = state.get("training_modules", [])
        sims = state.get("phishing_scenarios", [])

        logger.info(
            "trainer_human_review_pending",
            extra={
                "modules": len(modules),
                "simulations": len(sims),
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Deliver ─────────────────────────────────────────────

    async def _node_deliver(
        self, state: SecurityTrainerAgentState
    ) -> dict[str, Any]:
        """Node 5: Schedule training module and simulation delivery."""
        modules = state.get("training_modules", [])
        sims = state.get("phishing_scenarios", [])
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")

        logger.info(
            "trainer_deliver",
            extra={"modules": len(modules), "sims": len(sims)},
        )

        # Create delivery schedule (staggered over weeks)
        schedule: list[dict[str, Any]] = []
        base_date = datetime.now(timezone.utc)

        for i, module in enumerate(modules):
            delivery_date = base_date + timedelta(weeks=i)
            schedule.append({
                "module_topic": module.get("topic", ""),
                "module_title": module.get("title", ""),
                "target_date": delivery_date.date().isoformat(),
                "audience": module.get("target_audience", "all_employees"),
                "type": "training",
            })

        # Schedule simulations after training
        sim_start = base_date + timedelta(weeks=len(modules))
        for i, sim in enumerate(sims):
            sim_date = sim_start + timedelta(weeks=i)
            schedule.append({
                "simulation_name": sim.get("name", ""),
                "target_date": sim_date.date().isoformat(),
                "audience": sim.get("target_department", "all"),
                "type": "phishing_simulation",
            })

        # Write insight to shared brain
        self.store_insight(InsightData(
            insight_type="security_training",
            title=f"Training Plan: {company} — "
                  f"{len(modules)} modules, {len(sims)} simulations",
            content=(
                f"Security training plan for {company}: "
                f"{len(modules)} training modules and {len(sims)} phishing "
                f"simulations scheduled. Human risk score: "
                f"{state.get('human_risk_score', 0)}/100. "
                f"Current phishing click rate: "
                f"{state.get('phishing_click_rate', 0):.0%}."
            ),
            confidence=0.80,
            metadata={
                "company": company,
                "module_count": len(modules),
                "simulation_count": len(sims),
                "risk_score": state.get("human_risk_score", 0),
            },
        ))

        return {
            "current_node": "deliver",
            "delivery_schedule": schedule,
            "modules_delivered": len(modules),
            "simulations_launched": len(sims),
            "content_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: SecurityTrainerAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate security training plan report."""
        now = datetime.now(timezone.utc).isoformat()
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")

        risk_score = state.get("human_risk_score", 0)
        risk_cat = "low"
        for cat, cfg in RISK_CATEGORIES.items():
            if risk_score >= cfg["min_score"]:
                risk_cat = cat
                break

        sections = [
            "# Security Awareness Training Report",
            f"*Generated: {now}*\n",
            f"## Organization: {company}",
            f"- **Employee Count:** {state.get('employee_count', 'N/A')}",
            f"\n## Human Risk Assessment",
            f"- **Risk Score:** {risk_score}/100 ({risk_cat.upper()})",
            f"- **Phishing Click Rate:** {state.get('phishing_click_rate', 0):.0%}",
            f"- **Training Completion:** {state.get('training_completion_rate', 0):.0%}",
            f"- **Recommended Action:** {RISK_CATEGORIES.get(risk_cat, {}).get('action', 'N/A')}",
        ]

        # Department risks
        dept_risks = state.get("department_risks", [])
        if dept_risks:
            sections.append("\n## Department Risk Levels")
            for dept in dept_risks:
                sections.append(
                    f"- **{dept['department']}:** {dept['risk_level'].upper()}"
                )

        # Training modules
        modules = state.get("training_modules", [])
        sections.append(f"\n## Training Modules ({len(modules)})")
        for m in modules:
            sections.append(
                f"- **{m.get('title', 'N/A')}** "
                f"({m.get('difficulty', 'N/A')}, "
                f"{m.get('duration_minutes', 0)} min) — "
                f"Audience: {m.get('target_audience', 'all')}"
            )

        # Phishing simulations
        sims = state.get("phishing_scenarios", [])
        mix = state.get("phishing_difficulty_mix", {})
        sections.append(f"\n## Phishing Simulations ({len(sims)})")
        sections.append(
            f"- **Difficulty Mix:** Easy: {mix.get('easy', 0)}, "
            f"Medium: {mix.get('medium', 0)}, Hard: {mix.get('hard', 0)}"
        )
        for s in sims:
            sections.append(
                f"- **{s.get('name', 'N/A')}** ({s.get('difficulty', 'N/A')}) — "
                f"Target: {s.get('target_department', 'all')}"
            )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: SecurityTrainerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<SecurityTrainerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
