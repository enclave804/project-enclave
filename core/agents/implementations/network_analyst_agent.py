"""
Network Analyst Agent — The Perimeter Scout.

Maps the external attack surface of a target domain by analyzing
exposed services, open ports, DNS records, and subdomains. Produces
an attack surface score and network topology assessment.

Architecture (LangGraph State Machine):
    gather_data -> analyze_topology -> assess_attack_surface ->
    human_review -> report -> END

Trigger Events:
    - new_assessment: Triggered alongside VulnScanner for a domain
    - scheduled (weekly): Periodic perimeter monitoring
    - manual: On-demand network analysis

Shared Brain Integration:
    - Reads: previous network assessments, known infrastructure patterns
    - Writes: attack surface scores, exposed service inventory, topology data

Safety:
    - NEVER performs active port scanning -- relies on cached/passive data
    - All assessments require human_review before finalization
    - Only analyzes publicly accessible information
    - Does not attempt to connect to discovered services

Usage:
    agent = NetworkAnalystAgent(config, db, embedder, llm)
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
from core.agents.state import NetworkAnalystAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

HIGH_RISK_PORTS = {21, 22, 23, 25, 445, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 27017}

SERVICE_RISK_WEIGHTS = {
    "telnet": 3.0,
    "ftp": 2.5,
    "rdp": 2.5,
    "smb": 2.0,
    "mysql": 2.0,
    "postgres": 2.0,
    "redis": 2.5,
    "mongodb": 2.5,
    "ssh": 1.0,
    "smtp": 1.5,
    "http": 0.5,
    "https": 0.2,
}

DNS_RECORD_TYPES = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA"]

NETWORK_ANALYST_SYSTEM_PROMPT = """\
You are an expert network security analyst for {company_name}. \
You evaluate the external attack surface of {domain} based on \
DNS records, open ports, exposed services, and subdomain enumeration.

Given the reconnaissance data below, produce a JSON object with:
{{
    "attack_surface_score": 0.0-10.0,
    "topology_summary": "Description of the network architecture",
    "attack_vectors": [
        {{
            "vector": "Short name",
            "severity": "critical|high|medium|low",
            "description": "How this could be exploited",
            "affected_hosts": ["host1", "host2"]
        }}
    ],
    "recommendations": [
        {{
            "priority": "high|medium|low",
            "action": "What to do",
            "reasoning": "Why"
        }}
    ]
}}

Rules:
- Score considers cumulative exposure across all hosts
- Database ports exposed to internet are always critical
- Multiple subdomains increase the attack surface
- Consider service version information for known vulnerabilities

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("network_analyst")
class NetworkAnalystAgent(BaseAgent):
    """
    External perimeter and attack surface analysis agent.

    Nodes:
        1. gather_data          -- Collect DNS, ports, services, subdomains
        2. analyze_topology     -- Map the network topology
        3. assess_attack_surface -- Score attack surface, identify vectors
        4. human_review         -- Gate: review assessment before finalization
        5. report               -- Summary + write insights to Hive Mind
    """

    def build_graph(self) -> Any:
        """Build the Network Analyst's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(NetworkAnalystAgentState)

        workflow.add_node("gather_data", self._node_gather_data)
        workflow.add_node("analyze_topology", self._node_analyze_topology)
        workflow.add_node("assess_attack_surface", self._node_assess_attack_surface)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("gather_data")

        workflow.add_edge("gather_data", "analyze_topology")
        workflow.add_edge("analyze_topology", "assess_attack_surface")
        workflow.add_edge("assess_attack_surface", "human_review")
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
    def get_state_class(cls) -> Type[NetworkAnalystAgentState]:
        return NetworkAnalystAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "company_domain": task.get("company_domain", ""),
            "company_name": task.get("company_name", ""),
            "dns_records": [],
            "subdomains": [],
            "open_ports": [],
            "exposed_services": [],
            "whois_data": {},
            "topology_map": {},
            "attack_surface_score": 0.0,
            "attack_vectors": [],
            "recommendations": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Gather Data ──────────────────────────────────────

    async def _node_gather_data(
        self, state: NetworkAnalystAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Collect DNS records, open ports, exposed services, and subdomains.

        Queries cached reconnaissance data from the database (Shodan, DNS
        resolution results) to build a comprehensive picture of the
        target's external footprint.
        """
        domain = state.get("company_domain", "")

        logger.info("network_gather: starting recon for %s", domain)

        dns_records: list[dict[str, Any]] = []
        subdomains: list[str] = []
        open_ports: list[dict[str, Any]] = []
        exposed_services: list[dict[str, Any]] = []
        whois_data: dict[str, Any] = {}

        # --- DNS Records ---
        try:
            result = (
                self.db.client.table("dns_cache")
                .select("*")
                .eq("domain", domain)
                .execute()
            )
            for record in (result.data or []):
                dns_records.append({
                    "type": record.get("record_type", ""),
                    "name": record.get("name", ""),
                    "value": record.get("value", ""),
                    "ttl": record.get("ttl", 0),
                })
        except Exception as e:
            logger.debug(f"DNS lookup failed for {domain}: {e}")

        # --- Subdomains ---
        try:
            result = (
                self.db.client.table("subdomain_cache")
                .select("subdomain")
                .eq("parent_domain", domain)
                .execute()
            )
            subdomains = [r.get("subdomain", "") for r in (result.data or []) if r.get("subdomain")]
        except Exception as e:
            logger.debug(f"Subdomain enum failed for {domain}: {e}")

        # --- Shodan data (ports and services) ---
        try:
            result = (
                self.db.client.table("tech_stack_cache")
                .select("shodan_data")
                .eq("domain", domain)
                .limit(1)
                .execute()
            )
            shodan = result.data[0].get("shodan_data", {}) if result.data else {}

            for port_info in shodan.get("services", []):
                port = port_info.get("port", 0)
                service = port_info.get("service", "unknown")
                version = port_info.get("version", "")

                open_ports.append({
                    "host": domain,
                    "port": port,
                    "service": service,
                    "version": version,
                })

                risk_level = "low"
                if port in HIGH_RISK_PORTS:
                    risk_level = "high"
                    if port in (23, 6379, 27017):
                        risk_level = "critical"

                exposed_services.append({
                    "service": service,
                    "version": version,
                    "host": domain,
                    "port": port,
                    "risk_level": risk_level,
                })

            # Fall back to raw port list
            if not open_ports and shodan.get("ports"):
                for port in shodan["ports"]:
                    open_ports.append({"host": domain, "port": port, "service": "unknown", "version": ""})
        except Exception as e:
            logger.debug(f"Shodan lookup failed for {domain}: {e}")

        # --- WHOIS ---
        try:
            result = (
                self.db.client.table("whois_cache")
                .select("*")
                .eq("domain", domain)
                .limit(1)
                .execute()
            )
            if result.data:
                whois_data = result.data[0]
        except Exception as e:
            logger.debug(f"WHOIS lookup failed for {domain}: {e}")

        logger.info(
            "network_gather_complete: dns=%d, subdomains=%d, ports=%d, services=%d",
            len(dns_records), len(subdomains), len(open_ports), len(exposed_services),
        )

        return {
            "current_node": "gather_data",
            "dns_records": dns_records,
            "subdomains": subdomains,
            "open_ports": open_ports,
            "exposed_services": exposed_services,
            "whois_data": whois_data,
        }

    # ─── Node 2: Analyze Topology ────────────────────────────────

    async def _node_analyze_topology(
        self, state: NetworkAnalystAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Build a network topology map from gathered data.

        Organizes hosts, services, and DNS relationships into a
        structured topology representation.
        """
        domain = state.get("company_domain", "")
        dns_records = state.get("dns_records", [])
        subdomains = state.get("subdomains", [])
        services = state.get("exposed_services", [])
        open_ports = state.get("open_ports", [])

        logger.info("network_topology: mapping for %s", domain)

        # Build topology map
        hosts: dict[str, dict[str, Any]] = {}

        # Primary domain
        hosts[domain] = {
            "type": "primary",
            "ports": [p["port"] for p in open_ports if p.get("host") == domain],
            "services": [s["service"] for s in services if s.get("host") == domain],
        }

        # Subdomains
        for sub in subdomains:
            full = f"{sub}.{domain}" if not sub.endswith(domain) else sub
            hosts[full] = {
                "type": "subdomain",
                "ports": [],
                "services": [],
            }

        # DNS-derived hosts
        for record in dns_records:
            if record.get("type") == "MX":
                mx_host = record.get("value", "")
                if mx_host and mx_host not in hosts:
                    hosts[mx_host] = {"type": "mail_server", "ports": [], "services": ["smtp"]}

        topology_map = {
            "domain": domain,
            "host_count": len(hosts),
            "hosts": hosts,
            "total_open_ports": len(open_ports),
            "total_services": len(services),
            "dns_record_count": len(dns_records),
            "has_mail_server": any(r.get("type") == "MX" for r in dns_records),
            "subdomain_count": len(subdomains),
        }

        return {
            "current_node": "analyze_topology",
            "topology_map": topology_map,
        }

    # ─── Node 3: Assess Attack Surface ───────────────────────────

    async def _node_assess_attack_surface(
        self, state: NetworkAnalystAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Score the attack surface and identify attack vectors.

        Computes a 0-10 attack surface score based on exposed services,
        open ports, subdomain count, and service risk weights.
        Uses LLM for detailed vector analysis.
        """
        domain = state.get("company_domain", "")
        services = state.get("exposed_services", [])
        open_ports = state.get("open_ports", [])
        subdomains = state.get("subdomains", [])
        topology = state.get("topology_map", {})

        logger.info("network_assess: scoring attack surface for %s", domain)

        # Compute attack surface score
        score = 0.0
        for svc in services:
            svc_name = svc.get("service", "unknown").lower()
            weight = SERVICE_RISK_WEIGHTS.get(svc_name, 0.5)
            if svc.get("risk_level") == "critical":
                weight *= 2.0
            score += weight

        # Subdomain sprawl penalty
        score += min(2.0, len(subdomains) * 0.2)

        # Non-standard port penalty
        non_standard = [p for p in open_ports if p.get("port", 0) not in (80, 443)]
        score += min(1.5, len(non_standard) * 0.3)

        attack_surface_score = min(10.0, round(score, 1))

        # Build attack vectors and recommendations via LLM
        attack_vectors: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []

        company_name = state.get("company_name", "") or self.config.params.get("company_name", "the target")
        try:
            recon_summary = json.dumps({
                "open_ports": [{"port": p.get("port"), "service": p.get("service")} for p in open_ports[:15]],
                "services": [{"service": s.get("service"), "risk": s.get("risk_level")} for s in services[:15]],
                "subdomains": subdomains[:20],
                "topology": {
                    "host_count": topology.get("host_count", 0),
                    "total_open_ports": topology.get("total_open_ports", 0),
                },
            }, indent=2)

            prompt = (
                f"Reconnaissance data for {domain}:\n{recon_summary}\n\n"
                f"Attack surface score: {attack_surface_score}/10\n\n"
                f"Generate the attack surface assessment JSON."
            )
            system = NETWORK_ANALYST_SYSTEM_PROMPT.format(
                company_name=company_name,
                domain=domain,
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
                    attack_vectors = parsed.get("attack_vectors", [])
                    recommendations = parsed.get("recommendations", [])
                    if parsed.get("attack_surface_score"):
                        attack_surface_score = float(parsed["attack_surface_score"])
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Failed to parse LLM network analysis response")
        except Exception as e:
            logger.debug(f"LLM network analysis failed: {e}")

        # Fallback rule-based vectors if LLM did not produce any
        if not attack_vectors:
            for svc in services:
                if svc.get("risk_level") in ("critical", "high"):
                    attack_vectors.append({
                        "vector": f"Exposed {svc.get('service', 'unknown')} on port {svc.get('port', 0)}",
                        "severity": svc.get("risk_level", "high"),
                        "description": f"{svc.get('service')} is publicly accessible.",
                        "affected_hosts": [svc.get("host", domain)],
                    })

        # Publish insight to Hive Mind
        self.store_insight(InsightData(
            insight_type="attack_surface_assessment",
            title=f"Network: {domain} — Attack Surface {attack_surface_score:.1f}/10",
            content=(
                f"Network analysis of {domain}: {len(open_ports)} open ports, "
                f"{len(services)} exposed services, {len(subdomains)} subdomains. "
                f"Attack surface score: {attack_surface_score:.1f}/10."
            ),
            confidence=0.80,
            metadata={
                "domain": domain,
                "attack_surface_score": attack_surface_score,
                "port_count": len(open_ports),
                "service_count": len(services),
            },
        ))

        logger.info(
            "network_assess_complete: score=%.1f, vectors=%d, recommendations=%d",
            attack_surface_score, len(attack_vectors), len(recommendations),
        )

        return {
            "current_node": "assess_attack_surface",
            "attack_surface_score": attack_surface_score,
            "attack_vectors": attack_vectors,
            "recommendations": recommendations,
            "knowledge_written": True,
        }

    # ─── Node 4: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: NetworkAnalystAgentState
    ) -> dict[str, Any]:
        """Node 4: Present network assessment for human approval."""
        logger.info(
            "network_human_review_pending: score=%.1f, vectors=%d",
            state.get("attack_surface_score", 0.0),
            len(state.get("attack_vectors", [])),
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: NetworkAnalystAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate network analysis summary report."""
        now = datetime.now(timezone.utc).isoformat()
        domain = state.get("company_domain", "")
        topology = state.get("topology_map", {})

        sections = [
            "# Network Analysis Report",
            f"**Domain:** {domain}",
            f"*Generated: {now}*\n",
            "## Attack Surface Score",
            f"**{state.get('attack_surface_score', 0.0):.1f} / 10.0**\n",
            "## Topology Overview",
            f"- **Hosts Discovered:** {topology.get('host_count', 0)}",
            f"- **Open Ports:** {topology.get('total_open_ports', 0)}",
            f"- **Exposed Services:** {topology.get('total_services', 0)}",
            f"- **Subdomains:** {topology.get('subdomain_count', 0)}",
            f"- **DNS Records:** {topology.get('dns_record_count', 0)}",
        ]

        vectors = state.get("attack_vectors", [])
        if vectors:
            sections.append(f"\n## Attack Vectors ({len(vectors)})")
            for v in vectors[:10]:
                sections.append(
                    f"- **[{v.get('severity', 'medium').upper()}]** "
                    f"{v.get('vector', 'Unknown')}: {v.get('description', '')}"
                )

        recs = state.get("recommendations", [])
        if recs:
            sections.append(f"\n## Recommendations ({len(recs)})")
            for r in recs[:10]:
                sections.append(
                    f"- **[{r.get('priority', 'medium').upper()}]** "
                    f"{r.get('action', '')}"
                )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: NetworkAnalystAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── Knowledge ───────────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<NetworkAnalystAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
