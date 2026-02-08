"""
LangGraph pipeline nodes for Project Enclave.

Each function is a node in the sales pipeline graph.
Nodes read from LeadState, perform work, and return updated state.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from anthropic import Anthropic

from core.config.schema import VerticalConfig
from core.graph.state import LeadState
from core.integrations.supabase_client import EnclaveDB
from core.rag.embeddings import EmbeddingEngine
from core.rag.ingestion import KnowledgeIngester
from core.rag.retrieval import KnowledgeRetriever

logger = logging.getLogger(__name__)


class PipelineNodes:
    """
    Container for all pipeline node functions.

    Each method is a LangGraph node that receives and returns LeadState.
    Dependencies (DB, RAG, APIs) are injected via the constructor.
    """

    def __init__(
        self,
        config: VerticalConfig,
        db: EnclaveDB,
        apollo: Any,  # ApolloClient or MockApolloClient
        retriever: KnowledgeRetriever,
        ingester: KnowledgeIngester,
        anthropic_client: Anthropic,
    ):
        self.config = config
        self.db = db
        self.apollo = apollo
        self.retriever = retriever
        self.ingester = ingester
        self.llm = anthropic_client

    # ------------------------------------------------------------------
    # Node 1: Check Duplicate
    # ------------------------------------------------------------------

    async def check_duplicate(self, state: LeadState) -> dict:
        """
        Check if we've contacted this lead within the cooldown period.

        Queries both the relational DB and the RAG store for prior interactions.
        """
        start = time.time()
        updates: dict[str, Any] = {"current_node": "check_duplicate"}

        email = state.get("contact_email", "")
        domain = state.get("company_domain", "")

        if not email:
            updates["is_duplicate"] = False
            return updates

        # Check relational DB for existing contact
        existing = self.db.get_contact_by_email(email)
        if existing:
            updates["contact_id"] = existing["id"]
            updates["company_id"] = existing.get("company_id", "")

            # Check recent outreach
            cooldown = self.config.pipeline.duplicate_cooldown_days
            recent = self.db.get_recent_outreach_for_contact(
                existing["id"], days=cooldown
            )
            if recent:
                updates["is_duplicate"] = True
                updates["last_contacted_at"] = recent[0].get("sent_at")
                updates["skip_reason"] = (
                    f"Contacted {len(recent)} times in last {cooldown} days"
                )
                logger.info(f"Duplicate: {email} contacted recently")
                return updates

        # Check RAG for any prior outreach context
        prior = await self.retriever.find_previous_outreach(domain, email)
        updates["previous_outreach"] = prior
        updates["is_duplicate"] = False

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="check_duplicate",
            status="completed",
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 2: Enrich Company
    # ------------------------------------------------------------------

    async def enrich_company(self, state: LeadState) -> dict:
        """
        Enrich company data from external sources.

        Pulls from Apollo enrichment and stores in Supabase.
        Shodan/BuiltWith enrichment is handled by vertical-specific
        enrichment modules (see verticals/enclave_guard/enrichment/).
        """
        start = time.time()
        updates: dict[str, Any] = {"current_node": "enrich_company"}

        domain = state.get("company_domain", "")
        if not domain:
            updates["error"] = "No company domain available for enrichment"
            return updates

        # Upsert company in DB
        company_data = {
            "name": state.get("company_name", ""),
            "domain": domain,
            "industry": state.get("company_industry", ""),
            "employee_count": state.get("company_size"),
            "tech_stack": state.get("tech_stack", {}),
            "apollo_id": state.get("apollo_org_id", ""),
            "website_url": state.get("raw_apollo_data", {}).get("company", {}).get("website_url", ""),
            "linkedin_url": state.get("raw_apollo_data", {}).get("company", {}).get("linkedin_url", ""),
            "source": "apollo",
        }
        company = self.db.upsert_company(company_data)
        updates["company_id"] = company.get("id", "")

        # Try Apollo company enrichment for additional data
        try:
            apollo_enrichment = await self.apollo.enrich_company(domain)
            org = apollo_enrichment.get("organization", {})
            if org:
                additional_tech = {}
                for tech in org.get("current_technologies", []):
                    if isinstance(tech, dict):
                        additional_tech[tech.get("name", "")] = tech.get("category", "")
                    elif isinstance(tech, str):
                        additional_tech[tech] = "unknown"

                merged_tech = {**state.get("tech_stack", {}), **additional_tech}
                updates["tech_stack"] = merged_tech
                updates["company_industry"] = org.get("industry", state.get("company_industry", ""))
                updates["company_size"] = org.get("estimated_num_employees", state.get("company_size", 0)) or 0

                # Update company in DB with enriched data
                self.db.upsert_company({
                    "domain": domain,
                    "tech_stack": merged_tech,
                    "industry": updates["company_industry"],
                    "employee_count": updates["company_size"],
                    "enrichment_data": org,
                    "enriched_at": datetime.now(timezone.utc).isoformat(),
                })

                enrichment_sources = list(state.get("enrichment_sources", []))
                if "apollo_enrichment" not in enrichment_sources:
                    enrichment_sources.append("apollo_enrichment")
                updates["enrichment_sources"] = enrichment_sources

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.warning(f"Apollo enrichment failed for {domain}: {e}")
            # Non-fatal: we still have the initial Apollo data

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="enrich_company",
            status="completed",
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 3: Enrich Contact
    # ------------------------------------------------------------------

    async def enrich_contact(self, state: LeadState) -> dict:
        """Upsert the contact record in Supabase."""
        start = time.time()
        updates: dict[str, Any] = {"current_node": "enrich_contact"}

        contact_data = {
            "company_id": state.get("company_id", ""),
            "name": state.get("contact_name", ""),
            "title": state.get("contact_title", ""),
            "email": state.get("contact_email", ""),
            "linkedin_url": state.get("raw_apollo_data", {}).get("contact", {}).get("linkedin_url", ""),
            "phone": state.get("raw_apollo_data", {}).get("contact", {}).get("phone"),
            "seniority": state.get("contact_seniority", ""),
            "apollo_id": state.get("apollo_person_id", ""),
        }
        contact = self.db.upsert_contact(contact_data)
        updates["contact_id"] = contact.get("id", "")

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="enrich_contact",
            status="completed",
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 4: Qualify Lead
    # ------------------------------------------------------------------

    async def qualify_lead(self, state: LeadState) -> dict:
        """
        Score and qualify the lead against the ICP from config.

        Checks:
        - Company size within range
        - Industry match
        - Positive signals present
        - No disqualifiers triggered
        """
        start = time.time()
        updates: dict[str, Any] = {"current_node": "qualify_lead"}

        icp = self.config.targeting.ideal_customer_profile
        score = 0.0
        max_score = 0.0
        matching_signals: list[str] = []
        matching_disqualifiers: list[str] = []

        # Company size check (0.3 weight)
        max_score += 0.3
        size = state.get("company_size", 0)
        if icp.company_size[0] <= size <= icp.company_size[1]:
            score += 0.3
            matching_signals.append(f"company_size_{size}")

        # Industry check (0.3 weight)
        max_score += 0.3
        industry = (state.get("company_industry", "") or "").lower()
        if any(ind.lower() in industry for ind in icp.industries):
            score += 0.3
            matching_signals.append(f"industry_{industry}")

        # Positive signals (0.3 weight, distributed)
        tech_stack = state.get("tech_stack", {})
        vulns = state.get("vulnerabilities", [])
        signal_weight = 0.3 / max(len(icp.signals), 1)
        max_score += 0.3

        # Pre-build searchable strings (used for both signals and disqualifiers)
        tech_str = " ".join(
            f"{k} {v}" for k, v in tech_stack.items()
        ).lower()
        vuln_str = " ".join(str(v) for v in vulns).lower()

        for signal in icp.signals:
            signal_lower = signal.lower()
            # Check tech stack
            if signal_lower in tech_str:
                score += signal_weight
                matching_signals.append(signal)
                continue
            # Check vulnerabilities
            if signal_lower in vuln_str:
                score += signal_weight
                matching_signals.append(signal)

        # Contact title match (0.1 weight)
        max_score += 0.1
        title = (state.get("contact_title", "") or "").lower()
        for persona in self.config.targeting.personas:
            for pattern in persona.title_patterns:
                if pattern.lower() in title:
                    score += 0.1
                    matching_signals.append(f"persona_{persona.id}")
                    break

        # Disqualifier check
        company_size = state.get("company_size", 0) or 0
        for disq in icp.disqualifiers:
            disq_lower = disq.lower()
            # Check if disqualifier mentions large security teams
            if "10plus" in disq_lower and "security_team" in disq_lower:
                # Companies with 500+ employees likely have internal security
                if company_size >= 500:
                    matching_disqualifiers.append(disq)
                continue
            # Check if disqualifier matches tech stack
            if disq_lower in tech_str:
                matching_disqualifiers.append(disq)

        # Normalize score
        normalized_score = score / max_score if max_score > 0 else 0.0
        qualified = normalized_score >= 0.3 and len(matching_disqualifiers) == 0

        updates["qualification_score"] = round(normalized_score, 3)
        updates["qualified"] = qualified
        updates["matching_signals"] = matching_signals
        updates["matching_disqualifiers"] = matching_disqualifiers

        if not qualified:
            reasons = []
            if normalized_score < 0.3:
                reasons.append(f"Low score: {normalized_score:.1%}")
            if matching_disqualifiers:
                reasons.append(f"Disqualifiers: {', '.join(matching_disqualifiers)}")
            updates["disqualification_reason"] = "; ".join(reasons)
            updates["skip_reason"] = updates["disqualification_reason"]

        logger.info(
            f"Qualified {state.get('contact_email')}: "
            f"score={normalized_score:.1%}, qualified={qualified}, "
            f"signals={matching_signals}"
        )

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="qualify_lead",
            status="completed",
            output_state={"score": normalized_score, "qualified": qualified},
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 5: Select Strategy
    # ------------------------------------------------------------------

    async def select_strategy(self, state: LeadState) -> dict:
        """
        Select the outreach persona, approach, and template.

        Uses RAG winning patterns + config persona matching to
        determine the best approach for this specific lead.
        """
        start = time.time()
        updates: dict[str, Any] = {"current_node": "select_strategy"}

        title = (state.get("contact_title", "") or "").lower()
        industry = state.get("company_industry", "")

        # Match persona from config
        selected_persona = None
        for persona in self.config.targeting.personas:
            size = state.get("company_size", 0)
            if persona.company_size[0] <= size <= persona.company_size[1]:
                for pattern in persona.title_patterns:
                    if pattern.lower() in title:
                        selected_persona = persona
                        break
            if selected_persona:
                break

        if not selected_persona:
            # Default to first persona if no match
            selected_persona = self.config.targeting.personas[0]

        updates["selected_persona"] = selected_persona.id
        updates["selected_approach"] = selected_persona.approach

        # Query RAG for winning patterns
        patterns = await self.retriever.find_winning_patterns(
            persona=selected_persona.id,
            industry=industry,
            limit=3,
        )
        updates["rag_patterns"] = patterns

        # If a winning pattern suggests a different approach, use it
        if patterns and patterns[0].get("metadata", {}).get("win_rate", 0) > 0.15:
            best_pattern = patterns[0]
            pattern_approach = best_pattern.get("metadata", {}).get("pattern_type")
            if pattern_approach:
                updates["selected_approach"] = pattern_approach
                logger.info(
                    f"RAG override: using {pattern_approach} approach "
                    f"(win rate: {best_pattern['metadata'].get('win_rate', 0):.0%})"
                )

        # Get vulnerability context for enriched email drafting
        vuln_context = await self.retriever.find_vulnerability_context(
            tech_stack=state.get("tech_stack", {}),
            vulnerabilities=state.get("vulnerabilities", []),
        )
        updates["vulnerability_context"] = vuln_context

        # Find best matching template
        templates = self.db.get_templates(
            approach_type=updates["selected_approach"],
            persona=selected_persona.id,
        )
        if templates:
            updates["template_id"] = templates[0]["id"]

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="select_strategy",
            status="completed",
            output_state={
                "persona": selected_persona.id,
                "approach": updates["selected_approach"],
                "patterns_found": len(patterns),
            },
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 6: Draft Outreach
    # ------------------------------------------------------------------

    async def draft_outreach(self, state: LeadState) -> dict:
        """
        Use Claude to draft a personalized outreach email.

        The prompt includes:
        - Contact details and company intel
        - Tech stack and vulnerability findings
        - RAG winning patterns for this persona/industry
        - Template structure (if available)
        - Previous feedback (if this is a re-draft after human rejection)
        """
        start = time.time()
        updates: dict[str, Any] = {"current_node": "draft_outreach"}

        # Build the context for Claude
        context_parts = [
            f"Contact: {state.get('contact_name')} ({state.get('contact_title')})",
            f"Email: {state.get('contact_email')}",
            f"Company: {state.get('company_name')} ({state.get('company_domain')})",
            f"Industry: {state.get('company_industry')}",
            f"Size: {state.get('company_size')} employees",
        ]

        tech = state.get("tech_stack", {})
        if tech:
            tech_str = ", ".join(f"{k}" for k in list(tech.keys())[:15])
            context_parts.append(f"Tech stack: {tech_str}")

        vulns = state.get("vulnerabilities", [])
        if vulns:
            vuln_strs = []
            for v in vulns[:5]:
                if isinstance(v, dict):
                    vuln_strs.append(f"- {v.get('type', '')}: {v.get('description', '')}")
                else:
                    vuln_strs.append(f"- {v}")
            context_parts.append(f"Security findings:\n" + "\n".join(vuln_strs))

        patterns = state.get("rag_patterns", [])
        if patterns:
            pattern_str = "\n".join(
                f"- {p.get('content', '')[:200]}" for p in patterns[:3]
            )
            context_parts.append(f"Winning patterns for this persona:\n{pattern_str}")

        approach = state.get("selected_approach", "vulnerability_alert")

        # Check for previous feedback (re-draft loop)
        feedback = state.get("human_feedback")
        feedback_instruction = ""
        if feedback:
            feedback_instruction = (
                f"\n\nPREVIOUS DRAFT WAS REJECTED. Feedback: {feedback}\n"
                f"Please revise based on this feedback."
            )

        prompt = f"""You are writing a cold outreach email for Enclave Guard, a cybersecurity consulting firm.

APPROACH: {approach}
PERSONA: {state.get('selected_persona', 'unknown')}

PROSPECT CONTEXT:
{chr(10).join(context_parts)}

INSTRUCTIONS:
1. Write a subject line (under 50 characters) and email body.
2. The email must be highly personalized using the specific data above.
3. Do NOT use generic phrases like "I noticed your company..." without specific details.
4. Every factual claim must be grounded in the data provided above.
5. Keep the email under 150 words. Busy executives don't read long emails.
6. Include a clear, low-commitment call-to-action (e.g., "Would a 15-minute call next week work?").
7. Tone: professional, direct, not salesy. You are a peer, not a vendor.
8. Do NOT include any information you are not certain about.
{feedback_instruction}

OUTPUT FORMAT:
SUBJECT: [subject line]
---
[email body]"""

        model = self.config.agent.model_routing.email_drafting
        temperature = self.config.agent.temperature.email_drafting

        response = self.llm.messages.create(
            model=model,
            max_tokens=500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Guard against empty or unexpected response from Claude
        if not response.content or not hasattr(response.content[0], "text"):
            logger.error("Claude returned empty or non-text response for email draft")
            updates["error"] = "LLM returned empty response"
            updates["error_node"] = "draft_outreach"
            updates["draft_email_subject"] = "[DRAFT FAILED]"
            updates["draft_email_body"] = "The LLM returned an empty response. Please retry."
            return updates

        response_text = response.content[0].text

        # Parse subject and body
        if "SUBJECT:" in response_text and "---" in response_text:
            parts = response_text.split("---", 1)
            subject = parts[0].replace("SUBJECT:", "").strip()
            body = parts[1].strip()
        else:
            # Fallback parsing
            lines = response_text.strip().split("\n")
            subject = lines[0].replace("SUBJECT:", "").strip()
            body = "\n".join(lines[1:]).strip()

        updates["draft_email_subject"] = subject
        updates["draft_email_body"] = body
        updates["draft_reasoning"] = f"Approach: {approach}, Persona: {state.get('selected_persona')}"

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="draft_outreach",
            status="completed",
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 7: Compliance Check
    # ------------------------------------------------------------------

    async def compliance_check(self, state: LeadState) -> dict:
        """
        Verify the outreach complies with CAN-SPAM / GDPR requirements.

        Checks:
        - Email not on suppression list
        - Not in excluded country
        - Has required elements (unsubscribe, physical address)
        - Subject line is not misleading
        """
        start = time.time()
        updates: dict[str, Any] = {"current_node": "compliance_check"}
        issues: list[str] = []

        email = state.get("contact_email", "")

        # Check suppression list
        if self.db.is_suppressed(email):
            issues.append("Email is on suppression list")
            updates["is_suppressed"] = True

        # Check excluded countries
        # (In practice, you'd need to determine the contact's country
        #  from Apollo data or email domain TLD)
        exclude = self.config.outreach.compliance.exclude_countries
        # Basic TLD check for EU domains
        if email:
            tld = email.rsplit(".", 1)[-1].upper() if "." in email else ""
            country_tlds = {"DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT",
                           "IE", "SE", "DK", "FI", "PL", "CZ", "UK", "GB"}
            if tld in country_tlds and tld in exclude:
                issues.append(f"Contact in excluded country: {tld}")

        # Check that email body exists
        body = state.get("draft_email_body", "")
        subject = state.get("draft_email_subject", "")
        if not body:
            issues.append("Email body is empty")
        if not subject:
            issues.append("Email subject is empty")

        # Check subject line length
        if subject and len(subject) > 80:
            issues.append(f"Subject line too long: {len(subject)} chars (max 80)")

        updates["compliance_passed"] = len(issues) == 0
        updates["compliance_issues"] = issues

        if issues:
            updates["skip_reason"] = f"Compliance: {'; '.join(issues)}"
            logger.warning(f"Compliance failed for {email}: {issues}")

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="compliance_check",
            status="completed" if not issues else "failed",
            output_state={"issues": issues},
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 8: Send Outreach
    # ------------------------------------------------------------------

    async def send_outreach(self, state: LeadState) -> dict:
        """
        Send the approved email via the email sending infrastructure.

        Attempts to send via EmailEngine (SendGrid/Mailgun) if configured.
        Falls back to record-only mode if no email provider key is set.
        Always creates the outreach event in the database.
        """
        import os

        start = time.time()
        updates: dict[str, Any] = {"current_node": "send_outreach"}

        # Use edited version if human made changes
        subject = state.get("edited_subject") or state.get("draft_email_subject", "")
        body = state.get("edited_body") or state.get("draft_email_body", "")
        to_email = state.get("contact_email", "")
        to_name = state.get("contact_name", "")

        # Attempt actual email sending via provider
        provider_id = None
        send_status = "recorded"  # default: logged but not sent

        sendgrid_key = os.environ.get("SENDGRID_API_KEY", "").strip()
        mailgun_key = os.environ.get("MAILGUN_API_KEY", "").strip()

        if sendgrid_key or mailgun_key:
            try:
                from core.outreach.email_engine import EmailEngine

                provider = "sendgrid" if sendgrid_key else "mailgun"
                email_cfg = self.config.outreach.email

                engine = EmailEngine(
                    provider=provider,
                    sending_domain=email_cfg.sending_domain,
                    reply_to=email_cfg.reply_to,
                )

                result = await engine.send_email(
                    to_email=to_email,
                    to_name=to_name,
                    subject=subject,
                    body_html=body,
                    body_text=body,  # plain text fallback
                    from_name=self.config.vertical_name,
                    tracking_id=state.get("lead_id"),
                )

                provider_id = result.get("message_id", "")
                send_status = "sent"
                logger.info(
                    f"Email sent via {provider} to {to_email} "
                    f"(message_id: {provider_id})"
                )
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.error(f"Email sending failed: {e}")
                send_status = "send_failed"
                updates["send_error"] = str(e)
        else:
            logger.warning(
                "No email provider configured (SENDGRID_API_KEY or MAILGUN_API_KEY). "
                f"Outreach to {to_email} recorded but NOT sent."
            )

        # Create outreach event in database
        event = self.db.create_outreach_event({
            "contact_id": state.get("contact_id"),
            "company_id": state.get("company_id"),
            "channel": "email",
            "direction": "outbound",
            "template_id": state.get("template_id"),
            "sequence_name": state.get("selected_approach"),
            "sequence_step": 1,
            "subject": subject,
            "body_preview": body[:500],
            "status": send_status,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        })

        # Increment template usage counter
        if state.get("template_id"):
            self.db.increment_template_usage(state["template_id"])

        updates["email_sent"] = send_status == "sent"
        updates["sent_at"] = datetime.now(timezone.utc).isoformat()
        updates["sending_provider_id"] = provider_id or "NO_PROVIDER"

        logger.info(
            f"Outreach {send_status} for {to_email} "
            f"(approach: {state.get('selected_approach')})"
        )

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="send_outreach",
            status="completed",
            duration_ms=duration,
        )
        return updates

    # ------------------------------------------------------------------
    # Node 9: Write to RAG
    # ------------------------------------------------------------------

    async def write_to_rag(self, state: LeadState) -> dict:
        """
        Write enrichment data and outreach event to the RAG knowledge base.

        This is the learning step — every interaction becomes future knowledge.
        """
        start = time.time()
        updates: dict[str, Any] = {
            "current_node": "write_to_rag",
            "knowledge_written": False,
        }

        # Store company intel
        if state.get("company_domain") and state.get("tech_stack"):
            try:
                await self.ingester.ingest_company_intel(
                    company_id=state.get("company_id", ""),
                    domain=state.get("company_domain", ""),
                    intel={
                        "tech_stack": state.get("tech_stack", {}),
                        "vulnerabilities": state.get("vulnerabilities", []),
                        "industry": state.get("company_industry", ""),
                        "employee_count": state.get("company_size", 0),
                    },
                )
            except (ConnectionError, TimeoutError, OSError, ValueError) as e:
                logger.warning(f"Failed to ingest company intel: {e}")

        # Store outreach result (if email was sent)
        if state.get("email_sent"):
            try:
                await self.ingester.ingest_outreach_result(
                    contact_email=state.get("contact_email", ""),
                    approach_type=state.get("selected_approach", ""),
                    persona=state.get("selected_persona", ""),
                    outcome="sent",  # outcome updated later via webhook
                    industry=state.get("company_industry", ""),
                    company_size=state.get("company_size", 0),
                    subject=state.get("draft_email_subject", ""),
                    body_preview=(state.get("draft_email_body", "") or "")[:200],
                )
            except (ConnectionError, TimeoutError, OSError, ValueError) as e:
                logger.warning(f"Failed to ingest outreach result: {e}")

        updates["knowledge_written"] = True

        duration = int((time.time() - start) * 1000)
        self.db.log_pipeline_run(
            lead_id=state.get("lead_id", ""),
            node_name="write_to_rag",
            status="completed",
            duration_ms=duration,
        )
        return updates
