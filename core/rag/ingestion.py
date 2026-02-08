"""
RAG ingestion pipeline for Project Enclave.

Handles chunking, embedding, and storing knowledge in the vector database.
Supports different chunk types: company intel, outreach results,
winning patterns, vulnerability knowledge, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from core.integrations.supabase_client import EnclaveDB
from core.rag.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class KnowledgeIngester:
    """
    Ingests knowledge into the RAG vector store.

    Handles chunking of different content types and batch embedding.
    """

    def __init__(self, db: EnclaveDB, embedder: EmbeddingEngine):
        self.db = db
        self.embedder = embedder

    async def ingest_company_intel(
        self,
        company_id: str,
        domain: str,
        intel: dict[str, Any],
    ) -> str:
        """
        Store company intelligence as a knowledge chunk.

        Args:
            company_id: UUID of the company.
            domain: Company domain.
            intel: Dict with keys like 'tech_stack', 'vulnerabilities',
                   'recent_news', 'funding', etc.

        Returns:
            ID of the created knowledge chunk.
        """
        # Build a natural language summary for embedding
        parts = [f"Company: {domain}"]
        if "tech_stack" in intel:
            tech = intel["tech_stack"]
            if isinstance(tech, dict):
                tech_str = ", ".join(f"{k}: {v}" for k, v in tech.items())
            else:
                tech_str = str(tech)
            parts.append(f"Technology stack: {tech_str}")

        if "vulnerabilities" in intel:
            vulns = intel["vulnerabilities"]
            if vulns:
                vuln_strs = []
                for v in vulns[:10]:  # limit to top 10
                    if isinstance(v, dict):
                        vuln_strs.append(
                            f"{v.get('type', 'unknown')}: {v.get('description', '')}"
                        )
                    else:
                        vuln_strs.append(str(v))
                parts.append(f"Security findings: {'; '.join(vuln_strs)}")

        if "industry" in intel:
            parts.append(f"Industry: {intel['industry']}")

        if "employee_count" in intel:
            parts.append(f"Employees: {intel['employee_count']}")

        content = "\n".join(parts)
        embedding = await self.embedder.embed_text(content)

        result = self.db.store_knowledge_chunk(
            content=content,
            embedding=embedding,
            chunk_type="company_intel",
            metadata={
                "domain": domain,
                "company_id": company_id,
                "intel_keys": list(intel.keys()),
            },
            source_id=company_id,
            source_type="company",
        )
        logger.info(f"Ingested company intel for {domain}")
        return result.get("id", "")

    async def ingest_outreach_result(
        self,
        contact_email: str,
        approach_type: str,
        persona: str,
        outcome: str,
        industry: str,
        company_size: int,
        subject: str,
        body_preview: str,
        reply_text: Optional[str] = None,
    ) -> str:
        """
        Store an outreach result for pattern learning.

        This is the raw data that the weekly pattern extraction job
        uses to identify winning approaches.
        """
        parts = [
            f"Outreach to {persona} in {industry} ({company_size} employees)",
            f"Approach: {approach_type}",
            f"Subject: {subject}",
            f"Preview: {body_preview[:200]}",
            f"Outcome: {outcome}",
        ]
        if reply_text:
            parts.append(f"Reply: {reply_text[:300]}")

        content = "\n".join(parts)
        embedding = await self.embedder.embed_text(content)

        result = self.db.store_knowledge_chunk(
            content=content,
            embedding=embedding,
            chunk_type="outreach_result",
            metadata={
                "contact_email": contact_email,
                "approach_type": approach_type,
                "persona": persona,
                "outcome": outcome,
                "industry": industry,
                "company_size": company_size,
            },
            source_type="outreach_event",
        )
        logger.info(f"Ingested outreach result: {outcome} for {persona}")
        return result.get("id", "")

    async def ingest_winning_pattern(
        self,
        pattern_type: str,
        pattern_description: str,
        win_rate: float,
        sample_size: int,
        target_persona: str,
        industry: str,
        example: Optional[str] = None,
    ) -> str:
        """
        Store a discovered winning pattern.

        These are extracted by the weekly analysis job from accumulated
        outreach results. They represent actionable insights like:
        "vulnerability_alert approach to CTOs in healthcare has 3x reply rate"
        """
        parts = [
            f"Winning pattern ({pattern_type}): {pattern_description}",
            f"Win rate: {win_rate:.1%} (sample: {sample_size})",
            f"Target: {target_persona} in {industry}",
        ]
        if example:
            parts.append(f"Example: {example[:200]}")

        content = "\n".join(parts)
        embedding = await self.embedder.embed_text(content)

        result = self.db.store_knowledge_chunk(
            content=content,
            embedding=embedding,
            chunk_type="winning_pattern",
            metadata={
                "pattern_type": pattern_type,
                "win_rate": win_rate,
                "sample_size": sample_size,
                "target_persona": target_persona,
                "industry": industry,
            },
            source_type="manual",
        )
        logger.info(
            f"Ingested winning pattern: {pattern_type} for {target_persona} "
            f"({win_rate:.0%} win rate)"
        )
        return result.get("id", "")

    async def ingest_vulnerability_knowledge(
        self,
        title: str,
        description: str,
        severity: str,
        affected_tech: list[str],
        remediation: Optional[str] = None,
    ) -> str:
        """
        Store cybersecurity vulnerability knowledge.

        This seeds the knowledge base with common vulnerabilities that
        the agent can reference when crafting outreach emails.
        """
        parts = [
            f"Vulnerability: {title}",
            f"Severity: {severity}",
            f"Affected technology: {', '.join(affected_tech)}",
            f"Description: {description}",
        ]
        if remediation:
            parts.append(f"Remediation: {remediation}")

        content = "\n".join(parts)
        embedding = await self.embedder.embed_text(content)

        result = self.db.store_knowledge_chunk(
            content=content,
            embedding=embedding,
            chunk_type="vulnerability_knowledge",
            metadata={
                "title": title,
                "severity": severity,
                "affected_tech": affected_tech,
            },
            source_type="manual",
        )
        logger.info(f"Ingested vulnerability knowledge: {title}")
        return result.get("id", "")

    async def load_seed_data(self, seed_path: str | Path) -> int:
        """
        Load seed knowledge from JSON files in the seed_data directory.

        Expected format:
        [
            {
                "type": "vulnerability_knowledge",
                "title": "...",
                "description": "...",
                "severity": "critical|high|medium|low",
                "affected_tech": ["WordPress", "Apache"],
                "remediation": "..."
            }
        ]

        Returns:
            Number of chunks ingested.
        """
        seed_path = Path(seed_path)
        if not seed_path.exists():
            logger.warning(f"Seed data path not found: {seed_path}")
            return 0

        count = 0
        for json_file in seed_path.glob("*.json"):
            logger.info(f"Loading seed data from {json_file.name}")
            with open(json_file) as f:
                items = json.load(f)

            for item in items:
                chunk_type = item.get("type", "vulnerability_knowledge")
                if chunk_type == "vulnerability_knowledge":
                    await self.ingest_vulnerability_knowledge(
                        title=item["title"],
                        description=item["description"],
                        severity=item.get("severity", "medium"),
                        affected_tech=item.get("affected_tech", []),
                        remediation=item.get("remediation"),
                    )
                    count += 1

        logger.info(f"Loaded {count} seed knowledge chunks")
        return count
