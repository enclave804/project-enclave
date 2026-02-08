"""
Genesis Interview Engine — Adaptive business discovery through structured Q&A.

The interview engine drives Stage 1 of the Genesis flow: gathering enough
business context from the user to produce a high-quality BusinessBlueprint.

Design principles:
- Adaptive: questions unlock based on previous answers (not a fixed list)
- Completeness-driven: tracks which fields of BusinessContext are populated
- Domain-aware: follow-up questions adapt to the business type
- Progressive: starts broad, drills into specifics
- Resilient: partial answers still accumulate valid context

Architecture:
    InterviewEngine manages a QuestionBank of categorized questions.
    Each question targets one or more BusinessContext fields.
    The engine tracks completion, suggests next questions, and can
    determine when enough context has been gathered to proceed.

Usage:
    engine = InterviewEngine()
    while not engine.is_complete(context):
        question = engine.get_next_question(context)
        # ... present question to user, get answer ...
        context = engine.apply_answer(context, question.id, answer)

    # Ready to generate blueprint
    assert engine.get_completion_score(context) >= 0.8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InterviewPhase(str, Enum):
    """Progressive interview phases — broad to specific."""
    IDENTITY = "identity"          # Who are you? What do you do?
    BUSINESS_MODEL = "business_model"  # How do you make money?
    TARGET_MARKET = "target_market"    # Who do you sell to?
    VALUE_PROP = "value_proposition"   # Why should they buy?
    OUTREACH = "outreach"             # How do we reach them?
    ENRICHMENT = "enrichment"         # What signals matter?
    CONTENT = "content"               # What content to create?
    REVIEW = "review"                 # Confirm and finalize


class QuestionPriority(str, Enum):
    """How critical a question is to generating a valid blueprint."""
    REQUIRED = "required"      # Blueprint cannot be generated without this
    IMPORTANT = "important"    # Significantly improves blueprint quality
    OPTIONAL = "optional"      # Nice-to-have, has sensible defaults


# ---------------------------------------------------------------------------
# Question Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterviewQuestion:
    """
    A single interview question targeting BusinessContext fields.

    Attributes:
        id: Unique question identifier (e.g., "q_business_name").
        phase: Which interview phase this belongs to.
        priority: How critical this question is.
        question: The question text presented to the user.
        hint: Helper text shown below the question.
        target_fields: BusinessContext field(s) this answer populates.
        follow_up_trigger: If set, only ask this question when the
            named field in context has a truthy value.
        multi_value: Whether the answer should be parsed as a list.
        examples: Example answers to help the user understand what's expected.
    """
    id: str
    phase: InterviewPhase
    priority: QuestionPriority
    question: str
    hint: str = ""
    target_fields: tuple[str, ...] = ()
    follow_up_trigger: Optional[str] = None
    multi_value: bool = False
    examples: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Question Bank — The complete set of discovery questions
# ---------------------------------------------------------------------------

# The order within each phase matters: questions are asked in definition order
# unless a follow_up_trigger redirects flow.

QUESTION_BANK: tuple[InterviewQuestion, ...] = (
    # ── Phase: IDENTITY ──────────────────────────────────────────
    InterviewQuestion(
        id="q_business_name",
        phase=InterviewPhase.IDENTITY,
        priority=QuestionPriority.REQUIRED,
        question="What's the name of your business?",
        hint="This will be used as the vertical name in the platform.",
        target_fields=("business_name",),
        examples=("Enclave Guard", "PrintBiz 3D", "SwiftRecruit"),
    ),
    InterviewQuestion(
        id="q_business_description",
        phase=InterviewPhase.IDENTITY,
        priority=QuestionPriority.REQUIRED,
        question="Describe what your business does in 2-3 sentences.",
        hint="Be specific about the service/product and who it's for.",
        target_fields=("business_description",),
        examples=(
            "We provide cybersecurity assessments and penetration testing "
            "for mid-size financial institutions.",
        ),
    ),
    InterviewQuestion(
        id="q_website",
        phase=InterviewPhase.IDENTITY,
        priority=QuestionPriority.OPTIONAL,
        question="Do you have an existing website? If so, what's the URL?",
        hint="We'll use this for enrichment and competitive analysis.",
        target_fields=("website",),
    ),
    InterviewQuestion(
        id="q_region",
        phase=InterviewPhase.IDENTITY,
        priority=QuestionPriority.IMPORTANT,
        question="What's your primary market/region?",
        hint="This affects compliance rules, language, and targeting.",
        target_fields=("region",),
        examples=("United States", "Europe", "United Kingdom", "Global"),
    ),

    # ── Phase: BUSINESS_MODEL ────────────────────────────────────
    InterviewQuestion(
        id="q_business_model",
        phase=InterviewPhase.BUSINESS_MODEL,
        priority=QuestionPriority.REQUIRED,
        question="How does your business make money? (B2B service, SaaS, marketplace, etc.)",
        hint="We need to understand the revenue model to design the right sales approach.",
        target_fields=("business_model",),
        examples=(
            "B2B service — we charge per project",
            "SaaS — monthly subscription",
            "Consulting — hourly or retainer",
        ),
    ),
    InterviewQuestion(
        id="q_price_range",
        phase=InterviewPhase.BUSINESS_MODEL,
        priority=QuestionPriority.REQUIRED,
        question="What's your typical deal size? (minimum and maximum in USD)",
        hint="Enter a range like '$500 - $5000' or '2000-50000'.",
        target_fields=("price_range",),
        examples=("$500 - $5,000", "$10,000 - $100,000"),
    ),
    InterviewQuestion(
        id="q_sales_cycle",
        phase=InterviewPhase.BUSINESS_MODEL,
        priority=QuestionPriority.IMPORTANT,
        question="How long does a typical sales cycle take? (days from first contact to close)",
        hint="This determines email sequence timing and follow-up cadence.",
        target_fields=("sales_cycle_days",),
        examples=("7 days", "30 days", "90 days"),
    ),
    InterviewQuestion(
        id="q_currency",
        phase=InterviewPhase.BUSINESS_MODEL,
        priority=QuestionPriority.OPTIONAL,
        question="What currency do you operate in?",
        hint="Default is USD.",
        target_fields=("currency",),
    ),

    # ── Phase: TARGET_MARKET ─────────────────────────────────────
    InterviewQuestion(
        id="q_target_industries",
        phase=InterviewPhase.TARGET_MARKET,
        priority=QuestionPriority.REQUIRED,
        question="What industries do your ideal customers work in?",
        hint="List the top 3-5 industries. We'll use this to find leads.",
        target_fields=("target_industries",),
        multi_value=True,
        examples=(
            "Financial Services, Healthcare, Manufacturing",
            "Architecture, Interior Design, Real Estate",
        ),
    ),
    InterviewQuestion(
        id="q_company_sizes",
        phase=InterviewPhase.TARGET_MARKET,
        priority=QuestionPriority.REQUIRED,
        question="What size companies do you target? (employee count range)",
        hint="Enter a range like '10-500' or '50-5000'.",
        target_fields=("target_company_sizes",),
        examples=("10-200", "50-500", "100-5000"),
    ),
    InterviewQuestion(
        id="q_target_titles",
        phase=InterviewPhase.TARGET_MARKET,
        priority=QuestionPriority.REQUIRED,
        question="What job titles do you typically sell to? (the decision makers)",
        hint="List the top 3-5 titles. These are the people who can say 'yes'.",
        target_fields=("target_titles",),
        multi_value=True,
        examples=(
            "CTO, VP Engineering, Head of Security",
            "Design Director, Principal Architect, Studio Manager",
        ),
    ),
    InterviewQuestion(
        id="q_target_locations",
        phase=InterviewPhase.TARGET_MARKET,
        priority=QuestionPriority.IMPORTANT,
        question="What geographic locations do you want to target?",
        hint="Countries or regions. Default is United States.",
        target_fields=("target_locations",),
        multi_value=True,
        examples=("United States, Canada", "United States, United Kingdom"),
    ),

    # ── Phase: VALUE_PROPOSITION ─────────────────────────────────
    InterviewQuestion(
        id="q_pain_points",
        phase=InterviewPhase.VALUE_PROP,
        priority=QuestionPriority.REQUIRED,
        question="What problems do your customers face that you solve?",
        hint="Think about the pain that makes someone pick up the phone.",
        target_fields=("pain_points",),
        multi_value=True,
        examples=(
            "They don't know their security vulnerabilities",
            "Their 3D models take weeks to prototype in-house",
        ),
    ),
    InterviewQuestion(
        id="q_value_propositions",
        phase=InterviewPhase.VALUE_PROP,
        priority=QuestionPriority.REQUIRED,
        question="What are your key value propositions? Why do customers choose you?",
        hint="What makes you the right solution? Be specific about outcomes.",
        target_fields=("value_propositions",),
        multi_value=True,
        examples=(
            "48-hour turnaround on security assessments",
            "50% cost reduction vs in-house 3D printing",
        ),
    ),
    InterviewQuestion(
        id="q_differentiators",
        phase=InterviewPhase.VALUE_PROP,
        priority=QuestionPriority.IMPORTANT,
        question="What makes your business different from competitors?",
        hint="Unique strengths, proprietary technology, special expertise, etc.",
        target_fields=("differentiators",),
        multi_value=True,
    ),

    # ── Phase: OUTREACH ──────────────────────────────────────────
    InterviewQuestion(
        id="q_sending_domain",
        phase=InterviewPhase.OUTREACH,
        priority=QuestionPriority.IMPORTANT,
        question="Do you have a dedicated email sending domain? (e.g., 'mail.company.com')",
        hint="A separate domain protects your main domain's reputation.",
        target_fields=("sending_domain",),
        examples=("mail.enclaveguard.com", "outreach.printbiz3d.com"),
    ),
    InterviewQuestion(
        id="q_reply_to",
        phase=InterviewPhase.OUTREACH,
        priority=QuestionPriority.IMPORTANT,
        question="What email address should replies go to?",
        hint="This is the address prospects will reply to.",
        target_fields=("reply_to_email",),
        examples=("hello@company.com", "sales@company.com"),
    ),
    InterviewQuestion(
        id="q_physical_address",
        phase=InterviewPhase.OUTREACH,
        priority=QuestionPriority.IMPORTANT,
        question="What's your physical business address? (required by CAN-SPAM)",
        hint="Every outreach email must include a physical address.",
        target_fields=("physical_address",),
    ),
    InterviewQuestion(
        id="q_daily_limit",
        phase=InterviewPhase.OUTREACH,
        priority=QuestionPriority.OPTIONAL,
        question="How many outreach emails should we send per day? (default: 25)",
        hint="Start low during warmup. We recommend 15-30 for new domains.",
        target_fields=("daily_outreach_limit",),
    ),

    # ── Phase: ENRICHMENT ────────────────────────────────────────
    InterviewQuestion(
        id="q_positive_signals",
        phase=InterviewPhase.ENRICHMENT,
        priority=QuestionPriority.IMPORTANT,
        question="What signals indicate a lead is a good fit? (buying signals)",
        hint="Things you'd look for when qualifying a lead manually.",
        target_fields=("positive_signals",),
        multi_value=True,
        examples=(
            "Recently raised funding, hiring for security roles, compliance deadline coming",
            "Active building projects, uses CAD software, won design awards",
        ),
    ),
    InterviewQuestion(
        id="q_disqualifiers",
        phase=InterviewPhase.ENRICHMENT,
        priority=QuestionPriority.IMPORTANT,
        question="What signals disqualify a lead? (deal breakers)",
        hint="Conditions that mean a company is not a good fit.",
        target_fields=("disqualifiers",),
        multi_value=True,
        examples=(
            "Already has in-house security team, less than 10 employees",
            "Has in-house 3D printing, no design department",
        ),
    ),

    # ── Phase: CONTENT ───────────────────────────────────────────
    InterviewQuestion(
        id="q_tone",
        phase=InterviewPhase.CONTENT,
        priority=QuestionPriority.OPTIONAL,
        question="What tone should our outreach and content use?",
        hint="How do you want to sound to prospects?",
        target_fields=("tone",),
        examples=(
            "Professional, authoritative",
            "Friendly, approachable, technical",
            "Direct, data-driven",
        ),
    ),
    InterviewQuestion(
        id="q_content_topics",
        phase=InterviewPhase.CONTENT,
        priority=QuestionPriority.OPTIONAL,
        question="What topics should we create content about? (for SEO and thought leadership)",
        hint="Blog topics, keywords, areas of expertise.",
        target_fields=("content_topics",),
        multi_value=True,
        examples=(
            "cybersecurity assessment, penetration testing, SOC 2 compliance",
            "3D printing for architecture, rapid prototyping, model visualization",
        ),
    ),
)


# Build lookup indices
_QUESTIONS_BY_ID: dict[str, InterviewQuestion] = {q.id: q for q in QUESTION_BANK}
_QUESTIONS_BY_PHASE: dict[InterviewPhase, list[InterviewQuestion]] = {}
for _q in QUESTION_BANK:
    _QUESTIONS_BY_PHASE.setdefault(_q.phase, []).append(_q)


# ---------------------------------------------------------------------------
# Required fields — the minimum set for a valid BusinessContext
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: frozenset[str] = frozenset({
    "business_name",
    "business_description",
    "business_model",
    "price_range",
    "target_industries",
    "target_company_sizes",
    "target_titles",
    "pain_points",
    "value_propositions",
})

IMPORTANT_FIELDS: frozenset[str] = frozenset({
    "region",
    "sales_cycle_days",
    "target_locations",
    "differentiators",
    "sending_domain",
    "reply_to_email",
    "physical_address",
    "positive_signals",
    "disqualifiers",
})

ALL_TRACKED_FIELDS: frozenset[str] = REQUIRED_FIELDS | IMPORTANT_FIELDS | frozenset({
    "website",
    "currency",
    "daily_outreach_limit",
    "tone",
    "content_topics",
})


# ---------------------------------------------------------------------------
# Interview Engine
# ---------------------------------------------------------------------------

@dataclass
class InterviewProgress:
    """Snapshot of interview completion status."""
    total_fields: int
    required_filled: int
    required_total: int
    important_filled: int
    important_total: int
    optional_filled: int
    optional_total: int
    completion_score: float
    current_phase: InterviewPhase
    questions_asked: int
    questions_remaining: int
    is_complete: bool
    missing_required: list[str] = field(default_factory=list)


class InterviewEngine:
    """
    Drives the adaptive interview process for business discovery.

    The engine maintains a question bank and tracks which BusinessContext
    fields have been populated. It adapts the question flow based on
    what's already known — never asks redundant questions.

    Thread-safe: all state is passed as context dicts, no mutable instance state.
    """

    # Minimum completion score to consider the interview sufficient
    MINIMUM_COMPLETION_SCORE: float = 0.75

    # Weights for completion scoring
    REQUIRED_WEIGHT: float = 0.6
    IMPORTANT_WEIGHT: float = 0.3
    OPTIONAL_WEIGHT: float = 0.1

    def __init__(
        self,
        minimum_completion: float = 0.75,
    ):
        """
        Args:
            minimum_completion: Override the minimum completion score
                required to proceed to blueprint generation.
        """
        self.minimum_completion = minimum_completion
        self._questions = QUESTION_BANK
        self._by_id = _QUESTIONS_BY_ID
        self._by_phase = _QUESTIONS_BY_PHASE

    # ── Core API ─────────────────────────────────────────────────

    def get_next_question(
        self,
        context: dict[str, Any],
        asked_ids: Optional[set[str]] = None,
    ) -> Optional[InterviewQuestion]:
        """
        Get the next best question to ask based on current context.

        Strategy:
        1. Required questions first (in phase order)
        2. Important questions next
        3. Optional questions last
        4. Skip questions whose target fields are already populated
        5. Skip questions whose follow_up_trigger is not met

        Args:
            context: Current BusinessContext as a dict.
            asked_ids: Set of question IDs already asked.

        Returns:
            Next InterviewQuestion, or None if interview is complete.
        """
        asked_ids = asked_ids or set()

        # Priority ordering: required → important → optional
        for priority in (
            QuestionPriority.REQUIRED,
            QuestionPriority.IMPORTANT,
            QuestionPriority.OPTIONAL,
        ):
            for question in self._questions:
                if question.priority != priority:
                    continue
                if question.id in asked_ids:
                    continue
                if self._is_answered(question, context):
                    continue
                if not self._trigger_met(question, context):
                    continue
                return question

        return None

    def get_questions_for_phase(
        self,
        phase: InterviewPhase,
        context: dict[str, Any],
        asked_ids: Optional[set[str]] = None,
    ) -> list[InterviewQuestion]:
        """
        Get all unanswered questions for a specific phase.

        Useful for presenting a batch of questions in a form-style UI.
        """
        asked_ids = asked_ids or set()
        result = []
        for q in self._by_phase.get(phase, []):
            if q.id in asked_ids:
                continue
            if self._is_answered(q, context):
                continue
            if not self._trigger_met(q, context):
                continue
            result.append(q)
        return result

    def get_all_phases(self) -> list[InterviewPhase]:
        """Return all interview phases in order."""
        return [
            InterviewPhase.IDENTITY,
            InterviewPhase.BUSINESS_MODEL,
            InterviewPhase.TARGET_MARKET,
            InterviewPhase.VALUE_PROP,
            InterviewPhase.OUTREACH,
            InterviewPhase.ENRICHMENT,
            InterviewPhase.CONTENT,
            InterviewPhase.REVIEW,
        ]

    def get_current_phase(
        self, context: dict[str, Any]
    ) -> InterviewPhase:
        """
        Determine which phase the interview is currently in.

        Returns the first phase that has unanswered required/important questions.
        """
        for phase in self.get_all_phases():
            questions = self._by_phase.get(phase, [])
            for q in questions:
                if q.priority in (
                    QuestionPriority.REQUIRED,
                    QuestionPriority.IMPORTANT,
                ):
                    if not self._is_answered(q, context):
                        return phase
        return InterviewPhase.REVIEW

    def get_completion_score(
        self, context: dict[str, Any]
    ) -> float:
        """
        Calculate weighted completion score (0.0 to 1.0).

        Weighting:
        - Required fields: 60% of score
        - Important fields: 30% of score
        - Optional fields: 10% of score
        """
        required_total = len(REQUIRED_FIELDS)
        important_total = len(IMPORTANT_FIELDS)
        optional_fields = ALL_TRACKED_FIELDS - REQUIRED_FIELDS - IMPORTANT_FIELDS
        optional_total = len(optional_fields)

        required_filled = sum(
            1 for f in REQUIRED_FIELDS if self._field_has_value(context, f)
        )
        important_filled = sum(
            1 for f in IMPORTANT_FIELDS if self._field_has_value(context, f)
        )
        optional_filled = sum(
            1 for f in optional_fields if self._field_has_value(context, f)
        )

        score = 0.0
        if required_total > 0:
            score += self.REQUIRED_WEIGHT * (required_filled / required_total)
        if important_total > 0:
            score += self.IMPORTANT_WEIGHT * (important_filled / important_total)
        if optional_total > 0:
            score += self.OPTIONAL_WEIGHT * (optional_filled / optional_total)

        return round(score, 3)

    def is_complete(self, context: dict[str, Any]) -> bool:
        """
        Check if the interview has enough context to proceed.

        Requirements:
        1. All REQUIRED fields must be populated
        2. Overall completion score >= minimum_completion
        """
        # All required fields must be present
        for f in REQUIRED_FIELDS:
            if not self._field_has_value(context, f):
                return False

        return self.get_completion_score(context) >= self.minimum_completion

    def get_progress(
        self,
        context: dict[str, Any],
        asked_ids: Optional[set[str]] = None,
    ) -> InterviewProgress:
        """Get a complete progress snapshot."""
        asked_ids = asked_ids or set()
        optional_fields = ALL_TRACKED_FIELDS - REQUIRED_FIELDS - IMPORTANT_FIELDS

        required_filled = sum(
            1 for f in REQUIRED_FIELDS if self._field_has_value(context, f)
        )
        important_filled = sum(
            1 for f in IMPORTANT_FIELDS if self._field_has_value(context, f)
        )
        optional_filled = sum(
            1 for f in optional_fields if self._field_has_value(context, f)
        )

        missing_required = [
            f for f in sorted(REQUIRED_FIELDS)
            if not self._field_has_value(context, f)
        ]

        # Count remaining questions
        remaining = 0
        for q in self._questions:
            if q.id not in asked_ids and not self._is_answered(q, context):
                if self._trigger_met(q, context):
                    remaining += 1

        return InterviewProgress(
            total_fields=len(ALL_TRACKED_FIELDS),
            required_filled=required_filled,
            required_total=len(REQUIRED_FIELDS),
            important_filled=important_filled,
            important_total=len(IMPORTANT_FIELDS),
            optional_filled=optional_filled,
            optional_total=len(optional_fields),
            completion_score=self.get_completion_score(context),
            current_phase=self.get_current_phase(context),
            questions_asked=len(asked_ids),
            questions_remaining=remaining,
            is_complete=self.is_complete(context),
            missing_required=missing_required,
        )

    # ── Question Metadata ────────────────────────────────────────

    def get_question(self, question_id: str) -> Optional[InterviewQuestion]:
        """Look up a question by ID."""
        return self._by_id.get(question_id)

    def get_all_questions(self) -> tuple[InterviewQuestion, ...]:
        """Return the full question bank."""
        return self._questions

    def get_required_questions(self) -> list[InterviewQuestion]:
        """Return only required-priority questions."""
        return [
            q for q in self._questions
            if q.priority == QuestionPriority.REQUIRED
        ]

    def get_phase_count(self) -> dict[InterviewPhase, int]:
        """Return the number of questions per phase."""
        return {
            phase: len(questions)
            for phase, questions in self._by_phase.items()
        }

    # ── Answer Parsing ───────────────────────────────────────────

    @staticmethod
    def parse_list_answer(answer: str) -> list[str]:
        """
        Parse a user's multi-value answer into a list of strings.

        Handles:
        - Comma-separated: "A, B, C"
        - Newline-separated: "A\\nB\\nC"
        - Semicolon-separated: "A; B; C"
        - Numbered: "1. A\\n2. B\\n3. C"
        - Bulleted: "- A\\n- B\\n- C"
        """
        import re

        # Strip bullet points and numbering
        answer = re.sub(r"^\s*[-*•]\s*", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"^\s*\d+[.)]\s*", "", answer, flags=re.MULTILINE)

        # Split on newlines first (highest priority delimiter)
        if "\n" in answer:
            items = answer.split("\n")
        elif ";" in answer:
            items = answer.split(";")
        else:
            items = answer.split(",")

        # Clean up each item
        result = []
        for item in items:
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)

        return result

    @staticmethod
    def parse_range_answer(answer: str) -> Optional[tuple[int, int]]:
        """
        Parse a range answer like "$500 - $5000" or "10-500".

        Returns (min, max) tuple or None if unparseable.
        """
        import re

        # Remove currency symbols and commas
        cleaned = re.sub(r"[$€£,]", "", answer)

        # Try to find two numbers
        numbers = re.findall(r"\d+", cleaned)
        if len(numbers) >= 2:
            return (int(numbers[0]), int(numbers[1]))
        elif len(numbers) == 1:
            # Single number — use as both min and max
            n = int(numbers[0])
            return (n, n)

        return None

    @staticmethod
    def parse_int_answer(answer: str) -> Optional[int]:
        """Parse a single integer from an answer string."""
        import re

        cleaned = re.sub(r"[,$]", "", answer.strip())
        numbers = re.findall(r"\d+", cleaned)
        if numbers:
            return int(numbers[0])
        return None

    # ── Summary Generation ───────────────────────────────────────

    def generate_context_summary(
        self, context: dict[str, Any]
    ) -> str:
        """
        Generate a human-readable summary of collected context.

        Used in the review phase and for the ArchitectAgent to reason over.
        """
        lines = []
        lines.append(f"Business: {context.get('business_name', 'Unknown')}")
        lines.append(f"Description: {context.get('business_description', 'N/A')}")
        lines.append(f"Model: {context.get('business_model', 'N/A')}")

        if context.get("price_range"):
            pr = context["price_range"]
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                lines.append(f"Deal Size: ${pr[0]:,} - ${pr[1]:,}")

        if context.get("target_industries"):
            lines.append(f"Industries: {', '.join(context['target_industries'])}")

        if context.get("target_titles"):
            lines.append(f"Target Titles: {', '.join(context['target_titles'])}")

        if context.get("pain_points"):
            lines.append("Pain Points:")
            for pp in context["pain_points"]:
                lines.append(f"  - {pp}")

        if context.get("value_propositions"):
            lines.append("Value Props:")
            for vp in context["value_propositions"]:
                lines.append(f"  - {vp}")

        return "\n".join(lines)

    # ── Prompt Generation (for ArchitectAgent) ───────────────────

    def generate_question_prompt(
        self,
        question: InterviewQuestion,
        context: dict[str, Any],
    ) -> str:
        """
        Generate a formatted prompt for the ArchitectAgent to ask the user.

        Includes the question, hint, examples, and progress context.
        """
        parts = [question.question]

        if question.hint:
            parts.append(f"\n💡 {question.hint}")

        if question.examples:
            parts.append("\nExamples:")
            for ex in question.examples:
                parts.append(f"  → {ex}")

        # Add context awareness
        progress = self.get_progress(context)
        parts.append(
            f"\n[Phase: {progress.current_phase.value} | "
            f"Progress: {progress.completion_score:.0%}]"
        )

        return "\n".join(parts)

    # ── Private Helpers ──────────────────────────────────────────

    def _is_answered(
        self,
        question: InterviewQuestion,
        context: dict[str, Any],
    ) -> bool:
        """Check if a question's target fields are already populated."""
        if not question.target_fields:
            return False
        return all(
            self._field_has_value(context, field)
            for field in question.target_fields
        )

    def _trigger_met(
        self,
        question: InterviewQuestion,
        context: dict[str, Any],
    ) -> bool:
        """Check if a question's follow_up_trigger condition is met."""
        if question.follow_up_trigger is None:
            return True
        return self._field_has_value(context, question.follow_up_trigger)

    @staticmethod
    def _field_has_value(context: dict[str, Any], field_name: str) -> bool:
        """Check if a context field has a meaningful value."""
        value = context.get(field_name)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return False
        return True
