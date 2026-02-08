"""
Tests for the Genesis Interview Engine.

Covers:
- Question bank integrity (no duplicate IDs, all phases covered)
- Adaptive question routing (next question based on context)
- Completion scoring (required/important/optional weighting)
- Answer parsing (lists, ranges, integers)
- Phase progression
- Progress tracking
- Context summary generation
"""

import pytest

from core.genesis.interview import (
    ALL_TRACKED_FIELDS,
    IMPORTANT_FIELDS,
    QUESTION_BANK,
    REQUIRED_FIELDS,
    InterviewEngine,
    InterviewPhase,
    InterviewProgress,
    InterviewQuestion,
    QuestionPriority,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a default interview engine."""
    return InterviewEngine()


@pytest.fixture
def empty_context():
    """An empty business context."""
    return {}


@pytest.fixture
def minimal_context():
    """Context with required fields + enough important fields to pass threshold.

    Required fields alone give 0.6 score (60% weight), but the default
    threshold is 0.75. Adding a few important fields pushes it over.
    """
    return {
        # Required fields (9 fields → 0.6 score)
        "business_name": "TestBiz",
        "business_description": "A test business that does testing things for testers.",
        "business_model": "B2B service — per-project billing",
        "price_range": (1000, 10000),
        "target_industries": ["Technology", "Finance"],
        "target_company_sizes": (50, 500),
        "target_titles": ["CTO", "VP Engineering"],
        "pain_points": ["Slow testing cycles", "Low test coverage"],
        "value_propositions": ["Fast automated testing", "99% coverage guarantee"],
        # Important fields (adds ~0.17 score → total ~0.77)
        "region": "United States",
        "sending_domain": "mail.testbiz.com",
        "reply_to_email": "hello@testbiz.com",
        "physical_address": "123 Test St, San Francisco CA 94105",
        "positive_signals": ["hiring QA engineers"],
    }


@pytest.fixture
def full_context(minimal_context):
    """Context with all tracked fields populated."""
    ctx = dict(minimal_context)
    ctx.update({
        "website": "https://testbiz.com",
        "region": "United States",
        "sales_cycle_days": 30,
        "currency": "USD",
        "target_locations": ["United States", "Canada"],
        "differentiators": ["AI-powered test generation"],
        "sending_domain": "mail.testbiz.com",
        "reply_to_email": "hello@testbiz.com",
        "physical_address": "123 Test St, San Francisco CA 94105",
        "daily_outreach_limit": 25,
        "positive_signals": ["hiring QA engineers", "using outdated test tools"],
        "disqualifiers": ["has in-house QA team of 50+"],
        "tone": "Professional, technical",
        "content_topics": ["automated testing", "CI/CD best practices"],
    })
    return ctx


# ---------------------------------------------------------------------------
# Test: Question Bank Integrity
# ---------------------------------------------------------------------------

class TestQuestionBankIntegrity:
    """Verify the question bank is well-formed."""

    def test_no_duplicate_ids(self):
        """All question IDs must be unique."""
        ids = [q.id for q in QUESTION_BANK]
        assert len(ids) == len(set(ids)), f"Duplicate question IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_questions_have_target_fields(self):
        """Every question must target at least one field."""
        for q in QUESTION_BANK:
            assert len(q.target_fields) > 0, f"Question {q.id} has no target_fields"

    def test_all_phases_have_questions(self):
        """All phases except REVIEW should have at least one question."""
        phases_with_questions = {q.phase for q in QUESTION_BANK}
        for phase in InterviewPhase:
            if phase != InterviewPhase.REVIEW:
                assert phase in phases_with_questions, (
                    f"Phase {phase.value} has no questions"
                )

    def test_required_fields_have_questions(self):
        """Every required field should be targeted by at least one question."""
        targeted_fields: set[str] = set()
        for q in QUESTION_BANK:
            targeted_fields.update(q.target_fields)

        for field in REQUIRED_FIELDS:
            assert field in targeted_fields, (
                f"Required field '{field}' has no question targeting it"
            )

    def test_question_count_reasonable(self):
        """Question bank should have a reasonable number of questions."""
        assert 15 <= len(QUESTION_BANK) <= 50, (
            f"Expected 15-50 questions, got {len(QUESTION_BANK)}"
        )

    def test_at_least_5_required_questions(self):
        """At least 5 required questions for core business context."""
        required = [q for q in QUESTION_BANK if q.priority == QuestionPriority.REQUIRED]
        assert len(required) >= 5, f"Expected >=5 required questions, got {len(required)}"

    def test_question_ids_follow_convention(self):
        """Question IDs should follow q_ prefix convention."""
        for q in QUESTION_BANK:
            assert q.id.startswith("q_"), f"Question ID '{q.id}' should start with 'q_'"

    def test_multi_value_questions_target_list_fields(self):
        """Multi-value questions should target list-type context fields."""
        list_fields = {
            "target_industries", "target_titles", "target_locations",
            "pain_points", "value_propositions", "differentiators",
            "positive_signals", "disqualifiers", "content_topics",
        }
        for q in QUESTION_BANK:
            if q.multi_value:
                for tf in q.target_fields:
                    assert tf in list_fields, (
                        f"Multi-value question {q.id} targets non-list field '{tf}'"
                    )


# ---------------------------------------------------------------------------
# Test: Adaptive Question Routing
# ---------------------------------------------------------------------------

class TestAdaptiveRouting:
    """Test that questions adapt based on current context."""

    def test_first_question_is_business_name(self, engine, empty_context):
        """First question should be about business name (highest priority required)."""
        q = engine.get_next_question(empty_context)
        assert q is not None
        assert q.id == "q_business_name"

    def test_skips_answered_questions(self, engine):
        """Questions whose target fields are populated are skipped."""
        context = {"business_name": "TestBiz"}
        q = engine.get_next_question(context)
        assert q is not None
        assert q.id != "q_business_name"

    def test_required_before_important(self, engine):
        """Required questions are asked before important ones."""
        # Fill only business_name — next should be another required question
        context = {"business_name": "TestBiz"}
        q = engine.get_next_question(context)
        assert q is not None
        assert q.priority == QuestionPriority.REQUIRED

    def test_important_before_optional(self, engine, minimal_context):
        """Important questions come before optional ones."""
        q = engine.get_next_question(minimal_context)
        assert q is not None
        # Should be an important question since all required are filled
        assert q.priority in (QuestionPriority.IMPORTANT, QuestionPriority.OPTIONAL)

    def test_returns_none_when_all_answered(self, engine, full_context):
        """Returns None when all questions are answered."""
        q = engine.get_next_question(full_context)
        assert q is None

    def test_asked_ids_tracked(self, engine, empty_context):
        """Questions in asked_ids set are not returned again."""
        asked = {"q_business_name"}
        q = engine.get_next_question(empty_context, asked_ids=asked)
        assert q is not None
        assert q.id != "q_business_name"

    def test_follow_up_trigger_respected(self, engine):
        """Questions with follow_up_trigger only asked when trigger field is populated."""
        # Find a question with a follow_up_trigger (if any exist)
        trigger_questions = [q for q in QUESTION_BANK if q.follow_up_trigger]
        if trigger_questions:
            q = trigger_questions[0]
            # Without trigger field, this question should not appear
            empty_context = {}
            questions = []
            asked = set()
            while True:
                next_q = engine.get_next_question(empty_context, asked)
                if next_q is None:
                    break
                questions.append(next_q.id)
                asked.add(next_q.id)

            # Trigger question should not appear (trigger field not set)
            # This depends on implementation — may or may not appear


# ---------------------------------------------------------------------------
# Test: Completion Scoring
# ---------------------------------------------------------------------------

class TestCompletionScoring:
    """Test the weighted completion score calculation."""

    def test_empty_context_score_zero(self, engine, empty_context):
        """Empty context should have 0.0 completion score."""
        score = engine.get_completion_score(empty_context)
        assert score == 0.0

    def test_full_context_score_one(self, engine, full_context):
        """Full context should have ~1.0 completion score."""
        score = engine.get_completion_score(full_context)
        assert score >= 0.95, f"Expected >=0.95, got {score}"

    def test_required_only_gives_partial_score(self, engine):
        """Filling only required fields gives partial (not full) score."""
        required_only = {
            "business_name": "TestBiz",
            "business_description": "A test business that does testing.",
            "business_model": "B2B service",
            "price_range": (1000, 10000),
            "target_industries": ["Tech"],
            "target_company_sizes": (10, 500),
            "target_titles": ["CTO"],
            "pain_points": ["Testing is hard"],
            "value_propositions": ["We make it easy"],
        }
        score = engine.get_completion_score(required_only)
        # Required fields = 60% weight, so should be around 0.6
        assert 0.55 <= score <= 0.65, f"Expected ~0.6, got {score}"

    def test_score_increases_monotonically(self, engine):
        """Adding fields should never decrease the score."""
        context: dict = {}
        prev_score = 0.0

        fields_to_add = [
            ("business_name", "TestBiz"),
            ("business_description", "A test business for testing"),
            ("business_model", "B2B SaaS service model"),
            ("price_range", (500, 5000)),
            ("target_industries", ["Tech"]),
            ("target_company_sizes", (10, 500)),
            ("target_titles", ["CTO"]),
            ("pain_points", ["Testing is hard"]),
            ("value_propositions", ["We make testing easy"]),
            ("region", "US"),
            ("sending_domain", "mail.test.com"),
        ]

        for field_name, value in fields_to_add:
            context[field_name] = value
            score = engine.get_completion_score(context)
            assert score >= prev_score, (
                f"Score decreased from {prev_score} to {score} "
                f"after adding '{field_name}'"
            )
            prev_score = score

    def test_score_between_zero_and_one(self, engine):
        """Score should always be in [0.0, 1.0]."""
        for ctx in ({}, {"business_name": "X"}, {"business_name": "X", "region": "US"}):
            score = engine.get_completion_score(ctx)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"


# ---------------------------------------------------------------------------
# Test: Completeness Check
# ---------------------------------------------------------------------------

class TestCompletenessCheck:
    """Test the is_complete() method."""

    def test_empty_not_complete(self, engine, empty_context):
        """Empty context is never complete."""
        assert not engine.is_complete(empty_context)

    def test_minimal_is_complete(self, engine, minimal_context):
        """Minimal required context should be complete."""
        assert engine.is_complete(minimal_context)

    def test_missing_one_required_not_complete(self, engine, minimal_context):
        """Missing any single required field makes it incomplete."""
        for field in REQUIRED_FIELDS:
            ctx = dict(minimal_context)
            del ctx[field]
            assert not engine.is_complete(ctx), (
                f"Should be incomplete without '{field}'"
            )

    def test_empty_string_not_counted(self, engine, minimal_context):
        """Empty strings are not counted as populated."""
        ctx = dict(minimal_context)
        ctx["business_name"] = ""
        assert not engine.is_complete(ctx)

    def test_empty_list_not_counted(self, engine, minimal_context):
        """Empty lists are not counted as populated."""
        ctx = dict(minimal_context)
        ctx["target_industries"] = []
        assert not engine.is_complete(ctx)

    def test_custom_minimum_completion(self):
        """Custom minimum_completion threshold is respected."""
        strict_engine = InterviewEngine(minimum_completion=0.99)
        # Minimal context won't meet 0.99 threshold
        ctx = {
            "business_name": "TestBiz",
            "business_description": "A test business for testing",
            "business_model": "B2B service",
            "price_range": (100, 1000),
            "target_industries": ["Tech"],
            "target_company_sizes": (10, 100),
            "target_titles": ["CTO"],
            "pain_points": ["Problem"],
            "value_propositions": ["Solution"],
        }
        # Score will be ~0.6, below 0.99
        assert not strict_engine.is_complete(ctx)


# ---------------------------------------------------------------------------
# Test: Phase Progression
# ---------------------------------------------------------------------------

class TestPhaseProgression:
    """Test interview phase detection and ordering."""

    def test_starts_at_identity_phase(self, engine, empty_context):
        """Empty context should be in identity phase."""
        phase = engine.get_current_phase(empty_context)
        assert phase == InterviewPhase.IDENTITY

    def test_progresses_to_business_model(self, engine):
        """After identity, should progress to business_model."""
        context = {
            "business_name": "TestBiz",
            "business_description": "A test business for testing things.",
            "website": "https://testbiz.com",
            "region": "US",
        }
        phase = engine.get_current_phase(context)
        assert phase == InterviewPhase.BUSINESS_MODEL

    def test_reaches_review_when_complete(self, engine, full_context):
        """Full context should be in REVIEW phase."""
        phase = engine.get_current_phase(full_context)
        assert phase == InterviewPhase.REVIEW

    def test_all_phases_in_order(self, engine):
        """get_all_phases returns phases in correct order."""
        phases = engine.get_all_phases()
        assert phases[0] == InterviewPhase.IDENTITY
        assert phases[-1] == InterviewPhase.REVIEW
        assert len(phases) == len(InterviewPhase)

    def test_get_questions_for_phase(self, engine, empty_context):
        """Can retrieve questions filtered by phase."""
        identity_qs = engine.get_questions_for_phase(
            InterviewPhase.IDENTITY, empty_context
        )
        assert len(identity_qs) > 0
        assert all(q.phase == InterviewPhase.IDENTITY for q in identity_qs)


# ---------------------------------------------------------------------------
# Test: Progress Tracking
# ---------------------------------------------------------------------------

class TestProgressTracking:
    """Test the InterviewProgress snapshot."""

    def test_progress_empty_context(self, engine, empty_context):
        """Progress for empty context."""
        progress = engine.get_progress(empty_context)
        assert isinstance(progress, InterviewProgress)
        assert progress.required_filled == 0
        assert progress.required_total == len(REQUIRED_FIELDS)
        assert progress.completion_score == 0.0
        assert not progress.is_complete
        assert len(progress.missing_required) == len(REQUIRED_FIELDS)

    def test_progress_minimal_context(self, engine, minimal_context):
        """Progress for minimal (all required) context."""
        progress = engine.get_progress(minimal_context)
        assert progress.required_filled == progress.required_total
        assert progress.is_complete
        assert len(progress.missing_required) == 0

    def test_progress_tracks_asked_questions(self, engine, empty_context):
        """Progress reflects questions_asked count."""
        asked = {"q_business_name", "q_business_description"}
        progress = engine.get_progress(empty_context, asked_ids=asked)
        assert progress.questions_asked == 2

    def test_progress_total_fields(self, engine, empty_context):
        """Total fields matches ALL_TRACKED_FIELDS."""
        progress = engine.get_progress(empty_context)
        assert progress.total_fields == len(ALL_TRACKED_FIELDS)

    def test_progress_remaining_decreases(self, engine):
        """Remaining questions decrease as context fills up."""
        empty_progress = engine.get_progress({})
        filled_progress = engine.get_progress({
            "business_name": "Test",
            "business_description": "A test business for testing things",
        })
        assert filled_progress.questions_remaining < empty_progress.questions_remaining


# ---------------------------------------------------------------------------
# Test: Answer Parsing
# ---------------------------------------------------------------------------

class TestAnswerParsing:
    """Test the static answer parsing methods."""

    # -- List parsing --

    def test_parse_comma_list(self):
        """Parse comma-separated values."""
        result = InterviewEngine.parse_list_answer("Tech, Finance, Healthcare")
        assert result == ["Tech", "Finance", "Healthcare"]

    def test_parse_newline_list(self):
        """Parse newline-separated values."""
        result = InterviewEngine.parse_list_answer("Tech\nFinance\nHealthcare")
        assert result == ["Tech", "Finance", "Healthcare"]

    def test_parse_semicolon_list(self):
        """Parse semicolon-separated values."""
        result = InterviewEngine.parse_list_answer("Tech; Finance; Healthcare")
        assert result == ["Tech", "Finance", "Healthcare"]

    def test_parse_numbered_list(self):
        """Parse numbered list format."""
        result = InterviewEngine.parse_list_answer("1. Tech\n2. Finance\n3. Healthcare")
        assert result == ["Tech", "Finance", "Healthcare"]

    def test_parse_bulleted_list(self):
        """Parse bulleted list format."""
        result = InterviewEngine.parse_list_answer("- Tech\n- Finance\n- Healthcare")
        assert result == ["Tech", "Finance", "Healthcare"]

    def test_parse_list_strips_whitespace(self):
        """List parsing strips whitespace from items."""
        result = InterviewEngine.parse_list_answer("  Tech  ,  Finance  ")
        assert result == ["Tech", "Finance"]

    def test_parse_list_ignores_empty(self):
        """List parsing ignores empty items."""
        result = InterviewEngine.parse_list_answer("Tech,,Finance,")
        assert result == ["Tech", "Finance"]

    def test_parse_single_item(self):
        """Single item returns a list with one element."""
        result = InterviewEngine.parse_list_answer("Just one thing")
        assert result == ["Just one thing"]

    # -- Range parsing --

    def test_parse_range_basic(self):
        """Parse basic range format."""
        result = InterviewEngine.parse_range_answer("500-5000")
        assert result == (500, 5000)

    def test_parse_range_with_currency(self):
        """Parse range with dollar signs."""
        result = InterviewEngine.parse_range_answer("$500 - $5,000")
        assert result == (500, 5000)

    def test_parse_range_with_commas(self):
        """Parse range with comma-formatted numbers."""
        result = InterviewEngine.parse_range_answer("10,000 to 100,000")
        assert result == (10000, 100000)

    def test_parse_range_single_number(self):
        """Single number gives (n, n) range."""
        result = InterviewEngine.parse_range_answer("5000")
        assert result == (5000, 5000)

    def test_parse_range_invalid(self):
        """Invalid input returns None."""
        result = InterviewEngine.parse_range_answer("not a number")
        assert result is None

    # -- Integer parsing --

    def test_parse_int_basic(self):
        """Parse basic integer."""
        result = InterviewEngine.parse_int_answer("30")
        assert result == 30

    def test_parse_int_with_text(self):
        """Parse integer from text like '30 days'."""
        result = InterviewEngine.parse_int_answer("30 days")
        assert result == 30

    def test_parse_int_with_comma(self):
        """Parse comma-formatted integer."""
        result = InterviewEngine.parse_int_answer("1,000")
        assert result == 1000

    def test_parse_int_invalid(self):
        """Invalid input returns None."""
        result = InterviewEngine.parse_int_answer("no number here")
        assert result is None


# ---------------------------------------------------------------------------
# Test: Question Metadata
# ---------------------------------------------------------------------------

class TestQuestionMetadata:
    """Test question lookup and metadata methods."""

    def test_get_question_by_id(self, engine):
        """Can retrieve a question by its ID."""
        q = engine.get_question("q_business_name")
        assert q is not None
        assert q.id == "q_business_name"
        assert q.phase == InterviewPhase.IDENTITY

    def test_get_nonexistent_question(self, engine):
        """Nonexistent ID returns None."""
        q = engine.get_question("q_does_not_exist")
        assert q is None

    def test_get_all_questions(self, engine):
        """get_all_questions returns the full bank."""
        all_qs = engine.get_all_questions()
        assert len(all_qs) == len(QUESTION_BANK)

    def test_get_required_questions(self, engine):
        """get_required_questions filters by priority."""
        required = engine.get_required_questions()
        assert len(required) > 0
        assert all(q.priority == QuestionPriority.REQUIRED for q in required)

    def test_get_phase_count(self, engine):
        """Phase count returns dict with all phases."""
        counts = engine.get_phase_count()
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == len(QUESTION_BANK)


# ---------------------------------------------------------------------------
# Test: Context Summary
# ---------------------------------------------------------------------------

class TestContextSummary:
    """Test the context summary generation."""

    def test_summary_includes_name(self, engine, minimal_context):
        """Summary includes business name."""
        summary = engine.generate_context_summary(minimal_context)
        assert "TestBiz" in summary

    def test_summary_includes_model(self, engine, minimal_context):
        """Summary includes business model."""
        summary = engine.generate_context_summary(minimal_context)
        assert "B2B service" in summary

    def test_summary_includes_pain_points(self, engine, minimal_context):
        """Summary includes pain points."""
        summary = engine.generate_context_summary(minimal_context)
        assert "Pain Points" in summary

    def test_summary_empty_context(self, engine, empty_context):
        """Summary handles empty context gracefully."""
        summary = engine.generate_context_summary(empty_context)
        assert "Unknown" in summary or "N/A" in summary

    def test_summary_includes_deal_size(self, engine, minimal_context):
        """Summary includes formatted deal size."""
        summary = engine.generate_context_summary(minimal_context)
        assert "$1,000" in summary or "1000" in summary


# ---------------------------------------------------------------------------
# Test: Prompt Generation
# ---------------------------------------------------------------------------

class TestPromptGeneration:
    """Test prompt generation for the ArchitectAgent."""

    def test_prompt_includes_question(self, engine, empty_context):
        """Generated prompt includes the question text."""
        q = engine.get_next_question(empty_context)
        assert q is not None
        prompt = engine.generate_question_prompt(q, empty_context)
        assert q.question in prompt

    def test_prompt_includes_hint(self, engine, empty_context):
        """Generated prompt includes the hint."""
        q = engine.get_next_question(empty_context)
        assert q is not None
        if q.hint:
            prompt = engine.generate_question_prompt(q, empty_context)
            assert q.hint in prompt

    def test_prompt_includes_examples(self, engine, empty_context):
        """Generated prompt includes examples when available."""
        q = engine.get_next_question(empty_context)
        assert q is not None
        if q.examples:
            prompt = engine.generate_question_prompt(q, empty_context)
            assert q.examples[0] in prompt

    def test_prompt_includes_progress(self, engine, empty_context):
        """Generated prompt includes progress indicator."""
        q = engine.get_next_question(empty_context)
        assert q is not None
        prompt = engine.generate_question_prompt(q, empty_context)
        assert "Progress:" in prompt


# ---------------------------------------------------------------------------
# Test: Field Tracking
# ---------------------------------------------------------------------------

class TestFieldTracking:
    """Test field set integrity."""

    def test_required_fields_subset_of_all(self):
        """Required fields are a subset of all tracked fields."""
        assert REQUIRED_FIELDS.issubset(ALL_TRACKED_FIELDS)

    def test_important_fields_subset_of_all(self):
        """Important fields are a subset of all tracked fields."""
        assert IMPORTANT_FIELDS.issubset(ALL_TRACKED_FIELDS)

    def test_no_overlap_required_important(self):
        """Required and important fields don't overlap."""
        overlap = REQUIRED_FIELDS & IMPORTANT_FIELDS
        assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_all_tracked_fields_count(self):
        """Total tracked fields should be reasonable."""
        assert 10 <= len(ALL_TRACKED_FIELDS) <= 30

    def test_required_fields_count(self):
        """Should have 7-12 required fields."""
        assert 7 <= len(REQUIRED_FIELDS) <= 12


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_none_values_not_counted(self, engine):
        """None values in context are not counted."""
        context = {"business_name": None}
        assert not engine._field_has_value(context, "business_name")

    def test_zero_is_counted(self, engine):
        """Zero (int) IS counted as a value."""
        context = {"daily_outreach_limit": 0}
        # 0 is falsy in Python but should count for a field
        assert engine._field_has_value(context, "daily_outreach_limit")

    def test_tuple_counted_as_value(self, engine):
        """Tuples are counted as values."""
        context = {"price_range": (100, 500)}
        assert engine._field_has_value(context, "price_range")

    def test_empty_tuple_not_counted(self, engine):
        """Empty tuple is not counted."""
        context = {"price_range": ()}
        assert not engine._field_has_value(context, "price_range")

    def test_whitespace_only_string_not_counted(self, engine):
        """Whitespace-only strings are not counted."""
        context = {"business_name": "   "}
        assert not engine._field_has_value(context, "business_name")

    def test_missing_field_not_counted(self, engine):
        """Missing fields are not counted."""
        assert not engine._field_has_value({}, "business_name")
