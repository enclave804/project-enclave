"""
Tests for Phase 13: The Hive Mind (Shared Intelligence).

Covers:
    - HiveMind (publish, query, digest, flow, boost/decay, cache)
    - LeadScorer (heuristic, ML, batch, features, tiers, persistence)
    - ExperimentEngine (start, track, results, Bayesian, winner, batch)
    - BaseAgent integration (hive property, consult_hive, publish_to_hive)
    - Brain Dashboard helpers (insight stats, experiment stats, score dist,
      knowledge flow, brain health, formatting)
    - YAML configs (cs.yaml verified — hive agents use existing configs)
    - DB migration schema (010_hive_mind.sql)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# HiveMind Tests
# ══════════════════════════════════════════════════════════════════════


class TestHiveMindImport:
    """Basic import and construction tests."""

    def test_import(self):
        from core.memory.hive_mind import HiveMind
        assert HiveMind is not None

    def test_categories_defined(self):
        from core.memory.hive_mind import INSIGHT_CATEGORIES
        assert isinstance(INSIGHT_CATEGORIES, dict)
        assert len(INSIGHT_CATEGORIES) >= 10
        assert "email_performance" in INSIGHT_CATEGORIES
        assert "deal_patterns" in INSIGHT_CATEGORIES
        assert "competitive_intel" in INSIGHT_CATEGORIES

    def test_construct(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        embedder = MagicMock()
        hive = HiveMind(db=db, embedder=embedder, vertical_id="test_vert")
        assert hive.vertical_id == "test_vert"
        assert hive.db is db
        assert hive.embedder is embedder

    def test_repr(self):
        from core.memory.hive_mind import HiveMind
        hive = HiveMind(db=MagicMock(), embedder=MagicMock())
        r = repr(hive)
        assert "HiveMind" in r
        assert "cache_size" in r


class TestHiveMindPublish:
    """Tests for HiveMind.publish()."""

    def _make_hive(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        db.store_insight = MagicMock(return_value={"id": "ins-1", "content": "test"})
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.1] * 1536)
        return HiveMind(db=db, embedder=embedder, vertical_id="test")

    def test_publish_success(self):
        hive = self._make_hive()
        result = hive.publish(
            source_agent="outreach",
            topic="email_performance",
            content="Short subjects get 2x opens",
            confidence=0.85,
        )
        assert result is not None
        hive.db.store_insight.assert_called_once()
        call_kwargs = hive.db.store_insight.call_args
        assert call_kwargs[1]["source_agent_id"] == "outreach"
        assert call_kwargs[1]["insight_type"] == "email_performance"

    def test_publish_with_evidence(self):
        hive = self._make_hive()
        result = hive.publish(
            source_agent="outreach",
            topic="email_performance",
            content="Test insight",
            evidence={"sample_size": 100},
            title="Open Rate Finding",
        )
        assert result is not None
        call_kwargs = hive.db.store_insight.call_args[1]
        metadata = call_kwargs["metadata"]
        assert metadata["evidence"]["sample_size"] == 100

    def test_publish_clamps_confidence(self):
        hive = self._make_hive()
        # Should not raise, just clamp
        hive.publish(source_agent="test", topic="test", content="c", confidence=1.5)
        hive.publish(source_agent="test", topic="test", content="c", confidence=-0.5)
        assert hive.db.store_insight.call_count == 2

    def test_publish_unknown_topic_maps_to_market_signal(self):
        hive = self._make_hive()
        hive.publish(source_agent="test", topic="totally_unknown_thing", content="c")
        call_kwargs = hive.db.store_insight.call_args[1]
        assert call_kwargs["insight_type"] == "market_signal"

    def test_publish_invalidates_cache(self):
        hive = self._make_hive()
        cache_key = "email_performance:0.7:5"
        hive._cache[cache_key] = [{"id": "old"}]
        hive.publish(source_agent="test", topic="email_performance", content="c")
        assert cache_key not in hive._cache

    def test_publish_logs_flow(self):
        hive = self._make_hive()
        hive.publish(source_agent="outreach", topic="email_performance", content="c")
        assert len(hive._flow_log) == 1
        assert hive._flow_log[0]["direction"] == "publish"
        assert hive._flow_log[0]["agent"] == "outreach"

    def test_publish_embedding_failure_continues(self):
        hive = self._make_hive()
        hive.embedder.embed_query.side_effect = RuntimeError("GPU OOM")
        result = hive.publish(source_agent="test", topic="test", content="c")
        # Should still succeed (embedding is optional)
        assert result is not None

    def test_publish_db_failure_returns_none(self):
        hive = self._make_hive()
        hive.db.store_insight.side_effect = RuntimeError("DB down")
        result = hive.publish(source_agent="test", topic="test", content="c")
        assert result is None


class TestHiveMindQuery:
    """Tests for HiveMind.query() and query_by_text()."""

    def _make_hive(self, search_results=None):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        db.search_insights = MagicMock(return_value=search_results or [])
        db.client = MagicMock()
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.1] * 1536)
        return HiveMind(db=db, embedder=embedder, vertical_id="test")

    def test_query_empty(self):
        hive = self._make_hive([])
        results = hive.query(topic="email_performance", consumer_agent="social")
        assert results == []

    def test_query_returns_results(self):
        insights = [
            {"id": "1", "confidence_score": 0.9, "source_agent_id": "outreach", "content": "test"},
            {"id": "2", "confidence_score": 0.5, "source_agent_id": "ads", "content": "test2"},
        ]
        hive = self._make_hive(insights)
        results = hive.query(topic="email_performance", min_confidence=0.7)
        # Only insight with confidence >= 0.7 should pass
        assert len(results) == 1
        assert results[0]["id"] == "1"

    def test_query_exclude_own(self):
        insights = [
            {"id": "1", "confidence_score": 0.9, "source_agent_id": "outreach"},
            {"id": "2", "confidence_score": 0.9, "source_agent_id": "social"},
        ]
        hive = self._make_hive(insights)
        results = hive.query(
            topic="test", consumer_agent="outreach", exclude_own=True, min_confidence=0.5,
        )
        assert len(results) == 1
        assert results[0]["source_agent_id"] == "social"

    def test_query_by_text_delegates(self):
        hive = self._make_hive([])
        results = hive.query_by_text("What pricing works?", consumer_agent="proposal")
        assert results == []
        hive.embedder.embed_query.assert_called()

    def test_query_caching(self):
        insights = [{"id": "1", "confidence_score": 0.9}]
        hive = self._make_hive(insights)
        # First call hits DB
        r1 = hive.query(topic="t", min_confidence=0.5, limit=5)
        # Second call should use cache
        r2 = hive.query(topic="t", min_confidence=0.5, limit=5)
        assert hive.db.search_insights.call_count == 1
        assert r1 == r2

    def test_query_cache_clear(self):
        hive = self._make_hive([])
        hive._cache["test:0.5:5"] = [{"id": "cached"}]
        hive.clear_cache()
        assert len(hive._cache) == 0

    def test_query_logs_consumption(self):
        hive = self._make_hive([{"id": "1", "confidence_score": 0.9}])
        hive.query(topic="test", consumer_agent="social", min_confidence=0.5)
        assert len(hive._flow_log) == 1
        assert hive._flow_log[0]["direction"] == "consume"
        assert hive._flow_log[0]["agent"] == "social"

    def test_query_embedding_failure_returns_empty(self):
        hive = self._make_hive([])
        hive.embedder.embed_query.side_effect = RuntimeError("embed fail")
        results = hive.query(topic="test")
        assert results == []

    def test_query_source_agents_filter(self):
        insights = [
            {"id": "1", "confidence_score": 0.9, "source_agent_id": "outreach"},
            {"id": "2", "confidence_score": 0.9, "source_agent_id": "ads"},
            {"id": "3", "confidence_score": 0.9, "source_agent_id": "social"},
        ]
        hive = self._make_hive(insights)
        results = hive.query(
            topic="test", source_agents=["outreach", "social"],
            min_confidence=0.5,
        )
        agent_ids = {r["source_agent_id"] for r in results}
        assert "ads" not in agent_ids


class TestHiveMindDigest:
    """Tests for HiveMind.get_digest()."""

    def test_digest_structure(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.client = MagicMock()
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.1] * 10)
        hive = HiveMind(db=db, embedder=embedder)
        digest = hive.get_digest(days=7)
        assert "period_days" in digest
        assert "topics_covered" in digest
        assert "total_insights" in digest
        assert "insights_by_topic" in digest
        assert "generated_at" in digest


class TestHiveMindFlow:
    """Tests for knowledge flow tracking."""

    def test_get_knowledge_flow(self):
        from core.memory.hive_mind import HiveMind
        hive = HiveMind(db=MagicMock(), embedder=MagicMock())
        hive._flow_log.append({"direction": "publish", "agent": "a", "topic": "t"})
        flow = hive.get_knowledge_flow()
        assert len(flow) == 1

    def test_get_agent_connections(self):
        from core.memory.hive_mind import HiveMind
        hive = HiveMind(db=MagicMock(), embedder=MagicMock())
        hive._flow_log = [
            {"direction": "publish", "agent": "outreach", "topic": "email"},
            {"direction": "consume", "agent": "social", "topic": "email"},
            {"direction": "publish", "agent": "outreach", "topic": "deals"},
        ]
        connections = hive.get_agent_connections()
        assert connections["publishers"]["outreach"] == 2
        assert connections["consumers"]["social"] == 1
        assert "email" in connections["topics_by_agent"]["outreach"]


class TestHiveMindReinforcement:
    """Tests for boost/decay."""

    def test_boost_calls_rpc(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        hive = HiveMind(db=db, embedder=MagicMock())
        hive.boost_insight("ins-1", 0.05)
        db.client.rpc.assert_called_once()

    def test_decay_calls_rpc(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        hive = HiveMind(db=db, embedder=MagicMock())
        hive.decay_insight("ins-1", 0.02)
        db.client.rpc.assert_called_once()

    def test_boost_failure_silent(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        db.client.rpc.side_effect = RuntimeError("RPC fail")
        hive = HiveMind(db=db, embedder=MagicMock())
        # Should not raise
        hive.boost_insight("ins-1")

    def test_decay_failure_silent(self):
        from core.memory.hive_mind import HiveMind
        db = MagicMock()
        db.client.rpc.side_effect = RuntimeError("RPC fail")
        hive = HiveMind(db=db, embedder=MagicMock())
        hive.decay_insight("ins-1")


# ══════════════════════════════════════════════════════════════════════
# LeadScorer Tests
# ══════════════════════════════════════════════════════════════════════


class TestLeadScorerImport:
    """Basic import and construction tests."""

    def test_import(self):
        from core.analytics.lead_scorer import LeadScorer
        assert LeadScorer is not None

    def test_feature_names(self):
        from core.analytics.lead_scorer import FEATURE_NAMES
        assert isinstance(FEATURE_NAMES, list)
        assert len(FEATURE_NAMES) == 15
        assert "title_score" in FEATURE_NAMES
        assert "proposal_amount" in FEATURE_NAMES

    def test_title_scores(self):
        from core.analytics.lead_scorer import TITLE_SCORES
        assert TITLE_SCORES["ceo"] == 10
        assert TITLE_SCORES["manager"] == 5
        assert TITLE_SCORES["intern"] == 1

    def test_target_industries(self):
        from core.analytics.lead_scorer import TARGET_INDUSTRIES
        assert "fintech" in TARGET_INDUSTRIES
        assert "healthcare" in TARGET_INDUSTRIES

    def test_construct(self):
        from core.analytics.lead_scorer import LeadScorer
        with tempfile.TemporaryDirectory() as tmp:
            scorer = LeadScorer(db=None, model_path=Path(tmp))
            assert scorer._using_ml is False
            assert scorer._training_samples == 0

    def test_repr(self):
        from core.analytics.lead_scorer import LeadScorer
        with tempfile.TemporaryDirectory() as tmp:
            scorer = LeadScorer(db=None, model_path=Path(tmp))
            r = repr(scorer)
            assert "LeadScorer" in r
            assert "Heuristic" in r


class TestLeadScorerHeuristic:
    """Tests for heuristic scoring."""

    def _scorer(self):
        from core.analytics.lead_scorer import LeadScorer
        return LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))

    def test_empty_lead_scores_zero(self):
        scorer = self._scorer()
        score = scorer.predict_score({})
        assert score == 0

    def test_ceo_fintech_scores_high(self):
        scorer = self._scorer()
        score = scorer.predict_score({
            "title": "CEO",
            "industry": "fintech",
            "company_size": 200,
            "has_email": True,
            "has_replied": True,
            "meetings_booked": 1,
        })
        assert score >= 60  # Should be high

    def test_intern_no_engagement_scores_low(self):
        scorer = self._scorer()
        score = scorer.predict_score({
            "title": "Intern",
            "industry": "unknown",
            "company_size": 5,
        })
        assert score < 20

    def test_proposal_sent_boosts_score(self):
        scorer = self._scorer()
        base = scorer.predict_score({"title": "Director", "industry": "saas"})
        with_proposal = scorer.predict_score({
            "title": "Director", "industry": "saas",
            "proposal_sent": True, "proposal_amount": 25000,
        })
        assert with_proposal > base

    def test_score_range_0_100(self):
        scorer = self._scorer()
        for title in ["CEO", "Intern", ""]:
            for industry in ["fintech", "unknown"]:
                score = scorer.predict_score({"title": title, "industry": industry})
                assert 0 <= score <= 100

    def test_meetings_boost(self):
        scorer = self._scorer()
        s1 = scorer.predict_score({"title": "VP", "industry": "tech"})
        s2 = scorer.predict_score({"title": "VP", "industry": "tech", "meetings_booked": 3})
        assert s2 > s1


class TestLeadScorerBatch:
    """Tests for batch scoring."""

    def test_predict_batch(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        leads = [
            {"title": "CEO", "industry": "fintech", "company_size": 500},
            {"title": "Intern", "company_size": 5},
        ]
        results = scorer.predict_batch(leads)
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
        assert "tier" in results[0]
        assert "factors" in results[0]

    def test_tier_assignment(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        assert scorer._score_to_tier(90) == "hot"
        assert scorer._score_to_tier(70) == "warm"
        assert scorer._score_to_tier(50) == "lukewarm"
        assert scorer._score_to_tier(20) == "cold"


class TestLeadScorerExplain:
    """Tests for score explainability."""

    def test_explain_ceo(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        factors = scorer._explain_score({"title": "CEO", "industry": "fintech"}, 80)
        assert any("Decision-maker" in f for f in factors)

    def test_explain_reply(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        factors = scorer._explain_score({"has_replied": True}, 60)
        assert any("replied" in f.lower() for f in factors)

    def test_explain_enterprise(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        factors = scorer._explain_score({"company_size": 500}, 50)
        assert any("Enterprise" in f for f in factors)


class TestLeadScorerTrain:
    """Tests for model training."""

    def test_train_insufficient_data(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        data = [{"title": "CEO", "outcome": "won"} for _ in range(10)]
        result = scorer.train(data)
        assert result["status"] == "heuristic"
        assert scorer._using_ml is False

    def test_train_sufficient_data(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        data = []
        for i in range(60):
            data.append({
                "title": "CEO" if i % 2 == 0 else "Intern",
                "industry": "fintech" if i % 3 == 0 else "other",
                "company_size": 200 + i * 10,
                "has_email": True,
                "has_replied": i % 2 == 0,
                "meetings_booked": 1 if i % 4 == 0 else 0,
                "outcome": "won" if i % 2 == 0 else "lost",
            })
        result = scorer.train(data)
        assert result["status"] == "trained"
        assert scorer._using_ml is True
        assert "top_features" in result

    def test_train_without_sklearn_falls_back(self):
        """When sklearn is unavailable, training returns heuristic."""
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        data = [{"title": "CEO", "outcome": "won"} for _ in range(60)]
        result = scorer.train(data)
        # Either trained (sklearn available) or heuristic (not available)
        assert result["status"] in ("trained", "heuristic")

    def test_ml_predict_after_training(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        data = []
        for i in range(60):
            data.append({
                "title": "CEO" if i % 2 == 0 else "Intern",
                "company_size": 500 if i % 2 == 0 else 10,
                "outcome": "won" if i % 2 == 0 else "lost",
            })
        scorer.train(data)
        score = scorer.predict_score({"title": "CEO", "company_size": 500})
        assert 0 <= score <= 100

    def test_model_persistence(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from core.analytics.lead_scorer import LeadScorer
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scorer = LeadScorer(db=None, model_path=tmp_path)
            data = []
            for i in range(60):
                data.append({
                    "title": "CEO" if i % 2 == 0 else "Intern",
                    "outcome": "won" if i % 2 == 0 else "lost",
                })
            scorer.train(data)
            # Should save model
            model_file = tmp_path / "lead_scorer_enclave_guard.pkl"
            assert model_file.exists()

            # New scorer should load model
            scorer2 = LeadScorer(db=None, model_path=tmp_path)
            assert scorer2._using_ml is True

    def test_get_model_stats(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        stats = scorer.get_model_stats()
        assert stats["using_ml"] is False
        assert stats["feature_count"] == 15


class TestLeadScorerFeatures:
    """Tests for feature extraction."""

    def test_extract_all_defaults(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        features = scorer._extract_features({})
        assert len(features) == 15
        assert all(isinstance(f, float) for f in features)

    def test_extract_title_score(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        features = scorer._extract_features({"title": "CTO"})
        assert features[0] == 10.0  # CTO = 10

    def test_extract_industry_match(self):
        from core.analytics.lead_scorer import LeadScorer
        scorer = LeadScorer(db=None, model_path=Path(tempfile.mkdtemp()))
        f1 = scorer._extract_features({"industry": "fintech"})
        f2 = scorer._extract_features({"industry": "basket weaving"})
        assert f1[2] == 1.0  # Match
        assert f2[2] == 0.0  # No match


# ══════════════════════════════════════════════════════════════════════
# ExperimentEngine Tests
# ══════════════════════════════════════════════════════════════════════


class TestExperimentEngineImport:
    """Basic import and construction."""

    def test_import(self):
        from core.optimization.experiment_engine import ExperimentEngine
        assert ExperimentEngine is not None

    def test_outcomes_defined(self):
        from core.optimization.experiment_engine import POSITIVE_OUTCOMES, NEGATIVE_OUTCOMES
        assert "reply" in POSITIVE_OUTCOMES
        assert "ignore" in NEGATIVE_OUTCOMES
        assert len(POSITIVE_OUTCOMES) >= 5
        assert len(NEGATIVE_OUTCOMES) >= 5

    def test_constants(self):
        from core.optimization.experiment_engine import MIN_OBSERVATIONS, WIN_THRESHOLD
        assert MIN_OBSERVATIONS == 30
        assert WIN_THRESHOLD == 0.95

    def test_construct(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        assert len(engine._experiments) == 0

    def test_repr(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        assert "ExperimentEngine" in repr(engine)


class TestExperimentLifecycle:
    """Tests for experiment creation and management."""

    def _engine(self):
        from core.optimization.experiment_engine import ExperimentEngine
        return ExperimentEngine()

    def test_start_experiment(self):
        engine = self._engine()
        exp_id = engine.start_experiment(
            name="Subject Line Test",
            variants=["With Emoji", "Without Emoji"],
            agent_id="outreach",
            metric="open_rate",
        )
        assert exp_id.startswith("exp_")
        assert len(engine._experiments) == 1

    def test_start_requires_two_variants(self):
        engine = self._engine()
        with pytest.raises(ValueError, match="at least 2"):
            engine.start_experiment(name="Bad", variants=["Only One"])

    def test_start_requires_unique_variants(self):
        engine = self._engine()
        with pytest.raises(ValueError, match="unique"):
            engine.start_experiment(name="Bad", variants=["A", "A"])

    def test_pause_resume(self):
        engine = self._engine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        assert engine.pause_experiment(exp_id) is True
        assert engine._experiments[exp_id].status == "paused"
        assert engine.resume_experiment(exp_id) is True
        assert engine._experiments[exp_id].status == "active"

    def test_pause_nonexistent(self):
        engine = self._engine()
        assert engine.pause_experiment("nope") is False

    def test_conclude_experiment(self):
        engine = self._engine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        result = engine.conclude_experiment(exp_id)
        assert result is not None
        assert engine._experiments[exp_id].status == "concluded"

    def test_conclude_nonexistent(self):
        engine = self._engine()
        assert engine.conclude_experiment("nope") is None


class TestExperimentTracking:
    """Tests for outcome tracking."""

    def _engine_with_exp(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        return engine, exp_id

    def test_track_outcome(self):
        engine, exp_id = self._engine_with_exp()
        assert engine.track_outcome(exp_id, "A", "reply") is True
        assert engine.track_outcome(exp_id, "B", "ignore") is True
        exp = engine._experiments[exp_id]
        assert len(exp.outcomes["A"]) == 1
        assert len(exp.outcomes["B"]) == 1

    def test_track_invalid_experiment(self):
        engine, _ = self._engine_with_exp()
        assert engine.track_outcome("nope", "A", "reply") is False

    def test_track_invalid_variant(self):
        engine, exp_id = self._engine_with_exp()
        assert engine.track_outcome(exp_id, "C", "reply") is False

    def test_track_paused_experiment(self):
        engine, exp_id = self._engine_with_exp()
        engine.pause_experiment(exp_id)
        assert engine.track_outcome(exp_id, "A", "reply") is False

    def test_track_batch(self):
        engine, exp_id = self._engine_with_exp()
        results = [
            {"variant": "A", "outcome": "reply"},
            {"variant": "B", "outcome": "ignore"},
            {"variant": "A", "outcome": "click"},
        ]
        tracked = engine.track_batch(exp_id, results)
        assert tracked == 3

    def test_total_observations(self):
        engine, exp_id = self._engine_with_exp()
        for _ in range(5):
            engine.track_outcome(exp_id, "A", "reply")
        for _ in range(3):
            engine.track_outcome(exp_id, "B", "ignore")
        assert engine._experiments[exp_id].total_observations == 8


class TestExperimentAnalysis:
    """Tests for Bayesian analysis and results."""

    def _engine_with_data(self, a_replies=35, a_ignores=15, b_replies=20, b_ignores=30):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        for _ in range(a_replies):
            engine.track_outcome(exp_id, "A", "reply")
        for _ in range(a_ignores):
            engine.track_outcome(exp_id, "A", "ignore")
        for _ in range(b_replies):
            engine.track_outcome(exp_id, "B", "reply")
        for _ in range(b_ignores):
            engine.track_outcome(exp_id, "B", "ignore")
        return engine, exp_id

    def test_get_results_structure(self):
        engine, exp_id = self._engine_with_data()
        results = engine.get_results(exp_id)
        assert results is not None
        assert "variant_stats" in results
        assert "win_probabilities" in results
        assert "has_winner" in results
        assert "total_observations" in results

    def test_variant_stats(self):
        engine, exp_id = self._engine_with_data(a_replies=30, a_ignores=10)
        results = engine.get_results(exp_id)
        assert results["variant_stats"]["A"]["positive"] == 30
        assert results["variant_stats"]["A"]["negative"] == 10
        assert results["variant_stats"]["A"]["rate"] == 0.75

    def test_win_probabilities_sum_to_one(self):
        engine, exp_id = self._engine_with_data()
        results = engine.get_results(exp_id)
        probs = results["win_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_clear_winner_detected(self):
        # A is much better: 90% vs 20%
        engine, exp_id = self._engine_with_data(
            a_replies=45, a_ignores=5, b_replies=10, b_ignores=40,
        )
        results = engine.get_results(exp_id)
        assert results["has_winner"] is True
        assert results["winner"] == "A"
        assert results["winner_confidence"] >= 0.95

    def test_no_winner_when_close(self):
        # A and B are very similar: 50% vs 50%
        engine, exp_id = self._engine_with_data(
            a_replies=25, a_ignores=25, b_replies=25, b_ignores=25,
        )
        results = engine.get_results(exp_id)
        assert results["has_winner"] is False

    def test_no_winner_insufficient_data(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        # Only 5 observations per variant
        for _ in range(5):
            engine.track_outcome(exp_id, "A", "reply")
            engine.track_outcome(exp_id, "B", "ignore")
        results = engine.get_results(exp_id)
        assert results["min_observations_met"] is False
        assert results["has_winner"] is False

    def test_get_winner_returns_name(self):
        engine, exp_id = self._engine_with_data(
            a_replies=45, a_ignores=5, b_replies=10, b_ignores=40,
        )
        winner = engine.get_winner(exp_id)
        assert winner == "A"

    def test_get_winner_returns_none_when_no_winner(self):
        engine, exp_id = self._engine_with_data(
            a_replies=25, a_ignores=25, b_replies=25, b_ignores=25,
        )
        winner = engine.get_winner(exp_id)
        assert winner is None

    def test_get_results_nonexistent(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        assert engine.get_results("nope") is None

    def test_equal_probability_with_no_data(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B", "C"])
        results = engine.get_results(exp_id)
        probs = results["win_probabilities"]
        for p in probs.values():
            assert abs(p - 1/3) < 0.01


class TestExperimentListing:
    """Tests for listing and getting experiments."""

    def test_list_all(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        engine.start_experiment(name="T1", variants=["A", "B"], agent_id="outreach")
        engine.start_experiment(name="T2", variants=["A", "B"], agent_id="social")
        assert len(engine.list_experiments()) == 2

    def test_list_by_agent(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        engine.start_experiment(name="T1", variants=["A", "B"], agent_id="outreach")
        engine.start_experiment(name="T2", variants=["A", "B"], agent_id="social")
        assert len(engine.list_experiments(agent_id="outreach")) == 1

    def test_list_by_status(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp1 = engine.start_experiment(name="T1", variants=["A", "B"])
        engine.start_experiment(name="T2", variants=["A", "B"])
        engine.pause_experiment(exp1)
        active = engine.list_experiments(status="active")
        assert len(active) == 1
        paused = engine.list_experiments(status="paused")
        assert len(paused) == 1

    def test_get_active(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        engine.start_experiment(name="T1", variants=["A", "B"])
        exp2 = engine.start_experiment(name="T2", variants=["A", "B"])
        engine.conclude_experiment(exp2)
        assert len(engine.get_active_experiments()) == 1

    def test_get_experiment(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T1", variants=["A", "B"])
        exp = engine.get_experiment(exp_id)
        assert exp is not None
        assert exp["name"] == "T1"

    def test_get_experiment_nonexistent(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        assert engine.get_experiment("nope") is None


class TestExperimentVariantAssignment:
    """Tests for variant assignment."""

    def test_assign_with_entity_id(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        # Same entity always gets same variant (deterministic)
        v1 = engine.assign_variant(exp_id, entity_id="lead-123")
        v2 = engine.assign_variant(exp_id, entity_id="lead-123")
        assert v1 == v2

    def test_assign_round_robin(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        v1 = engine.assign_variant(exp_id)
        engine.track_outcome(exp_id, v1, "reply")
        v2 = engine.assign_variant(exp_id)
        # After 1 observation, should get next variant
        assert v2 != v1

    def test_assign_paused_returns_none(self):
        from core.optimization.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        exp_id = engine.start_experiment(name="T", variants=["A", "B"])
        engine.pause_experiment(exp_id)
        assert engine.assign_variant(exp_id) is None


class TestExperimentToDict:
    """Tests for Experiment.to_dict()."""

    def test_to_dict(self):
        from core.optimization.experiment_engine import Experiment
        exp = Experiment(
            experiment_id="exp_123",
            name="Test",
            variants=["A", "B"],
            agent_id="outreach",
        )
        d = exp.to_dict()
        assert d["experiment_id"] == "exp_123"
        assert d["name"] == "Test"
        assert d["variants"] == ["A", "B"]
        assert d["status"] == "active"
        assert d["total_observations"] == 0


# ══════════════════════════════════════════════════════════════════════
# BaseAgent Integration Tests
# ══════════════════════════════════════════════════════════════════════


class TestBaseAgentHive:
    """Tests for BaseAgent.hive, consult_hive, publish_to_hive."""

    def _make_agent(self):
        """Create a mock agent with required attributes."""
        from core.agents.base import BaseAgent
        from core.agents.state import BaseAgentState
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="test_agent",
            agent_type="outreach",
            name="Test Agent",
            vertical_id="test_vert",
        )
        db = MagicMock()
        db.store_insight = MagicMock(return_value={"id": "1"})
        db.search_insights = MagicMock(return_value=[])
        db.client = MagicMock()
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.1] * 1536)

        # Create a concrete subclass for testing
        class TestAgent(BaseAgent):
            def build_graph(self):
                return MagicMock()

            def get_tools(self):
                return []

            def get_state_class(self):
                return BaseAgentState

        agent = TestAgent(
            config=config, db=db, embedder=embedder,
            anthropic_client=MagicMock(),
        )
        return agent

    def test_hive_property_type(self):
        agent = self._make_agent()
        from core.memory.hive_mind import HiveMind
        assert isinstance(agent.hive, HiveMind)

    def test_hive_lazy_init(self):
        agent = self._make_agent()
        assert agent._hive is None
        _ = agent.hive
        assert agent._hive is not None

    def test_hive_singleton(self):
        agent = self._make_agent()
        h1 = agent.hive
        h2 = agent.hive
        assert h1 is h2

    def test_consult_hive(self):
        agent = self._make_agent()
        results = agent.consult_hive("What pricing works?")
        assert isinstance(results, list)

    def test_publish_to_hive(self):
        agent = self._make_agent()
        result = agent.publish_to_hive(
            topic="email_performance",
            content="Short subjects work",
            confidence=0.85,
        )
        assert result is not None


# ══════════════════════════════════════════════════════════════════════
# Brain Dashboard Helpers Tests
# ══════════════════════════════════════════════════════════════════════


class TestInsightStats:
    """Tests for compute_insight_stats."""

    def test_empty(self):
        from dashboard.pages._brain_helpers import compute_insight_stats
        stats = compute_insight_stats([])
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_basic(self):
        from dashboard.pages._brain_helpers import compute_insight_stats
        insights = [
            {"insight_type": "email_performance", "source_agent_id": "outreach",
             "confidence_score": 0.9, "metadata": {"topic": "email"}},
            {"insight_type": "deal_patterns", "source_agent_id": "proposal",
             "confidence_score": 0.7, "metadata": {"topic": "deals"}},
        ]
        stats = compute_insight_stats(insights)
        assert stats["total"] == 2
        assert stats["by_type"]["email_performance"] == 1
        assert stats["by_agent"]["outreach"] == 1
        assert stats["topics_covered"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.8)

    def test_high_confidence_count(self):
        from dashboard.pages._brain_helpers import compute_insight_stats
        insights = [
            {"confidence_score": 0.95, "insight_type": "t", "source_agent_id": "a"},
            {"confidence_score": 0.85, "insight_type": "t", "source_agent_id": "a"},
            {"confidence_score": 0.5, "insight_type": "t", "source_agent_id": "a"},
        ]
        stats = compute_insight_stats(insights)
        assert stats["high_confidence_count"] == 2

    def test_metadata_as_string(self):
        from dashboard.pages._brain_helpers import compute_insight_stats
        insights = [
            {"insight_type": "t", "source_agent_id": "a",
             "confidence_score": 0.8, "metadata": '{"topic": "email"}'},
        ]
        stats = compute_insight_stats(insights)
        assert stats["topics_covered"] == 1


class TestExperimentStats:
    """Tests for compute_experiment_stats."""

    def test_empty(self):
        from dashboard.pages._brain_helpers import compute_experiment_stats
        stats = compute_experiment_stats([])
        assert stats["total"] == 0
        assert stats["active"] == 0

    def test_basic(self):
        from dashboard.pages._brain_helpers import compute_experiment_stats
        experiments = [
            {"status": "active", "total_observations": 50},
            {"status": "concluded", "total_observations": 100},
            {"status": "paused", "total_observations": 20},
        ]
        stats = compute_experiment_stats(experiments)
        assert stats["total"] == 3
        assert stats["active"] == 1
        assert stats["concluded"] == 1
        assert stats["paused"] == 1
        assert stats["total_observations"] == 170


class TestScoreDistribution:
    """Tests for compute_score_distribution."""

    def test_empty(self):
        from dashboard.pages._brain_helpers import compute_score_distribution
        stats = compute_score_distribution([])
        assert stats["total"] == 0
        assert stats["avg"] == 0

    def test_distribution(self):
        from dashboard.pages._brain_helpers import compute_score_distribution
        scores = [90, 85, 70, 65, 50, 45, 30, 20]
        stats = compute_score_distribution(scores)
        assert stats["total"] == 8
        assert stats["hot"] == 2  # 90, 85
        assert stats["warm"] == 2  # 70, 65
        assert stats["lukewarm"] == 2  # 50, 45
        assert stats["cold"] == 2  # 30, 20

    def test_median_odd(self):
        from dashboard.pages._brain_helpers import compute_score_distribution
        stats = compute_score_distribution([10, 50, 90])
        assert stats["median"] == 50

    def test_median_even(self):
        from dashboard.pages._brain_helpers import compute_score_distribution
        stats = compute_score_distribution([10, 50, 60, 90])
        assert stats["median"] == 55


class TestKnowledgeFlow:
    """Tests for compute_knowledge_flow."""

    def test_empty(self):
        from dashboard.pages._brain_helpers import compute_knowledge_flow
        flow = compute_knowledge_flow([])
        assert flow["total_flows"] == 0

    def test_basic(self):
        from dashboard.pages._brain_helpers import compute_knowledge_flow
        log = [
            {"agent": "outreach", "direction": "publish"},
            {"agent": "outreach", "direction": "publish"},
            {"agent": "social", "direction": "consume"},
        ]
        flow = compute_knowledge_flow(log)
        assert flow["total_flows"] == 3
        assert flow["publishers"]["outreach"] == 2
        assert flow["consumers"]["social"] == 1
        assert flow["most_active_publisher"] == "outreach"
        assert flow["most_active_consumer"] == "social"


class TestBrainHealth:
    """Tests for compute_brain_health."""

    def test_all_zeros(self):
        from dashboard.pages._brain_helpers import compute_brain_health
        health = compute_brain_health(
            {"topics_covered": 0, "avg_confidence": 0},
            {"active": 0, "total_observations": 0},
            {"total": 0},
        )
        assert health["score"] == 0
        assert health["grade"] == "F"

    def test_good_health(self):
        from dashboard.pages._brain_helpers import compute_brain_health
        health = compute_brain_health(
            {"topics_covered": 8, "avg_confidence": 0.85},
            {"active": 3, "total_observations": 200},
            {"total": 50},
        )
        assert health["score"] > 60
        assert health["grade"] in ("A", "B", "C")

    def test_all_components_present(self):
        from dashboard.pages._brain_helpers import compute_brain_health
        health = compute_brain_health(
            {"topics_covered": 5, "avg_confidence": 0.7},
            {"active": 2, "total_observations": 100},
            {"total": 30},
        )
        assert "knowledge_score" in health
        assert "experiment_score" in health
        assert "prediction_score" in health
        assert 0 <= health["score"] <= 100


class TestBrainFormatters:
    """Tests for formatting helper functions."""

    def test_format_insight_type_known(self):
        from dashboard.pages._brain_helpers import format_insight_type
        text, color = format_insight_type("email_performance")
        assert text == "Email"
        assert color.startswith("#")

    def test_format_insight_type_unknown(self):
        from dashboard.pages._brain_helpers import format_insight_type
        text, color = format_insight_type("something_new")
        assert text == "Something New"
        assert color == "#8B8B8B"

    def test_format_confidence_levels(self):
        from dashboard.pages._brain_helpers import format_confidence
        assert format_confidence(0.95)[0] == "Very High"
        assert format_confidence(0.85)[0] == "High"
        assert format_confidence(0.75)[0] == "Good"
        assert format_confidence(0.55)[0] == "Moderate"
        assert format_confidence(0.3)[0] == "Low"

    def test_format_experiment_status(self):
        from dashboard.pages._brain_helpers import format_experiment_status
        assert format_experiment_status("active")[0] == "Active"
        assert format_experiment_status("concluded")[0] == "Concluded"
        assert format_experiment_status("paused")[0] == "Paused"
        text, _ = format_experiment_status("unknown")
        assert text == "Unknown"

    def test_format_score_tier(self):
        from dashboard.pages._brain_helpers import format_score_tier
        assert format_score_tier(90)[0] == "Hot"
        assert format_score_tier(70)[0] == "Warm"
        assert format_score_tier(50)[0] == "Lukewarm"
        assert format_score_tier(20)[0] == "Cold"


# ══════════════════════════════════════════════════════════════════════
# YAML Config Tests
# ══════════════════════════════════════════════════════════════════════


class TestYAMLConfigs:
    """Verify existing agent YAML configs are valid."""

    def test_cs_yaml_exists(self):
        path = Path("verticals/enclave_guard/agents/cs.yaml")
        assert path.exists()

    def test_cs_yaml_parses(self):
        import yaml
        path = Path("verticals/enclave_guard/agents/cs.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        assert config["agent_id"] == "cs_v1"
        assert config["agent_type"] == "cs"
        assert config["enabled"] is True
        assert "onboarding_checklist" in config["params"]
        assert "risk_thresholds" in config["params"]
        assert "checkin_schedule" in config["params"]

    def test_cs_risk_thresholds(self):
        import yaml
        path = Path("verticals/enclave_guard/agents/cs.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        thresholds = config["params"]["risk_thresholds"]
        assert thresholds["low"] < thresholds["moderate"]
        assert thresholds["moderate"] < thresholds["high"]
        assert thresholds["high"] < thresholds["critical"]


# ══════════════════════════════════════════════════════════════════════
# DB Migration Schema Tests
# ══════════════════════════════════════════════════════════════════════


class TestMigration010:
    """Verify 010_hive_mind.sql schema."""

    def _read_migration(self):
        path = Path("infrastructure/migrations/010_hive_mind.sql")
        return path.read_text()

    def test_migration_exists(self):
        assert Path("infrastructure/migrations/010_hive_mind.sql").exists()

    def test_experiments_table(self):
        sql = self._read_migration()
        assert "CREATE TABLE IF NOT EXISTS experiments" in sql
        assert "experiment_id" in sql
        assert "vertical_id" in sql
        assert "variants" in sql
        assert "JSONB" in sql

    def test_experiment_outcomes_table(self):
        sql = self._read_migration()
        assert "CREATE TABLE IF NOT EXISTS experiment_outcomes" in sql
        assert "variant" in sql
        assert "outcome" in sql

    def test_lead_scores_table(self):
        sql = self._read_migration()
        assert "CREATE TABLE IF NOT EXISTS lead_scores" in sql
        assert "score" in sql
        assert "tier" in sql
        assert "factors" in sql

    def test_boost_insight_rpc(self):
        sql = self._read_migration()
        assert "boost_insight_confidence" in sql
        assert "LEAST" in sql  # Caps at 1.0

    def test_decay_insight_rpc(self):
        sql = self._read_migration()
        assert "decay_insight_confidence" in sql
        assert "GREATEST" in sql  # Floors at 0.0

    def test_brain_stats_rpc(self):
        sql = self._read_migration()
        assert "get_brain_stats" in sql
        assert "total_insights" in sql
        assert "active_experiments" in sql

    def test_indexes(self):
        sql = self._read_migration()
        assert "idx_experiments_vertical" in sql
        assert "idx_experiments_status" in sql
        assert "idx_lead_scores_vertical" in sql
        assert "idx_lead_scores_tier" in sql

    def test_status_constraints(self):
        sql = self._read_migration()
        assert "'active'" in sql
        assert "'concluded'" in sql
        assert "'paused'" in sql

    def test_score_constraints(self):
        sql = self._read_migration()
        assert "score >= 0" in sql
        assert "score <= 100" in sql


# ══════════════════════════════════════════════════════════════════════
# Brain Dashboard Page Tests
# ══════════════════════════════════════════════════════════════════════


class TestBrainDashboard:
    """Verify Brain Dashboard page structure."""

    def test_page_exists(self):
        assert Path("dashboard/pages/6_Brain.py").exists()

    def test_page_imports(self):
        content = Path("dashboard/pages/6_Brain.py").read_text()
        assert "import streamlit as st" in content
        assert "from dashboard.auth import require_auth" in content
        assert "from dashboard.theme import" in content
        assert "from dashboard.pages._brain_helpers import" in content

    def test_page_tabs(self):
        content = Path("dashboard/pages/6_Brain.py").read_text()
        assert "Insight Feed" in content
        assert "Experiments" in content
        assert "Lead Scoring" in content
        assert "Knowledge Flow" in content

    def test_page_queries_tables(self):
        content = Path("dashboard/pages/6_Brain.py").read_text()
        assert "shared_insights" in content
        assert "experiments" in content
        assert "lead_scores" in content

    def test_brain_health_banner(self):
        content = Path("dashboard/pages/6_Brain.py").read_text()
        assert "Brain Health" in content
        assert "Knowledge Depth" in content
        assert "Experimentation" in content
        assert "Prediction Quality" in content
