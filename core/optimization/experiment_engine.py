"""
The Lab — A/B Testing & Experiment Engine.

Provides a formal framework for agents to run experiments:
    - Email subject lines (Emoji vs No-Emoji)
    - Pricing tiers (Flat fee vs Hourly)
    - Ad creatives (Image A vs Image B)
    - Outreach timing (Morning vs Afternoon)

Uses Bayesian probability to declare winners with statistical confidence.
No external dependencies — pure Python + numpy.

Experiment Lifecycle:
    1. start_experiment() — Register variants
    2. track_outcome() — Log results per variant
    3. get_results() — Check stats and confidence
    4. get_winner() — Declare winner when confident

Usage:
    engine = ExperimentEngine(db=db)

    # Start an experiment
    exp_id = engine.start_experiment(
        name="Subject Line Emoji Test",
        variants=["With Emoji", "Without Emoji"],
        agent_id="outreach",
        metric="reply_rate",
    )

    # Track outcomes
    engine.track_outcome(exp_id, "With Emoji", "reply")
    engine.track_outcome(exp_id, "Without Emoji", "ignore")
    engine.track_outcome(exp_id, "With Emoji", "reply")

    # Check results
    results = engine.get_results(exp_id)
    winner = engine.get_winner(exp_id)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Outcome Types ─────────────────────────────────────────────

POSITIVE_OUTCOMES = {"reply", "click", "open", "convert", "book", "accept", "win", "engage"}
NEGATIVE_OUTCOMES = {"ignore", "bounce", "unsubscribe", "reject", "lose", "skip"}
ALL_OUTCOMES = POSITIVE_OUTCOMES | NEGATIVE_OUTCOMES

# Minimum observations per variant before declaring a winner
MIN_OBSERVATIONS = 30
# Minimum probability advantage to declare a winner (95%)
WIN_THRESHOLD = 0.95


# ── Experiment Data ───────────────────────────────────────────

class Experiment:
    """In-memory representation of an experiment."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        variants: list[str],
        agent_id: str = "",
        metric: str = "conversion",
        metadata: Optional[dict] = None,
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.variants = variants
        self.agent_id = agent_id
        self.metric = metric
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "active"  # active, concluded, paused

        # Per-variant outcome tracking
        self.outcomes: dict[str, list[str]] = {v: [] for v in variants}

    @property
    def total_observations(self) -> int:
        return sum(len(o) for o in self.outcomes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "variants": self.variants,
            "agent_id": self.agent_id,
            "metric": self.metric,
            "status": self.status,
            "total_observations": self.total_observations,
            "outcomes": {k: len(v) for k, v in self.outcomes.items()},
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# ── The Experiment Engine ─────────────────────────────────────

class ExperimentEngine:
    """
    A/B Testing framework with Bayesian analysis.

    Stores experiments in-memory with optional DB persistence.
    Uses Beta-Binomial conjugate model for fast winner detection.
    """

    def __init__(
        self,
        db: Any = None,
        vertical_id: str = "enclave_guard",
    ):
        self.db = db
        self.vertical_id = vertical_id

        # In-memory experiment store
        self._experiments: dict[str, Experiment] = {}

    # ── Create & Manage ───────────────────────────────────────

    def start_experiment(
        self,
        name: str,
        variants: list[str],
        agent_id: str = "",
        metric: str = "conversion",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Register a new A/B test experiment.

        Args:
            name: Human-readable experiment name.
            variants: List of variant names (at least 2).
            agent_id: Agent that owns this experiment.
            metric: What we're measuring (conversion, reply_rate, etc.).
            metadata: Additional experiment context.

        Returns:
            Experiment ID string.
        """
        if len(variants) < 2:
            raise ValueError("Experiments require at least 2 variants")
        if len(variants) != len(set(variants)):
            raise ValueError("Variant names must be unique")

        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            agent_id=agent_id,
            metric=metric,
            metadata=metadata,
        )

        self._experiments[experiment_id] = experiment

        # Persist to DB if available
        self._persist_experiment(experiment)

        logger.info(
            "experiment_started",
            extra={
                "experiment_id": experiment_id,
                "name": name,
                "variants": variants,
                "agent_id": agent_id,
            },
        )

        return experiment_id

    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an active experiment."""
        exp = self._experiments.get(experiment_id)
        if exp and exp.status == "active":
            exp.status = "paused"
            return True
        return False

    def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        exp = self._experiments.get(experiment_id)
        if exp and exp.status == "paused":
            exp.status = "active"
            return True
        return False

    def conclude_experiment(self, experiment_id: str) -> Optional[dict]:
        """
        Conclude an experiment and return final results.

        Marks the experiment as concluded and returns the winner analysis.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        exp.status = "concluded"
        return self.get_results(experiment_id)

    # ── Track Outcomes ────────────────────────────────────────

    def track_outcome(
        self,
        experiment_id: str,
        variant: str,
        outcome: str,
    ) -> bool:
        """
        Log an outcome for a specific variant.

        Args:
            experiment_id: Which experiment.
            variant: Which variant was shown.
            outcome: Result — "reply", "click", "ignore", etc.

        Returns:
            True if tracked successfully.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            logger.warning(f"Experiment {experiment_id} not found")
            return False

        if exp.status != "active":
            logger.warning(f"Experiment {experiment_id} is {exp.status}, not tracking")
            return False

        if variant not in exp.outcomes:
            logger.warning(f"Variant {variant!r} not in experiment {experiment_id}")
            return False

        exp.outcomes[variant].append(outcome)

        # Persist to DB
        self._persist_outcome(experiment_id, variant, outcome)

        return True

    def track_batch(
        self,
        experiment_id: str,
        results: list[dict[str, str]],
    ) -> int:
        """
        Track multiple outcomes at once.

        Each result dict should have 'variant' and 'outcome' keys.
        Returns count of successfully tracked.
        """
        tracked = 0
        for result in results:
            if self.track_outcome(
                experiment_id,
                result.get("variant", ""),
                result.get("outcome", ""),
            ):
                tracked += 1
        return tracked

    # ── Analysis ──────────────────────────────────────────────

    def get_results(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """
        Get current results with Bayesian analysis.

        Returns variant stats, win probabilities, and whether
        a winner can be confidently declared.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        variant_stats = {}
        for variant in exp.variants:
            outcomes = exp.outcomes[variant]
            total = len(outcomes)
            positive = sum(1 for o in outcomes if o in POSITIVE_OUTCOMES)
            negative = total - positive
            rate = positive / total if total > 0 else 0.0

            variant_stats[variant] = {
                "total": total,
                "positive": positive,
                "negative": negative,
                "rate": round(rate, 4),
            }

        # Bayesian win probabilities
        win_probs = self._bayesian_win_probability(exp)

        # Determine winner
        ready_to_call = all(
            stats["total"] >= MIN_OBSERVATIONS
            for stats in variant_stats.values()
        )
        best_variant = max(win_probs, key=win_probs.get) if win_probs else None
        best_prob = win_probs.get(best_variant, 0) if best_variant else 0
        has_winner = ready_to_call and best_prob >= WIN_THRESHOLD

        return {
            "experiment_id": experiment_id,
            "name": exp.name,
            "status": exp.status,
            "metric": exp.metric,
            "total_observations": exp.total_observations,
            "variant_stats": variant_stats,
            "win_probabilities": {k: round(v, 4) for k, v in win_probs.items()},
            "has_winner": has_winner,
            "winner": best_variant if has_winner else None,
            "winner_confidence": round(best_prob, 4) if best_variant else 0,
            "min_observations_met": ready_to_call,
        }

    def get_winner(self, experiment_id: str) -> Optional[str]:
        """
        Get the winning variant, if one can be declared.

        Returns the variant name if confident, None otherwise.
        """
        results = self.get_results(experiment_id)
        if results and results.get("has_winner"):
            return results["winner"]
        return None

    # ── Bayesian Analysis ─────────────────────────────────────

    def _bayesian_win_probability(
        self,
        experiment: Experiment,
        n_simulations: int = 10000,
    ) -> dict[str, float]:
        """
        Calculate win probability for each variant using Monte Carlo
        simulation of Beta distributions.

        Uses Beta(alpha=successes+1, beta=failures+1) prior.
        Simulates n_simulations draws and counts how often each
        variant has the highest conversion rate.
        """
        if experiment.total_observations == 0:
            # Equal probability when no data
            n = len(experiment.variants)
            return {v: 1.0 / n for v in experiment.variants}

        # Build Beta distribution parameters
        alphas = {}
        betas = {}
        for variant in experiment.variants:
            outcomes = experiment.outcomes[variant]
            successes = sum(1 for o in outcomes if o in POSITIVE_OUTCOMES)
            failures = len(outcomes) - successes
            alphas[variant] = successes + 1  # +1 prior (uniform)
            betas[variant] = failures + 1

        # Monte Carlo simulation
        rng = np.random.default_rng(42)
        win_counts: dict[str, int] = {v: 0 for v in experiment.variants}

        samples = {
            v: rng.beta(alphas[v], betas[v], size=n_simulations)
            for v in experiment.variants
        }

        # Stack and find argmax for each simulation
        variant_list = list(experiment.variants)
        stacked = np.stack([samples[v] for v in variant_list])
        winners = np.argmax(stacked, axis=0)

        for idx in range(len(variant_list)):
            win_counts[variant_list[idx]] = int(np.sum(winners == idx))

        # Convert to probabilities
        total = sum(win_counts.values())
        return {
            v: count / total if total > 0 else 0
            for v, count in win_counts.items()
        }

    # ── Listing ───────────────────────────────────────────────

    def list_experiments(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all experiments, optionally filtered."""
        results = []
        for exp in self._experiments.values():
            if agent_id and exp.agent_id != agent_id:
                continue
            if status and exp.status != status:
                continue
            results.append(exp.to_dict())
        return results

    def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get all currently active experiments."""
        return self.list_experiments(status="active")

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """Get a single experiment's summary."""
        exp = self._experiments.get(experiment_id)
        return exp.to_dict() if exp else None

    # ── Variant Assignment ────────────────────────────────────

    def assign_variant(self, experiment_id: str, entity_id: str = "") -> Optional[str]:
        """
        Assign a variant for a given entity (lead, email, etc.).

        Uses simple round-robin for balanced assignment.
        Falls back to random if no entity_id.
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != "active":
            return None

        if entity_id:
            # Deterministic: hash entity_id to variant index
            idx = hash(entity_id) % len(exp.variants)
        else:
            # Round-robin based on total observations
            idx = exp.total_observations % len(exp.variants)

        return exp.variants[idx]

    # ── Persistence (best-effort) ─────────────────────────────

    def _persist_experiment(self, experiment: Experiment) -> None:
        """Save experiment to DB if available."""
        if not self.db:
            return
        try:
            self.db.client.table("experiments").insert({
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "variants": experiment.variants,
                "agent_id": experiment.agent_id,
                "metric": experiment.metric,
                "status": experiment.status,
                "vertical_id": self.vertical_id,
                "metadata": experiment.metadata,
            }).execute()
        except Exception as e:
            logger.debug(f"Experiment persist failed (non-critical): {e}")

    def _persist_outcome(
        self,
        experiment_id: str,
        variant: str,
        outcome: str,
    ) -> None:
        """Save outcome to DB if available."""
        if not self.db:
            return
        try:
            self.db.client.table("experiment_outcomes").insert({
                "experiment_id": experiment_id,
                "variant": variant,
                "outcome": outcome,
                "vertical_id": self.vertical_id,
            }).execute()
        except Exception as e:
            logger.debug(f"Outcome persist failed (non-critical): {e}")

    def __repr__(self) -> str:
        active = sum(1 for e in self._experiments.values() if e.status == "active")
        return (
            f"ExperimentEngine(active={active}, "
            f"total={len(self._experiments)}, "
            f"vertical={self.vertical_id!r})"
        )
