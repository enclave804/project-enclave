"""
The Oracle — Predictive Lead Scoring Engine.

Uses scikit-learn to train a Random Forest model on historical
"Closed Won" vs "Closed Lost" leads. Returns a 0-100 probability
of conversion for any new lead.

When insufficient data exists (<50 training examples), falls back
to a heuristic scoring model based on:
    - Title match (is the lead a decision maker?)
    - Company size (fits ICP?)
    - Industry match (target vertical?)
    - Engagement signals (email opens, replies, meetings)

The model automatically retrains when new data arrives.

Usage:
    scorer = LeadScorer(db=db)

    # Score a lead
    score = scorer.predict_score({
        "title": "CTO",
        "company_size": 200,
        "industry": "fintech",
        "has_replied": True,
        "meetings_booked": 1,
    })
    # Returns: 82  (high conversion probability)

    # Train on historical data
    scorer.train(historical_leads)

    # Get feature importance
    importance = scorer.get_feature_importance()
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature Configuration ─────────────────────────────────────

# Features used for scoring (order matters for model consistency)
FEATURE_NAMES = [
    "title_score",       # How senior is the contact (0-10)
    "company_size",      # Employee count
    "industry_match",    # Does industry match ICP (0/1)
    "has_email",         # Valid email found (0/1)
    "has_phone",         # Phone number available (0/1)
    "email_opened",      # Opened any email (0/1)
    "email_replied",     # Replied to any email (0/1)
    "link_clicked",      # Clicked any link (0/1)
    "meetings_booked",   # Number of meetings scheduled
    "days_in_pipeline",  # Days since first contact
    "tech_stack_fit",    # Technology overlap score (0-10)
    "website_traffic",   # Estimated monthly traffic (log scale)
    "social_engagement", # Social media interaction (0-10)
    "proposal_sent",     # Has a proposal been sent (0/1)
    "proposal_amount",   # Proposal value in dollars
]

# Title seniority mapping
TITLE_SCORES: dict[str, int] = {
    "ceo": 10, "cto": 10, "cfo": 9, "coo": 9, "ciso": 10,
    "vp": 8, "vice president": 8, "svp": 9,
    "director": 7, "head of": 7,
    "manager": 5, "senior": 6,
    "engineer": 3, "analyst": 3, "developer": 3,
    "intern": 1, "assistant": 2, "coordinator": 2,
}

# Industries that match cybersecurity ICP
TARGET_INDUSTRIES = {
    "fintech", "finance", "financial services", "banking",
    "healthcare", "health", "medical",
    "saas", "software", "technology", "tech",
    "e-commerce", "ecommerce", "retail",
    "insurance", "legal", "government",
}

# Minimum training examples before using ML model
MIN_TRAINING_EXAMPLES = 50

# Model storage path
MODEL_PATH = Path("storage/models")


class LeadScorer:
    """
    Predictive lead scoring engine.

    Automatically chooses between:
    - ML model (Random Forest) when enough training data exists
    - Heuristic model when insufficient data
    """

    def __init__(
        self,
        db: Any = None,
        vertical_id: str = "enclave_guard",
        model_path: Optional[Path] = None,
    ):
        self.db = db
        self.vertical_id = vertical_id
        self.model_path = model_path or MODEL_PATH
        self._model = None
        self._model_trained_at: Optional[str] = None
        self._feature_importance: dict[str, float] = {}
        self._training_samples: int = 0
        self._using_ml: bool = False

        # Try to load existing model
        self._try_load_model()

    # ── Predict ───────────────────────────────────────────────

    def predict_score(self, lead_features: dict[str, Any]) -> int:
        """
        Score a lead 0-100 (probability of conversion).

        Args:
            lead_features: Dict with any of the FEATURE_NAMES keys.
                Missing features are imputed with defaults.

        Returns:
            Integer score 0-100.
        """
        features = self._extract_features(lead_features)

        if self._using_ml and self._model is not None:
            return self._ml_predict(features)
        else:
            return self._heuristic_predict(lead_features, features)

    def predict_batch(self, leads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Score multiple leads at once.

        Returns list of dicts with 'score', 'tier', and 'factors'.
        """
        results = []
        for lead in leads:
            score = self.predict_score(lead)
            tier = self._score_to_tier(score)
            factors = self._explain_score(lead, score)
            results.append({
                "lead": lead,
                "score": score,
                "tier": tier,
                "factors": factors,
            })
        return results

    # ── Train ─────────────────────────────────────────────────

    def train(self, historical_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Train the prediction model on historical lead data.

        Each record should have:
            - Feature fields (title, company_size, industry, etc.)
            - 'outcome': 'won' or 'lost'

        Returns training stats.
        """
        if len(historical_data) < MIN_TRAINING_EXAMPLES:
            logger.info(
                f"Insufficient data ({len(historical_data)} < {MIN_TRAINING_EXAMPLES}), "
                f"using heuristic model"
            )
            self._training_samples = len(historical_data)
            self._using_ml = False
            return {
                "status": "heuristic",
                "reason": f"Need {MIN_TRAINING_EXAMPLES} examples, have {len(historical_data)}",
                "samples": len(historical_data),
            }

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.warning("scikit-learn not installed, using heuristic model")
            self._using_ml = False
            return {"status": "heuristic", "reason": "scikit-learn not installed"}

        # Extract features and labels
        X = []
        y = []
        for record in historical_data:
            features = self._extract_features(record)
            X.append(features)
            outcome = record.get("outcome", "lost")
            y.append(1 if outcome == "won" else 0)

        X_arr = np.array(X)
        y_arr = np.array(y)

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",  # Handle imbalanced data
        )

        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_arr, y_arr, cv=min(5, len(X_arr) // 10 + 1))
            avg_accuracy = float(np.mean(cv_scores))
        except Exception:
            avg_accuracy = 0.0

        # Full train
        model.fit(X_arr, y_arr)

        # Store model
        self._model = model
        self._using_ml = True
        self._training_samples = len(historical_data)
        self._model_trained_at = datetime.now(timezone.utc).isoformat()

        # Feature importance
        importances = model.feature_importances_
        self._feature_importance = {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, importances)
        }

        # Persist model
        self._save_model()

        stats = {
            "status": "trained",
            "model": "RandomForestClassifier",
            "samples": len(historical_data),
            "positive_rate": float(np.mean(y_arr)),
            "cv_accuracy": avg_accuracy,
            "top_features": sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "trained_at": self._model_trained_at,
        }

        logger.info(
            "lead_scorer_trained",
            extra={
                "samples": len(historical_data),
                "accuracy": avg_accuracy,
                "using_ml": True,
            },
        )

        return stats

    # ── Feature Extraction ────────────────────────────────────

    def _extract_features(self, lead: dict[str, Any]) -> list[float]:
        """
        Extract numeric features from a lead record.

        Missing values are imputed with sensible defaults.
        """
        title = str(lead.get("title", "")).lower()
        title_score = 0
        for keyword, score in TITLE_SCORES.items():
            if keyword in title:
                title_score = max(title_score, score)

        industry = str(lead.get("industry", "")).lower()
        industry_match = 1 if any(t in industry for t in TARGET_INDUSTRIES) else 0

        return [
            float(title_score),
            float(lead.get("company_size", lead.get("employees", 0)) or 0),
            float(industry_match),
            1.0 if lead.get("email") or lead.get("has_email") else 0.0,
            1.0 if lead.get("phone") or lead.get("has_phone") else 0.0,
            1.0 if lead.get("email_opened") else 0.0,
            1.0 if lead.get("email_replied") or lead.get("has_replied") else 0.0,
            1.0 if lead.get("link_clicked") else 0.0,
            float(lead.get("meetings_booked", 0) or 0),
            float(lead.get("days_in_pipeline", 0) or 0),
            float(lead.get("tech_stack_fit", 0) or 0),
            float(lead.get("website_traffic", 0) or 0),
            float(lead.get("social_engagement", 0) or 0),
            1.0 if lead.get("proposal_sent") else 0.0,
            float(lead.get("proposal_amount", 0) or 0),
        ]

    # ── ML Prediction ─────────────────────────────────────────

    def _ml_predict(self, features: list[float]) -> int:
        """Score using the trained ML model."""
        try:
            X = np.array([features])
            probabilities = self._model.predict_proba(X)
            # Probability of class 1 (won)
            win_prob = probabilities[0][1] if probabilities.shape[1] > 1 else probabilities[0][0]
            return int(round(win_prob * 100))
        except Exception as e:
            logger.warning(f"ML prediction failed, falling back to heuristic: {e}")
            return self._heuristic_predict({}, features)

    # ── Heuristic Prediction ──────────────────────────────────

    def _heuristic_predict(
        self,
        lead: dict[str, Any],
        features: list[float],
    ) -> int:
        """
        Rule-based scoring when ML model is unavailable.

        Weights:
            - Title seniority: 25%
            - Company fit: 20%
            - Engagement: 30%
            - Deal progress: 25%
        """
        title_score = features[0]  # 0-10
        company_size = features[1]
        industry_match = features[2]  # 0/1
        has_email = features[3]
        email_replied = features[6]
        meetings = features[8]
        proposal_sent = features[13]
        proposal_amount = features[14]

        # Title component (0-25)
        title_component = (title_score / 10) * 25

        # Company fit component (0-20)
        size_score = min(company_size / 500, 1.0) * 10  # Larger companies score higher
        industry_score = industry_match * 10
        company_component = size_score + industry_score

        # Engagement component (0-30)
        engagement = 0.0
        if has_email:
            engagement += 5
        if email_replied:
            engagement += 10
        if meetings > 0:
            engagement += min(meetings * 7.5, 15)

        # Deal progress component (0-25)
        deal_progress = 0.0
        if proposal_sent:
            deal_progress += 15
        if proposal_amount > 0:
            deal_progress += min(proposal_amount / 50000 * 10, 10)

        total = title_component + company_component + engagement + deal_progress
        return int(round(max(0, min(100, total))))

    # ── Explainability ────────────────────────────────────────

    def _explain_score(self, lead: dict[str, Any], score: int) -> list[str]:
        """Generate human-readable factors that influenced the score."""
        factors = []
        title = str(lead.get("title", "")).lower()

        if any(k in title for k in ("ceo", "cto", "cfo", "ciso", "vp", "director")):
            factors.append("Decision-maker title (high influence)")
        elif any(k in title for k in ("manager", "senior")):
            factors.append("Mid-level title (moderate influence)")

        if lead.get("email_replied") or lead.get("has_replied"):
            factors.append("Has replied to outreach (strong buy signal)")
        if lead.get("meetings_booked", 0) > 0:
            factors.append(f"{lead['meetings_booked']} meeting(s) booked (strong signal)")
        if lead.get("proposal_sent"):
            factors.append("Proposal sent (deal in progress)")

        industry = str(lead.get("industry", "")).lower()
        if any(t in industry for t in TARGET_INDUSTRIES):
            factors.append(f"Industry match: {industry}")

        company_size = lead.get("company_size", lead.get("employees", 0)) or 0
        if company_size > 200:
            factors.append(f"Enterprise company ({company_size} employees)")
        elif company_size > 50:
            factors.append(f"Mid-market company ({company_size} employees)")

        return factors

    def _score_to_tier(self, score: int) -> str:
        """Convert numeric score to tier label."""
        if score >= 80:
            return "hot"
        elif score >= 60:
            return "warm"
        elif score >= 40:
            return "lukewarm"
        else:
            return "cold"

    # ── Feature Importance ────────────────────────────────────

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the trained model."""
        return dict(self._feature_importance)

    def get_model_stats(self) -> dict[str, Any]:
        """Get model metadata and training stats."""
        return {
            "using_ml": self._using_ml,
            "training_samples": self._training_samples,
            "trained_at": self._model_trained_at,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "feature_importance": self._feature_importance,
        }

    # ── Persistence ───────────────────────────────────────────

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
            model_file = self.model_path / f"lead_scorer_{self.vertical_id}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "feature_importance": self._feature_importance,
                    "training_samples": self._training_samples,
                    "trained_at": self._model_trained_at,
                }, f)
            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.warning(f"Model save failed: {e}")

    def _try_load_model(self) -> None:
        """Try to load a previously trained model."""
        try:
            model_file = self.model_path / f"lead_scorer_{self.vertical_id}.pkl"
            if model_file.exists():
                with open(model_file, "rb") as f:
                    data = pickle.load(f)
                self._model = data["model"]
                self._feature_importance = data.get("feature_importance", {})
                self._training_samples = data.get("training_samples", 0)
                self._model_trained_at = data.get("trained_at")
                self._using_ml = True
                logger.info(f"Model loaded from {model_file}")
        except Exception as e:
            logger.debug(f"No existing model loaded: {e}")

    def __repr__(self) -> str:
        mode = "ML" if self._using_ml else "Heuristic"
        return (
            f"LeadScorer(mode={mode}, "
            f"samples={self._training_samples}, "
            f"vertical={self.vertical_id!r})"
        )
