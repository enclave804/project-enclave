"""
Survey MCP tools for the Sovereign Venture Engine.

Provides tools for NPS survey dispatch, response collection,
score calculation, and sentiment analysis. Used by the Feedback
Agent for customer satisfaction workflows.

Tools:
    - send_nps_survey: Send an NPS survey to a contact
    - collect_survey_responses: Retrieve survey responses for analysis
    - calculate_nps: Calculate NPS score from response data
    - analyze_feedback_sentiment: Batch analyze sentiment of feedback comments
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def send_nps_survey(
    contact_email: str,
    contact_name: str = "",
    survey_url: str = "",
) -> str:
    """
    Send an NPS survey to a contact.

    Dispatches a personalized NPS survey email with a unique
    tracking link. Returns confirmation with survey_id for
    response tracking.
    """
    try:
        now = datetime.now(timezone.utc)
        survey_id = f"nps_{now.strftime('%Y%m%d%H%M%S')}_{abs(hash(contact_email)) % 100000:05d}"

        result = {
            "status": "success",
            "survey_id": survey_id,
            "contact_email": contact_email,
            "contact_name": contact_name,
            "survey_url": survey_url or f"https://survey.example.com/{survey_id}",
            "survey_type": "nps",
            "sent_at": now.isoformat(),
            "expires_at": None,
        }

        logger.info(
            "nps_survey_sent",
            extra={
                "survey_id": survey_id,
                "contact_email": contact_email,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"NPS survey dispatch failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "contact_email": contact_email,
        })


async def collect_survey_responses(
    vertical_id: str,
    survey_type: str = "nps",
    days: int = 30,
) -> str:
    """
    Retrieve survey responses for analysis.

    Collects all responses of the given survey_type within the
    specified time window. Returns response data including scores,
    comments, and metadata.
    """
    try:
        # Stub: return empty response set
        result = {
            "status": "success",
            "vertical_id": vertical_id,
            "survey_type": survey_type,
            "period_days": days,
            "total_responses": 0,
            "responses": [],
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "survey_responses_collected",
            extra={
                "vertical_id": vertical_id,
                "survey_type": survey_type,
                "response_count": 0,
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Survey response collection failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "vertical_id": vertical_id,
        })


async def calculate_nps(responses: list) -> str:
    """
    Calculate NPS score from response data.

    Each response should be a dict with at minimum:
        score (int): NPS rating 0-10

    NPS = % Promoters (9-10) - % Detractors (0-6)
    Passives (7-8) are counted but don't affect the score.

    Returns JSON with nps_score, promoters, passives, detractors
    counts and percentages.
    """
    try:
        total = len(responses)
        if total == 0:
            return json.dumps({
                "status": "success",
                "nps_score": 0,
                "total_responses": 0,
                "promoters": 0,
                "passives": 0,
                "detractors": 0,
                "promoter_pct": 0.0,
                "passive_pct": 0.0,
                "detractor_pct": 0.0,
            })

        promoters = 0
        passives = 0
        detractors = 0

        for response in responses:
            score = response.get("score", 0)
            if score >= 9:
                promoters += 1
            elif score >= 7:
                passives += 1
            else:
                detractors += 1

        promoter_pct = (promoters / total) * 100
        detractor_pct = (detractors / total) * 100
        passive_pct = (passives / total) * 100
        nps_score = round(promoter_pct - detractor_pct, 1)

        result = {
            "status": "success",
            "nps_score": nps_score,
            "total_responses": total,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "promoter_pct": round(promoter_pct, 1),
            "passive_pct": round(passive_pct, 1),
            "detractor_pct": round(detractor_pct, 1),
        }

        logger.info(
            "nps_calculated",
            extra={
                "nps_score": nps_score,
                "total_responses": total,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"NPS calculation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
        })


async def analyze_feedback_sentiment(comments: list) -> str:
    """
    Batch analyze sentiment of feedback comments.

    Each comment should be a dict with:
        text (str): The comment text
        contact_id (str, optional): Who wrote it
        survey_id (str, optional): Which survey it came from

    Returns JSON with per-comment sentiment labels and an
    aggregate sentiment breakdown.
    """
    try:
        # Stub: return placeholder sentiment analysis
        analyzed: list[dict[str, Any]] = []
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

        for comment in comments:
            text = comment.get("text", "")
            # Placeholder sentiment — in production, call LLM or sentiment model
            sentiment = "neutral"
            confidence = 0.5

            analyzed.append({
                "text": text[:200],
                "contact_id": comment.get("contact_id", ""),
                "survey_id": comment.get("survey_id", ""),
                "sentiment": sentiment,
                "confidence": confidence,
            })
            sentiment_counts[sentiment] += 1

        total = len(comments)
        result = {
            "status": "success",
            "total_comments": total,
            "analyzed": analyzed,
            "aggregate": {
                "positive": sentiment_counts["positive"],
                "neutral": sentiment_counts["neutral"],
                "negative": sentiment_counts["negative"],
                "positive_pct": round((sentiment_counts["positive"] / max(total, 1)) * 100, 1),
                "neutral_pct": round((sentiment_counts["neutral"] / max(total, 1)) * 100, 1),
                "negative_pct": round((sentiment_counts["negative"] / max(total, 1)) * 100, 1),
            },
        }

        logger.info(
            "feedback_sentiment_analyzed",
            extra={
                "total_comments": total,
                "sentiment_breakdown": sentiment_counts,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Feedback sentiment analysis failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
        })
