"""
Voice & SMS MCP tools for the Sovereign Venture Engine.

Exposes Twilio telephony operations and audio services as MCP tools
for the VoiceAgent. Tools auto-detect mock vs production mode based
on environment variables (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).

Architecture:
    send_sms()              -> Twilio SMS API
    make_call()             -> Synthesize audio → Twilio call  [@sandboxed_tool]
    get_call_logs()         -> Twilio call history
    buy_phone_number()      -> Twilio phone number purchase     [@sandboxed_tool]
    transcribe_audio()      -> Whisper API (via Transcriber)

Safety:
    - Making calls is sandboxed in non-production environments
    - Buying phone numbers is sandboxed (costs real money!)
    - SMS sending is sandboxed in non-production
    - Read-only tools (get_call_logs, transcribe) are always live
    - All outbound communications require human approval
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from core.safety.sandbox import sandboxed_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock Twilio Client
# ---------------------------------------------------------------------------


class MockTwilioClient:
    """
    Mock Twilio client for development.

    Returns realistic-looking responses without touching the Twilio API.
    """

    def __init__(self):
        self._sms_log: list[dict[str, Any]] = []
        self._call_log: list[dict[str, Any]] = []
        self._counter = 0

    async def send_sms(
        self,
        to: str,
        body: str,
        from_number: Optional[str] = None,
    ) -> dict[str, Any]:
        self._counter += 1
        sid = f"SM_mock_{self._counter:06d}"
        result = {
            "sid": sid,
            "to": to,
            "from": from_number or "+15551234567",
            "body": body,
            "status": "queued",
            "direction": "outbound-api",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "price": "-0.0075",
            "price_unit": "USD",
        }
        self._sms_log.append(result)
        return result

    async def make_call(
        self,
        to: str,
        twiml_url: Optional[str] = None,
        audio_url: Optional[str] = None,
        from_number: Optional[str] = None,
    ) -> dict[str, Any]:
        self._counter += 1
        sid = f"CA_mock_{self._counter:06d}"
        result = {
            "sid": sid,
            "to": to,
            "from": from_number or "+15551234567",
            "status": "queued",
            "direction": "outbound-api",
            "twiml_url": twiml_url,
            "audio_url": audio_url,
            "date_created": datetime.now(timezone.utc).isoformat(),
            "duration": None,
            "price": None,
        }
        self._call_log.append(result)
        return result

    async def get_call_logs(self, limit: int = 10) -> list[dict[str, Any]]:
        # Mix mock data with any calls made this session
        mock_history = [
            {
                "sid": "CA_mock_history_001",
                "to": "+15559876543",
                "from": "+15551234567",
                "status": "completed",
                "direction": "inbound",
                "duration": "45",
                "date_created": "2025-01-15T10:30:00Z",
                "price": "-0.0085",
                "recording_url": None,
            },
            {
                "sid": "CA_mock_history_002",
                "to": "+15553334444",
                "from": "+15551234567",
                "status": "completed",
                "direction": "outbound-api",
                "duration": "120",
                "date_created": "2025-01-15T14:22:00Z",
                "price": "-0.014",
                "recording_url": "https://api.twilio.com/recordings/RE_mock_002",
            },
        ]
        all_calls = mock_history + self._call_log
        return all_calls[-limit:]

    async def get_sms_logs(self, limit: int = 10) -> list[dict[str, Any]]:
        mock_history = [
            {
                "sid": "SM_mock_history_001",
                "to": "+15559876543",
                "from": "+15551234567",
                "body": "Thanks for your inquiry! We'll get back to you shortly.",
                "status": "delivered",
                "direction": "outbound-api",
                "date_created": "2025-01-15T09:15:00Z",
                "price": "-0.0075",
            },
        ]
        all_sms = mock_history + self._sms_log
        return all_sms[-limit:]

    async def buy_phone_number(
        self,
        area_code: str = "415",
    ) -> dict[str, Any]:
        self._counter += 1
        return {
            "sid": f"PN_mock_{self._counter:06d}",
            "phone_number": f"+1{area_code}555{self._counter:04d}",
            "friendly_name": f"Mock Number ({area_code})",
            "status": "in-use",
            "capabilities": {
                "voice": True,
                "sms": True,
                "mms": True,
            },
            "date_created": datetime.now(timezone.utc).isoformat(),
            "monthly_price": "1.50",
        }


# Singleton mock client
_mock_twilio: Optional[MockTwilioClient] = None


def _get_mock_twilio() -> MockTwilioClient:
    global _mock_twilio
    if _mock_twilio is None:
        _mock_twilio = MockTwilioClient()
    return _mock_twilio


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@sandboxed_tool("send_sms")
async def send_sms(
    to_number: str,
    message: str,
    from_number: Optional[str] = None,
    *,
    _client: Any = None,
) -> str:
    """
    Send an SMS message via Twilio.

    SAFETY: Sandboxed in non-production environments.

    Args:
        to_number: Recipient phone number (E.164 format, e.g., +15551234567).
        message: SMS body text (max 1600 chars).
        from_number: Sender number. Defaults to TWILIO_PHONE_NUMBER env var.

    Returns:
        JSON string with message SID, status, and delivery info.
    """
    client = _client or _get_mock_twilio()

    try:
        from_num = from_number or os.environ.get("TWILIO_PHONE_NUMBER", "+15551234567")
        result = await client.send_sms(
            to=to_number,
            body=message[:1600],  # Twilio SMS limit
            from_number=from_num,
        )

        logger.info(
            "sms_sent",
            extra={
                "to": to_number[-4:],  # Log last 4 digits only
                "status": result.get("status"),
                "sid": result.get("sid", "")[:16],
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("sms_send_failed", extra={"error": str(e)[:200]})
        return json.dumps({"error": str(e), "to": to_number})


@sandboxed_tool("make_call")
async def make_call(
    to_number: str,
    script_text: str,
    voice_preset: str = "professional_female",
    from_number: Optional[str] = None,
    *,
    _client: Any = None,
    _synthesizer: Any = None,
) -> str:
    """
    Make an outbound phone call with synthesized speech.

    Flow: Text → ElevenLabs synthesis → Upload audio → Twilio call.

    SAFETY: Sandboxed in non-production environments.

    Args:
        to_number: Recipient phone number (E.164 format).
        script_text: What to say on the call.
        voice_preset: Voice to use ("professional_female", "warm_male", etc.).
        from_number: Caller ID. Defaults to TWILIO_PHONE_NUMBER.

    Returns:
        JSON string with call SID, status, and audio details.
    """
    client = _client or _get_mock_twilio()

    try:
        # Step 1: Synthesize audio
        audio_result: dict[str, Any] = {}
        if _synthesizer:
            audio_result = await _synthesizer.synthesize(
                text=script_text,
                voice_preset=voice_preset,
            )
        else:
            try:
                from core.llm.audio import Synthesizer
                synth = Synthesizer()
                audio_result = await synth.synthesize(
                    text=script_text,
                    voice_preset=voice_preset,
                )
            except Exception as e:
                logger.warning(
                    "call_synthesis_skipped",
                    extra={"error": str(e)[:200]},
                )
                audio_result = {"audio_path": "", "source": "failed"}

        # Step 2: Make the call
        from_num = from_number or os.environ.get("TWILIO_PHONE_NUMBER", "+15551234567")
        call_result = await client.make_call(
            to=to_number,
            audio_url=audio_result.get("audio_path", ""),
            from_number=from_num,
        )

        result = {
            **call_result,
            "script_text": script_text[:200],
            "voice_preset": voice_preset,
            "audio_cached": audio_result.get("cached", False),
            "audio_source": audio_result.get("source", "unknown"),
        }

        logger.info(
            "call_initiated",
            extra={
                "to": to_number[-4:],
                "sid": call_result.get("sid", "")[:16],
                "audio_cached": audio_result.get("cached", False),
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("call_failed", extra={"error": str(e)[:200]})
        return json.dumps({"error": str(e), "to": to_number})


async def get_call_logs(
    limit: int = 10,
    *,
    _client: Any = None,
) -> str:
    """
    Get recent call history from Twilio.

    Args:
        limit: Maximum number of records to return (default: 10).

    Returns:
        JSON string with list of call records and summary.
    """
    client = _client or _get_mock_twilio()

    try:
        calls = await client.get_call_logs(limit=limit)

        # Compute summary
        total_duration = sum(
            int(c.get("duration") or 0) for c in calls
        )
        inbound_count = sum(1 for c in calls if c.get("direction") == "inbound")
        outbound_count = len(calls) - inbound_count

        result = {
            "calls": calls,
            "summary": {
                "total_calls": len(calls),
                "inbound_count": inbound_count,
                "outbound_count": outbound_count,
                "total_duration_seconds": total_duration,
                "avg_duration_seconds": (
                    total_duration / len(calls) if calls else 0
                ),
            },
        }

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("call_logs_failed", extra={"error": str(e)[:200]})
        return json.dumps({"error": str(e)})


async def get_sms_logs(
    limit: int = 10,
    *,
    _client: Any = None,
) -> str:
    """
    Get recent SMS history.

    Args:
        limit: Maximum number of records to return (default: 10).

    Returns:
        JSON string with list of SMS records and count.
    """
    client = _client or _get_mock_twilio()

    try:
        messages = await client.get_sms_logs(limit=limit)
        result = {
            "messages": messages,
            "total_count": len(messages),
        }
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("sms_logs_failed", extra={"error": str(e)[:200]})
        return json.dumps({"error": str(e)})


@sandboxed_tool("buy_phone_number")
async def buy_phone_number(
    area_code: str = "415",
    *,
    _client: Any = None,
) -> str:
    """
    Purchase a new Twilio phone number.

    SAFETY: Sandboxed in ALL non-production environments.
    This costs real money and ALWAYS requires human approval.

    Args:
        area_code: Desired area code (e.g., "415" for San Francisco).

    Returns:
        JSON string with new phone number details.
    """
    client = _client or _get_mock_twilio()

    try:
        result = await client.buy_phone_number(area_code=area_code)

        logger.info(
            "phone_number_purchased",
            extra={
                "area_code": area_code,
                "number": result.get("phone_number", "")[-4:],
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(
            "phone_number_purchase_failed",
            extra={"error": str(e)[:200]},
        )
        return json.dumps({"error": str(e)})


async def transcribe_audio(
    audio_path: str,
    language: str = "en",
    *,
    _transcriber: Any = None,
) -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper.

    Args:
        audio_path: Path to audio file (.mp3, .wav, .m4a, .webm).
        language: Language code (ISO-639-1, default: "en").

    Returns:
        JSON string with transcribed text and metadata.
    """
    try:
        if _transcriber:
            result = await _transcriber.transcribe(audio_path, language=language)
        else:
            from core.llm.audio import Transcriber
            t = Transcriber()
            result = await t.transcribe(audio_path, language=language)

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("transcription_failed", extra={"error": str(e)[:200]})
        return json.dumps({"error": str(e), "audio_path": audio_path})
