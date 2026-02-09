"""
Audio services for the Sovereign Venture Engine.

Provides text-to-speech synthesis (ElevenLabs) and speech-to-text
transcription (OpenAI Whisper) with intelligent caching to avoid
paying to synthesize the same greeting twice.

Architecture:
    Transcriber: Audio file → text (via OpenAI Whisper API)
    Synthesizer: Text → audio file (via ElevenLabs API)

Both services degrade gracefully:
    - Missing API keys → mock responses for development
    - Network errors → error dicts with context
    - Cache prevents duplicate synthesis charges

Usage:
    from core.llm.audio import Transcriber, Synthesizer

    transcriber = Transcriber()
    text = await transcriber.transcribe("/path/to/recording.mp3")

    synth = Synthesizer()
    audio_path = await synth.synthesize("Thank you for calling!")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio Cache
# ---------------------------------------------------------------------------


class AudioCache:
    """
    Simple file-based cache for synthesized audio.

    Prevents paying to synthesize the same text twice.
    Cache key = SHA-256(text + voice_id + stability + similarity_boost).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.environ.get(
            "AUDIO_CACHE_DIR", ".audio_cache"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the cache index from disk."""
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self) -> None:
        """Persist the cache index to disk."""
        try:
            self._index_path.write_text(json.dumps(self._index, indent=2))
        except IOError as e:
            logger.warning("audio_cache_save_failed", extra={"error": str(e)})

    @staticmethod
    def _cache_key(text: str, voice_id: str = "", **kwargs: Any) -> str:
        """Generate a deterministic cache key."""
        raw = f"{text}|{voice_id}|{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def get(self, text: str, voice_id: str = "", **kwargs: Any) -> Optional[str]:
        """
        Look up cached audio for the given text.

        Returns:
            Path to cached audio file, or None if not cached.
        """
        key = self._cache_key(text, voice_id, **kwargs)
        entry = self._index.get(key)
        if entry and Path(entry["path"]).exists():
            logger.debug("audio_cache_hit", extra={"key": key[:8]})
            return entry["path"]
        return None

    def put(
        self,
        text: str,
        audio_data: bytes,
        voice_id: str = "",
        fmt: str = "mp3",
        **kwargs: Any,
    ) -> str:
        """
        Store synthesized audio in the cache.

        Returns:
            Path to the cached audio file.
        """
        key = self._cache_key(text, voice_id, **kwargs)
        audio_path = self.cache_dir / f"{key}.{fmt}"
        audio_path.write_bytes(audio_data)

        self._index[key] = {
            "path": str(audio_path),
            "text": text[:100],  # Store truncated text for debugging
            "voice_id": voice_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(audio_data),
        }
        self._save_index()

        logger.info(
            "audio_cache_stored",
            extra={"key": key[:8], "size": len(audio_data)},
        )
        return str(audio_path)

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._index)

    def clear(self) -> int:
        """Clear all cached audio. Returns count of entries removed."""
        count = len(self._index)
        for entry in self._index.values():
            path = Path(entry["path"])
            if path.exists():
                path.unlink()
        self._index = {}
        self._save_index()
        logger.info("audio_cache_cleared", extra={"entries_removed": count})
        return count


# ---------------------------------------------------------------------------
# Transcriber (Speech-to-Text via OpenAI Whisper)
# ---------------------------------------------------------------------------


class Transcriber:
    """
    Transcribes audio files to text using OpenAI Whisper API.

    Falls back to mock transcription when OPENAI_API_KEY is not set.

    Usage:
        t = Transcriber()
        result = await t.transcribe("recording.mp3")
        print(result["text"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        language: str = "en",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.language = language
        self._mock_mode = not bool(self.api_key)

        if self._mock_mode:
            logger.warning(
                "transcriber_mock_mode: OPENAI_API_KEY not set — "
                "using mock transcriptions"
            )

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (.mp3, .wav, .m4a, .webm, .mp4).
            language: Override language (ISO-639-1).
            prompt: Optional prompt to guide transcription.

        Returns:
            Dict with 'text', 'language', 'duration_seconds', 'source'.
        """
        if self._mock_mode:
            return self._mock_transcribe(audio_path)

        try:
            return await self._api_transcribe(
                audio_path,
                language=language or self.language,
                prompt=prompt,
            )
        except Exception as e:
            logger.error(
                "transcription_failed",
                extra={"audio_path": audio_path, "error": str(e)[:200]},
            )
            return {
                "text": "",
                "error": str(e),
                "source": "whisper_api",
                "audio_path": audio_path,
            }

    async def _api_transcribe(
        self,
        audio_path: str,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call OpenAI Whisper API for transcription."""
        import httpx

        path = Path(audio_path)
        if not path.exists():
            return {"text": "", "error": f"File not found: {audio_path}"}

        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
            ".mp4": "audio/mp4",
            ".ogg": "audio/ogg",
        }
        content_type = mime_types.get(path.suffix.lower(), "audio/mpeg")

        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {
                "file": (path.name, path.read_bytes(), content_type),
            }
            data: dict[str, str] = {
                "model": self.model,
                "language": language,
                "response_format": "json",
            }
            if prompt:
                data["prompt"] = prompt

            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data,
            )
            resp.raise_for_status()
            result = resp.json()

        logger.info(
            "transcription_complete",
            extra={
                "audio_path": audio_path,
                "text_length": len(result.get("text", "")),
            },
        )

        return {
            "text": result.get("text", ""),
            "language": language,
            "source": "whisper_api",
            "model": self.model,
            "audio_path": audio_path,
        }

    def _mock_transcribe(self, audio_path: str) -> dict[str, Any]:
        """Return mock transcription for development."""
        mock_texts = {
            "support": (
                "Hi, I'm calling because I have a question about my recent order. "
                "The tracking number doesn't seem to work. Can someone help me?"
            ),
            "sales": (
                "Hello, I saw your website and I'm interested in learning more "
                "about your cybersecurity assessment services. We're a mid-size "
                "company with about 200 employees."
            ),
            "urgent": (
                "This is urgent — I need to speak with someone about a security "
                "incident that happened this morning. Please call me back as soon "
                "as possible."
            ),
        }

        # Cycle through mock texts based on filename
        filename = Path(audio_path).stem.lower()
        for keyword, text in mock_texts.items():
            if keyword in filename:
                return {
                    "text": text,
                    "language": "en",
                    "source": "mock",
                    "audio_path": audio_path,
                }

        return {
            "text": mock_texts["support"],
            "language": "en",
            "source": "mock",
            "audio_path": audio_path,
        }


# ---------------------------------------------------------------------------
# Synthesizer (Text-to-Speech via ElevenLabs)
# ---------------------------------------------------------------------------


class Synthesizer:
    """
    Synthesizes speech from text using ElevenLabs API.

    Features:
        - Intelligent caching (same text + voice → cached audio)
        - Multiple voice support
        - Falls back to mock when ELEVENLABS_API_KEY is not set

    Usage:
        s = Synthesizer()
        result = await s.synthesize("Thank you for calling!")
        print(result["audio_path"])  # Path to MP3 file
    """

    # Default ElevenLabs voice IDs
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" — professional female
    VOICE_PRESETS = {
        "professional_female": "21m00Tcm4TlvDq8ikWAM",
        "professional_male": "29vD33N1CtxCmqQRPOHJ",
        "warm_female": "EXAVITQu4vr4xnSDxMaL",
        "warm_male": "VR6AewLTigWG4xSOukaG",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_monolingual_v1",
        cache_dir: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or self.DEFAULT_VOICE_ID
        self.model_id = model_id
        self.cache = AudioCache(cache_dir=cache_dir)
        self._mock_mode = not bool(self.api_key)

        if self._mock_mode:
            logger.warning(
                "synthesizer_mock_mode: ELEVENLABS_API_KEY not set — "
                "using mock synthesis"
            )

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_preset: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech (max ~5000 chars).
            voice_id: ElevenLabs voice ID (overrides default).
            voice_preset: Named voice preset ("professional_female", etc.).
            stability: Voice stability (0.0 = more variable, 1.0 = more stable).
            similarity_boost: Voice similarity boost (0.0 = more creative).
            output_dir: Where to write audio file (defaults to cache dir).

        Returns:
            Dict with 'audio_path', 'text', 'voice_id', 'duration_estimate',
            'cached', 'source'.
        """
        # Resolve voice
        vid = voice_id
        if not vid and voice_preset:
            vid = self.VOICE_PRESETS.get(voice_preset, self.voice_id)
        if not vid:
            vid = self.voice_id

        # Check cache
        cached_path = self.cache.get(
            text, vid, stability=stability, similarity_boost=similarity_boost
        )
        if cached_path:
            return {
                "audio_path": cached_path,
                "text": text[:100],
                "voice_id": vid,
                "duration_estimate": len(text) * 0.065,  # ~65ms per char
                "cached": True,
                "source": "cache",
            }

        if self._mock_mode:
            return self._mock_synthesize(text, vid, stability, similarity_boost)

        try:
            return await self._api_synthesize(
                text, vid, stability, similarity_boost, output_dir
            )
        except Exception as e:
            logger.error(
                "synthesis_failed",
                extra={"text_length": len(text), "error": str(e)[:200]},
            )
            return {
                "audio_path": "",
                "text": text[:100],
                "voice_id": vid,
                "error": str(e),
                "cached": False,
                "source": "elevenlabs_api",
            }

    async def _api_synthesize(
        self,
        text: str,
        voice_id: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call ElevenLabs API for speech synthesis."""
        import httpx

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text,
                    "model_id": self.model_id,
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                    },
                },
            )
            resp.raise_for_status()
            audio_data = resp.content

        # Cache the result
        audio_path = self.cache.put(
            text,
            audio_data,
            voice_id=voice_id,
            stability=stability,
            similarity_boost=similarity_boost,
        )

        logger.info(
            "synthesis_complete",
            extra={
                "text_length": len(text),
                "audio_size": len(audio_data),
                "voice_id": voice_id[:8],
            },
        )

        return {
            "audio_path": audio_path,
            "text": text[:100],
            "voice_id": voice_id,
            "duration_estimate": len(text) * 0.065,
            "size_bytes": len(audio_data),
            "cached": False,
            "source": "elevenlabs_api",
        }

    def _mock_synthesize(
        self,
        text: str,
        voice_id: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> dict[str, Any]:
        """Return mock synthesis for development."""
        # Generate a tiny placeholder "audio" file
        mock_audio = b"MOCK_AUDIO_DATA:" + text[:50].encode("utf-8")
        audio_path = self.cache.put(
            text, mock_audio, voice_id=voice_id, fmt="mp3",
            stability=stability, similarity_boost=similarity_boost,
        )

        return {
            "audio_path": audio_path,
            "text": text[:100],
            "voice_id": voice_id,
            "duration_estimate": len(text) * 0.065,
            "size_bytes": len(mock_audio),
            "cached": False,
            "source": "mock",
        }

    def get_available_voices(self) -> dict[str, str]:
        """Return available voice presets."""
        return dict(self.VOICE_PRESETS)
