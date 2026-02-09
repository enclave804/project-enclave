"""
Tests for Audio Services — Transcriber, Synthesizer, and AudioCache.

Covers:
- AudioCache: put/get/clear, deterministic keys, disk persistence
- Transcriber: Mock mode, filename-based transcript selection, error handling
- Synthesizer: Mock synthesis, cache hit/miss, voice presets, get_available_voices
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.llm.audio import AudioCache, Synthesizer, Transcriber


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════
# 1. AudioCache
# ═══════════════════════════════════════════════════════════════════════


class TestAudioCache:
    """Tests for the file-based audio cache."""

    def test_creates_cache_directory(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "audio_cache"))
        assert Path(cache.cache_dir).exists()

    def test_empty_cache_has_zero_size(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        assert cache.size == 0

    def test_put_stores_audio_file(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        path = cache.put("Hello world", b"fake_audio_data", voice_id="v1")
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"fake_audio_data"

    def test_put_increments_size(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        cache.put("text1", b"data1", voice_id="v1")
        cache.put("text2", b"data2", voice_id="v1")
        assert cache.size == 2

    def test_get_returns_cached_path(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        cache.put("Hello", b"data", voice_id="v1")
        result = cache.get("Hello", voice_id="v1")
        assert result is not None
        assert Path(result).exists()

    def test_get_returns_none_for_miss(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        result = cache.get("not_cached", voice_id="v1")
        assert result is None

    def test_deterministic_cache_key(self):
        """Same inputs produce the same cache key."""
        key1 = AudioCache._cache_key("Hello", "voice1", stability=0.5)
        key2 = AudioCache._cache_key("Hello", "voice1", stability=0.5)
        assert key1 == key2

    def test_different_text_different_key(self):
        key1 = AudioCache._cache_key("Hello", "voice1")
        key2 = AudioCache._cache_key("World", "voice1")
        assert key1 != key2

    def test_different_voice_different_key(self):
        key1 = AudioCache._cache_key("Hello", "voice1")
        key2 = AudioCache._cache_key("Hello", "voice2")
        assert key1 != key2

    def test_clear_removes_all_entries(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        cache.put("a", b"1", voice_id="v")
        cache.put("b", b"2", voice_id="v")
        assert cache.size == 2

        removed = cache.clear()
        assert removed == 2
        assert cache.size == 0

    def test_clear_deletes_files(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        path = cache.put("text", b"audio", voice_id="v")
        assert Path(path).exists()

        cache.clear()
        assert not Path(path).exists()

    def test_index_persists_to_disk(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache1 = AudioCache(cache_dir=cache_dir)
        cache1.put("persist_test", b"data", voice_id="v")

        # Reload cache from disk
        cache2 = AudioCache(cache_dir=cache_dir)
        assert cache2.size == 1
        result = cache2.get("persist_test", voice_id="v")
        assert result is not None

    def test_custom_format(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        path = cache.put("text", b"data", voice_id="v", fmt="wav")
        assert path.endswith(".wav")

    def test_truncates_text_in_index(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        long_text = "x" * 200
        cache.put(long_text, b"data", voice_id="v")
        index = json.loads((cache.cache_dir / "index.json").read_text())
        for entry in index.values():
            assert len(entry["text"]) <= 100

    def test_corrupted_index_recovers(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "index.json").write_text("not valid json{{{")

        cache = AudioCache(cache_dir=str(cache_dir))
        assert cache.size == 0  # Recovered from corruption


# ═══════════════════════════════════════════════════════════════════════
# 2. Transcriber
# ═══════════════════════════════════════════════════════════════════════


class TestTranscriber:
    """Tests for the OpenAI Whisper transcriber (mock mode)."""

    def test_enters_mock_mode_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            assert t._mock_mode is True

    def test_live_mode_with_api_key(self):
        t = Transcriber(api_key="sk-test-key")
        assert t._mock_mode is False

    def test_mock_transcribe_returns_text(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            result = _run(t.transcribe("/path/to/generic_recording.mp3"))
            assert "text" in result
            assert len(result["text"]) > 0
            assert result["source"] == "mock"

    def test_mock_detects_support_keyword(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            result = _run(t.transcribe("/calls/support_issue.mp3"))
            assert "order" in result["text"].lower() or "tracking" in result["text"].lower()

    def test_mock_detects_sales_keyword(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            result = _run(t.transcribe("/calls/sales_inquiry.wav"))
            assert "interested" in result["text"].lower() or "cybersecurity" in result["text"].lower()

    def test_mock_detects_urgent_keyword(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            result = _run(t.transcribe("/calls/urgent_message.m4a"))
            assert "urgent" in result["text"].lower() or "security" in result["text"].lower()

    def test_mock_defaults_to_support(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            result = _run(t.transcribe("/calls/random_name.mp3"))
            # Default mock text is the support one
            assert result["source"] == "mock"
            assert len(result["text"]) > 10

    def test_transcription_result_has_required_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            t = Transcriber()
            result = _run(t.transcribe("/path/file.mp3"))
            assert "text" in result
            assert "language" in result
            assert "source" in result
            assert "audio_path" in result

    def test_default_language_is_english(self):
        t = Transcriber()
        assert t.language == "en"

    def test_custom_model(self):
        t = Transcriber(model="whisper-2")
        assert t.model == "whisper-2"


# ═══════════════════════════════════════════════════════════════════════
# 3. Synthesizer
# ═══════════════════════════════════════════════════════════════════════


class TestSynthesizer:
    """Tests for the ElevenLabs text-to-speech synthesizer (mock mode)."""

    def test_enters_mock_mode_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            assert s._mock_mode is True

    def test_live_mode_with_api_key(self):
        s = Synthesizer(api_key="el-test-key", cache_dir=tempfile.mkdtemp())
        assert s._mock_mode is False

    def test_mock_synthesis_creates_file(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result = _run(s.synthesize("Thank you for calling!"))
            assert "audio_path" in result
            assert Path(result["audio_path"]).exists()
            assert result["source"] == "mock"
            assert result["cached"] is False

    def test_mock_audio_contains_text(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result = _run(s.synthesize("Hello World"))
            data = Path(result["audio_path"]).read_bytes()
            assert b"MOCK_AUDIO_DATA:" in data
            assert b"Hello World" in data

    def test_cache_hit_on_second_call(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result1 = _run(s.synthesize("Same text"))
            result2 = _run(s.synthesize("Same text"))
            assert result2["cached"] is True
            assert result2["source"] == "cache"
            assert result1["audio_path"] == result2["audio_path"]

    def test_different_text_no_cache_hit(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result1 = _run(s.synthesize("Text A"))
            result2 = _run(s.synthesize("Text B"))
            assert result2["cached"] is False
            assert result1["audio_path"] != result2["audio_path"]

    def test_voice_preset_resolves_to_voice_id(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result = _run(s.synthesize("Test", voice_preset="warm_male"))
            assert result["voice_id"] == Synthesizer.VOICE_PRESETS["warm_male"]

    def test_explicit_voice_id_overrides_preset(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result = _run(s.synthesize("Test", voice_id="custom_id", voice_preset="warm_male"))
            assert result["voice_id"] == "custom_id"

    def test_duration_estimate(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            text = "Hello World"  # 11 chars
            result = _run(s.synthesize(text))
            assert result["duration_estimate"] == pytest.approx(11 * 0.065, abs=0.01)

    def test_get_available_voices(self):
        s = Synthesizer(cache_dir=tempfile.mkdtemp())
        voices = s.get_available_voices()
        assert "professional_female" in voices
        assert "professional_male" in voices
        assert "warm_female" in voices
        assert "warm_male" in voices
        assert len(voices) == 4

    def test_default_voice_id(self):
        s = Synthesizer(cache_dir=tempfile.mkdtemp())
        assert s.voice_id == Synthesizer.DEFAULT_VOICE_ID

    def test_custom_voice_id(self):
        s = Synthesizer(voice_id="my_custom_voice", cache_dir=tempfile.mkdtemp())
        assert s.voice_id == "my_custom_voice"

    def test_synthesis_result_has_required_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result = _run(s.synthesize("Test"))
            required = ["audio_path", "text", "voice_id", "duration_estimate", "cached", "source"]
            for field in required:
                assert field in result, f"Missing field: {field}"

    def test_text_truncated_in_result(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            long_text = "x" * 200
            result = _run(s.synthesize(long_text))
            assert len(result["text"]) <= 100

    def test_size_bytes_in_mock_result(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            s = Synthesizer(cache_dir=tempfile.mkdtemp())
            result = _run(s.synthesize("Hi"))
            assert "size_bytes" in result
            assert result["size_bytes"] > 0
