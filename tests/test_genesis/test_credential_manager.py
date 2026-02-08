"""
Tests for the Genesis Credential Manager.

Covers:
- Encryption/decryption round-trip
- Key derivation
- In-memory credential storage (set, get, delete, has)
- Empty/invalid credential handling
- Required credential listing (platform + blueprint extras)
- Credential report generation
- Environment variable export
- Environment variable injection
- Credential deletion
- Missing master key error
- Invalid encrypted data handling
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from core.genesis.credential_manager import (
    MASTER_KEY_ENV,
    OPTIONAL_CREDENTIALS,
    PLATFORM_CREDENTIALS,
    CredentialManager,
    CredentialReport,
    CredentialStatus,
    _derive_fernet_key,
    decrypt_value,
    encrypt_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_MASTER_KEY = "test-master-key-for-enclave-genesis-engine-42"


@pytest.fixture
def cm() -> CredentialManager:
    """Create a credential manager with in-memory storage."""
    return CredentialManager(master_key=TEST_MASTER_KEY)


@pytest.fixture
def cm_with_creds(cm: CredentialManager) -> CredentialManager:
    """Credential manager with some pre-stored credentials."""
    cm.set_credential("test_vertical", "ANTHROPIC_API_KEY", "sk-ant-test123")
    cm.set_credential("test_vertical", "SUPABASE_URL", "https://test.supabase.co")
    cm.set_credential("test_vertical", "CUSTOM_KEY", "custom-secret")
    return cm


# ---------------------------------------------------------------------------
# Test: Key Derivation
# ---------------------------------------------------------------------------

class TestKeyDerivation:
    """Test Fernet key derivation from master key."""

    def test_derives_consistent_key(self):
        """Same master key always produces same derived key."""
        key1 = _derive_fernet_key("my-secret")
        key2 = _derive_fernet_key("my-secret")
        assert key1 == key2

    def test_different_keys_produce_different_derivation(self):
        """Different master keys produce different derived keys."""
        key1 = _derive_fernet_key("key-a")
        key2 = _derive_fernet_key("key-b")
        assert key1 != key2

    def test_derived_key_is_32_bytes_base64(self):
        """Fernet requires 32 url-safe base64-encoded bytes."""
        key = _derive_fernet_key("test-key")
        assert isinstance(key, bytes)
        # Fernet keys are 44 bytes (32 bytes base64-encoded)
        assert len(key) == 44


# ---------------------------------------------------------------------------
# Test: Encryption Round-Trip
# ---------------------------------------------------------------------------

class TestEncryptionRoundTrip:
    """Test encrypt/decrypt cycle."""

    def test_roundtrip_basic(self):
        """Encrypt then decrypt returns original value."""
        original = "sk-ant-test1234567890"
        encrypted = encrypt_value(original, TEST_MASTER_KEY)
        decrypted = decrypt_value(encrypted, TEST_MASTER_KEY)
        assert decrypted == original

    def test_roundtrip_unicode(self):
        """Unicode values survive encryption round-trip."""
        original = "api-key-with-ñ-and-ü"
        encrypted = encrypt_value(original, TEST_MASTER_KEY)
        decrypted = decrypt_value(encrypted, TEST_MASTER_KEY)
        assert decrypted == original

    def test_roundtrip_long_value(self):
        """Long values survive encryption round-trip."""
        original = "x" * 10000
        encrypted = encrypt_value(original, TEST_MASTER_KEY)
        decrypted = decrypt_value(encrypted, TEST_MASTER_KEY)
        assert decrypted == original

    def test_encrypted_differs_from_plaintext(self):
        """Encrypted value is not the same as plaintext."""
        original = "my-api-key"
        encrypted = encrypt_value(original, TEST_MASTER_KEY)
        assert encrypted != original

    def test_different_encryptions_differ(self):
        """Two encryptions of the same value produce different ciphertext (IV)."""
        original = "my-api-key"
        e1 = encrypt_value(original, TEST_MASTER_KEY)
        e2 = encrypt_value(original, TEST_MASTER_KEY)
        assert e1 != e2  # Fernet uses random IV

    def test_wrong_key_fails(self):
        """Decryption with wrong key raises error."""
        from cryptography.fernet import InvalidToken

        encrypted = encrypt_value("secret", TEST_MASTER_KEY)
        with pytest.raises(InvalidToken):
            decrypt_value(encrypted, "wrong-key")

    def test_corrupted_data_fails(self):
        """Corrupted encrypted data raises error."""
        from cryptography.fernet import InvalidToken

        with pytest.raises((InvalidToken, Exception)):
            decrypt_value("not-valid-encrypted-data", TEST_MASTER_KEY)

    def test_missing_master_key_raises(self):
        """Encryption without master key (and no env var) raises."""
        with patch.dict(os.environ, {MASTER_KEY_ENV: ""}, clear=False):
            with pytest.raises(EnvironmentError, match="ENCLAVE_MASTER_KEY"):
                encrypt_value("secret")

    def test_env_master_key_used(self):
        """Master key from environment variable is used when not passed."""
        with patch.dict(os.environ, {MASTER_KEY_ENV: TEST_MASTER_KEY}, clear=False):
            encrypted = encrypt_value("test-secret")
            decrypted = decrypt_value(encrypted)
            assert decrypted == "test-secret"


# ---------------------------------------------------------------------------
# Test: In-Memory Storage
# ---------------------------------------------------------------------------

class TestInMemoryStorage:
    """Test credential CRUD with in-memory backend."""

    def test_set_and_get(self, cm):
        """Store and retrieve a credential."""
        cm.set_credential("v1", "MY_KEY", "secret123")
        result = cm.get_credential("v1", "MY_KEY")
        assert result == "secret123"

    def test_get_nonexistent_returns_none(self, cm):
        """Getting a missing credential returns None."""
        result = cm.get_credential("v1", "NOPE")
        assert result is None

    def test_has_credential_true(self, cm):
        """has_credential returns True for stored credentials."""
        cm.set_credential("v1", "KEY", "val")
        assert cm.has_credential("v1", "KEY") is True

    def test_has_credential_false(self, cm):
        """has_credential returns False for missing credentials."""
        assert cm.has_credential("v1", "MISSING") is False

    def test_delete_credential(self, cm):
        """Deleted credentials cannot be retrieved."""
        cm.set_credential("v1", "KEY", "val")
        assert cm.has_credential("v1", "KEY") is True

        result = cm.delete_credential("v1", "KEY")
        assert result is True
        assert cm.has_credential("v1", "KEY") is False
        assert cm.get_credential("v1", "KEY") is None

    def test_delete_nonexistent(self, cm):
        """Deleting a nonexistent credential returns False."""
        assert cm.delete_credential("v1", "NOPE") is False

    def test_overwrite_credential(self, cm):
        """Storing a credential twice overwrites the first."""
        cm.set_credential("v1", "KEY", "original")
        cm.set_credential("v1", "KEY", "updated")
        assert cm.get_credential("v1", "KEY") == "updated"

    def test_empty_value_raises(self, cm):
        """Empty credential values are rejected."""
        with pytest.raises(ValueError, match="empty credential"):
            cm.set_credential("v1", "KEY", "")

    def test_whitespace_only_raises(self, cm):
        """Whitespace-only credential values are rejected."""
        with pytest.raises(ValueError, match="empty credential"):
            cm.set_credential("v1", "KEY", "   ")

    def test_vertical_isolation(self, cm):
        """Credentials are scoped to their vertical."""
        cm.set_credential("v1", "KEY", "secret-v1")
        cm.set_credential("v2", "KEY", "secret-v2")

        assert cm.get_credential("v1", "KEY") == "secret-v1"
        assert cm.get_credential("v2", "KEY") == "secret-v2"

    def test_multiple_credentials_per_vertical(self, cm):
        """A vertical can store multiple different credentials."""
        cm.set_credential("v1", "KEY_A", "val-a")
        cm.set_credential("v1", "KEY_B", "val-b")

        assert cm.get_credential("v1", "KEY_A") == "val-a"
        assert cm.get_credential("v1", "KEY_B") == "val-b"


# ---------------------------------------------------------------------------
# Test: Required Credentials Listing
# ---------------------------------------------------------------------------

class TestRequiredCredentials:
    """Test credential requirement resolution."""

    def test_platform_credentials_always_included(self, cm):
        """Platform credentials are always in the list."""
        creds = cm.get_required_credentials("v1")
        env_vars = [c["env_var_name"] for c in creds]

        assert "ANTHROPIC_API_KEY" in env_vars
        assert "SUPABASE_URL" in env_vars
        assert "SUPABASE_SERVICE_KEY" in env_vars
        assert "OPENAI_API_KEY" in env_vars
        assert "APOLLO_API_KEY" in env_vars

    def test_platform_credential_count(self, cm):
        """Default platform credentials count."""
        creds = cm.get_required_credentials("v1")
        assert len(creds) == len(PLATFORM_CREDENTIALS)

    def test_blueprint_extras_added(self, cm):
        """Blueprint-specific env vars are added."""
        creds = cm.get_required_credentials(
            "v1",
            blueprint_env_vars=["SENDGRID_API_KEY", "CUSTOM_API_KEY"],
        )
        env_vars = [c["env_var_name"] for c in creds]

        assert "SENDGRID_API_KEY" in env_vars
        assert "CUSTOM_API_KEY" in env_vars

    def test_no_duplicates(self, cm):
        """Duplicate env vars are not listed twice."""
        creds = cm.get_required_credentials(
            "v1",
            # ANTHROPIC_API_KEY is already a platform credential
            blueprint_env_vars=["ANTHROPIC_API_KEY", "NEW_KEY"],
        )
        env_vars = [c["env_var_name"] for c in creds]
        assert env_vars.count("ANTHROPIC_API_KEY") == 1

    def test_known_optional_has_instructions(self, cm):
        """Known optional credentials have proper instructions."""
        creds = cm.get_required_credentials(
            "v1",
            blueprint_env_vars=["SENDGRID_API_KEY"],
        )
        sendgrid = next(
            c for c in creds if c["env_var_name"] == "SENDGRID_API_KEY"
        )
        assert sendgrid["instructions"]  # Non-empty
        assert sendgrid["required"] is False

    def test_unknown_credential_auto_named(self, cm):
        """Unknown env vars get auto-generated names."""
        creds = cm.get_required_credentials(
            "v1",
            blueprint_env_vars=["MY_CUSTOM_THING"],
        )
        custom = next(
            c for c in creds if c["env_var_name"] == "MY_CUSTOM_THING"
        )
        assert custom["credential_name"]  # Auto-generated
        assert custom["required"] is True  # Unknown = required


# ---------------------------------------------------------------------------
# Test: Credential Report
# ---------------------------------------------------------------------------

class TestCredentialReport:
    """Test credential report generation."""

    def test_report_structure(self, cm):
        """Report has expected structure."""
        report = cm.get_credential_report("v1")
        assert isinstance(report, CredentialReport)
        assert report.vertical_id == "v1"
        assert len(report.credentials) > 0

    def test_report_all_missing(self, cm):
        """Report shows all missing when nothing is set."""
        report = cm.get_credential_report("v1")
        assert report.all_required_set is False
        assert len(report.missing_required) > 0

    def test_report_tracks_set_credentials(self, cm_with_creds):
        """Report reflects credentials that are set."""
        report = cm_with_creds.get_credential_report("test_vertical")
        set_vars = [c.env_var_name for c in report.credentials if c.is_set]
        assert "ANTHROPIC_API_KEY" in set_vars
        assert "SUPABASE_URL" in set_vars

    def test_report_checks_env_vars(self, cm):
        """Report checks os.environ for credentials."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "from-env"}, clear=False):
            report = cm.get_credential_report("v1")
            anthropic = next(
                c for c in report.credentials
                if c.env_var_name == "ANTHROPIC_API_KEY"
            )
            assert anthropic.is_set is True

    def test_report_total_set(self, cm_with_creds):
        """total_set counts correctly."""
        report = cm_with_creds.get_credential_report("test_vertical")
        assert report.total_set >= 2  # At least ANTHROPIC and SUPABASE_URL

    def test_report_total_required(self, cm):
        """total_required counts platform credentials."""
        report = cm.get_credential_report("v1")
        assert report.total_required == len(PLATFORM_CREDENTIALS)

    def test_credential_status_fields(self, cm):
        """CredentialStatus has all expected fields."""
        report = cm.get_credential_report("v1")
        cred = report.credentials[0]
        assert isinstance(cred, CredentialStatus)
        assert cred.env_var_name
        assert cred.credential_name
        assert isinstance(cred.is_set, bool)
        assert isinstance(cred.required, bool)

    def test_missing_optional_list(self, cm):
        """missing_optional returns non-required missing creds."""
        report = cm.get_credential_report(
            "v1",
            blueprint_env_vars=["SENDGRID_API_KEY"],
        )
        optional_missing = report.missing_optional
        env_vars = [c.env_var_name for c in optional_missing]
        assert "SENDGRID_API_KEY" in env_vars


# ---------------------------------------------------------------------------
# Test: Export / Inject
# ---------------------------------------------------------------------------

class TestExportAndInject:
    """Test credential export and environment injection."""

    def test_export_as_env(self, cm_with_creds):
        """Export returns decrypted key-value pairs."""
        env = cm_with_creds.export_as_env("test_vertical")
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-test123"
        assert env["SUPABASE_URL"] == "https://test.supabase.co"
        assert env["CUSTOM_KEY"] == "custom-secret"

    def test_export_empty_vertical(self, cm):
        """Export for empty vertical returns empty dict."""
        env = cm.export_as_env("nonexistent")
        assert env == {}

    def test_export_vertical_isolation(self, cm_with_creds):
        """Export only returns credentials for the specified vertical."""
        cm_with_creds.set_credential("other_vertical", "OTHER_KEY", "other-val")
        env = cm_with_creds.export_as_env("test_vertical")
        assert "OTHER_KEY" not in env

    def test_inject_into_env(self, cm_with_creds):
        """inject_into_env sets os.environ variables."""
        # Clear any existing env vars
        for key in ["ANTHROPIC_API_KEY", "SUPABASE_URL", "CUSTOM_KEY"]:
            os.environ.pop(key, None)

        count = cm_with_creds.inject_into_env("test_vertical")
        assert count == 3
        assert os.environ.get("CUSTOM_KEY") == "custom-secret"

        # Clean up
        for key in ["ANTHROPIC_API_KEY", "SUPABASE_URL", "CUSTOM_KEY"]:
            os.environ.pop(key, None)

    def test_inject_returns_count(self, cm):
        """inject_into_env returns 0 for empty vertical."""
        count = cm.inject_into_env("empty_vertical")
        assert count == 0


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters_in_value(self, cm):
        """Special characters survive encryption round-trip."""
        cm.set_credential("v1", "KEY", 'p@ss="w0rd"&more<>')
        assert cm.get_credential("v1", "KEY") == 'p@ss="w0rd"&more<>'

    def test_newlines_in_value(self, cm):
        """Multi-line values (PEM keys) survive encryption."""
        pem = "-----BEGIN RSA KEY-----\nMIIE...base64...\n-----END RSA KEY-----"
        cm.set_credential("v1", "RSA_KEY", pem)
        assert cm.get_credential("v1", "RSA_KEY") == pem

    def test_credential_name_auto_generated(self, cm):
        """set_credential works without explicit credential_name."""
        cm.set_credential("v1", "MY_API_KEY", "val")
        # Should not raise — name is auto-handled internally

    def test_wrong_master_key_returns_none(self):
        """Getting credential with wrong master key returns None gracefully."""
        cm1 = CredentialManager(master_key="key-one")
        cm1.set_credential("v1", "KEY", "secret")

        # Access same memory store but with different key
        cm2 = CredentialManager(master_key="key-two")
        cm2._memory_store = cm1._memory_store  # Share the store

        # Should return None (decryption fails gracefully)
        result = cm2.get_credential("v1", "KEY")
        assert result is None

    def test_platform_credentials_have_instructions(self):
        """All platform credentials have non-empty instructions."""
        for cred in PLATFORM_CREDENTIALS:
            assert cred["instructions"], f"{cred['env_var_name']} missing instructions"

    def test_optional_credentials_all_documented(self):
        """All optional credentials have instructions."""
        for env_var, info in OPTIONAL_CREDENTIALS.items():
            assert info["instructions"], f"{env_var} missing instructions"
            assert info["credential_name"], f"{env_var} missing name"


# ---------------------------------------------------------------------------
# Test: Database Backend (mocked)
# ---------------------------------------------------------------------------

class TestDatabaseBackend:
    """Test credential storage with mocked database."""

    def test_set_stores_in_db(self):
        """set_credential writes to database table."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])

        cm = CredentialManager(db=mock_db, master_key=TEST_MASTER_KEY)
        cm.set_credential("v1", "KEY", "secret")

        mock_db.client.table.assert_called_with("vertical_credentials")
        mock_table.upsert.assert_called_once()
        call_args = mock_table.upsert.call_args[0][0]
        assert call_args["vertical_id"] == "v1"
        assert call_args["env_var_name"] == "KEY"
        assert call_args["is_set"] is True
        # Encrypted value should NOT be the plaintext
        assert call_args["encrypted_value"] != "secret"

    def test_get_reads_from_db(self):
        """get_credential reads from database table."""
        encrypted = encrypt_value("db-secret", TEST_MASTER_KEY)

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"encrypted_value": encrypted}]

        mock_chain = MagicMock()
        mock_chain.select.return_value = mock_chain
        mock_chain.eq.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        mock_chain.execute.return_value = mock_result

        mock_db.client.table.return_value = mock_chain

        cm = CredentialManager(db=mock_db, master_key=TEST_MASTER_KEY)
        result = cm.get_credential("v1", "KEY")
        assert result == "db-secret"

    def test_db_failure_falls_back_to_memory(self):
        """If DB write fails, credential is stored in memory."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.client.table.return_value = mock_table
        # DB write fails
        mock_table.upsert.side_effect = Exception("DB down")
        # DB read also fails (so get_credential falls back to memory)
        mock_table.select.side_effect = Exception("DB down")

        cm = CredentialManager(db=mock_db, master_key=TEST_MASTER_KEY)
        cm.set_credential("v1", "KEY", "fallback-secret")

        # Should still be retrievable from memory
        assert cm.get_credential("v1", "KEY") == "fallback-secret"
