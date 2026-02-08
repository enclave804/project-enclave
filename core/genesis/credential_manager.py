"""
Genesis Credential Manager — Encrypted per-vertical secret storage.

Handles the secure storage and retrieval of API keys and secrets that
each vertical needs to function. Uses Fernet symmetric encryption
derived from a master key.

Security model:
    - Plaintext credentials are NEVER stored at rest
    - Encryption key is derived from ENCLAVE_MASTER_KEY env var
    - Each credential is independently encrypted
    - Validation checks credentials against their provider (where possible)
    - Credential instructions guide users to obtain keys

Usage:
    from core.genesis.credential_manager import CredentialManager

    cm = CredentialManager()
    cm.set_credential("my_vertical", "APOLLO_API_KEY", "secret123")
    value = cm.get_credential("my_vertical", "APOLLO_API_KEY")
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER_KEY_ENV = "ENCLAVE_MASTER_KEY"

# Well-known credentials the platform always needs
PLATFORM_CREDENTIALS: list[dict[str, Any]] = [
    {
        "credential_name": "Anthropic API Key",
        "env_var_name": "ANTHROPIC_API_KEY",
        "required": True,
        "instructions": (
            "Get your key from https://console.anthropic.com/settings/keys. "
            "This powers all AI-driven agent decisions."
        ),
    },
    {
        "credential_name": "Supabase URL",
        "env_var_name": "SUPABASE_URL",
        "required": True,
        "instructions": (
            "Your Supabase project URL (e.g., https://xxx.supabase.co). "
            "Found in Project Settings → API."
        ),
    },
    {
        "credential_name": "Supabase Service Key",
        "env_var_name": "SUPABASE_SERVICE_KEY",
        "required": True,
        "instructions": (
            "The service_role key from Supabase Project Settings → API. "
            "This bypasses RLS for backend operations."
        ),
    },
    {
        "credential_name": "OpenAI API Key",
        "env_var_name": "OPENAI_API_KEY",
        "required": True,
        "instructions": (
            "Get your key from https://platform.openai.com/api-keys. "
            "Used for text embeddings (RAG knowledge base)."
        ),
    },
    {
        "credential_name": "Apollo API Key",
        "env_var_name": "APOLLO_API_KEY",
        "required": True,
        "instructions": (
            "Get your key from https://developer.apollo.io/. "
            "Used for lead sourcing and enrichment."
        ),
    },
]

# Optional integrations and their credential details
OPTIONAL_CREDENTIALS: dict[str, dict[str, Any]] = {
    "SENDGRID_API_KEY": {
        "credential_name": "SendGrid API Key",
        "required": False,
        "instructions": (
            "Get your key from https://app.sendgrid.com/settings/api_keys. "
            "Required for production email sending."
        ),
    },
    "MAILGUN_API_KEY": {
        "credential_name": "Mailgun API Key",
        "required": False,
        "instructions": (
            "Get your key from https://app.mailgun.com/app/account/security/api_keys. "
            "Alternative email provider."
        ),
    },
    "SHODAN_API_KEY": {
        "credential_name": "Shodan API Key",
        "required": False,
        "instructions": (
            "Get your key from https://account.shodan.io/. "
            "Enhances tech stack scanning capabilities."
        ),
    },
}


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class CredentialStatus:
    """Status of a single credential."""
    env_var_name: str
    credential_name: str
    is_set: bool
    required: bool
    instructions: str = ""
    validation_status: str = "unknown"  # unknown, valid, invalid, expired


@dataclass
class CredentialReport:
    """Report on all credentials for a vertical."""
    vertical_id: str
    credentials: list[CredentialStatus] = field(default_factory=list)

    @property
    def all_required_set(self) -> bool:
        """Check if all required credentials are set."""
        return all(
            c.is_set for c in self.credentials if c.required
        )

    @property
    def missing_required(self) -> list[CredentialStatus]:
        """Get list of missing required credentials."""
        return [c for c in self.credentials if c.required and not c.is_set]

    @property
    def missing_optional(self) -> list[CredentialStatus]:
        """Get list of missing optional credentials."""
        return [c for c in self.credentials if not c.required and not c.is_set]

    @property
    def total_set(self) -> int:
        """Count of credentials that are set."""
        return sum(1 for c in self.credentials if c.is_set)

    @property
    def total_required(self) -> int:
        """Count of required credentials."""
        return sum(1 for c in self.credentials if c.required)


# ---------------------------------------------------------------------------
# Encryption Utilities
# ---------------------------------------------------------------------------

def _derive_fernet_key(master_key: str) -> bytes:
    """
    Derive a Fernet-compatible key from a master key string.

    Uses SHA-256 hash of the master key, base64-encoded to 32 bytes.
    Fernet requires exactly 32 url-safe base64-encoded bytes.
    """
    hashed = hashlib.sha256(master_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(hashed)


def _get_fernet(master_key: Optional[str] = None) -> Fernet:
    """
    Get a Fernet instance from the master key.

    Args:
        master_key: Optional explicit key. If None, reads from env.

    Raises:
        EnvironmentError: If no master key is available.
    """
    key = master_key or os.environ.get(MASTER_KEY_ENV, "").strip()
    if not key:
        raise EnvironmentError(
            f"{MASTER_KEY_ENV} environment variable is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return Fernet(_derive_fernet_key(key))


def encrypt_value(plaintext: str, master_key: Optional[str] = None) -> str:
    """
    Encrypt a plaintext credential value.

    Returns a base64-encoded encrypted string that can be stored safely.
    """
    f = _get_fernet(master_key)
    encrypted = f.encrypt(plaintext.encode("utf-8"))
    return encrypted.decode("utf-8")


def decrypt_value(encrypted_text: str, master_key: Optional[str] = None) -> str:
    """
    Decrypt an encrypted credential value.

    Raises:
        InvalidToken: If the encrypted text is invalid or the key is wrong.
    """
    f = _get_fernet(master_key)
    decrypted = f.decrypt(encrypted_text.encode("utf-8"))
    return decrypted.decode("utf-8")


# ---------------------------------------------------------------------------
# Credential Manager
# ---------------------------------------------------------------------------

class CredentialManager:
    """
    Manages encrypted credentials for verticals.

    In production, stores encrypted values in the vertical_credentials table.
    Also supports an in-memory mode for testing and local dev.
    """

    def __init__(
        self,
        db: Any = None,
        master_key: Optional[str] = None,
    ):
        """
        Args:
            db: Optional database client (EnclaveDB) for persistent storage.
                If None, uses in-memory storage only.
            master_key: Optional explicit master key. If None, reads from env.
        """
        self._db = db
        self._master_key = master_key
        # In-memory store: {(vertical_id, env_var): encrypted_value}
        self._memory_store: dict[tuple[str, str], str] = {}

    # --- Core Operations ---

    def set_credential(
        self,
        vertical_id: str,
        env_var_name: str,
        plaintext_value: str,
        credential_name: Optional[str] = None,
        instructions: str = "",
        required: bool = True,
    ) -> None:
        """
        Store an encrypted credential for a vertical.

        Args:
            vertical_id: The vertical this credential belongs to.
            env_var_name: Environment variable name (e.g., "APOLLO_API_KEY").
            plaintext_value: The actual secret value to encrypt.
            credential_name: Human-readable name (auto-generated if None).
            instructions: How to obtain this credential.
            required: Whether this credential is required for the vertical.
        """
        if not plaintext_value or not plaintext_value.strip():
            raise ValueError(
                f"Cannot store empty credential for {env_var_name}"
            )

        encrypted = encrypt_value(plaintext_value, self._master_key)

        if self._db is not None:
            try:
                self._db.client.table("vertical_credentials").upsert(
                    {
                        "vertical_id": vertical_id,
                        "env_var_name": env_var_name,
                        "credential_name": credential_name or env_var_name,
                        "encrypted_value": encrypted,
                        "is_set": True,
                        "instructions": instructions,
                        "required": required,
                        "validation_status": "unknown",
                    },
                    on_conflict="vertical_id,env_var_name",
                ).execute()
            except Exception as e:
                logger.warning(f"DB storage failed, using memory: {e}")
                self._memory_store[(vertical_id, env_var_name)] = encrypted
        else:
            self._memory_store[(vertical_id, env_var_name)] = encrypted

        logger.info(
            "credential_stored",
            extra={
                "vertical_id": vertical_id,
                "env_var": env_var_name,
                "storage": "db" if self._db else "memory",
            },
        )

    def get_credential(
        self,
        vertical_id: str,
        env_var_name: str,
    ) -> Optional[str]:
        """
        Retrieve and decrypt a credential.

        Returns None if the credential is not set.
        """
        encrypted = None

        if self._db is not None:
            try:
                result = (
                    self._db.client.table("vertical_credentials")
                    .select("encrypted_value")
                    .eq("vertical_id", vertical_id)
                    .eq("env_var_name", env_var_name)
                    .eq("is_set", True)
                    .limit(1)
                    .execute()
                )
                if result.data:
                    encrypted = result.data[0]["encrypted_value"]
            except Exception as e:
                logger.warning(f"DB retrieval failed, checking memory: {e}")

        if encrypted is None:
            encrypted = self._memory_store.get(
                (vertical_id, env_var_name)
            )

        if encrypted is None:
            return None

        try:
            return decrypt_value(encrypted, self._master_key)
        except InvalidToken:
            logger.error(
                "credential_decryption_failed",
                extra={
                    "vertical_id": vertical_id,
                    "env_var": env_var_name,
                },
            )
            return None

    def delete_credential(
        self,
        vertical_id: str,
        env_var_name: str,
    ) -> bool:
        """
        Remove a stored credential.

        Returns True if the credential was found and removed.
        """
        removed = False

        if self._db is not None:
            try:
                self._db.client.table("vertical_credentials").update(
                    {
                        "encrypted_value": None,
                        "is_set": False,
                        "validation_status": "unknown",
                    }
                ).eq("vertical_id", vertical_id).eq(
                    "env_var_name", env_var_name
                ).execute()
                removed = True
            except Exception as e:
                logger.warning(f"DB deletion failed: {e}")

        key = (vertical_id, env_var_name)
        if key in self._memory_store:
            del self._memory_store[key]
            removed = True

        return removed

    def has_credential(
        self,
        vertical_id: str,
        env_var_name: str,
    ) -> bool:
        """Check if a credential is stored (without decrypting)."""
        if self._db is not None:
            try:
                result = (
                    self._db.client.table("vertical_credentials")
                    .select("is_set")
                    .eq("vertical_id", vertical_id)
                    .eq("env_var_name", env_var_name)
                    .eq("is_set", True)
                    .limit(1)
                    .execute()
                )
                if result.data:
                    return True
            except Exception:
                pass

        return (vertical_id, env_var_name) in self._memory_store

    # --- Bulk Operations ---

    def get_required_credentials(
        self,
        vertical_id: str,
        blueprint_env_vars: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Get the list of required credentials for a vertical.

        Combines platform defaults with blueprint-specific integrations.

        Args:
            vertical_id: The vertical ID.
            blueprint_env_vars: Additional env vars from the blueprint.

        Returns:
            List of credential dicts with name, env_var, required, instructions.
        """
        credentials = []
        seen_vars: set[str] = set()

        # Platform credentials
        for cred in PLATFORM_CREDENTIALS:
            credentials.append({**cred})
            seen_vars.add(cred["env_var_name"])

        # Blueprint-specific credentials
        if blueprint_env_vars:
            for env_var in blueprint_env_vars:
                if env_var in seen_vars:
                    continue
                seen_vars.add(env_var)

                # Check if it's a known optional credential
                if env_var in OPTIONAL_CREDENTIALS:
                    info = OPTIONAL_CREDENTIALS[env_var]
                    credentials.append({
                        "credential_name": info["credential_name"],
                        "env_var_name": env_var,
                        "required": info.get("required", False),
                        "instructions": info.get("instructions", ""),
                    })
                else:
                    # Unknown credential — still list it
                    credentials.append({
                        "credential_name": env_var.replace("_", " ").title(),
                        "env_var_name": env_var,
                        "required": True,
                        "instructions": f"Set the {env_var} environment variable.",
                    })

        return credentials

    def get_credential_report(
        self,
        vertical_id: str,
        blueprint_env_vars: Optional[list[str]] = None,
    ) -> CredentialReport:
        """
        Generate a comprehensive report on credential status for a vertical.

        Args:
            vertical_id: The vertical ID.
            blueprint_env_vars: Additional env vars from the blueprint.

        Returns:
            CredentialReport with status of each credential.
        """
        required_creds = self.get_required_credentials(
            vertical_id, blueprint_env_vars
        )

        report = CredentialReport(vertical_id=vertical_id)

        for cred_info in required_creds:
            env_var = cred_info["env_var_name"]
            is_set = self.has_credential(vertical_id, env_var)

            # Also check environment (some creds are set at system level)
            if not is_set and os.environ.get(env_var, "").strip():
                is_set = True

            report.credentials.append(
                CredentialStatus(
                    env_var_name=env_var,
                    credential_name=cred_info["credential_name"],
                    is_set=is_set,
                    required=cred_info["required"],
                    instructions=cred_info.get("instructions", ""),
                )
            )

        return report

    def export_as_env(
        self,
        vertical_id: str,
    ) -> dict[str, str]:
        """
        Export all stored credentials for a vertical as env var dict.

        Useful for injecting into subprocess environments for agent launch.

        Returns:
            Dict of {ENV_VAR_NAME: decrypted_value}.
        """
        env_vars: dict[str, str] = {}

        if self._db is not None:
            try:
                result = (
                    self._db.client.table("vertical_credentials")
                    .select("env_var_name, encrypted_value")
                    .eq("vertical_id", vertical_id)
                    .eq("is_set", True)
                    .execute()
                )
                for row in result.data:
                    try:
                        value = decrypt_value(
                            row["encrypted_value"], self._master_key
                        )
                        env_vars[row["env_var_name"]] = value
                    except (InvalidToken, Exception):
                        continue
            except Exception as e:
                logger.warning(f"DB export failed: {e}")

        # Also check memory store
        for (vid, env_var), encrypted in self._memory_store.items():
            if vid == vertical_id and env_var not in env_vars:
                try:
                    env_vars[env_var] = decrypt_value(
                        encrypted, self._master_key
                    )
                except (InvalidToken, Exception):
                    continue

        return env_vars

    def inject_into_env(
        self,
        vertical_id: str,
    ) -> int:
        """
        Inject stored credentials into the current process environment.

        Returns the number of credentials injected.

        WARNING: This modifies os.environ. Use with care.
        """
        env_vars = self.export_as_env(vertical_id)
        for key, value in env_vars.items():
            os.environ[key] = value
        return len(env_vars)
