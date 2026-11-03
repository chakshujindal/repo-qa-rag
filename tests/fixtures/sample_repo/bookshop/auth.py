"""Password hashing and signed session tokens."""

import hashlib
import hmac
import os
import time

PBKDF2_ROUNDS = 310_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-SHA256 and a random salt, for storage in the users table."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a login attempt against the stored salt$digest string in constant time."""
    salt_hex, digest_hex = stored.split("$")
    candidate = hash_password(password, bytes.fromhex(salt_hex)).split("$")[1]
    return hmac.compare_digest(candidate, digest_hex)


def issue_token(user_id: int, secret: bytes, ttl_seconds: int = 3600) -> str:
    """Create a session token: user id and expiry, signed with HMAC so it cannot be forged."""
    expires = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{expires}"
    signature = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def verify_token(token: str, secret: bytes) -> int | None:
    """Return the user id if the token signature is valid and it has not expired."""
    user_id, expires, signature = token.split(":")
    expected = hmac.new(secret, f"{user_id}:{expires}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected) or int(expires) < time.time():
        return None
    return int(user_id)
