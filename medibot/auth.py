"""Login and signed session tokens.

The role is decided by the server at login and travels ONLY inside a signed JWT. On every
protected call the server verifies the signature and reads the role from the token — never from
the request body or a header the client can edit. (A peer submission was rejected for exactly
that gap: role kept in localStorage and resent by the browser.)
"""

from __future__ import annotations

import hashlib
import hmac
import time

import jwt

from medibot.config import JWT_HOURS, JWT_SECRET, ROLE_COLLECTIONS

ALGORITHM = "HS256"
PBKDF2_ROUNDS = 100_000

# Demo accounts, one per role (passwords are documented in the README; only hashes are kept in memory).
DEMO_ACCOUNTS: dict[str, tuple[str, str]] = {
    "dr.mehta": ("doctor", "doctor123"),
    "nurse.priya": ("nurse", "nurse123"),
    "billing.ravi": ("billing_executive", "billing123"),
    "tech.anand": ("technician", "tech123"),
    "admin.sys": ("admin", "admin123"),
}


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ROUNDS).hex()


_USERS: dict[str, dict[str, str]] = {
    username: {"role": role, "salt": f"medbud:{username}", "hash": _hash(password, f"medbud:{username}")}
    for username, (role, password) in DEMO_ACCOUNTS.items()
}


class AuthError(Exception):
    pass


def authenticate(username: str, password: str) -> str:
    """Return the user's role if the credentials are right; raise AuthError otherwise."""
    user = _USERS.get(username)
    if user is None or not hmac.compare_digest(_hash(password, user["salt"]), user["hash"]):
        raise AuthError("invalid username or password")
    return user["role"]


def issue_token(username: str, role: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "role": role, "iat": now, "exp": now + JWT_HOURS * 3600}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """Decode + verify signature and expiry. Returns the claims; raises AuthError if anything is off."""
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("session expired, please log in again") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("invalid session token") from exc
    if claims.get("role") not in ROLE_COLLECTIONS:
        raise AuthError("token carries an unknown role")
    return claims
