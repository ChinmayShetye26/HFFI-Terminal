"""Security helpers for the HFFI FastAPI service.

The project is a decision-support terminal that handles household financial
inputs. This module keeps security controls local and dependency-light:
HMAC-signed bearer tokens, role checks, rate limiting, safe audit logging, and
security headers. It is intentionally simple enough for a capstone prototype
while still matching production security patterns.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

security_logger = logging.getLogger("hffi.security")
security_logger.setLevel(logging.INFO)
security_logger.propagate = False
if not security_logger.handlers:
    handler = RotatingFileHandler(
        LOG_DIR / "security_audit.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    security_logger.addHandler(handler)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AuthenticatedUser(BaseModel):
    username: str
    role: str


bearer_scheme = HTTPBearer(auto_error=False)
_rate_buckets: dict[tuple[str, str], Deque[float]] = defaultdict(deque)


def auth_enabled() -> bool:
    return os.getenv("HFFI_AUTH_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def token_ttl_seconds() -> int:
    try:
        minutes = int(os.getenv("HFFI_TOKEN_TTL_MINUTES", "480"))
    except ValueError:
        minutes = 480
    return max(5, min(minutes, 24 * 60)) * 60


def allowed_origins() -> list[str]:
    raw = os.getenv(
        "HFFI_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def security_config() -> dict[str, Any]:
    return {
        "authEnabled": auth_enabled(),
        "tokenTtlMinutes": token_ttl_seconds() // 60,
        "allowedOrigins": allowed_origins(),
    }


def _secret_key() -> bytes:
    secret = os.getenv("HFFI_SECRET_KEY", "dev-only-hffi-secret-change-before-deployment")
    return secret.encode("utf-8")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def hash_password(password: str, *, iterations: int = 260_000) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def _verify_hash(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_raw, salt, expected = encoded.split("$", 3)
        iterations = int(iterations_raw)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return hmac.compare_digest(digest.hex(), expected)


def _configured_users() -> dict[str, dict[str, str]]:
    users: dict[str, dict[str, str]] = {}

    admin_username = os.getenv("HFFI_ADMIN_USERNAME", "admin")
    admin_password = os.getenv("HFFI_ADMIN_PASSWORD", "change-me-now")
    admin_hash = os.getenv("HFFI_ADMIN_PASSWORD_HASH", "")
    users[admin_username] = {
        "role": "admin",
        "password": admin_password,
        "password_hash": admin_hash,
    }

    viewer_username = os.getenv("HFFI_VIEWER_USERNAME", "")
    viewer_password = os.getenv("HFFI_VIEWER_PASSWORD", "")
    viewer_hash = os.getenv("HFFI_VIEWER_PASSWORD_HASH", "")
    if viewer_username and (viewer_password or viewer_hash):
        users[viewer_username] = {
            "role": "viewer",
            "password": viewer_password,
            "password_hash": viewer_hash,
        }

    analyst_username = os.getenv("HFFI_ANALYST_USERNAME", "")
    analyst_password = os.getenv("HFFI_ANALYST_PASSWORD", "")
    analyst_hash = os.getenv("HFFI_ANALYST_PASSWORD_HASH", "")
    if analyst_username and (analyst_password or analyst_hash):
        users[analyst_username] = {
            "role": "analyst",
            "password": analyst_password,
            "password_hash": analyst_hash,
        }

    return users


def authenticate_user(username: str, password: str) -> AuthenticatedUser | None:
    user = _configured_users().get(username)
    if not user:
        return None
    configured_hash = user.get("password_hash") or ""
    if configured_hash:
        ok = _verify_hash(password, configured_hash)
    else:
        ok = hmac.compare_digest(password, user.get("password", ""))
    if not ok:
        return None
    return AuthenticatedUser(username=username, role=user["role"])


def create_access_token(user: AuthenticatedUser) -> str:
    now = int(time.time())
    payload = {
        "sub": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + token_ttl_seconds(),
    }
    payload_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url(payload_raw)
    signature = hmac.new(_secret_key(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url(signature)}"


def decode_access_token(token: str) -> AuthenticatedUser:
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        expected = hmac.new(_secret_key(), payload_b64.encode("ascii"), hashlib.sha256).digest()
        supplied = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected, supplied):
            raise ValueError("bad signature")
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.") from exc
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication token expired.")
    return AuthenticatedUser(username=str(payload.get("sub", "")), role=str(payload.get("role", "viewer")))


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> AuthenticatedUser:
    if not auth_enabled():
        return AuthenticatedUser(username="local-dev", role="admin")
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return decode_access_token(credentials.credentials)


def require_roles(*roles: str) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    allowed = set(roles)

    def dependency(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions.")
        return user

    return dependency


def rate_limit(bucket: str, limit: int, window_seconds: int = 60) -> Callable[[Request, AuthenticatedUser], None]:
    def dependency(request: Request, user: AuthenticatedUser = Depends(get_current_user)) -> None:
        client = request.client.host if request.client else "unknown"
        key = (bucket, f"{client}:{user.username}")
        now = time.time()
        q = _rate_buckets[key]
        while q and now - q[0] > window_seconds:
            q.popleft()
        if len(q) >= limit:
            security_logger.warning(
                "rate_limit bucket=%s user=%s role=%s client=%s path=%s",
                bucket,
                user.username,
                user.role,
                client,
                request.url.path,
            )
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Try again shortly.")
        q.append(now)

    return dependency


def public_rate_limit(bucket: str, limit: int, window_seconds: int = 60) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        key = (bucket, client)
        now = time.time()
        q = _rate_buckets[key]
        while q and now - q[0] > window_seconds:
            q.popleft()
        if len(q) >= limit:
            security_logger.warning(
                "rate_limit bucket=%s client=%s path=%s",
                bucket,
                client,
                request.url.path,
            )
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Try again shortly.")
        q.append(now)

    return dependency


def audit_event(event: str, **fields: Any) -> None:
    safe_fields = " ".join(f"{key}={value}" for key, value in fields.items())
    security_logger.info("%s %s", event, safe_fields)


def payload_fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def security_headers() -> dict[str, str]:
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        "Cache-Control": "no-store",
    }
