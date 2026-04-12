"""
auth.py – Core Authentication Module (Security Hardened)
=========================================================
Changes from original:
    1. JWT Token now set as HttpOnly+SameSite Cookie (not returned in JSON body)
    2. get_current_user reads token from Cookie instead of Authorization header
    3. Added set_auth_cookie() and clear_auth_cookie() helpers
    4. Added log_audit() helper for immutable audit logging
    5. Password hashing remains PBKDF2-SHA256 (unchanged)
"""

import os
import base64
import hashlib
import hmac
import secrets
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from .database import get_db
from . import models

# Secret key to encode the JWT token
SECRET_KEY = os.getenv("SECRET_KEY", "7b4c6e9d2f1a8c5b3e0d4f9a7c6b5a4d3f2e1b0c9a8b7c6d5e4f3a2b1c0d9e8f")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
PASSWORD_HASH_SCHEME = "pbkdf2_sha256"
PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "390000"))
SALT_SIZE = 16

# Cookie configuration
COOKIE_NAME = "tax_session"
COOKIE_MAX_AGE = ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # True in production (HTTPS)
COOKIE_SAMESITE = "lax"
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", None)  # None = current domain only

# Password recovery configuration
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "30"))
FRONTEND_RESET_URL = os.getenv("FRONTEND_RESET_URL", "http://localhost:3000/pages/reset-password.html")
PASSWORD_OUTBOX_PATH = os.getenv(
    "PASSWORD_OUTBOX_PATH",
    str(Path(__file__).resolve().parents[1] / ".otp_outbox.log"),
)
SMTP_ENABLED = os.getenv("SMTP_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply-taxinspector@gdt.gov.vn")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
ALLOW_OUTBOX_FALLBACK = os.getenv("ALLOW_OUTBOX_FALLBACK", "true").lower() == "true"


# =============================================================================
# PASSWORD HASHING (unchanged)
# =============================================================================

def _verify_pbkdf2_password(plain_password: str, hashed_password: str) -> bool:
    try:
        scheme, iterations_str, salt_b64, digest_b64 = hashed_password.split("$", 3)
        if scheme != PASSWORD_HASH_SCHEME:
            return False

        iterations = int(iterations_str)
        salt = base64.b64decode(salt_b64.encode("ascii"), validate=True)
        stored_digest = base64.b64decode(digest_b64.encode("ascii"), validate=True)
        candidate_digest = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(candidate_digest, stored_digest)
    except Exception:
        return False


def _verify_legacy_bcrypt_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password.startswith("$2"):
        return False

    try:
        import bcrypt

        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if hashed_password.startswith(f"{PASSWORD_HASH_SCHEME}$"):
        return _verify_pbkdf2_password(plain_password, hashed_password)

    # Backward compatibility for accounts created before hash migration.
    return _verify_legacy_bcrypt_password(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    salt = os.urandom(SALT_SIZE)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"{PASSWORD_HASH_SCHEME}${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def validate_new_password(password: str):
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải có ít nhất 8 ký tự.")
    if len(password) > 128:
        raise HTTPException(status_code=400, detail="Mật khẩu mới không hợp lệ.")


# =============================================================================
# JWT TOKEN (unchanged logic, new cookie delivery)
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# =============================================================================
# HTTPONLY COOKIE MANAGEMENT
# =============================================================================

def set_auth_cookie(response: Response, token: str):
    """Set JWT token as an HttpOnly cookie on the response."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,          # JS cannot read this cookie (XSS-proof)
        secure=COOKIE_SECURE,   # Only sent over HTTPS (set True in production)
        samesite=COOKIE_SAMESITE,
        max_age=COOKIE_MAX_AGE,
        path="/",
        domain=COOKIE_DOMAIN,
    )


def clear_auth_cookie(response: Response):
    """Clear the auth cookie (logout)."""
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        domain=COOKIE_DOMAIN,
    )


# =============================================================================
# GET CURRENT USER (reads from Cookie instead of Authorization header)
# =============================================================================

def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> models.User:
    """
    Decode JWT token from HttpOnly cookie and return the corresponding User.
    Falls back to Authorization header for API tools like Swagger UI.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn.",
    )

    # Priority 1: HttpOnly Cookie
    token = request.cookies.get(COOKIE_NAME)

    # Priority 2: Authorization Bearer header (for Swagger UI / API tools)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        badge_id: str = payload.get("sub")
        if badge_id is None:
            raise credentials_exception
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
        )
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.badge_id == badge_id).first()
    if user is None:
        raise credentials_exception
    return user


# =============================================================================
# AUDIT LOGGING
# =============================================================================

def log_audit(
    db: Session,
    action: str,
    request: Request = None,
    user_id: int = None,
    badge_id: str = None,
    detail: str = None,
):
    """
    Record an immutable audit log entry.
    Actions: LOGIN_SUCCESS, LOGIN_FAILED, LOGOUT, REGISTER,
             FACE_SETUP, FACE_RESET, FACE_LOGIN_SUCCESS, FACE_LOGIN_FAILED,
             CCCD_SETUP, CCCD_RESET, CCCD_LOGIN_SUCCESS, CCCD_LOGIN_FAILED,
             SIGNATURE_SETUP, SIGNATURE_RESET, SIGNATURE_LOGIN_SUCCESS, SIGNATURE_LOGIN_FAILED,
             PHONE_UPDATE
    """
    ip = None
    ua = None
    if request:
        ip = request.client.host if request.client else None
        ua = request.headers.get("user-agent", "")[:500]

    entry = models.AuditLog(
        user_id=user_id,
        badge_id=badge_id,
        action=action,
        detail=detail,
        ip_address=ip,
        user_agent=ua,
    )
    db.add(entry)
    db.commit()


# =============================================================================
# PASSWORD RESET (forgot/reset password)
# =============================================================================

def generate_password_reset_token() -> str:
    """Generate a URL-safe random token for password reset."""
    return secrets.token_urlsafe(32)


def hash_password_reset_token(token: str) -> str:
    """Hash token before storing in DB (never store raw reset token)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def build_password_reset_link(token: str) -> str:
    sep = "&" if "?" in FRONTEND_RESET_URL else "?"
    return f"{FRONTEND_RESET_URL}{sep}token={token}"


def get_password_reset_expiry() -> datetime:
    return datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)


def _append_password_outbox(email: str, badge_id: str, reset_link: str, expires_at: datetime):
    outbox_file = Path(PASSWORD_OUTBOX_PATH)
    outbox_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"[{datetime.utcnow().isoformat()}] PASSWORD_RESET",
        f"email={email}",
        f"badge_id={badge_id}",
        f"expires_utc={expires_at.isoformat()}",
        f"reset_link={reset_link}",
        "-",
    ]
    with outbox_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _try_send_reset_email(email: str, reset_link: str) -> bool:
    if not (SMTP_ENABLED and SMTP_HOST):
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = email
    msg["Subject"] = "TaxInspector - Dat lai mat khau"
    msg.set_content(
        "Yeu cau dat lai mat khau cua ban:\n\n"
        f"{reset_link}\n\n"
        f"Lien ket nay het han sau {RESET_TOKEN_EXPIRE_MINUTES} phut."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        if SMTP_USE_TLS:
            server.starttls()
        if SMTP_USER:
            server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    return True


def deliver_password_reset(email: str, badge_id: str, token: str, expires_at: datetime) -> str:
    """
    Deliver reset link by SMTP if configured, otherwise write to local outbox file.
    Returns channel name: 'smtp' or 'outbox'.
    """
    reset_link = build_password_reset_link(token)

    try:
        if _try_send_reset_email(email, reset_link):
            return "smtp"

        if SMTP_ENABLED and not ALLOW_OUTBOX_FALLBACK:
            raise RuntimeError("SMTP delivery unavailable and outbox fallback disabled.")
    except Exception:
        if not ALLOW_OUTBOX_FALLBACK:
            raise

    _append_password_outbox(email, badge_id, reset_link, expires_at)
    return "outbox"


# =============================================================================
# TEMPORARY TOKEN (for 2FA intermediate step: face/cccd → signature)
# =============================================================================

TEMP_TOKEN_EXPIRE_MINUTES = 5  # 5 minutes to complete signature step

def create_temp_token(badge_id: str) -> str:
    """Create a short-lived JWT for the 2FA intermediate step."""
    expire = datetime.utcnow() + timedelta(minutes=TEMP_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": badge_id, "type": "2fa_temp", "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def verify_temp_token(token: str) -> str:
    """Verify a temp token and return the badge_id. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "2fa_temp":
            raise HTTPException(status_code=401, detail="Token không hợp lệ.")
        badge_id = payload.get("sub")
        if not badge_id:
            raise HTTPException(status_code=401, detail="Token không hợp lệ.")
        return badge_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token xác thực đã hết hạn (5 phút). Vui lòng thử lại.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ.")


# =============================================================================
# SIGNATURE COMPARISON (dHash - Fuzzy Matching)
# =============================================================================

def _preprocess_signature(img):
    from PIL import ImageOps, Image, ImageFilter
    # Invert to make background black (0) and strokes white (255)
    inv = ImageOps.invert(img)
    bbox = inv.getbbox()
    if bbox:
        # Pad slightly
        padding = 20
        width, height = img.size
        # Unpack bbox (left, upper, right, lower)
        left = max(0, bbox[0] - padding)
        upper = max(0, bbox[1] - padding)
        right = min(width, bbox[2] + padding)
        lower = min(height, bbox[3] + padding)
        
        c = img.crop((left, upper, right, lower))
        
        # Make squarely padded to prevent aspect ratio distortion during hash resize
        cw, ch = c.size
        size = max(cw, ch)
        sq = Image.new("L", (size, size), "WHITE")
        sq.paste(c, ((size - cw)//2, (size - ch)//2))
        
        # Blur to thicken strokes, making dHash more robust to handwriting jitter
        sq = sq.filter(ImageFilter.GaussianBlur(1.5))
        return sq
    return img

def compare_signatures(stored_b64: str, candidate_b64: str, threshold: int = 16) -> tuple:
    import io
    import base64
    from PIL import Image, ImageChops
    import imagehash

    try:
        stored_bytes = base64.b64decode(stored_b64)
        candidate_bytes = base64.b64decode(candidate_b64)

        img_stored = Image.open(io.BytesIO(stored_bytes)).convert("RGBA")
        img_candidate = Image.open(io.BytesIO(candidate_bytes)).convert("RGBA")

        # Paste over white background
        bg_stored = Image.new("RGBA", img_stored.size, "WHITE")
        bg_stored.paste(img_stored, (0, 0), img_stored)
        img_stored_L = bg_stored.convert('L')
        
        bg_cand = Image.new("RGBA", img_candidate.size, "WHITE")
        bg_cand.paste(img_candidate, (0, 0), img_candidate)
        img_candidate_L = bg_cand.convert('L')

        # Crop to square bounding box and blur
        img_stored_final = _preprocess_signature(img_stored_L)
        img_candidate_final = _preprocess_signature(img_candidate_L)

        # Hash: use dHash (difference hash) which is better for strokes, size 8 = 64bit
        hash_stored = imagehash.dhash(img_stored_final, hash_size=8)
        hash_candidate = imagehash.dhash(img_candidate_final, hash_size=8)

        distance = hash_stored - hash_candidate

        return distance <= threshold, distance
    except Exception as e:
        print("[Auth Error] compare_signatures error:", e)
        return False, -1
