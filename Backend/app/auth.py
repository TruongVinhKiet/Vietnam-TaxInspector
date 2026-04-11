import os
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional
import jwt

# Secret key to encode the JWT token
SECRET_KEY = os.getenv("SECRET_KEY", "7b4c6e9d2f1a8c5b3e0d4f9a7c6b5a4d3f2e1b0c9a8b7c6d5e4f3a2b1c0d9e8f")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
PASSWORD_HASH_SCHEME = "pbkdf2_sha256"
PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "390000"))
SALT_SIZE = 16


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

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
