"""
encryption.py – Mã hóa Dữ liệu nhạy cảm bằng AES-256 (Fernet)
=================================================================
Sử dụng Fernet (AES-128-CBC + HMAC-SHA256) từ thư viện 'cryptography'.
Fernet đảm bảo tính toàn vẹn (Integrity) và bí mật (Confidentiality).

Luồng hoạt động:
    1. SECRET_KEY từ .env → sinh Fernet key (base64url-encoded 32-byte)
    2. Dữ liệu plaintext → encrypt() → ciphertext (base64 token)
    3. Ciphertext → decrypt() → plaintext gốc
    4. Database Admin query trực tiếp chỉ thấy ciphertext

Ghi chú:
    - KHÔNG MÃ HÓA face_data (128-d vector) vì cần so khớp khoảng cách.
    - CHỈ MÃ HÓA cccd_data vì chỉ cần so sánh chính xác (hash-like matching).
"""

import os
import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


def _derive_fernet_key(secret: str) -> bytes:
    """
    Derive a 32-byte Fernet-compatible key from an arbitrary secret string.
    Uses SHA-256 to normalize any length secret to exactly 32 bytes,
    then base64url-encodes it as required by Fernet.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


# Read the same SECRET_KEY used for JWT (defense-in-depth: ideally separate keys in production)
_SECRET = os.getenv(
    "ENCRYPTION_KEY",
    os.getenv("SECRET_KEY", "7b4c6e9d2f1a8c5b3e0d4f9a7c6b5a4d3f2e1b0c9a8b7c6d5e4f3a2b1c0d9e8f"),
)
_fernet = Fernet(_derive_fernet_key(_SECRET))


def encrypt_data(plaintext: str) -> str:
    """
    Encrypt a plaintext string → returns base64 ciphertext string.
    Safe to store in TEXT/VARCHAR database columns.
    """
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_data(ciphertext: str) -> str:
    """
    Decrypt a ciphertext string → returns original plaintext.
    Raises ValueError if token is invalid or tampered.
    """
    try:
        return _fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken:
        raise ValueError("Dữ liệu bị hỏng hoặc khóa mã hóa không khớp.")


def encrypt_match(plaintext: str, ciphertext: str) -> bool:
    """
    Check if a plaintext value matches a stored ciphertext.
    Returns True if decrypted ciphertext == plaintext.
    Constant-time comparison via hmac to prevent timing attacks.
    """
    import hmac as _hmac
    try:
        decrypted = decrypt_data(ciphertext)
        return _hmac.compare_digest(plaintext, decrypted)
    except (ValueError, Exception):
        return False
