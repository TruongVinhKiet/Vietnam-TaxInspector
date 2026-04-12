"""
routers/auth.py – Authentication Router (Security Hardened + 2FA Signature)
=============================================================================
Features:
    1. All login endpoints → set HttpOnly cookie (no JSON token response)
    2. /logout endpoint to clear cookie
    3. CCCD data encrypted with AES-256 before storing
    4. Audit logging on all auth events
    5. Rate limiting via @limiter.limit() decorator
    6. Digital Signature 2FA: face/cccd login → temp_token → signature verify → full session
    7. Signature setup/reset with AES-256 encrypted image storage
    8. Phone number update
"""

from datetime import datetime, timedelta
import base64
import binascii
import json
import math
import re
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from ..database import get_db
from .. import schemas, models, auth
from ..encryption import encrypt_data, decrypt_data, encrypt_match
from ..security import limiter

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# --- Face matching config ---
FACE_TOLERANCE = 0.6  # Euclidean distance threshold (0.6 is standard for face-api.js)
AVATAR_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
AVATAR_ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg"}


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _find_face_match(db: Session, descriptor: List[float], exclude_user_id: int = None):
    """Find a user whose stored face matches the given descriptor within tolerance."""
    users = db.query(models.User).filter(
        models.User.face_verified == True,
        models.User.face_data.isnot(None),
    ).all()

    best_match = None
    best_distance = float('inf')

    for user in users:
        if exclude_user_id and user.id == exclude_user_id:
            continue
        try:
            stored = json.loads(user.face_data)
            dist = _euclidean_distance(stored, descriptor)
            if dist < best_distance:
                best_distance = dist
                best_match = user
        except (json.JSONDecodeError, TypeError):
            continue

    if best_match and best_distance <= FACE_TOLERANCE:
        return best_match, best_distance
    return None, best_distance


def _create_and_set_cookie(user: models.User, response: Response) -> str:
    """Create JWT token and set it as HttpOnly cookie. Returns token for internal use."""
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.badge_id, "role": user.role},
        expires_delta=access_token_expires,
    )
    auth.set_auth_cookie(response, access_token)
    return access_token


def _strip_base64_prefix(data: str) -> str:
    """Remove 'data:image/png;base64,' prefix if present."""
    if "," in data:
        return data.split(",", 1)[1]
    return data


def _decode_avatar_bytes(avatar_image: str) -> bytes:
    """
    Decode and validate avatar payload.
    Accepts either raw Base64 or data URL format.
    """
    if not avatar_image:
        raise HTTPException(status_code=400, detail="Ảnh đại diện không hợp lệ.")

    data = avatar_image.strip()

    if data.startswith("data:"):
        try:
            header, payload = data.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Dữ liệu ảnh không hợp lệ.")

        if ";base64" not in header:
            raise HTTPException(status_code=400, detail="Dữ liệu ảnh không đúng định dạng Base64.")

        mime_type = header[5:].split(";", 1)[0].lower()
        if mime_type not in AVATAR_ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh PNG hoặc JPEG.")

        data = payload.strip()

    try:
        image_bytes = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Ảnh tải lên không phải Base64 hợp lệ.")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Ảnh đại diện không hợp lệ.")

    if len(image_bytes) > AVATAR_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Dung lượng ảnh vượt quá giới hạn 5MB.")

    return image_bytes


def _normalize_avatar_data_url(image_bytes: bytes) -> str:
    """
    Normalize avatar to a square JPEG for consistent rendering and safer payload size.
    Returns data URL string.
    """
    from io import BytesIO
    from PIL import Image, ImageOps

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Tệp ảnh không hợp lệ hoặc bị hỏng.")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
            width, height = image.size

            if min(width, height) < 80:
                raise HTTPException(status_code=400, detail="Ảnh quá nhỏ. Vui lòng chọn ảnh rõ hơn.")

            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))

            resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            image = image.resize((320, 320), resampling)

            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            normalized_bytes = output.getvalue()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể xử lý ảnh đại diện.")

    if len(normalized_bytes) > AVATAR_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Ảnh sau xử lý vẫn vượt quá giới hạn 5MB.")

    normalized_b64 = base64.b64encode(normalized_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{normalized_b64}"


# =============================================================================
# REGISTER (unchanged logic, added audit log)
# =============================================================================

@router.post("/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(
    request: Request,
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
):
    # Check if badge_id or email already exists
    db_user_badge = db.query(models.User).filter(models.User.badge_id == user.badge_id).first()
    if db_user_badge:
        raise HTTPException(status_code=400, detail="Mã số cán bộ đã được đăng ký trong hệ thống.")
    
    db_user_email = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email công vụ đã được đăng ký.")

    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        badge_id=user.badge_id,
        full_name=user.full_name,
        department=user.department,
        email=user.email,
        phone=user.phone,
        password_hash=hashed_password,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Audit log
    auth.log_audit(db, "REGISTER", request, user_id=db_user.id, badge_id=db_user.badge_id)

    return db_user


# =============================================================================
# LOGIN (password) – Returns HttpOnly cookie directly (no 2FA for password login)
# =============================================================================

@router.post("/login")
@limiter.limit("5/minute")
def login(
    request: Request,
    response: Response,
    user_credentials: schemas.LoginRequest,
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(models.User.badge_id == user_credentials.badge_id).first()
    
    if not user:
        auth.log_audit(db, "LOGIN_FAILED", request, badge_id=user_credentials.badge_id,
                       detail="Tài khoản không tồn tại.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản cán bộ không tồn tại.",
        )
    if not auth.verify_password(user_credentials.password, user.password_hash):
        auth.log_audit(db, "LOGIN_FAILED", request, user_id=user.id, badge_id=user.badge_id,
                       detail="Sai mật khẩu.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản hoặc mật khẩu không chính xác.",
        )
    
    _create_and_set_cookie(user, response)
    auth.log_audit(db, "LOGIN_SUCCESS", request, user_id=user.id, badge_id=user.badge_id)

    return {"success": True, "message": "Đăng nhập thành công."}


# =============================================================================
# PASSWORD RECOVERY (forgot/reset/change)
# =============================================================================

@router.post("/forgot-password", response_model=schemas.GenericMessage)
@limiter.limit("3/minute")
def forgot_password(
    request: Request,
    body: schemas.ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Create a one-time password reset token.
    Always returns a generic message to avoid account enumeration.
    """
    generic_message = (
        "Nếu email tồn tại, hướng dẫn đặt lại mật khẩu đã được gửi. "
        "Trong môi trường local, vui lòng kiểm tra file Backend/.otp_outbox.log."
    )

    email = body.email.strip().lower()
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        auth.log_audit(db, "PASSWORD_FORGOT_REQUEST", request, detail="Email not found")
        return {"success": True, "message": generic_message}

    now = datetime.utcnow()

    # Invalidate previous active reset tokens for this user.
    db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.user_id == user.id,
        models.PasswordResetToken.used == False,
    ).update(
        {"used": True, "used_at": now},
        synchronize_session=False,
    )

    raw_token = auth.generate_password_reset_token()
    token_hash = auth.hash_password_reset_token(raw_token)
    expires_at = auth.get_password_reset_expiry()

    reset_entry = models.PasswordResetToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        used=False,
    )
    db.add(reset_entry)
    db.commit()

    try:
        channel = auth.deliver_password_reset(user.email, user.badge_id, raw_token, expires_at)
    except Exception as exc:
        auth.log_audit(
            db,
            "PASSWORD_FORGOT_REQUEST",
            request,
            user_id=user.id,
            badge_id=user.badge_id,
            detail=f"Reset delivery failed: {type(exc).__name__}",
        )
        return {"success": True, "message": generic_message}

    auth.log_audit(
        db,
        "PASSWORD_FORGOT_REQUEST",
        request,
        user_id=user.id,
        badge_id=user.badge_id,
        detail=f"Reset link delivered via {channel}",
    )
    return {"success": True, "message": generic_message}


@router.post("/reset-password", response_model=schemas.GenericMessage)
@limiter.limit("5/minute")
def reset_password(
    request: Request,
    body: schemas.ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    if body.new_password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Mật khẩu xác nhận không khớp.")

    auth.validate_new_password(body.new_password)

    token_hash = auth.hash_password_reset_token(body.token)

    reset_entry = db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.token_hash == token_hash,
        models.PasswordResetToken.used == False,
    ).first()

    if not reset_entry:
        auth.log_audit(db, "PASSWORD_RESET_FAILED", request, detail="Invalid or used reset token")
        raise HTTPException(status_code=400, detail="Liên kết đặt lại mật khẩu không hợp lệ hoặc đã được sử dụng.")

    now = datetime.now(reset_entry.expires_at.tzinfo) if reset_entry.expires_at.tzinfo else datetime.utcnow()

    if reset_entry.expires_at < now:
        reset_entry.used = True
        reset_entry.used_at = now
        db.commit()
        auth.log_audit(db, "PASSWORD_RESET_FAILED", request, detail="Expired reset token")
        raise HTTPException(status_code=400, detail="Liên kết đặt lại mật khẩu đã hết hạn.")

    user = db.query(models.User).filter(models.User.id == reset_entry.user_id).first()
    if not user:
        reset_entry.used = True
        reset_entry.used_at = now
        db.commit()
        auth.log_audit(db, "PASSWORD_RESET_FAILED", request, detail="User missing for token")
        raise HTTPException(status_code=400, detail="Tài khoản không tồn tại hoặc đã bị xóa.")

    if auth.verify_password(body.new_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải khác mật khẩu hiện tại.")

    user.password_hash = auth.get_password_hash(body.new_password)
    reset_entry.used = True
    reset_entry.used_at = now

    # Revoke all remaining active reset tokens after password was changed.
    db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.user_id == user.id,
        models.PasswordResetToken.used == False,
    ).update(
        {"used": True, "used_at": now},
        synchronize_session=False,
    )

    db.commit()
    auth.log_audit(db, "PASSWORD_RESET_SUCCESS", request, user_id=user.id, badge_id=user.badge_id)
    return {"success": True, "message": "Đặt lại mật khẩu thành công. Bạn có thể đăng nhập bằng mật khẩu mới."}


@router.post("/change-password", response_model=schemas.GenericMessage)
@limiter.limit("10/minute")
def change_password(
    request: Request,
    body: schemas.ChangePasswordRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    if body.new_password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Mật khẩu xác nhận không khớp.")

    auth.validate_new_password(body.new_password)

    if not auth.verify_password(body.current_password, current_user.password_hash):
        auth.log_audit(
            db,
            "PASSWORD_CHANGE_FAILED",
            request,
            user_id=current_user.id,
            badge_id=current_user.badge_id,
            detail="Current password mismatch",
        )
        raise HTTPException(status_code=401, detail="Mật khẩu hiện tại không chính xác.")

    if auth.verify_password(body.new_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải khác mật khẩu hiện tại.")

    current_user.password_hash = auth.get_password_hash(body.new_password)
    db.commit()

    auth.log_audit(db, "PASSWORD_CHANGE_SUCCESS", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Đổi mật khẩu thành công."}


# =============================================================================
# LOGOUT – Clear cookie
# =============================================================================

@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """Clear auth cookie to end the session."""
    try:
        user = auth.get_current_user(request, db)
        auth.log_audit(db, "LOGOUT", request, user_id=user.id, badge_id=user.badge_id)
    except Exception:
        auth.log_audit(db, "LOGOUT", request, detail="Anonymous/expired session")

    auth.clear_auth_cookie(response)
    return {"success": True, "message": "Đăng xuất thành công."}


# =============================================================================
# ME (unchanged behavior)
# =============================================================================

@router.get("/me", response_model=schemas.UserResponse)
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    """Return the profile of the currently authenticated officer."""
    return current_user


# =============================================================================
# UPDATE PHONE
# =============================================================================

@router.put("/update-phone")
def update_phone(
    request: Request,
    body: schemas.UpdatePhoneRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Update the phone number of the currently authenticated user."""
    # Validate phone format (Vietnamese phone: 10-11 digits starting with 0)
    phone_clean = re.sub(r"[\s\-\.]", "", body.phone)
    if not re.match(r"^0\d{9,10}$", phone_clean):
        raise HTTPException(status_code=400, detail="Số điện thoại không hợp lệ. Phải bắt đầu bằng 0 và có 10-11 chữ số.")

    current_user.phone = phone_clean
    db.commit()

    auth.log_audit(db, "PHONE_UPDATE", request, user_id=current_user.id, badge_id=current_user.badge_id,
                   detail=f"SĐT cập nhật: {phone_clean[:4]}****")
    return {"success": True, "message": "Cập nhật số điện thoại thành công.", "phone": phone_clean}


@router.put("/update-avatar")
@limiter.limit("10/minute")
def update_avatar(
    request: Request,
    body: schemas.UpdateAvatarRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Update avatar for the authenticated user (PNG/JPEG, max 5MB)."""
    try:
        image_bytes = _decode_avatar_bytes(body.avatar_image)
        normalized_avatar = _normalize_avatar_data_url(image_bytes)
    except HTTPException as exc:
        auth.log_audit(
            db,
            "AVATAR_UPDATE_FAILED",
            request,
            user_id=current_user.id,
            badge_id=current_user.badge_id,
            detail=f"Validation failed: {exc.detail}",
        )
        raise
    except Exception as exc:
        auth.log_audit(
            db,
            "AVATAR_UPDATE_FAILED",
            request,
            user_id=current_user.id,
            badge_id=current_user.badge_id,
            detail=f"Unexpected error: {type(exc).__name__}",
        )
        raise HTTPException(status_code=500, detail="Không thể cập nhật ảnh đại diện lúc này.")

    current_user.avatar_data = normalized_avatar
    db.commit()

    auth.log_audit(
        db,
        "AVATAR_UPDATE_SUCCESS",
        request,
        user_id=current_user.id,
        badge_id=current_user.badge_id,
        detail=f"Avatar updated ({len(normalized_avatar)} chars)",
    )

    return {
        "success": True,
        "message": "Cập nhật ảnh đại diện thành công.",
        "avatar_data": normalized_avatar,
    }


# =============================================================================
# BIOMETRIC FACE ENDPOINTS
# =============================================================================

@router.post("/setup-face")
def setup_face(
    request: Request,
    body: schemas.BiometricSetupRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Store face descriptor for the authenticated user. Enforces one-face-one-account."""
    if len(body.descriptor) != 128:
        raise HTTPException(status_code=400, detail="Vector khuôn mặt phải có đúng 128 chiều.")

    match, dist = _find_face_match(db, body.descriptor, exclude_user_id=current_user.id)
    if match:
        auth.log_audit(db, "FACE_SETUP_REJECTED", request, user_id=current_user.id,
                       badge_id=current_user.badge_id, detail="Khuôn mặt đã đăng ký trên tài khoản khác.")
        raise HTTPException(
            status_code=409,
            detail="Khuôn mặt này đã được đăng ký trên tài khoản khác. Mỗi khuôn mặt chỉ được thiết lập cho 1 tài khoản.",
        )

    current_user.face_data = json.dumps(body.descriptor)
    current_user.face_verified = True
    db.commit()

    auth.log_audit(db, "FACE_SETUP", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Thiết lập xác minh khuôn mặt thành công."}


@router.delete("/reset-face")
def reset_face(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Reset face verification for the authenticated user."""
    current_user.face_data = None
    current_user.face_verified = False
    db.commit()
    auth.log_audit(db, "FACE_RESET", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Đã xóa dữ liệu xác minh khuôn mặt."}


@router.post("/login-face")
@limiter.limit("5/minute")
def login_face(
    request: Request,
    response: Response,
    body: schemas.BiometricLoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticate by face. If user has a signature set up, returns temp_token for 2FA step 2.
    Otherwise grants full session immediately.
    """
    if len(body.descriptor) != 128:
        raise HTTPException(status_code=400, detail="Vector khuôn mặt phải có đúng 128 chiều.")

    match, dist = _find_face_match(db, body.descriptor)
    if not match:
        auth.log_audit(db, "FACE_LOGIN_FAILED", request,
                       detail=f"Không tìm thấy khuôn mặt khớp (Khoảng cách gần nhất: {dist:.4f}).")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Không tìm thấy khuôn mặt khớp trong hệ thống. Vui lòng thiết lập xác minh khuôn mặt trước.",
        )

    # 2FA check: does user have a verified signature?
    if match.signature_verified and match.signature_data:
        temp_token = auth.create_temp_token(match.badge_id)
        auth.log_audit(db, "FACE_LOGIN_STEP1", request, user_id=match.id, badge_id=match.badge_id,
                       detail=f"Distance: {dist:.4f}. Chuyển sang bước 2 (chữ ký).")
        return {
            "success": True,
            "require_signature": True,
            "temp_token": temp_token,
            "message": "Xác thực khuôn mặt thành công. Vui lòng ký xác nhận để hoàn tất đăng nhập.",
        }

    # No signature → grant full session
    _create_and_set_cookie(match, response)
    auth.log_audit(db, "FACE_LOGIN_SUCCESS", request, user_id=match.id, badge_id=match.badge_id,
                   detail=f"Distance: {dist:.4f}")
    return {"success": True, "require_signature": False, "message": "Xác thực khuôn mặt thành công."}


# =============================================================================
# BIOMETRIC CCCD ENDPOINTS (with AES-256 encryption)
# =============================================================================

@router.post("/setup-cccd")
def setup_cccd(
    request: Request,
    body: schemas.CccdSetupRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Store CCCD number (encrypted) for the authenticated user."""
    cccd_users = db.query(models.User).filter(
        models.User.cccd_verified == True,
        models.User.cccd_data.isnot(None),
        models.User.id != current_user.id,
    ).all()

    for u in cccd_users:
        if encrypt_match(body.cccd_number, u.cccd_data):
            auth.log_audit(db, "CCCD_SETUP_REJECTED", request, user_id=current_user.id,
                           badge_id=current_user.badge_id, detail="CCCD đã đăng ký trên tài khoản khác.")
            raise HTTPException(
                status_code=409,
                detail="Số CCCD này đã được đăng ký trên tài khoản khác. Mỗi CCCD chỉ được thiết lập cho 1 tài khoản.",
            )

    current_user.cccd_data = encrypt_data(body.cccd_number)
    current_user.cccd_verified = True
    db.commit()

    auth.log_audit(db, "CCCD_SETUP", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Xác minh CCCD thành công."}


@router.delete("/reset-cccd")
def reset_cccd(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Reset CCCD verification for the authenticated user."""
    current_user.cccd_data = None
    current_user.cccd_verified = False
    db.commit()
    auth.log_audit(db, "CCCD_RESET", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Đã xóa dữ liệu xác minh CCCD."}


@router.post("/login-cccd")
@limiter.limit("5/minute")
def login_cccd(
    request: Request,
    response: Response,
    body: schemas.CccdLoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticate by CCCD. If user has a signature set up, returns temp_token for 2FA step 2.
    Otherwise grants full session immediately.
    """
    cccd_users = db.query(models.User).filter(
        models.User.cccd_verified == True,
        models.User.cccd_data.isnot(None),
    ).all()

    matched_user = None
    for u in cccd_users:
        if encrypt_match(body.cccd_number, u.cccd_data):
            matched_user = u
            break

    if not matched_user:
        auth.log_audit(db, "CCCD_LOGIN_FAILED", request, detail="CCCD không tìm thấy.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Số CCCD không tìm thấy trong hệ thống. Vui lòng xác minh CCCD trong trang Tài khoản trước.",
        )

    # 2FA check: does user have a verified signature?
    if matched_user.signature_verified and matched_user.signature_data:
        temp_token = auth.create_temp_token(matched_user.badge_id)
        auth.log_audit(db, "CCCD_LOGIN_STEP1", request, user_id=matched_user.id,
                       badge_id=matched_user.badge_id, detail="Chuyển sang bước 2 (chữ ký).")
        return {
            "success": True,
            "require_signature": True,
            "temp_token": temp_token,
            "message": "Xác thực CCCD thành công. Vui lòng ký xác nhận để hoàn tất đăng nhập.",
        }

    # No signature → grant full session
    _create_and_set_cookie(matched_user, response)
    auth.log_audit(db, "CCCD_LOGIN_SUCCESS", request, user_id=matched_user.id, badge_id=matched_user.badge_id)
    return {"success": True, "require_signature": False, "message": "Xác thực CCCD thành công."}


# =============================================================================
# DIGITAL SIGNATURE ENDPOINTS
# =============================================================================

@router.post("/setup-signature")
def setup_signature(
    request: Request,
    body: schemas.SignatureSetupRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Store signature image (encrypted) for the authenticated user."""
    raw_b64 = _strip_base64_prefix(body.signature_image)

    if len(raw_b64) < 100:
        raise HTTPException(status_code=400, detail="Chữ ký quá đơn giản. Vui lòng ký tên đầy đủ hơn.")

    # Encrypt the Base64 signature image before storing
    current_user.signature_data = encrypt_data(raw_b64)
    current_user.signature_verified = True
    db.commit()

    auth.log_audit(db, "SIGNATURE_SETUP", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Thiết lập chữ ký số thành công."}


@router.delete("/reset-signature")
def reset_signature(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Reset signature for the authenticated user."""
    current_user.signature_data = None
    current_user.signature_verified = False
    db.commit()
    auth.log_audit(db, "SIGNATURE_RESET", request, user_id=current_user.id, badge_id=current_user.badge_id)
    return {"success": True, "message": "Đã xóa dữ liệu chữ ký số."}


@router.post("/login-signature")
@limiter.limit("10/minute")
def login_signature(
    request: Request,
    response: Response,
    body: schemas.SignatureLoginRequest,
    db: Session = Depends(get_db),
):
    """
    2FA Step 2: Verify signature against stored signature. Requires temp_token from Step 1.
    On success, grants full HttpOnly cookie session.
    """
    # Verify temp token
    badge_id = auth.verify_temp_token(body.temp_token)

    user = db.query(models.User).filter(models.User.badge_id == badge_id).first()
    if not user or not user.signature_verified or not user.signature_data:
        raise HTTPException(status_code=400, detail="Tài khoản chưa thiết lập chữ ký số.")

    # Decrypt stored signature
    try:
        stored_sig_b64 = decrypt_data(user.signature_data)
    except ValueError:
        raise HTTPException(status_code=500, detail="Không thể giải mã chữ ký lưu trữ.")

    # Compare signatures using perceptual hashing
    candidate_b64 = _strip_base64_prefix(body.signature_image)
    is_match, distance = auth.compare_signatures(stored_sig_b64, candidate_b64)

    if not is_match:
        auth.log_audit(db, "SIGNATURE_LOGIN_FAILED", request, user_id=user.id, badge_id=user.badge_id,
                       detail=f"Hamming distance: {distance}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chữ ký không khớp. Vui lòng ký lại cẩn thận hơn.",
        )

    # Signature matched → grant full session
    _create_and_set_cookie(user, response)
    auth.log_audit(db, "SIGNATURE_LOGIN_SUCCESS", request, user_id=user.id, badge_id=user.badge_id,
                   detail=f"Hamming distance: {distance}")

    return {"success": True, "message": "Xác thực chữ ký thành công. Đăng nhập hoàn tất."}
