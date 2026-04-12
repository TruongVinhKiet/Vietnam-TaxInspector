from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Literal
from datetime import date, datetime


# --- Auth/Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    badge_id: Optional[str] = None


class LoginRequest(BaseModel):
    badge_id: str
    password: str = Field(min_length=1, max_length=128)


class GenericMessage(BaseModel):
    success: bool
    message: str


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=8, max_length=100)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=20, max_length=256)
    new_password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)

# --- User Schemas ---
class UserBase(BaseModel):
    badge_id: str
    full_name: str
    department: str
    email: str
    phone: Optional[str] = None
    role: Literal["viewer", "analyst", "inspector", "admin"] = "viewer"

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    avatar_data: Optional[str] = None
    face_verified: bool = False
    cccd_verified: bool = False
    signature_verified: bool = False
    created_at: datetime


# --- Biometric Auth Schemas ---
class BiometricSetupRequest(BaseModel):
    descriptor: List[float] = Field(..., description="128-d face descriptor vector or CCCD data as list")

class CccdSetupRequest(BaseModel):
    cccd_number: str = Field(..., min_length=9, max_length=12, description="Số CCCD 12 chữ số")

class BiometricLoginRequest(BaseModel):
    descriptor: List[float] = Field(..., description="128-d face descriptor vector for matching")

class CccdLoginRequest(BaseModel):
    cccd_number: str = Field(..., min_length=9, max_length=12)


# --- Signature Auth Schemas ---
class SignatureSetupRequest(BaseModel):
    signature_image: str = Field(..., description="Base64 PNG image of the signature drawn on canvas")

class SignatureLoginRequest(BaseModel):
    temp_token: str = Field(..., description="Temporary token from Step 1 (face/cccd)")
    signature_image: str = Field(..., description="Base64 PNG image of the signature for verification")


# --- Phone Update Schema ---
class UpdatePhoneRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15, description="Số điện thoại mới")


class UpdateAvatarRequest(BaseModel):
    avatar_image: str = Field(
        ...,
        min_length=64,
        max_length=8_000_000,
        description="Data URL hoặc raw Base64 PNG/JPEG (toi da 5MB).",
    )



# --- Company Schemas ---
class CompanyBase(BaseModel):
    tax_code: str
    name: str
    industry: Optional[str] = None
    registration_date: Optional[date] = None


class CompanyCreate(CompanyBase):
    pass


class CompanyResponse(CompanyBase):
    model_config = ConfigDict(from_attributes=True)

    risk_score: float
    is_active: bool


# --- ML Prediction Schemas ---
class FraudRiskPrediction(BaseModel):
    tax_code: str
    risk_score: float
    red_flags: List[str]


class VATGraphNode(BaseModel):
    id: str
    label: str
    group: Optional[str] = "default"


class VATGraphEdge(BaseModel):
    from_node: str
    to_node: str
    value: float


class GraphResponse(BaseModel):
    nodes: List[VATGraphNode]
    edges: List[VATGraphEdge]
    suspicious_clusters: List[List[str]]


class DelinquencyPrediction(BaseModel):
    tax_code: str
    company_name: str
    probability: float
    cluster: str
