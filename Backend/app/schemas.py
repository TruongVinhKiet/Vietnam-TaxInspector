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
    created_at: datetime


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
