"""Pydantic request/response models for the Firasa inference API."""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

Number = Union[float, int]


class MonthlyStatement(BaseModel):
    """One customer-month of already row-engineered features.

    'Row-engineered' means: the per-row transforms notebook 01 applies to a
    single statement -- the *_was_missing indicator flags and the *_log
    values -- are already computed by the caller. What this API does NOT
    redo is the cross-month aggregation (mean/std/min/max/last across a
    customer's whole history) that Branch 1 needs; that still has to come
    from upstream feature preparation, same as it does in the notebooks.

    - features: any of the ~359 numeric Branch-2 sequence columns (see
      GET /model-info -> branch2_sequence_features). Missing keys default
      to 0.0, matching how a short/incomplete statement would look.
    - D_63 / D_64: the two categorical columns. Pass either the raw string
      category (e.g. "CL", "O" -- the API applies the same OrdinalEncoder
      used in training) or the already-encoded integer code.
    """

    features: Dict[str, Number] = Field(default_factory=dict)
    D_63: Optional[Union[str, Number]] = None
    D_64: Optional[Union[str, Number]] = None


class PredictRequest(BaseModel):
    customer_id: Optional[str] = Field(
        default=None, description="Optional label for the response only; not used in scoring."
    )
    xgb_features: Dict[str, Number] = Field(
        description=(
            "The 1,802 Branch-1 aggregated features (see GET /model-info -> "
            "branch1_features), already aggregated across the customer's "
            "statement history exactly as notebook 01 does it. Missing keys "
            "default to 0.0. D_63/D_64 may be given as the raw category "
            "string or the pre-encoded integer code."
        )
    )
    monthly_statements: List[MonthlyStatement] = Field(
        description=(
            "The customer's statement history, oldest first, up to 13 months. "
            "Fewer than 13 is fine (Branch 2 zero-pads the earlier months, "
            "same as training); more than 13 keeps only the most recent 13."
        ),
        min_length=1,
    )

    @field_validator("monthly_statements")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("monthly_statements must contain at least one month")
        return v


class PredictResponse(BaseModel):
    customer_id: Optional[str]
    p_xgb: float
    p_gru: float
    p_fused: float
    risk_level: str
    action: str
    xgb_contribution_pct: float
    gru_contribution_pct: float
    warnings: List[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    branch1_n_features: int
    branch1_features: List[str]
    branch2_sequence_features: List[str]
    branch2_categorical_features: List[str]
    branch2_categorical_categories: Dict[str, List[str]]
    max_sequence_length: int
    risk_bins: List[float]
    risk_levels: List[str]
    risk_actions: Dict[str, str]
    fusion_coefficients: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    branch1_loaded: bool
    branch2_loaded: bool
    fusion_loaded: bool
