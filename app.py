"""
MSME Financial Health Score API
================================
FastAPI backend that loads the trained model (model.pkl / scaler.pkl /
sector_encoder.pkl / feature_names.pkl - produced by train_model.py) and
serves a /predict endpoint that turns a business's alternative digital
data (UPI + GST + cash-flow signals) into a 0-100 Financial Health Score,
a risk category, and a loan recommendation.

DEPLOY ON RENDER:
    Build command : pip install -r requirements.txt
    Start command  : uvicorn app:app --host 0.0.0.0 --port $PORT

Make sure model.pkl, scaler.pkl, sector_encoder.pkl and feature_names.pkl
are committed to the repo alongside this file (they are small, <5MB).
"""

import os
import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List

# ---------------------------------------------------------------------------
# Human-readable labels for each feature, used in the explainability output
# so a credit officer sees "GST Filing Regularity" instead of "gst_filing_regularity"
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "sector_encoded": "Business Sector",
    "business_vintage_months": "Business Vintage",
    "monthly_upi_inflow": "Monthly UPI Inflow",
    "monthly_upi_outflow": "Monthly UPI Outflow",
    "upi_transaction_count": "UPI Transaction Count",
    "avg_transaction_value": "Average Transaction Value",
    "upi_inflow_growth_rate": "UPI Inflow Growth Rate",
    "cash_flow_volatility": "Cash Flow Volatility",
    "avg_bank_balance": "Average Bank Balance",
    "gst_monthly_turnover": "GST Monthly Turnover",
    "gst_filing_regularity": "GST Filing Regularity",
    "gst_late_filings_count": "Late GST Filings",
    "digital_invoice_count_monthly": "Digital Invoice Volume",
    "on_time_payment_ratio": "On-Time Payment Ratio",
    "bounced_payment_count": "Bounced Payments",
    "existing_loan_emi_ratio": "Existing Loan EMI Ratio",
}

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    sector_encoder = joblib.load(os.path.join(BASE_DIR, "sector_encoder.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    MODEL_LOADED = True
    MODEL_LOAD_ERROR = None
except Exception as e:
    # Don't crash the whole app on import - surface the error via /health
    # so Render logs show a helpful message instead of a bare stack trace.
    model = scaler = sector_encoder = feature_names = None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

VALID_SECTORS = list(sector_encoder.keys()) if sector_encoder else []

# ---------------------------------------------------------------------------
# Explainable AI: SHAP TreeExplainer
# ---------------------------------------------------------------------------
# Works directly with tree-based models (Gradient Boosting / Random Forest /
# XGBoost) with no extra training step - it introspects the trees themselves.
# If the winning model in model.pkl is ever swapped for something SHAP's
# TreeExplainer doesn't support (e.g. the MLP neural net), this falls back
# to None and /predict simply omits the explanation rather than failing.
try:
    explainer = shap.TreeExplainer(model) if MODEL_LOADED else None
except Exception:
    explainer = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MSME Financial Health Score API",
    description=(
        "Aggregates UPI transaction behaviour, GST filing history, and "
        "digital cash-flow signals for an MSME with no audited balance "
        "sheet, and returns an AI-generated Financial Health Score (0-100) "
        "banks can use for credit decisioning."
    ),
    version="1.0.0",
)

# Allow calls from any frontend (adjust origins for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class MSMEInput(BaseModel):
    sector: str = Field(
        ..., description=f"Business sector. One of: {VALID_SECTORS}",
        json_schema_extra={"example": "grocery_retail"},
    )
    business_vintage_months: float = Field(
        ..., ge=0, description="How many months the business has been operating",
        json_schema_extra={"example": 36},
    )
    monthly_upi_inflow: float = Field(
        ..., ge=0, description="Average monthly UPI inflow (₹)",
        json_schema_extra={"example": 250000},
    )
    monthly_upi_outflow: float = Field(
        ..., ge=0, description="Average monthly UPI outflow / expenses (₹)",
        json_schema_extra={"example": 180000},
    )
    upi_transaction_count: float = Field(
        ..., ge=0, description="Number of UPI transactions per month",
        json_schema_extra={"example": 620},
    )
    avg_transaction_value: float = Field(
        ..., ge=0, description="Average value per UPI transaction (₹)",
        json_schema_extra={"example": 400},
    )
    upi_inflow_growth_rate: float = Field(
        ..., description="% growth in UPI inflow over the last 6 months (can be negative)",
        json_schema_extra={"example": 8.5},
    )
    cash_flow_volatility: float = Field(
        ..., ge=0, description="Coefficient of variation of monthly inflow (0 = perfectly stable)",
        json_schema_extra={"example": 0.18},
    )
    avg_bank_balance: float = Field(
        ..., ge=0, description="Average bank account balance (₹)",
        json_schema_extra={"example": 45000},
    )
    gst_monthly_turnover: float = Field(
        ..., ge=0, description="Monthly turnover as declared in GST filings (₹)",
        json_schema_extra={"example": 230000},
    )
    gst_filing_regularity: float = Field(
        ..., ge=0, le=1, description="Fraction of GST returns filed on time (0-1)",
        json_schema_extra={"example": 0.9},
    )
    gst_late_filings_count: float = Field(
        ..., ge=0, description="Number of late GST filings in the last 12 months",
        json_schema_extra={"example": 1},
    )
    digital_invoice_count_monthly: float = Field(
        ..., ge=0, description="Number of digital invoices generated per month",
        json_schema_extra={"example": 150},
    )
    on_time_payment_ratio: float = Field(
        ..., ge=0, le=1, description="Fraction of supplier/vendor payments made on time (0-1)",
        json_schema_extra={"example": 0.85},
    )
    bounced_payment_count: float = Field(
        ..., ge=0, description="Number of bounced payments in the last 6 months",
        json_schema_extra={"example": 0},
    )
    existing_loan_emi_ratio: float = Field(
        ..., ge=0, le=2, description="Existing loan EMI as a fraction of monthly inflow",
        json_schema_extra={"example": 0.15},
    )


class FeatureContribution(BaseModel):
    feature: str
    value_provided: float
    impact: float
    direction: Literal["increased", "decreased"]


class PredictionResponse(BaseModel):
    financial_health_score: float
    risk_category: Literal["Low Risk", "Medium Risk", "High Risk"]
    loan_recommendation: str
    model_used: str
    base_score: float
    top_factors: List[FeatureContribution]


# ---------------------------------------------------------------------------
# Helper: risk category + human-readable recommendation
# ---------------------------------------------------------------------------
def get_risk_category(score: float) -> str:
    if score >= 70:
        return "Low Risk"
    elif score >= 45:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(score: float, category: str) -> str:
    if category == "Low Risk":
        return (
            f"Strong financial health score ({score:.1f}/100). Consistent digital "
            "cash flows and clean compliance history. Recommended for approval, "
            "including expansion/working-capital loans."
        )
    elif category == "Medium Risk":
        return (
            f"Moderate financial health score ({score:.1f}/100). Business shows "
            "reasonable digital footprint but has some inconsistencies (filing "
            "delays, payment gaps, or cash flow volatility). Consider approval "
            "with a smaller ticket size, higher collateral, or closer monitoring."
        )
    else:
        return (
            f"Weak financial health score ({score:.1f}/100). Significant red flags "
            "in cash flow stability, compliance, or repayment behaviour. "
            "Manual underwriting review strongly recommended before approval."
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "MSME Financial Health Score API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "error": MODEL_LOAD_ERROR,
        "model_type": type(model).__name__ if MODEL_LOADED else None,
        "valid_sectors": VALID_SECTORS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: MSMEInput):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded on server: {MODEL_LOAD_ERROR}",
        )

    if data.sector not in sector_encoder:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sector '{data.sector}'. Must be one of: {VALID_SECTORS}",
        )

    sector_encoded = sector_encoder[data.sector]

    # Build the feature vector in EXACTLY the order the model was trained on
    feature_values = {
        "sector_encoded": sector_encoded,
        "business_vintage_months": data.business_vintage_months,
        "monthly_upi_inflow": data.monthly_upi_inflow,
        "monthly_upi_outflow": data.monthly_upi_outflow,
        "upi_transaction_count": data.upi_transaction_count,
        "avg_transaction_value": data.avg_transaction_value,
        "upi_inflow_growth_rate": data.upi_inflow_growth_rate,
        "cash_flow_volatility": data.cash_flow_volatility,
        "avg_bank_balance": data.avg_bank_balance,
        "gst_monthly_turnover": data.gst_monthly_turnover,
        "gst_filing_regularity": data.gst_filing_regularity,
        "gst_late_filings_count": data.gst_late_filings_count,
        "digital_invoice_count_monthly": data.digital_invoice_count_monthly,
        "on_time_payment_ratio": data.on_time_payment_ratio,
        "bounced_payment_count": data.bounced_payment_count,
        "existing_loan_emi_ratio": data.existing_loan_emi_ratio,
    }

    try:
        x = np.array([[feature_values[f] for f in feature_names]])
        x_scaled = scaler.transform(x)
        raw_score = float(model.predict(x_scaled)[0])
        score = float(np.clip(raw_score, 0, 100))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    category = get_risk_category(score)
    recommendation = get_recommendation(score, category)

    # ------------------------------------------------------------------
    # Explainable AI: compute per-feature SHAP contributions for THIS
    # specific business, so a credit officer can see exactly why the
    # score came out the way it did (not just a black-box number).
    # ------------------------------------------------------------------
    base_score = score
    top_factors: List[FeatureContribution] = []

    if explainer is not None:
        try:
            shap_values = explainer.shap_values(x_scaled)[0]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = np.asarray(expected_value).ravel()[0]
            base_score = round(float(expected_value), 2)

            ranked = sorted(
                zip(feature_names, shap_values),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )[:5]  # top 5 drivers of this specific score

            for feat, impact in ranked:
                top_factors.append(
                    FeatureContribution(
                        feature=FEATURE_LABELS.get(feat, feat),
                        value_provided=float(feature_values[feat]),
                        impact=round(float(impact), 2),
                        direction="increased" if impact >= 0 else "decreased",
                    )
                )
        except Exception:
            # Explainability is a bonus, never let it break the core prediction
            top_factors = []

    return PredictionResponse(
        financial_health_score=round(score, 2),
        risk_category=category,
        loan_recommendation=recommendation,
        model_used=type(model).__name__,
        base_score=base_score,
        top_factors=top_factors,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
