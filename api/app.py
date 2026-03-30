"""
Assignment 03 — Standalone FastAPI server for Student Risk Prediction
AIE1014 | Onyekachi Odunze
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load model on startup ──────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"
model = None

# Exact 44 feature columns the RandomForest was trained on (Assignment 02)
MODEL_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "avg_score", "total_clicks",
    "completion_rate", "is_repeat_student", "engagement_score", "credits_per_attempt",
    "gender_m",
    "code_module_bbb", "code_module_ccc", "code_module_ddd",
    "code_module_eee", "code_module_fff", "code_module_ggg",
    "region_east midlands region", "region_ireland", "region_london region",
    "region_north region", "region_north western region", "region_scotland",
    "region_south east region", "region_south region", "region_south west region",
    "region_wales", "region_west midlands region", "region_yorkshire region",
    "highest_education_he qualification", "highest_education_lower than a level",
    "highest_education_no formal quals", "highest_education_post graduate qualification",
    "imd_band_10-20%", "imd_band_20-30%", "imd_band_30-40%", "imd_band_40-50%",
    "imd_band_50-60%", "imd_band_60-70%", "imd_band_70-80%", "imd_band_80-90%",
    "imd_band_90-100%", "imd_band_unknown",
    "age_band_35-55", "age_band_55<=", "disability_y",
]

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class StudentFeatures(BaseModel):
    """All features needed for a student risk prediction."""

    # Core academic features (required — no defaults)
    avg_score: float = Field(ge=0, le=100,
                             description="Average assessment score (0-100)")
    completion_rate: float = Field(ge=0, le=1,
                                   description="Assessment completion rate (0-1)")
    total_clicks: int = Field(ge=0,
                              description="Total VLE interaction clicks")
    studied_credits: int = Field(ge=0,
                                 description="Total credits currently studying")
    num_of_prev_attempts: int = Field(ge=0,
                                      description="Number of previous module attempts")

    # Module flags (leave all False for module AAA — the baseline)
    module_BBB: bool = Field(default=False, description="Enrolled in module BBB")
    module_CCC: bool = Field(default=False, description="Enrolled in module CCC")
    module_DDD: bool = Field(default=False, description="Enrolled in module DDD")
    module_EEE: bool = Field(default=False, description="Enrolled in module EEE")
    module_FFF: bool = Field(default=False, description="Enrolled in module FFF")
    module_GGG: bool = Field(default=False, description="Enrolled in module GGG")

    # Demographics
    gender: str = Field(default="M", description="Gender: M or F")
    region: str = Field(default="South East Region",
                        description="UK region (e.g. 'South East Region')")
    highest_education: str = Field(default="A Level or Equivalent",
                                   description="Highest qualification")
    imd_band: str = Field(default="50-60%",
                          description="Index of Multiple Deprivation band")
    age_band: str = Field(default="0-35", description="Age band: 0-35, 35-55, or 55<=")
    disability: str = Field(default="N", description="Disability status: Y or N")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "avg_score": 45.0, "completion_rate": 0.35, "total_clicks": 150,
                "studied_credits": 60, "num_of_prev_attempts": 0,
                "module_BBB": False, "module_CCC": False, "module_DDD": False,
                "module_EEE": False, "module_FFF": False, "module_GGG": False,
                "gender": "M", "region": "South East Region",
                "highest_education": "A Level or Equivalent",
                "imd_band": "50-60%", "age_band": "0-35", "disability": "N",
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = Not at risk, 1 = At risk")
    risk_level: str = Field(description="low / medium / high")
    confidence: float = Field(description="Model confidence (0-1)")
    probability_at_risk: float = Field(description="Raw probability of being at risk")


# ── Feature engineering (mirrors Assignment 02 Stage 4 pipeline) ──────────────
def build_feature_vector(s: StudentFeatures) -> pd.DataFrame:
    """Convert StudentFeatures into the 44-column vector the model expects."""

    # Start with all zeros — fills any one-hot columns not set
    row = {f: 0 for f in MODEL_FEATURES}

    # Raw numeric features
    row["avg_score"] = s.avg_score
    row["completion_rate"] = min(s.completion_rate, 1.0)  # clip outliers (max was 5.0 in raw data)
    row["total_clicks"] = s.total_clicks
    row["studied_credits"] = s.studied_credits
    row["num_of_prev_attempts"] = s.num_of_prev_attempts

    # Engineered features — same formulas used in Assignment 02 Stage 4
    row["is_repeat_student"] = int(s.num_of_prev_attempts > 0)
    row["engagement_score"] = s.avg_score * min(s.completion_rate, 1.0)
    row["credits_per_attempt"] = s.studied_credits / (s.num_of_prev_attempts + 1)

    # Gender one-hot (baseline = F, drop_first=True)
    row["gender_m"] = int(s.gender.strip().upper() == "M")

    # Module one-hot (baseline = AAA, drop_first=True)
    for mod in ["BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]:
        key = f"code_module_{mod.lower()}"
        if key in row:
            row[key] = int(getattr(s, f"module_{mod}", False))

    # Region one-hot
    region_key = f"region_{s.region.strip().lower()}"
    if region_key in row:
        row[region_key] = 1

    # Highest education one-hot (baseline = A Level or Equivalent)
    edu_map = {
        "he qualification": "highest_education_he qualification",
        "lower than a level": "highest_education_lower than a level",
        "no formal quals": "highest_education_no formal quals",
        "post graduate qualification": "highest_education_post graduate qualification",
    }
    edu_key = edu_map.get(s.highest_education.strip().lower())
    if edu_key and edu_key in row:
        row[edu_key] = 1

    # IMD band one-hot
    imd_key = f"imd_band_{s.imd_band.strip().lower()}"
    if imd_key in row:
        row[imd_key] = 1

    # Age band one-hot (baseline = 0-35)
    age_map = {"35-55": "age_band_35-55", "55<=": "age_band_55<="}
    age_key = age_map.get(s.age_band.strip())
    if age_key and age_key in row:
        row[age_key] = 1

    # Disability one-hot (baseline = N)
    row["disability_y"] = int(s.disability.strip().upper() == "Y")

    return pd.DataFrame([row])[MODEL_FEATURES]


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Risk Prediction API",
    version="1.0.0",
    description="Predicts whether a student is at risk of failing or withdrawing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health_check():
    """Returns API and model status."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/info", tags=["Info"])
def model_info():
    """Returns metadata about the loaded model."""
    return {
        "model_type": type(model).__name__ if model else "None",
        "version": "1.0.0",
        "target": "at_risk (0 = not at risk, 1 = at risk)",
        "features_expected": MODEL_FEATURES,
        "n_features": len(MODEL_FEATURES),
        "trained_on": "OULAD dataset — 32,176 records after cleaning",
        "cv_accuracy": 0.9062,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentFeatures):
    """Predict student risk level from their academic and demographic features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Ensure api/model.pkl exists.")

    try:
        X = build_feature_vector(student)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        confidence = float(max(model.predict_proba(X)[0]))

        # Translate probability into a human-readable risk tier
        if proba < 0.35:
            risk_level = "low"
        elif proba < 0.65:
            risk_level = "medium"
        else:
            risk_level = "high"

        return PredictionResponse(
            prediction=pred,
            risk_level=risk_level,
            confidence=confidence,
            probability_at_risk=proba,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
