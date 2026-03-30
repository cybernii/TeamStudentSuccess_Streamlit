"""
Assignment 03 — Standalone FastAPI server for Student Risk Prediction
AIE1014 | Onyekachi Odunze
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared.prediction import MODEL_FEATURES, load_model, predict_payload

# ── Load model on startup ──────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"
model = None

try:
    model = load_model()
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
        result = predict_payload(student.model_dump(), model=model)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
