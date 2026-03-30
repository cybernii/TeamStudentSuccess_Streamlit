from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "api" / "model.pkl"

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


def load_model():
    return joblib.load(MODEL_PATH)


def build_feature_vector(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert payload data into the 44-column vector the model expects."""
    row = {feature: 0 for feature in MODEL_FEATURES}

    row["avg_score"] = float(payload["avg_score"])
    row["completion_rate"] = min(float(payload["completion_rate"]), 1.0)
    row["total_clicks"] = int(payload["total_clicks"])
    row["studied_credits"] = int(payload["studied_credits"])
    row["num_of_prev_attempts"] = int(payload["num_of_prev_attempts"])

    row["is_repeat_student"] = int(row["num_of_prev_attempts"] > 0)
    row["engagement_score"] = row["avg_score"] * row["completion_rate"]
    row["credits_per_attempt"] = row["studied_credits"] / (row["num_of_prev_attempts"] + 1)

    row["gender_m"] = int(str(payload.get("gender", "")).strip().upper() == "M")

    for module_code in ["BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]:
        feature_name = f"code_module_{module_code.lower()}"
        row[feature_name] = int(bool(payload.get(f"module_{module_code}", False)))

    region_key = f"region_{str(payload.get('region', '')).strip().lower()}"
    if region_key in row:
        row[region_key] = 1

    education_map = {
        "he qualification": "highest_education_he qualification",
        "lower than a level": "highest_education_lower than a level",
        "no formal quals": "highest_education_no formal quals",
        "post graduate qualification": "highest_education_post graduate qualification",
    }
    education_key = education_map.get(str(payload.get("highest_education", "")).strip().lower())
    if education_key:
        row[education_key] = 1

    imd_key = f"imd_band_{str(payload.get('imd_band', '')).strip().lower()}"
    if imd_key in row:
        row[imd_key] = 1

    age_map = {"35-55": "age_band_35-55", "55<=": "age_band_55<="}
    age_key = age_map.get(str(payload.get("age_band", "")).strip())
    if age_key:
        row[age_key] = 1

    row["disability_y"] = int(str(payload.get("disability", "")).strip().upper() == "Y")

    return pd.DataFrame([row])[MODEL_FEATURES]


def predict_payload(payload: Mapping[str, Any], model=None) -> dict[str, Any]:
    model = model or load_model()
    features = build_feature_vector(payload)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    probability_at_risk = float(probabilities[1])
    confidence = float(max(probabilities))

    if probability_at_risk < 0.35:
        risk_level = "low"
    elif probability_at_risk < 0.65:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "confidence": confidence,
        "probability_at_risk": probability_at_risk,
    }
