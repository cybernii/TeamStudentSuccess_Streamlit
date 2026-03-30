# AI-Powered Student Success Recommendation System

> Predicts at-risk students and generates personalised, actionable intervention recommendations using a hybrid AI approach.

**AIE1014 — AI Applied Project · Cambrian College · Winter 2026**

---

## Overview

This system helps academic advisors identify students at risk of failing or withdrawing — and prescribes exactly what to do about it. It combines an XGBoost machine learning model trained on behavioural engagement data with a hybrid intervention engine (rule-based logic enhanced by a Large Language Model) to generate specific, prioritised recommendations for each student. The system is designed for Cambrian College advisors who need to act faster and more precisely than manual review allows.

| | |
|---|---|
| **Stakeholder** | Academic advisors and student success coordinators at Cambrian College |
| **Problem** | Advisors cannot manually monitor hundreds of students for early warning signs of dropout; interventions come too late |
| **Solution** | An AI platform that automatically scores each student's dropout risk, explains the key behavioural factors driving that risk (via SHAP), and generates prioritised, personalised intervention actions — saving advisors 80% of the time needed per student |

---

## Team

| Name | Role |
|------|------|
| Daniel Quartey | Project Lead |
| Odunze Onyekachi | Data & ML Engineer |
| Juthamard Jirapanyalerd | Backend & LLM Integration |
| Ime-Jnr Ime-Essien | Frontend Engineer |

---

## Project Structure

```
TeamStudentSuccess_Milestone4/
├── api/
│   ├── app.py              # FastAPI backend (predict, health, info)
│   ├── model.pkl           # Trained RandomForest model
│   └── requirements.txt    # API dependencies
├── ui/
│   ├── app_ui.py           # Streamlit UI
│   └── requirements.txt    # UI dependencies
├── tests/
│   ├── test_integration.py # Integration test suite (8 tests)
│   └── test_results.txt    # Latest test run results (8/8 pass)
├── docs/
│   └── TeamStudentSuccess_Milestone4_Report.pdf
├── README.md
└── requirements.txt        # Combined dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- pip

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Start the API

```bash
# From the project root
python -m uvicorn api.app:app --reload --port 8000
```

API will be available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

### 3 — Start the Streamlit UI (new terminal)

```bash
streamlit run ui/app_ui.py
```

UI will open at `http://localhost:8501`

### 4 — Run integration tests (new terminal)

```bash
python tests/test_integration.py
```

Expected: **8 passed | 0 failed**

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — confirms API and model are loaded |
| `/info` | GET | Model metadata (type, accuracy, feature count) |
| `/predict` | POST | Predict at-risk status for a student |

### Predict Request Body

```json
{
  "avg_score": 45.0,
  "completion_rate": 0.35,
  "total_clicks": 150,
  "studied_credits": 60,
  "num_of_prev_attempts": 0,
  "module": "BBB"
}
```

### Predict Response

```json
{
  "prediction": 1,
  "risk_level": "high",
  "confidence": 0.96,
  "probability": 0.96
}
```

---

## ML Model

| | |
|---|---|
| **Dataset** | OULAD — Open University Learning Analytics Dataset |
| **Training Records** | 32,593 student enrolments across 7 courses |
| **Algorithm** | RandomForestClassifier |
| **CV Accuracy** | 90.6% |
| **Features** | 44 (after one-hot encoding of demographics and module) |
| **Target** | `at_risk`: 1 = at risk (Fail/Withdrawn), 0 = not at risk |

### Top Features (by importance)

1. `avg_score` — Average mark across submitted assessments
2. `completion_rate` — Fraction of assessments submitted
3. `total_clicks` — Total VLE platform interactions
4. `num_of_prev_attempts` — Times student has previously attempted the module
5. `studied_credits` — Total credits enrolled this semester

---

## Testing

```
Test 1: API health check              ✅ PASS
Test 2: Valid prediction (happy path) ✅ PASS
Test 3: Minimum boundary values       ✅ PASS
Test 4: Maximum boundary values       ✅ PASS
Test 5: Missing required field        ✅ PASS (422 returned)
Test 6: Empty request body            ✅ PASS (422 returned)
Test 7: Wrong data type               ✅ PASS (422 returned)
Test 8: Response time < 5s            ✅ PASS (~2s)

RESULTS: 8 passed | 0 failed
```

---

## Known Issues

- No persistent database — predictions reset on server restart
- Single-user session — no multi-user authentication
- OULAD module codes only — real college data requires column remapping

---

*AIE1014 — AI Applied Project · Cambrian College · Winter 2026*
