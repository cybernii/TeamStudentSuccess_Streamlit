"""
Assignment 03 - Streamlit UI for Student Risk Prediction
AIE1014 | Onyekachi Odunze
"""

import streamlit as st
import pandas as pd
from shared.prediction import MODEL_FEATURES, MODEL_PATH, load_model as load_prediction_model, predict_payload

# Configuration
APP_TITLE = "Student Risk Prediction"
APP_SUBTITLE = "Enter a student's details to predict whether they are at risk of failing or withdrawing."

@st.cache_resource(show_spinner=False)
def load_model():
    return load_prediction_model()


def predict_locally(payload: dict) -> dict:
    """Run a local prediction (no FastAPI required)."""
    try:
        return {"success": True, "data": predict_payload(payload, model=load_model())}
    except FileNotFoundError:
        return {"success": False, "error": f"Model file not found at {MODEL_PATH}."}
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}


# Page setup - MUST be the very first Streamlit command in the script
st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 " + APP_TITLE)
st.write(APP_SUBTITLE)
st.divider()

with st.sidebar:
    st.header("🔌 System Status")
    try:
        _ = load_model()
        st.success("✅ Model Loaded (local)")
    except Exception:
        st.error("❌ Model not loaded")

    st.divider()

    st.header("📊 Model Information")
    st.write("**Type:** RandomForestClassifier")
    st.write("**CV Accuracy:** 90.6%")
    st.write(f"**Features:** {len(MODEL_FEATURES)}")
    with st.expander("View all features"):
        for feat in MODEL_FEATURES:
            st.write(f"• {feat}")

    st.divider()
    st.header("ℹ️ How to Use")
    st.write(
        "1. Fill in the student details\n"
        "2. Click **Get Prediction**\n"
        "3. Review the risk level result"
    )


# Input form
st.subheader("📝 Enter Student Details")

# Wrapping inputs in st.form prevents Streamlit from re-running (and calling the API)
# on every single widget change - it only fires when the submit button is clicked.
with st.form("prediction_form"):

    st.markdown("##### 📚 Academic Performance")
    col1, col2 = st.columns(2)

    with col1:
        avg_score = st.number_input(
            "Average Score (%)",
            min_value=0.0, max_value=100.0, value=50.0, step=1.0,
            help="Student's average assessment score out of 100",
        )
        completion_rate = st.slider(
            "Completion Rate",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Proportion of assessments submitted (0 = none, 1 = all)",
        )
        total_clicks = st.number_input(
            "Total VLE Clicks",
            min_value=0, max_value=30000, value=500, step=50,
            help="Total number of interactions with the Virtual Learning Environment",
        )

    with col2:
        studied_credits = st.number_input(
            "Studied Credits",
            min_value=0, max_value=700, value=60, step=10,
            help="Total credits the student is currently enrolled in",
        )
        num_of_prev_attempts = st.number_input(
            "Previous Attempts",
            min_value=0, max_value=6, value=0, step=1,
            help="Number of times the student has previously attempted this module",
        )

    st.markdown("##### 🏫 Module")
    module = st.selectbox(
        "Current Module",
        options=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        help="Module code the student is enrolled in (AAA is the baseline)",
    )

    st.markdown("##### 👤 Demographics")
    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox("Gender", options=["M", "F"])
        age_band = st.selectbox(
            "Age Band",
            options=["0-35", "35-55", "55<="],
            help="Student age range",
        )
        disability = st.selectbox(
            "Disability",
            options=["N", "Y"],
            help="Does the student have a registered disability?",
        )

    with col4:
        region = st.selectbox(
            "Region",
            options=[
                "South East Region", "London Region", "North Western Region",
                "Yorkshire Region", "West Midlands Region", "East Midlands Region",
                "South West Region", "South Region", "North Region",
                "Scotland", "Ireland", "Wales",
            ],
        )
        highest_education = st.selectbox(
            "Highest Education",
            options=[
                "A Level or Equivalent", "HE Qualification",
                "Lower Than A Level", "Post Graduate Qualification", "No Formal Quals",
            ],
        )
        imd_band = st.selectbox(
            "IMD Band",
            options=[
                "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown",
            ],
            index=5,
            help="Index of Multiple Deprivation (lower % = more deprived area)",
        )

    submitted = st.form_submit_button("🔮 Get Prediction", use_container_width=True)

# Prediction output
result = {}
if submitted:
    # Build module boolean flags from the selected module dropdown
    payload = {
        "avg_score": avg_score,
        "completion_rate": completion_rate,
        "total_clicks": int(total_clicks),
        "studied_credits": int(studied_credits),
        "num_of_prev_attempts": int(num_of_prev_attempts),
        "module_BBB": module == "BBB",
        "module_CCC": module == "CCC",
        "module_DDD": module == "DDD",
        "module_EEE": module == "EEE",
        "module_FFF": module == "FFF",
        "module_GGG": module == "GGG",
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "disability": disability,
    }

    with st.spinner("🔮 Getting prediction..."):
        result = predict_locally(payload)

    if result["success"]:
        data = result["data"]
        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Level", data.get("risk_level", "?").upper())
        with col2:
            st.metric("Confidence", f"{data.get('confidence', 0):.1%}")

        risk_level = data.get("risk_level", "")
        prob = data.get("probability_at_risk", 0)

        # Plain-language interpretation - advisors should not need to interpret numbers
        if risk_level == "low":
            st.success(
                f"✅ Low Risk - This student is performing well "
                f"(risk probability: {prob:.1%}). Continue routine monitoring."
            )
        elif risk_level == "medium":
            st.warning(
                f"⚠️ Medium Risk - This student shows some warning signs "
                f"(risk probability: {prob:.1%}). Consider a proactive check-in."
            )
        else:
            st.error(
                f"🚨 High Risk - This student needs immediate support "
                f"(risk probability: {prob:.1%}). Escalate to an academic advisor."
            )

        # Collapsible raw response - useful for debugging without cluttering the main view
        with st.expander("🔍 View full prediction output"):
            st.json(data)

    else:
        # Display a friendly error - users must never see a Python traceback
        st.error(f"❌ {result['error']}")

# Prediction history - persists within the same browser session, resets on refresh
if "history" not in st.session_state:
    st.session_state.history = []

if submitted and result.get("success"):
    st.session_state.history.append({
        "avg_score": avg_score,
        "completion_rate": round(completion_rate, 2),
        "total_clicks": total_clicks,
        "module": module,
        "risk_level": result["data"].get("risk_level"),
        "probability": f"{result['data'].get('probability_at_risk', 0):.1%}",
    })

if st.session_state.history:
    st.divider()
    st.subheader("📋 Prediction History (this session)")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
