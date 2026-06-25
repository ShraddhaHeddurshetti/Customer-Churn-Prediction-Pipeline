import sys
import os
from pathlib import Path

# ── Ensure project root is on sys.path so 'src' is importable ─
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide")

MODEL_PATH = ROOT / "models" / "xgboost.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "⚠️ Model not found. Run training first:\n\n"
            "```\npython -m src.train\n```"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("🔮 Customer Churn Predictor")
st.caption("Predict which customers are likely to leave — powered by XGBoost + MLflow")

# ── Mode Switch ───────────────────────────────────────────────
mode = st.radio("Mode", ["🧑 Single Customer", "📋 Batch Upload"], horizontal=True)
st.divider()


def churn_gauge(probability: float):
    """Plotly gauge chart for churn probability."""
    color = "#ef4444" if probability > 0.5 else "#10b981" if probability < 0.3 else "#f59e0b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Churn Risk", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 60], "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


# ─────────────────────────────────────────────────────────────
# MODE 1: SINGLE CUSTOMER
# ─────────────────────────────────────────────────────────────
if mode == "🧑 Single Customer":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless = st.checkbox("Paperless Billing", value=True)

    with col2:
        st.subheader("📞 Services")
        phone_service = st.checkbox("Phone Service", value=True)
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("💰 Billing")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                         float(tenure * monthly_charges))
        senior = st.checkbox("Senior Citizen")
        partner = st.checkbox("Partner")
        dependents = st.checkbox("Dependents")

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "PaymentMethod": payment,
            "PaperlessBilling": int(paperless),
            "PhoneService": int(phone_service),
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "SeniorCitizen": int(senior),
            "Partner": int(partner),
            "Dependents": int(dependents),
            "MultipleLines": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "gender": "Male",
        }])

        prob = model.predict_proba(input_data)[0][1]
        st.divider()

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(churn_gauge(prob), use_container_width=True)
        with res_col2:
            st.markdown("### Recommendation")
            if prob > 0.6:
                st.error(f"⚠️ HIGH RISK ({prob:.1%}) — Immediate action needed")
                st.markdown(
                    "**Suggested actions:**\n"
                    "- Offer contract upgrade discount\n"
                    "- Proactive support call\n"
                    "- Loyalty reward"
                )
            elif prob > 0.35:
                st.warning(f"⚡ MEDIUM RISK ({prob:.1%}) — Monitor closely")
                st.markdown(
                    "**Suggested actions:**\n"
                    "- Send satisfaction survey\n"
                    "- Offer annual plan promotion"
                )
            else:
                st.success(f"✅ LOW RISK ({prob:.1%}) — Customer is stable")
                st.markdown(
                    "**Suggested actions:**\n"
                    "- Upsell opportunity\n"
                    "- Referral program"
                )

# ─────────────────────────────────────────────────────────────
# MODE 2: BATCH UPLOAD
# ─────────────────────────────────────────────────────────────
else:
    st.subheader("📋 Batch Churn Scoring")
    st.caption("Upload a CSV with customer data — same columns as training data.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Drop target if present
        if "Churn" in df.columns:
            actual = df["Churn"].copy()
            df = df.drop(columns=["Churn"])
        else:
            actual = None

        st.write(f"Loaded {len(df):,} customers")
        st.dataframe(df.head(3), use_container_width=True)

        if st.button("🚀 Score All Customers", type="primary"):
            with st.spinner("Scoring..."):
                probs = model.predict_proba(df)[:, 1]
                df["churn_probability"] = probs.round(4)
                df["churn_risk"] = pd.cut(
                    probs, bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"]
                )

            st.success("Done!")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("High Risk", f"{(df['churn_risk'] == 'High').sum():,}")
            col_b.metric("Medium Risk", f"{(df['churn_risk'] == 'Medium').sum():,}")
            col_c.metric("Low Risk", f"{(df['churn_risk'] == 'Low').sum():,}")

            st.dataframe(
                df[["churn_probability", "churn_risk"]].join(df.iloc[:, :3]),
                use_container_width=True
            )

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV", csv_out,
                "churn_predictions.csv", "text/csv"
            )
