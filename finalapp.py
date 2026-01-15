import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

st.title("Mobile Money Fraud Detection System")
st.caption("Anomaly-Based Risk Scoring | Production-Ready Pipeline")

# -------------------------------------------------
# LOAD MODEL ARTIFACTS
# -------------------------------------------------
artifacts = joblib.load("isolation_forest_fraud_model.joblib")

model = artifacts["model"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]

FEATURES = [
    "amount",
    "hour",
    "is_foreign_number",
    "is_sim_recently_swapped",
    "has_multiple_accounts",
    "transaction_type",
    "location",
    "device_type",
    "network_provider",
    "user_type",
    "time_of_day"
]

# -------------------------------------------------
# RISK BAND FUNCTION
# -------------------------------------------------
def risk_band(score):
    if score > 0.75:
        return "High Risk"
    elif score > 0.45:
        return "Medium Risk"
    return "Low Risk"

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV with RAW (non-encoded) values",
    type=["csv"]
)

if uploaded_file:

    # ---------------- LOAD RAW DATA ----------------
    df_raw = pd.read_csv(uploaded_file)

    missing = set(FEATURES) - set(df_raw.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # -------------------------------------------------
    # IMMUTABLE RAW DATA (FOR UI & ANALYTICS)
    # -------------------------------------------------
    df_display = df_raw.copy()

    # -------------------------------------------------
    # MODEL DATA (ENCODE + SCALE ONLY)
    # -------------------------------------------------
    df_model = df_raw.copy()

    unknown_summary = {}

    for col, le in encoders.items():
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str)

            df_model[col] = df_model[col].apply(
                lambda x: x if x in le.classes_ else "UNKNOWN"
            )

            unknown_count = (df_model[col] == "UNKNOWN").sum()
            if unknown_count > 0:
                unknown_summary[col] = unknown_count

            df_model[col] = le.transform(df_model[col])

    if unknown_summary:
        st.warning("Unknown category values detected")
        st.json(unknown_summary)

    # ---------------- SCALE ----------------
    X = df_model[FEATURES]
    X_scaled = scaler.transform(X)

    # -------------------------------------------------
    # MODEL SCORING
    # -------------------------------------------------
    df_display["anomaly_score"] = -model.decision_function(X_scaled)
    df_display["fraud_flag"] = (model.predict(X_scaled) == -1).astype(int)
    df_display["risk_band"] = df_display["anomaly_score"].apply(risk_band)

    # -------------------------------------------------
    # KPI METRICS
    # -------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df_display))
    col2.metric("Fraud Alerts", int(df_display["fraud_flag"].sum()))
    col3.metric(
        "High Risk %",
        f"{(df_display['risk_band'].eq('High Risk').mean() * 100):.2f}%"
    )

    # -------------------------------------------------
    # TOP RISK TRANSACTIONS
    # -------------------------------------------------
    st.subheader("Top 20 Highest Risk Transactions")

    top_risk = (
        df_display
        .sort_values("anomaly_score", ascending=False)
        .head(20)
    )

    st.dataframe(top_risk, use_container_width=True)

    # -------------------------------------------------
    # CATEGORY BREAKDOWN (RAW VALUES)
    # -------------------------------------------------
    st.subheader("Risk Distribution by Transaction Type")

    risk_by_type = (
        df_display
        .groupby(["transaction_type", "risk_band"])
        .size()
        .reset_index(name="count")
    )

    st.dataframe(risk_by_type, use_container_width=True)

    # -------------------------------------------------
    # DOWNLOAD RESULTS
    # -------------------------------------------------
    st.subheader("Download Scored Dataset")

    csv = df_display.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Fraud Scored CSV",
        data=csv,
        file_name="fraud_scored_output.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin fraud analysis.")
