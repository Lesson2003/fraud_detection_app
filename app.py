import streamlit as st
import pandas as pd
import joblib

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    layout="wide"
)

st.title("Mobile Money Fraud Detection System")
st.caption("Isolation Forest – Anomaly Based Risk Scoring")
st.caption(
    "This project was done by Dorvale, Lesson, Alexander  and Abdul "
    "as part of the DataVerse Africa Internship — CAPS Team 3."
)
st.markdown("---")

# =============================
# LOAD MODEL ARTIFACTS
# =============================
artifacts = joblib.load("isolation_forest_fraud_model.joblib")

model = artifacts["model"]
scaler = artifacts["scaler"]
FEATURES = artifacts["features"]
encoders = artifacts["encoders"]

# =============================
# UI OPTIONS (RAW VALUES)
# =============================
location_options = [
    "Nairobi","Mombasa","Kisumu","Nakuru","Nyeri",
    "Machakos","Meru","Garissa","Eldoret","Thika","Unknown"
]

device_type_options = ["Android","iOS","Feature Phone"]
network_provider_options = ["Safaricom","Airtel","Telkom Kenya","Unknown"]
user_type_options = ["Individual","Agent"]
time_of_day_options = ["morning","afternoon","evening","night"]

transaction_type_options = [
    "Withdraw Cash","Send Money","Deposit Cash",
    "Lipa na M-Pesa","Buy Airtime","Pay Bill"
]

# =============================
# RISK BAND FUNCTION
# =============================
def risk_band(score):
    if score >= 0.75:
        return "High Risk"
    elif score >= 0.40:
        return "Medium Risk"
    return "Low Risk"

# =============================
# MODE SELECTION
# =============================
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Transaction", "Batch CSV Upload"]
)

# =====================================================
# SINGLE TRANSACTION MODE
# =====================================================
if mode == "Single Transaction":

    st.subheader("Transaction Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        amount = st.number_input("Transaction Amount", min_value=0.0)
        location = st.selectbox("Location", location_options)
        device_type = st.selectbox("Device Type", device_type_options)

    with c2:
        network_provider = st.selectbox("Network Provider", network_provider_options)
        user_type = st.selectbox("User Type", user_type_options)
        time_of_day = st.selectbox("Time of Day", time_of_day_options)

    with c3:
        transaction_type = st.selectbox("Transaction Type", transaction_type_options)
        is_foreign_number = st.selectbox("Foreign Number", ["Yes","No"])
        is_sim_recently_swapped = st.selectbox("SIM Recently Swapped", ["Yes","No"])
        has_multiple_accounts = st.selectbox("Multiple Accounts", ["Yes","No"])

    if st.button("Run Fraud Analysis"):

        df = pd.DataFrame([{
            "amount": amount,
            "hour": 12,
            "is_foreign_number": 1 if is_foreign_number == "Yes" else 0,
            "is_sim_recently_swapped": 1 if is_sim_recently_swapped == "Yes" else 0,
            "has_multiple_accounts": 1 if has_multiple_accounts == "Yes" else 0,
            "transaction_type": transaction_type,
            "location": location,
            "device_type": device_type,
            "network_provider": network_provider,
            "user_type": user_type,
            "time_of_day": time_of_day
        }])

        # Derived feature
        df["is_mobile_verified"] = (
            (df["is_foreign_number"] == 0) &
            (df["is_sim_recently_swapped"] == 0)
        ).astype(int)

        # Safe encoding
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else "UNKNOWN"
                )
                df[col] = le.transform(df[col])

        X = df[FEATURES]
        X_scaled = scaler.transform(X)

        anomaly_score = float(-model.decision_function(X_scaled)[0])
        fraud_flag = int(model.predict(X_scaled)[0] == -1)

        st.markdown("---")
        st.subheader("Fraud Assessment")

        st.metric("Anomaly Score", round(anomaly_score, 4))
        st.write("Risk Band:", risk_band(anomaly_score))

        if fraud_flag:
            st.error("⚠️ Likely a Fraud;Consider Further Investigation")
        else:
            st.success("✅ Transaction Appears Legitimate")

# =====================================================
# BATCH CSV MODE
# =====================================================
else:

    st.subheader("Batch Transaction Scoring")
    st.info(
        "CSV must contain raw (non-encoded) values for:\n"
        "amount, hour, is_foreign_number, is_sim_recently_swapped, "
        "has_multiple_accounts, transaction_type, location, device_type, "
        "network_provider, user_type, time_of_day"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Derived feature
        df["is_mobile_verified"] = (
            (df["is_foreign_number"] == 0) &
            (df["is_sim_recently_swapped"] == 0)
        ).astype(int)

        # Safe encoding
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else "UNKNOWN"
                )
                df[col] = le.transform(df[col])

        X = df[FEATURES]
        X_scaled = scaler.transform(X)

        df["anomaly_score"] = -model.decision_function(X_scaled)
        df["fraud_flag"] = (model.predict(X_scaled) == -1).astype(int)
        df["risk_band"] = df["anomaly_score"].apply(risk_band)

        st.success("Batch fraud scoring completed")

        st.dataframe(df.head())

        st.download_button(
            "Download Scored Dataset",
            df.to_csv(index=False),
            file_name="fraud_scored_transactions.csv",
            mime="text/csv"
        )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "Isolation Forest–based fraud analytics. "
    "Built for operational monitoring and Power BI integration."
)
