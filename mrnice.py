import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Mobile Money Fraud Detection System")
st.caption("Isolation Forest – Anomaly Based Risk Scoring")
st.caption(
    "Project by Dorvale, Lesson, Alexander & Abdul • "
    "DataVerse Africa Internship — CAPS Team 3"
)
st.markdown("---")

# =============================
# LOAD MODEL ARTIFACTS
# =============================
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("isolation_forest_fraud_model.joblib")
    return (
        artifacts["model"],
        artifacts["scaler"],
        artifacts["features"],
        artifacts["encoders"]
    )

model, scaler, FEATURES, encoders = load_artifacts()

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
# HELPERS
# =============================
def risk_band(score: float) -> str:
    if score >= 0.75:
        return "High Risk"
    elif score >= 0.40:
        return "Medium Risk"
    return "Low Risk"

def encode_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

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

    return df

# =============================
# SIDEBAR MODE
# =============================
mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Transaction", "Batch CSV Upload"]
)

# =====================================================
# SINGLE TRANSACTION MODE
# =====================================================
if mode == "Single Transaction":

    st.subheader("Single Transaction Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        amount = st.number_input("Transaction Amount (KES)", min_value=0.0, step=100.0)
        location = st.selectbox("Location", location_options)
        device_type = st.selectbox("Device Type", device_type_options)

    with c2:
        network_provider = st.selectbox("Network Provider", network_provider_options)
        user_type = st.selectbox("User Type", user_type_options)
        time_of_day = st.selectbox("Time of Day", time_of_day_options)

    with c3:
        transaction_type = st.selectbox("Transaction Type", transaction_type_options)
        hour = st.slider("Transaction Hour (0–23)", 0, 23, datetime.now().hour)
        is_foreign_number = st.selectbox("Foreign Number Used?", ["No", "Yes"])
        is_sim_recently_swapped = st.selectbox("SIM Recently Swapped?", ["No", "Yes"])
        has_multiple_accounts = st.selectbox("Multiple Accounts?", ["No", "Yes"])

    if st.button("Analyze Transaction", type="primary"):

        raw_df = pd.DataFrame([{
            "amount": amount,
            "hour": hour,
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

        encoded_df = encode_for_model(raw_df)
        X = encoded_df[FEATURES]
        X_scaled = scaler.transform(X)

        anomaly_score = float(-model.decision_function(X_scaled)[0])
        fraud_flag = int(model.predict(X_scaled)[0] == -1)

        st.markdown("---")
        st.subheader("Fraud Assessment Result")

        st.metric("Anomaly Score", f"{anomaly_score:.4f}")
        band = risk_band(anomaly_score)

        if band == "High Risk":
            st.error("🚨 High Risk – Immediate review required")
        elif band == "Medium Risk":
            st.warning("⚠️ Medium Risk – Review recommended")
        else:
            st.success("✅ Low Risk – Transaction appears normal")

# =====================================================
# BATCH CSV MODE
# =====================================================
else:

    st.subheader("Batch Transaction Scoring & Analytics")
    st.info("""
    Upload a CSV containing raw values with these required columns:
    amount, hour, is_foreign_number, is_sim_recently_swapped,
    has_multiple_accounts, transaction_type, location, device_type,
    network_provider, user_type, time_of_day
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)

        encoded_df = encode_for_model(raw_df)
        X = encoded_df[FEATURES]
        X_scaled = scaler.transform(X)

        raw_df["anomaly_score"] = -model.decision_function(X_scaled)
        raw_df["fraud_flag"] = (model.predict(X_scaled) == -1).astype(int)
        raw_df["risk_band"] = raw_df["anomaly_score"].apply(risk_band)

        st.success(f"Successfully scored {len(raw_df):,} transactions")

        # =============================
        # EXECUTIVE KPIs
        # =============================
        st.markdown("---")
        st.subheader("Executive Risk Overview")

        k1, k2, k3, k4 = st.columns(4)

        total_tx = len(raw_df)
        fraud_tx = raw_df["fraud_flag"].sum()
        high_risk_tx = (raw_df["risk_band"] == "High Risk").sum()
        avg_score = raw_df["anomaly_score"].mean()

        k1.metric("Total Transactions", f"{total_tx:,}")
        k2.metric("Fraud Flags", f"{fraud_tx:,}", f"{fraud_tx/total_tx:.1%}")
        k3.metric("High Risk", f"{high_risk_tx:,}")
        k4.metric("Avg Anomaly Score", f"{avg_score:.4f}")

        # =============================
        # RISK DISTRIBUTION
        # =============================
        st.markdown("### Risk Distribution")

        risk_counts = raw_df["risk_band"].value_counts().reindex(
            ["Low Risk","Medium Risk","High Risk"], fill_value=0
        )

        c1, c2 = st.columns(2)

        with c1:
            st.bar_chart(risk_counts)

        with c2:
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.4,
                title="Risk Proportion"
            )
            st.plotly_chart(fig, use_container_width=True)

        # =============================
        # FRAUD SPIKE ANALYSIS
        # =============================
        st.markdown("### Fraud Activity by Hour")

        hourly = (
            raw_df.groupby("hour")
            .agg(total=("fraud_flag","count"), fraud=("fraud_flag","sum"))
            .reset_index()
        )
        hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

        fig = px.line(
            hourly,
            x="hour",
            y="fraud_rate",
            markers=True,
            title="Hourly Fraud Rate"
        )
        st.plotly_chart(fig, use_container_width=True)

        if hourly["fraud_rate"].max() > 0.25:
            st.error("🚨 Fraud spike detected in specific hours")

        # =============================
        # AMOUNT VS RISK
        # =============================
        st.markdown("### Amount vs Anomaly Score")

        fig = px.scatter(
            raw_df,
            x="amount",
            y="anomaly_score",
            color="risk_band",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        # =============================
        # CATEGORY ANALYSIS
        # =============================
        st.markdown("### Risk Breakdown by Category")

        categories = [
            "transaction_type","location","device_type",
            "network_provider","user_type","time_of_day"
        ]

        selected = st.selectbox("Select Category", categories)

        breakdown = (
            raw_df.groupby(selected)["risk_band"]
            .value_counts()
            .unstack(fill_value=0)
        )

        st.dataframe(breakdown)
        st.bar_chart(breakdown)

        # =============================
        # TOP RISKY TRANSACTIONS
        # =============================
        st.markdown("### Top 20 Highest Risk Transactions")

        top_risk = raw_df.sort_values(
            "anomaly_score", ascending=False
        ).head(20)

        st.dataframe(
            top_risk[
                ["anomaly_score","risk_band","amount",
                 "transaction_type","location"]
            ],
            use_container_width=True
        )

        # =============================
        # DOWNLOAD
        # =============================
        st.markdown("---")
        st.download_button(
            "Download Full Scored Dataset",
            raw_df.to_csv(index=False),
            file_name="fraud_scored_transactions.csv",
            mime="text/csv"
        )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "Isolation Forest • Fraud Analytics • "
    f"Operational Monitoring • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
