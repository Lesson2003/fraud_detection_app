import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.express as px

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
def load_model_artifacts():
    try:
        artifacts = joblib.load("isolation_forest_fraud_model.joblib")
        return (
            artifacts["model"],
            artifacts["scaler"],
            artifacts["features"],
            artifacts["encoders"]
        )
    except FileNotFoundError:
        st.error("Model file 'isolation_forest_fraud_model.joblib' not found in app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, scaler, FEATURES, encoders = load_model_artifacts()

# =============================
# UI OPTIONS
# =============================
location_options = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Nyeri",
    "Machakos", "Meru", "Garissa", "Eldoret", "Thika", "Unknown"
]

device_type_options = ["Android", "iOS", "Feature Phone"]
network_provider_options = ["Safaricom", "Airtel", "Telkom Kenya", "Unknown"]
user_type_options = ["Individual", "Agent"]
time_of_day_options = ["morning", "afternoon", "evening", "night"]
transaction_type_options = [
    "Withdraw Cash", "Send Money", "Deposit Cash",
    "Lipa na M-Pesa", "Buy Airtime", "Pay Bill"
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

# =============================
# SIDEBAR – MODE SELECTION
# =============================
mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Transaction", "Batch CSV Upload"],
    index=0
)

# =====================================================
# SINGLE TRANSACTION MODE
# =====================================================
if mode == "Single Transaction":

    st.subheader("Enter Transaction Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        amount = st.number_input("Transaction Amount (KES)", min_value=0.0, step=100.0, format="%.2f")
        location = st.selectbox("Location", location_options, index=0)
        device_type = st.selectbox("Device Type", device_type_options)

    with c2:
        network_provider = st.selectbox("Network Provider", network_provider_options)
        user_type = st.selectbox("User Type", user_type_options)
        time_of_day = st.selectbox("Time of Day", time_of_day_options)

    with c3:
        transaction_type = st.selectbox("Transaction Type", transaction_type_options)
        trans_hour = st.slider("Transaction Hour (0–23)", 0, 23, datetime.now().hour)
        is_foreign_number = st.selectbox("Foreign Number Used?", ["No", "Yes"])
        is_sim_recently_swapped = st.selectbox("SIM Recently Swapped?", ["No", "Yes"])
        has_multiple_accounts = st.selectbox("Multiple Accounts Linked?", ["No", "Yes"])

    if st.button("Analyze Transaction", type="primary"):

        with st.spinner("Scoring transaction..."):

            df = pd.DataFrame([{
                "amount": amount,
                "hour": trans_hour,
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

            X = df[FEATURES].copy()
            X_scaled = scaler.transform(X)

            anomaly_score = float(-model.decision_function(X_scaled)[0])
            is_fraud = int(model.predict(X_scaled)[0] == -1)

            st.markdown("---")
            st.subheader("Fraud Assessment Result")

            col_score, col_risk = st.columns([1, 2])

            with col_score:
                st.metric("Anomaly Score", f"{anomaly_score:.4f}")

            with col_risk:
                risk = risk_band(anomaly_score)
                if risk == "High Risk":
                    st.error(f"**{risk}** – Immediate attention recommended")
                elif risk == "Medium Risk":
                    st.warning(f"**{risk}** – Review recommended")
                else:
                    st.success(f"**{risk}** – Appears normal")

            if is_fraud:
                st.error("⚠️ **Flagged as potential fraud** – Consider manual verification / block")
            else:
                st.success("✅ Transaction appears legitimate")

            with st.expander("Technical Details"):
                st.write("Model:", type(model).__name__)
                st.write("Features used:", ", ".join(FEATURES))

# =====================================================
# BATCH CSV UPLOAD + ANALYSIS MODE
# =====================================================
else:

    st.subheader("Batch Scoring & Analysis – Multiple Transactions")

    st.info("""
    Upload a CSV containing **raw values** with these columns (required):
    • amount
    • hour (0–23)
    • is_foreign_number (0/1)
    • is_sim_recently_swapped (0/1)
    • has_multiple_accounts (0/1)
    • transaction_type
    • location
    • device_type
    • network_provider
    • user_type
    • time_of_day

    Optional helpful columns: customer_id, transaction_id, date, account_id
    """)

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Processing batch and generating analysis..."):
            try:
                df_raw = pd.read_csv(uploaded_file)
                df = df_raw.copy()

                # Derived feature
                if all(col in df.columns for col in ["is_foreign_number", "is_sim_recently_swapped"]):
                    df["is_mobile_verified"] = (
                        (df["is_foreign_number"] == 0) &
                        (df["is_sim_recently_swapped"] == 0)
                    ).astype(int)
                else:
                    st.warning("Missing 'is_foreign_number' and/or 'is_sim_recently_swapped' → skipping 'is_mobile_verified'")

                # Safe encoding + track unknowns
                unknown_flags = {}
                for col, le in encoders.items():
                    if col in df.columns:
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in le.classes_ else "UNKNOWN"
                        )
                        transformed = le.transform(df[col])
                        unknown_count = (transformed == le.transform(["UNKNOWN"])[0]).sum()
                        if unknown_count > 0:
                            unknown_flags[col] = unknown_count
                        df[col] = transformed

                if unknown_flags:
                    st.warning("Values mapped to 'UNKNOWN': " + 
                               ", ".join(f"{k}: {v}" for k,v in unknown_flags.items()))

                # Scoring
                X = df[FEATURES].copy()
                X_scaled = scaler.transform(X)

                df["anomaly_score"] = -model.decision_function(X_scaled)
                df["fraud_flag"] = (model.predict(X_scaled) == -1).astype(int)
                df["risk_band"] = df["anomaly_score"].apply(risk_band)

                st.success(f"Successfully scored **{len(df):,}** transactions!")

                # ────────────────────────────────────────────────
                # ANALYSIS SECTION
                # ────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("Batch Analysis & Insights")

                tab1, tab2, tab3 = st.tabs([
                    "📊 Summary Overview",
                    "⚠️ High-Risk Transactions",
                    "🔍 Category Breakdown"
                ])

                with tab1:
                    colA, colB, colC = st.columns(3)

                    total_tx = len(df)
                    fraud_count = df["fraud_flag"].sum()
                    high_risk_count = (df["risk_band"] == "High Risk").sum()
                    avg_anomaly = df["anomaly_score"].mean()

                    colA.metric("Total Transactions", f"{total_tx:,}")
                    colB.metric("Flagged as Fraud", fraud_count,
                               delta=f"{fraud_count/total_tx:.1%}" if total_tx > 0 else "0%",
                               delta_color="inverse")
                    colC.metric("High Risk Transactions", high_risk_count,
                               delta=f"{high_risk_count/total_tx:.1%}" if total_tx > 0 else "0%")

                    st.markdown("**Risk Distribution**")
                    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
                    risk_counts = df["risk_band"].value_counts().reindex(risk_order, fill_value=0)

                    col_chartA, col_chartB = st.columns([3, 2])

                    with col_chartA:
                        st.bar_chart(risk_counts)

                    with col_chartB:
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            color=risk_counts.index,
                            color_discrete_map={
                                "Low Risk": "#4CAF50",
                                "Medium Risk": "#FFC107",
                                "High Risk": "#F44336"
                            },
                            title="Risk Band Proportion"
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("**Average Anomaly Score by Risk Band**")
                    avg_by_risk = df.groupby("risk_band")["anomaly_score"].mean().round(4)
                    st.dataframe(avg_by_risk.rename("Avg Anomaly Score"))

                with tab2:
                    st.markdown("**Top 10 Highest Anomaly Scores (Most Suspicious)**")

                    top_risky = df.sort_values("anomaly_score", ascending=False).head(10)

                    display_cols = ["anomaly_score", "risk_band", "fraud_flag", "amount"]
                    optional_cols = ["transaction_type", "location", "customer_id", "transaction_id"]
                    for col in optional_cols:
                        if col in df.columns:
                            display_cols.append(col)

                    st.dataframe(
                        top_risky[display_cols].style.format({
                            "anomaly_score": "{:.4f}",
                            "amount": "KES {:,.0f}"
                        }).highlight_between(
                            subset=["anomaly_score"],
                            left=0.75,
                            color="#ffcccc"
                        ),
                        use_container_width=True
                    )

                    if not top_risky.empty:
                        st.download_button(
                            "↓ Download Top 10 Riskiest Transactions",
                            top_risky.to_csv(index=False),
                            file_name="top_10_high_risk_transactions.csv",
                            mime="text/csv"
                        )

                with tab3:
                    st.markdown("**Risk Distribution by Category**")

                    cat_columns = ["transaction_type", "location", "device_type",
                                   "network_provider", "user_type", "time_of_day"]
                    available = [c for c in cat_columns if c in df.columns]

                    if available:
                        selected = st.selectbox("Select category", available, index=0)

                        if selected:
                            breakdown = df.groupby(selected)["risk_band"].value_counts().unstack(fill_value=0)
                            breakdown["Total"] = breakdown.sum(axis=1)
                            breakdown = breakdown.sort_values("Total", ascending=False)

                            st.dataframe(breakdown.style.format("{:,}"))

                            st.bar_chart(breakdown.drop(columns="Total", errors="ignore"))
                    else:
                        st.info("No suitable categorical columns found for breakdown.")

                # ────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("Full Scored Dataset (first 500 rows shown)")

                st.dataframe(df.head(500))

                csv_full = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Complete Scored Results",
                    data=csv_full,
                    file_name="fraud_scored_transactions_full.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.info("Common causes: missing required columns, wrong data types, or encoding issues. Try UTF-8 CSV.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "Isolation Forest • Anomaly Detection • "
    "Built for real-time monitoring & Power BI integration • "
    f"App running • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)