# Mobile Money Fraud Detection & Risk Analytics Across Africa
### Unsupervised Machine Learning & Power BI

**Organisation:** DataVerse Africa
**Type:** Internship Project Presentation
**Analyst:** Lesson Shepherd Karidza

---

## Overview

This project documents an end-to-end fraud analytics pipeline conducted during a Data Science internship at DataVerse Africa. The project applies unsupervised machine learning to detect hidden fraud patterns in mobile money transactions across Africa — without relying on pre-labeled fraud data.

The solution combines anomaly detection modelling with interactive Power BI dashboards to deliver real-time fraud intelligence to operations and risk teams.

---

## Problem Statement

Africa is the global leader in mobile money, processing billions of real-time digital transactions daily across platforms such as M-Pesa, MTN Mobile Money, and Airtel Money. The rapid growth of mobile money adoption has created an expanding fraud attack surface, with threat vectors including:

- SIM swap fraud
- Social engineering attacks
- Agent-assisted scams
- Compromised account takeovers

Traditional rule-based fraud detection systems are static and struggle to adapt to evolving fraud behaviours. This creates financial losses, delayed detection, and eroded customer trust — driving the need for intelligent, adaptive, data-driven fraud detection.

---

## Objectives

**Primary Objective**
Uncover hidden fraud patterns in mobile money transactions using unsupervised machine learning, without requiring labeled training data.

**Secondary Objectives**
- Transform raw transaction data into actionable fraud intelligence through advanced feature engineering and analytics
- Empower fraud teams with real-time insights by identifying high-risk users, agents, and transaction channels
- Deliver a scalable, visualised dashboard for ongoing fraud monitoring and investigation

---

## Methodology

| Stage | Description |
|---|---|
| **1. Data Collection & Preprocessing** | Raw mobile money transaction data ingested, deduplicated, type-cast, and cleaned of null records |
| **2. Feature Engineering & Normalisation** | Derived risk signals including transaction velocity, channel, device type, balance flow, verification status, and time-based features; features standardised with `StandardScaler` |
| **3. Unsupervised ML Modelling** | Isolation Forest (grid-search tuned) and Autoencoder models applied to detect anomalies without labels |
| **4. Dashboard & Visualisation** | Power BI dashboards built for real-time fraud investigation and stakeholder reporting |

---

## Dataset Summary

- **Total transactions analysed:** ~10,000
- **Anomalies flagged:** 199 (anomaly rate: 0.02%)
- **Estimated financial exposure:** $1.24M
- **Geographic coverage:** Multiple Kenyan cities including Kisumu, Nairobi, Thika, Eldoret, and Garissa
- **Channels covered:** Agent, App, USSD
- **Devices covered:** Android, iOS, Feature Phone

---

## Technical Implementation

The core anomaly-detection pipeline (`fraud_detect.ipynb`) was built in Python and follows these steps:

**1. Data ingestion & cleaning**
- Source data loaded from an Excel export (`kenya_fraud_detection.xlsx`)
- Dropped redundant index column, parsed `datetime`, removed null rows, and reset the index

**2. Feature engineering**
- Time-based features: `hour`, `dayofweek`, `month`, `is_weekend`
- Channel derivation: transactions classified into `Agent`, `App`, or `USSD` based on `user_type` and `device_type`
- Running balance tracking: `balance_before` / `balance_after` computed per user via cumulative sums of cash deposits
- Verification flag: `is_mobile_verified`, derived from `is_foreign_number` and `is_sim_recently_swapped`
- User behaviour aggregates: `user_txn_count`, `user_total_amount`, `user_avg_amount`
- Categorical encoding of `transaction_type`, `location`, `device_type`, `network_provider`, `user_type`, `time_of_day`, and `channel` via `LabelEncoder`

**3. Modelling — Isolation Forest**
- Feature matrix standardised with `StandardScaler`
- Grid search across `n_estimators` (100–300), `contamination` (0.005–0.02), `max_samples` (`auto`, 0.6, 0.8), and `max_features` (0.6, 0.8, 1.0), ranked by score stability (`std_score`) and `max_score`
- Final production configuration:
  ```
  n_estimators = 200
  contamination = 0.02
  max_samples = 'auto'
  max_features = 1.0
  random_state = 42
  ```
- Each transaction scored with `decision_function` (inverted to an `anomaly_score`) and flagged via `is_anomalous` where `predict() == -1`

**4. Modelling — Autoencoder**
- A neural network (TensorFlow/Keras) trained to reconstruct normal transaction patterns; transactions with high reconstruction error are flagged as anomalies

**5. Outputs**
- `fraud_detection_processed.csv` — cleaned, feature-engineered dataset
- `fraud_detection_with_anomalies.csv` — final dataset with `anomaly_score` and `is_anomalous` columns
- `isolation_forest_fraud_model.joblib` — serialized bundle containing the fitted model, scaler, feature list, and final hyperparameters (via `joblib`)

Both models operate without labeled fraud data — enabling deployment in environments where historical fraud labels are incomplete or unavailable.

---

## Key Findings

### Volume & Financial Exposure
- 199 of ~10,000 transactions flagged as anomalous — low frequency but high financial impact per event
- Estimated fraud-related losses of **$1.24M** confirm high-value exposure despite a low anomaly rate

### Temporal Patterns
- Fraud peaks occur on **weekends, particularly Sundays** — highest daily anomaly rate of 0.04 on June 2 and June 23
- Anomaly volumes concentrate during **early-morning and late-afternoon hours**, indicating deliberate exploitation of predictable operating windows
- Fraud is **equally active on weekdays and weekends** — confirming risk is continuous, not calendar-dependent

### Channel Risk
- The **Agent channel** is the dominant source of anomalies, followed by the App channel
- The **Agent + Android device combination** accounts for 40 fraud instances — 20% of all fraud transactions — the single highest-risk combination identified
- USSD shows concentrated risk among Feature Phone users

### Geographic Risk
- **Kisumu** has the highest anomaly count (25 instances, 2.64% anomaly rate) despite lower total transaction volume — the highest-risk region
- **Garissa** has the lowest anomaly count (17 instances, 1.6%) despite the highest transaction volume — suggesting isolated risk pockets rather than widespread regional vulnerability
- **Kisumu, Thika, and Eldoret** are the primary geographic fraud hotspots

### Transaction Type Risk
- **Buy Airtime** accounts for 17.42% of total transaction value and **26.7% of fraud cases** — the highest-risk transaction type
- **Send Money** follows as the second highest fraud category

### User & Agent Risk
- Fraud exposure is **highly concentrated among a small subset of users and agents**
- Top high-risk users show elevated anomaly scores despite low transaction counts — indicating quality-of-risk over volume, consistent with account misuse
- Risk concentration is most pronounced in Agent and USSD channels, with repeat appearances in specific locations suggesting **network-linked or behavioural fraud clusters**

### Device Risk
- **Android devices** drive the largest share of anomalies, particularly in Agent and App channels
- Feature phones show concentrated risk within USSD
- iOS exposure is material but secondary

### Verification Limitations
- The **majority of anomalous activity originates from mobile-verified users** — confirming that KYC verification alone is not a sufficient risk control and must be complemented with behavioural analytics

---

## Models Used

| Model | Type | Purpose |
|---|---|---|
| **Isolation Forest** | Unsupervised anomaly detection | Flags statistically rare transactions based on feature isolation depth; hyperparameters selected via grid search |
| **Autoencoder** | Neural network (unsupervised) | Learns normal transaction patterns and flags high reconstruction-error transactions as anomalies |

---

## Recommendations

### Short-Term (Immediate)
- Deploy unsupervised fraud detection for real-time anomaly alerting
- Use live Power BI dashboards to enable rapid fraud investigation and response
- Implement stricter controls and enhanced monitoring specifically on Agent and App channels
- Introduce device fingerprinting and device-specific thresholds — particularly for Android in Agent-mediated transactions
- Deploy time-based risk weighting and dynamic detection thresholds during identified fraud peak windows

### Long-Term (Strategic)
- Integrate MLOps pipelines for continuous model learning and improvement as new fraud patterns emerge
- Combine unsupervised and supervised models to build a resilient hybrid fraud defence system
- Adopt tiered user risk profiling with automated escalation for high-score or repeat-risk entities
- Enable geo-targeted alerts and regional performance reviews for hotspot locations
- Shift from reactive fraud response to a proactive, real-time, predictive risk operating model

---

## Implementation Priorities

| Priority | Action |
|---|---|
| Risk-weighted controls | Prioritise anomaly scores and loss severity over raw transaction volume |
| Channel-focused mitigation | Concentrate controls on Agent and App — not spread evenly across all channels |
| User-centric profiling | Tiered risk profiles with automated restrictions for high-risk users and agents |
| Device-aware monitoring | Device fingerprinting and behavioural baselining, especially Android + Agent |
| Time-adaptive detection | Dynamic thresholds during fraud peak hours — not static rule sets |
| Geo-targeted intervention | Location-level alerts for Kisumu, Thika, and Eldoret hotspots |
| Continuous behavioural monitoring | Complement KYC verification with velocity and behavioural analytics |
| Proactive operating model | Real-time alerts and predictive scoring before financial loss materialises |

---

## Conclusion

This project demonstrates that unsupervised machine learning can effectively detect fraud in mobile money systems without relying on labeled data. The combination of Isolation Forest and Autoencoder models, supported by real-time Power BI dashboards, delivers:

- Faster fraud detection before financial loss materialises
- Better fraud investigation through explainable risk signals
- Informed decision-making for fraud and compliance teams

The solution is scalable, adaptive, and well-suited for Africa's fast-growing digital finance ecosystem — where fraud patterns evolve rapidly and labeled training data is often scarce.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Data preprocessing, feature engineering, modelling |
| Isolation Forest (scikit-learn) | Unsupervised anomaly detection |
| Autoencoder (TensorFlow / Keras) | Deep learning anomaly detection |
| Power BI | Real-time fraud dashboard and visualisation |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Exploratory analysis and visualisation |
| joblib | Model serialization |

---

## Repository Contents

| File | Description |
|---|---|
| `fraud_detect.ipynb` | End-to-end notebook: data cleaning, feature engineering, Isolation Forest tuning/training, anomaly scoring, and model export |
| `fraud_detection_processed.csv` | Cleaned, feature-engineered dataset (pre-scoring) |
| `fraud_detection_with_anomalies.csv` | Final dataset with anomaly scores and flags |
| `isolation_forest_fraud_model.joblib` | Serialized model bundle (model, scaler, features, hyperparameters) |

---

## Presentation Structure

| Slide | Section |
|---|---|
| 1 | Title — Mobile Money Fraud Detection & Risk Analytics |
| 2 | Context and Justification |
| 3 | Project Objectives |
| 4 | Methodology |
| 5 | Data Insights 1/5 — Anomaly Rate, Temporal & Channel Risk |
| 6 | Data Insights 2/5 — Device Risk & Transaction Type Risk |
| 7 | Data Insights 3/5 — User Risk & Verification Limitations |
| 8 | Data Insights 4/5 — Geographic Hotspots & Agent Risk |
| 9 | Data Insights 5/5 — Time Patterns & Location-Level Analysis |
| 10 | Recommendations |
| 11 | Implementation Framework |
| 12 | Conclusion |
| 13 | Thank You |

---

## Author

**Lesson Shepherd Karidza**
Data Science Intern — DataVerse Africa
BSc Data Science | Adikavi Nannaya University
lessonshepherdkaridza@gmail.com

---

## Related Projects

- [AgriSense Zimbabwe](https://agri-sense-zimbabwe.streamlit.app) — Maize yield prediction and farmer-buyer platform
- Health Insurance Cost Prediction — Gradient Boosting regression model with Streamlit deployment
- Loan Approval Risk Assessment — Credit risk classification with feature engineering and explainability
