# Aegis: Clinical AI Command Center 🛡️
**Dual-Model Architecture for Predictive Patient Triage & Macro-Resource Forecasting**


Aegis is an enterprise-grade healthcare data product designed to bridge the gap between individual clinical risk and hospital-wide logistical strain. By integrating a micro-level gradient-boosted classifier with a macro-level time-series forecasting engine, Aegis provides hospital administrators and Chief Medical Officers with a unified, actionable mission control dashboard.

## 🚨 The Problem
Modern hospitals suffer from a structural disconnect between **clinical data** (patient vitals, history) and **logistical data** (bed availability, staffing). Traditional AI projects isolate these models—predicting a disease without considering if the hospital has the physical capacity to treat an influx of patients. This siloed approach leads to ER bottlenecks, ICU saturation, and reactive rather than proactive resource management.

## 💡 The Solution
Aegis solves this by processing both pipelines simultaneously:
1. **Micro-Level:** Point-of-care triage scoring for individual patients to determine immediate readmission/admission risk.
2. **Macro-Level:** Aggregated hospital-wide time-series forecasting to predict operational bottlenecks 7 days before they occur. 

---

## 📊 Datasets & Data Strategy
This project was trained and validated on real-world clinical datasets, rigorously preprocessed to handle categorical mapping and ensure robust feature engineering.

* **Clinical Risk Dataset (XGBoost):** Trained on a 32,300-record hospital readmission dataset. Features include patient demographics, primary vital markers, length of stay, number of diagnoses, and prior admission history.
* **Resource Forecasting Dataset (ARIMA):** Trained on a 10,000-record time-series dataset aggregated into 417 days of continuous daily hospital occupancy metrics, capturing strong weekly cyclicality.

## 📈 Quantifiable Outputs & Model Metrics
Both models achieved high-performance validation metrics, demonstrating strong clinical applicability rather than simple data memorization:

**Patient Triage Classifier (XGBoost):**
* **ROC-AUC Score:** `0.870` (Indicating exceptional separability between high-risk and low-risk clinical profiles)
* **Accuracy:** `78%` across 6,460 test records.
* **F1-Score:** `0.78` (Balanced precision and recall for binary readmission targeting).

**Hospital Capacity Forecaster (ARIMA):**
* **Configuration:** $ARIMA(p=7, d=1, q=1)$ natively capturing the 7-day hospital admission harmonic.
* **Fit Quality:** AIC: `1288.57` | BIC: `1324.84`
* **Performance:** Outputted a highly stable 7-day predictive corridor bounded within realistic physical thresholds.

---

## 🏗️ System Architecture

The application operates on a decoupled data pipeline, ensuring clinical inference and resource forecasting function concurrently without computational bottlenecking.

```mermaid
graph TD
    subgraph Data Layer
        A[Historical Bed Log / Daily Census] -->|Time Series 417d| B(ARIMA Engine)
        C[Live ER Patient Vitals & History] -->|Tabular Clinical Data| D(XGBoost Engine)
    end

    subgraph Aegis Dual-Model Processing
        B -->|Order: 7, 1, 1| E[7-Day Macro Occupancy Forecast]
        D -->|Feature Weighting & Logloss| F[Individual Readmission Risk Score]
    end

    subgraph Aegis Clinical Command Center
        E --> G[Capacity Horizon & Surge Alert Dashboard]
        F --> H[Live Triage Prioritization Queue]
        I[Clinical Copilot & Vision Screening] --> J[Decision Support System]
    end

    G --> K((Hospital Administrator / CMO))
    H --> K
    J --> K


 ---

🛠️ Tech Stack
Machine Learning Backend: Python, XGBoost, Scikit-Learn, Statsmodels (ARIMA)
Data Processing & Engineering: Pandas, NumPy
Frontend Application: Streamlit, Plotly Graph Objects (Interactive Data Viz)
Deployment Assets: Pickle serialization, Mermaid.js

---

Install required dependencies:
python -m pip install -r requirements.txt

Train and serialize the backend models:
python train_risk_model.py
python train_forecasting_model.py

Launch the Aegis Command Center Dashboard:
python -m streamlit run app.py