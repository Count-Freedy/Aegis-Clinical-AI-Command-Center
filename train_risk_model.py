import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

print("[Aegis] Training Micro-Triage Model (XGBoost)...")

file_path = 'dataset/hospital_readmission.csv'

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}. Please ensure it is in the 'dataset' folder.")
    exit()

# 1. Load the dataset
df = pd.read_csv(file_path)
print(f"-> Successfully loaded {file_path} ({len(df)} records found).")

# 2. Preprocessing
# Identify the target column 
target_candidates = [col for col in df.columns if 'readmit' in col.lower() or 'target' in col.lower()]
target_col = target_candidates[0] if target_candidates else df.columns[-1]
print(f"-> Detected target column: '{target_col}'")

# Robust Categorical Encoding: Catch ALL non-numeric columns
le = LabelEncoder()
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = le.fit_transform(df[col].astype(str))

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Guarantee XGBoost compatibility by enforcing strict numeric types
X = X.astype(float)

# If target is not binary (0/1), force it to binary for classification
if len(y.unique()) > 2:
    y = (y > y.median()).astype(int) 
else:
    y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Model
model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.08, 
    max_depth=4, 
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\n=== Model Validation Report ===")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.3f}")

# 5. Export Model & Feature Names for Streamlit UI
with open('xgboost_risk_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': list(X.columns)}, f)
print("-> Serialized: xgboost_risk_model.pkl\n")