# ==========================================
# Load Library
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
# ==========================================
# Load Dataset
# ==========================================
data = load_breast_cancer()
X = data.data  # Features (30 features)
y = data.target  # Labels (0: malignant, 1: benign)
# Split 80/20 training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

record_count = len(data.data)
target_count = len(data.target)
feature_count = len(data.feature_names)

# ==========================================
# Define Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}
# ==========================================
# Dataset upload option (CSV)
# ==========================================
st.header("Dataset Upload Option")
uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain 'target' column)",
    type=["csv"]
)

# ==========================================
# Model selection dropdown (if multiple models)
# ==========================================

st.header("Model Selection")

model_options_dropdown = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_options_dropdown]
st.text("Selected Model is : " + model_options_dropdown)
# Train Selected Model
if model_options_dropdown in ["Logistic Regression", "KNN"]:
    selected_model.fit(X_train_scaled, y_train)
    y_pred_test = selected_model.predict(X_test_scaled)
    y_prob_test = selected_model.predict_proba(X_test_scaled)[:, 1]
else:
    selected_model.fit(X_train, y_train)
    y_pred_test = selected_model.predict(X_test)
    y_prob_test = selected_model.predict_proba(X_test)[:, 1]

# ==========================================
# Uploaded Dataset Evaluation (First)
# ==========================================

if uploaded_file is not None:
    st.markdown("---")
    st.header("Evaluation on Uploaded Dataset")
    st.text("Uploaded File Path : " + str(uploaded_file))
    uploaded_data = pd.read_csv(uploaded_file)
    if "target" not in uploaded_data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        st.text("Uploaded CSV has a 'target' column.")
        X_uploadedData = uploaded_data.drop("target", axis=1)
        y_uploadedData = uploaded_data["target"]
        st.text("Uploaded File Path X_uploadedData: " + str(len(X_uploadedData)))
        st.text("Uploaded File Path y_uploadedData: " + str(len(X_uploadedData)))
else:
    st.text("No data")
