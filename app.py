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

print(len(data.data))
st.text("Record Count : " + str(record_count))
st.text("Target Count : " + str(target_count))
st.text("Feature Count : " + str(feature_count))
