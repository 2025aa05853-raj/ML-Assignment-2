# ==========================================
# ML Assignment 2 (Model Training and Evaluation Script)
# ==========================================

# ==========================================
# Load Library
# ==========================================

import os
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
    matthews_corrcoef
)

# ==========================================
# Step 1: Load Dataset
# ==========================================

data = load_breast_cancer()
X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset Shape:", X.shape)
print("Number of Features:", len(feature_names))

