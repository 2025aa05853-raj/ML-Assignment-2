**a. Problem statement**

This project implements and compares multiple Machine Learning classification models using a real-world medical dataset.

The objective is to evaluate how different algorithms perform on the same dataset using standard performance metrics and present the results through an interactive Streamlit web application.

The project demonstrates a complete end-to-end ML workflow including:
Data preprocessing
Model training
Model evaluation
Performance comparison
Web deployment using Streamlit

**b. Dataset Description**

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

The dataset was loaded using:
from sklearn.datasets import load_breast_cancer

Dataset Details:
569 total instances
30 numerical features

Binary classification problem:
0 → Malignant
1 → Benign

This dataset satisfies assignment requirements:
Minimum 500 instances
Minimum 12 features

Data Preprocessing

The following steps were performed before training:
Dataset loading
Train-test split (80% training, 20% testing)
Feature scaling using StandardScaler (for models requiring scaling)

**c. Models used**

The following six classification models were implemented:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest (Ensemble – Bagging)
XGBoost (Ensemble – Boosting)

Evaluation Metrics Used
Each model was evaluated using the following metrics:
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC Score)

Model results are stored in:
model/model_results.csv

**Model Performance Comparison**

| ML Model Name              | Accuracy  | AUC   | Precision | Recall | F1       | MCC   |
|----------------------------|-----------|-------|-----------|--------|----------|------ |
| Logistic Regression        | 0.97      | 0.99  | 0.98      | 0.97   | 0.97     | 0.94  |
| Decision Tree              | ...       | ...   | ...       | ...    | ...      | ...   |
| KNN                        | ...       | ...   | ...       | ...    | ...      | ...   |
| Naive Bayes                | ...       | ...   | ...       | ...    | ...      | ...   |
| Random Forest (Ensemble)   | ...       | ...   | ...       | ...    | ...      | ...   |
| XGBoost (Ensemble)         | ...       | ...   | ...       | ...    | ...      | ...   |


**Model Performance Comparison**

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Performs very well on this dataset due to good linear separability of features. Shows strong accuracy, precision, and AUC. Provides a good balance between bias and variance. |
| Decision Tree | Achieves good accuracy but may slightly overfit the training data. Performance is reasonable but less stable compared to ensemble methods. |
| K-Nearest Neighbors (KNN) | Performance improves significantly after feature scaling. Sensitive to the choice of K and distance metric. Works well but may be slower for larger datasets. |
| Naive Bayes | Performs moderately well. Since it assumes feature independence, it may not fully capture complex relationships between features, leading to slightly lower performance compared to other models. |
| Random Forest (Ensemble) | Shows strong and stable performance due to ensemble learning. Reduces overfitting by combining multiple decision trees and provides high generalization ability. |
| XGBoost (Ensemble) | Often achieves the best or near-best performance. Uses sequential boosting and regularization to reduce errors and overfitting. Captures complex patterns effectively and delivers high predictive power. |


**Streamlit Application Features**

The Streamlit web application includes:
  Dataset upload option (CSV with target column)
  Model selection dropdown
  Evaluation metrics display
  Confusion matrix
  Classification report
  Model performance comparison table

The app dynamically trains and evaluates models based on user selection.

**Project Structure**

ML_Assignment_2/
│
├── app.py                      # Streamlit web application
├── model/
│   ├── train_models.py         # Model training & evaluation script
│   └── model_results.csv       # Saved model performance results
│
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation


**How to Run the Project**
Run through Streamlit App URL (Mentioned below):
https://ml-assignment-2-ajffoyej9h3pkimlabefzr.streamlit.app/

Above URL is mentioned in the assignment attached PDF file.

Alternate option is to install dependency and call the model python file like below:
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Generate Model Results
python model/train_models.py

