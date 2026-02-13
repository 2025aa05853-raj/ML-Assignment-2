import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
record_count = len(data.data)
target_count = len(data.target)
print(len(data.data))
st.text("Record Count : " + str(record_count))
st.text("Target Count : " + str(target_count))
