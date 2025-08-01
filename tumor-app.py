# Import Libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


# Reading Dataset
data = pd.read_csv("data.csv")
X = data[["radius_mean", "perimeter_mean", "area_mean", "compactness_mean", "concavity_mean"]]  # Expanded features
y = data["diagnosis"].map({'M': 1, 'B': 0}).values


# Normalizing Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter Tuning with GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
model = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)


# Best Model
best_model = grid_search.best_estimator_
st.write(f"✅ Best C parameter: **{grid_search.best_params_['C']}**")
st.write(f"📊 Cross-validated accuracy: **{grid_search.best_score_:.2f}**")

def logit2prob(model, x_values):
    log_odds = model.coef_ @ x_values + model.intercept_
    odds = np.exp(log_odds)
    prob = odds / (1 + odds)
    return float(prob)


# UI Streamlit
st.title("Tumor Malignancy Predictor 🔬")
st.write("Predict whether a tumor is likely cancerous based on multiple features.")

tumor_radius = st.number_input("Enter tumor radius (in cm):", min_value=0.0, format="%.2f")
tumor_perimeter = st.number_input("Enter tumor perimeter (in cm):", min_value=0.0, format="%.2f")
tumor_area = st.number_input("Enter tumor area (in cm²):", min_value=0.0, format="%.2f")
compactness = st.number_input("Enter compactness:", min_value=0.0, format="%.4f")
concavity = st.number_input("Enter concavity:", min_value=0.0, format="%.4f")

if st.button("Predict"):
    input_data = np.array([[tumor_radius, tumor_perimeter, tumor_area, compactness, concavity]])
    input_data_scaled = scaler.transform(input_data)
    probability = logit2prob(best_model, input_data_scaled.T)
    st.write(f"📈 Probability of being cancerous: **{probability:.2f}**")

    if probability > 0.5:
        st.error("⚠️ Likely cancerous.")
    else:
        st.success("✅ Likely benign.")