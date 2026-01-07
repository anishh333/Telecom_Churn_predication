import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Telecom Churn Analysis",
    layout="centered"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------
# Title Section
# ---------------------------
st.markdown(
    "<div class='title-card'>"
    "<h1>Telecom Churn Prediction</h1>"
    "<p>Logistic Regression Model Dashboard</p>"
    "</div>",
    unsafe_allow_html=True
)



# ---------------------------
# Dataset Preview
# ---------------------------
st.subheader("Dataset Preview")
df = pd.read_csv("./churn.csv")
st.dataframe(df.head())


# ---------------------------
# Data Preprocessing
# ---------------------------
st.subheader("Data Preprocessing")

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

st.success("Preprocessing completed successfully.")


# ---------------------------
# Feature Selection
# ---------------------------
st.subheader("Input & Output Variables")

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

st.write("**Input Features:**", list(X.columns))
st.write("**Target Variable:** Churn")


# ---------------------------
# Train-Test Split & Scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------
# Model Training
# ---------------------------
st.subheader("Model Building")

model = LogisticRegression()
model.fit(X_train, y_train)

st.success("Logistic Regression model trained successfully.")


# ---------------------------
# Model Performance
# ---------------------------
st.subheader("Model Performance")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.metric(label="Accuracy", value=f"{accuracy:.2f}")

st.text("Classification Report")
st.text(report)


col1, col2 = st.columns(2)

with col1:
    st.metric("Customers Likely to Churn", cm[1, 1])

with col2:
    st.metric("Customers Likely to Stay", cm[0, 0])


# ---------------------------
# User Input & Prediction
# ---------------------------
st.subheader("Churn Prediction (User Input)")

st.markdown(
    "<div class='input-card'>"
    "<h4>Enter Customer Details</h4>",
    unsafe_allow_html=True
)

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)

if st.button("Predict Churn"):
    user_data = np.array([[tenure, monthly_charges, total_charges]])
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        st.markdown(
            f"<div class='result-box safe'>"
            f"<h4>Customer is likely to continue with the company</h4>"
            f"<p>Probability: {probability:.2f}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box danger'>"
            f"<h4>Customer is  likely to discontinue with the company</h4>"
            f"<p>Probability: {probability:.2f}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
