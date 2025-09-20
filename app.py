import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("credit_card_fraud.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

st.title("💳 Fraud Detection System")
st.write("This app predicts whether a transaction is **Fraudulent** or **Legitimate** using an XGBoost model.")

# Sidebar for user input
st.sidebar.header("Transaction Features")

def user_input_features():
    # Example features, adjust based on your dataset
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, max_value=100000.0, value=100.0)
    oldbalanceOrg = st.sidebar.number_input("Old Balance (Origin)", min_value=0.0, value=500.0)
    newbalanceOrg = st.sidebar.number_input("New Balance (Origin)", min_value=0.0, value=400.0)
    oldbalanceDest = st.sidebar.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.sidebar.number_input("New Balance (Destination)", min_value=0.0, value=100.0)
    transaction_type = st.sidebar.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])

    # Encode categorical (simple encoding, replace with actual preprocessing if needed)
    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    transaction_type_encoded = type_map[transaction_type]

    features = {
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrg": newbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "type": transaction_type_encoded
    }

    return pd.DataFrame([features])

# Get user input
input_df = user_input_features()

st.subheader("🔎 Transaction Details")
st.write(input_df)

# Make Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected with probability {prediction_proba[0][1]:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction with probability {prediction_proba[0][0]:.2f}")
