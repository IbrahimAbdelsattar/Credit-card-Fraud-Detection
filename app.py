import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("credit_card_fraud.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

# Title
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
    This app uses a trained **XGBoost model** on the **2023 Credit Card Fraud Dataset**  
    to predict whether a transaction is **Fraudulent (1)** or **Legitimate (0)**.  
    """
)

# --- Input Section ---
st.subheader("📝 Enter Transaction Features")

with st.form("input_form"):
    cols = st.columns(3)
    input_data = {}

    # Define features (must match training data)
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    for idx, feature in enumerate(features):
        col = cols[idx % 3]
        input_data[feature] = col.number_input(
            f"{feature}", 
            value=0.0, 
            step=0.01, 
            format="%.4f"
        )

    submitted = st.form_submit_button("🔍 Predict Fraud")

# --- Prediction ---
if submitted:
    # Convert input to DataFrame with exact column order
    input_df = pd.DataFrame([[input_data[f] for f in features]], columns=features)

    st.subheader("📊 Transaction Details")
    st.write(input_df)

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected with probability {prediction_proba[0][1]:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction with probability {prediction_proba[0][0]:.2%}")
