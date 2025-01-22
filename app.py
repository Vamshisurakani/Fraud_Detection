import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("c:/users/vamshi/full_stack/fraud_detection/models/fraud_detection_model.pkl")

# Ensure input data matches training features
def preprocess_input(data, model):
    """
    Preprocess input data to match the model's training features.
    """
    # Get features used during training
    required_features = model.feature_names_in_

    # Add missing features with default values
    for feature in required_features:
        if feature not in data:
            data[feature] = 0

    # Drop extra features
    data = data[required_features]
    return data

# Predict fraud
def predict_fraud(input_data):
    """
    Predict fraud using the trained model.
    """
    input_data = preprocess_input(input_data, model)  # Match input data to model's features
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    return prediction, prediction_proba

# Streamlit App Interface
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# Input form for user data
st.sidebar.header("Enter Transaction Details:")
amt = st.sidebar.number_input("Transaction Amount (USD):", min_value=0.0, step=1.0)
lat = st.sidebar.number_input("Latitude of Merchant:", step=0.0001)
long = st.sidebar.number_input("Longitude of Merchant:", step=0.0001)
merch_lat = st.sidebar.number_input("Merchant Latitude:", step=0.0001)
merch_long = st.sidebar.number_input("Merchant Longitude:", step=0.0001)
city_pop = st.sidebar.number_input("City Population:", min_value=0, step=1)

# Prepare input data
features = {
    "amt": amt,
    "lat": lat,
    "long": long,
    "merch_lat": merch_lat,
    "merch_long": merch_long,
    "city_pop": city_pop,
}

# Convert input data to DataFrame
input_df = pd.DataFrame([features])

# Predict button
if st.button("Predict Fraud"):
    try:
        prediction, prediction_proba = predict_fraud(input_df)
        if prediction[0] == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! Confidence: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(f"✅ Legitimate Transaction! Confidence: {prediction_proba[0][0]*100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write("**Note:** This model is trained on a sample dataset and may not reflect real-world accuracy.")
