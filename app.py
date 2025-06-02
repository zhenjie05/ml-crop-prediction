import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('final_xgb_model.pkl')

st.title("Crop Production Predictor")

st.header("Enter input features:")

# Example inputs – update with your real features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Crop Production: {prediction[0]:.2f}")