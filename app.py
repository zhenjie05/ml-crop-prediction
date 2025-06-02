import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# Load model and encoder objects
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")  # Make sure you saved this during training

st.set_page_config(page_title="Crop Prediction App", layout="centered")
st.title("🌾 Crop Production Prediction")
st.markdown("Predict crop production based on species, location, and season.")

# Input options (ensure these match training data)
crop_species_list = ['paddy', 'maize', 'cassava', 'sweet potato']  # update based on your data
district_list = ['kedah', 'selangor', 'perak', 'kelantan']         # update based on your data
crop_type_list = ['main', 'off']                                   # if included in model
season_list = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# User Inputs
crop_species = st.selectbox("Select Crop Species", crop_species_list)
district = st.selectbox("Select District", district_list)
crop_type = st.selectbox("Select Crop Type", crop_type_list)
season = st.selectbox("Select Month", season_list)

# Prepare input as DataFrame
input_data = pd.DataFrame({
    'crop_species': [crop_species],
    'district': [district]
})

# Encode categorical variables
input_encoded = encoder.transform(input_data)

# Add cyclic encoding for month
month_num = season_list.index(season) + 1
input_encoded['month_sin'] = np.sin(2 * np.pi * month_num / 12)
input_encoded['month_cos'] = np.cos(2 * np.pi * month_num / 12)

# Make prediction
if st.button("🔍 Predict Production"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
