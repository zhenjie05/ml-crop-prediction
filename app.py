import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load model and encoder ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")

# --- App Configuration ---
st.set_page_config(page_title="Crop Prediction App", layout="centered")
st.title("🌾 Crop Production Prediction")
st.markdown("Predict crop production based on species, location, and season.")

# --- Static Lists (replace with actual options if needed) ---
state_district_map = {
    "kedah": ["pendang", "pokok sena"],
    "selangor": ["kuala selangor", "sekinchan"],
    "perak": ["bagan serai", "parit buntar"],
    "kelantan": ["tumpat", "pasir mas"]
}

crop_type_species_map = {
    "main": ["paddy", "maize"],
    "off": ["cassava", "sweet potato"]
}

season_list = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# --- User Inputs ---
month = st.selectbox("Select Month", season_list)

state = st.selectbox("Select State", list(state_district_map.keys()))

districts = state_district_map[state]
district = st.selectbox("Select District", districts)

crop_type = st.selectbox("Select Crop Type", list(crop_type_species_map.keys()))

species_options = crop_type_species_map[crop_type]
crop_species = st.selectbox("Select Crop Species", species_options)

# --- Prepare Input for Model ---
input_data = pd.DataFrame({
    "crop_species": [crop_species],
    "district": [district],
})

# Apply Target Encoding
input_encoded = encoder.transform(input_data)

# Add cyclic encoding for month
month_num = season_list.index(month) + 1
input_encoded["month_sin"] = np.sin(2 * np.pi * month_num / 12)
input_encoded["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# --- Predict ---
if st.button("🔍 Predict Production"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
