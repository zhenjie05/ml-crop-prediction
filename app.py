import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load Model & Encoder ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")

# --- Load Preprocessed Data for Dynamic Options ---
@st.cache_data
def load_preprocessed_data():
    file_path = "crop_weather_preprocessed.csv"
    df = pd.read_csv(file_path)
    return df

df = load_preprocessed_data()

# --- Build Lookup Dictionaries ---
state_district_map = df.groupby('state')['district'].unique().apply(list).to_dict()
crop_type_species_map = df.groupby('crop_type')['crop_species'].unique().apply(list).to_dict()

# --- Static Month List ---
season_list = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# --- App UI ---
st.set_page_config(page_title="Crop Prediction App", layout="centered")
st.title("🌾 Crop Production Prediction")
st.markdown("Predict crop production based on species, location, and season.")

# --- Input Widgets ---
month = st.selectbox("Select Month", season_list)

state = st.selectbox("Select State", sorted(state_district_map.keys()))

district_options = sorted(state_district_map.get(state, []))
district = st.selectbox("Select District", district_options)

crop_type = st.selectbox("Select Crop Type", sorted(crop_type_species_map.keys()))

species_options = sorted(crop_type_species_map.get(crop_type, []))
crop_species = st.selectbox("Select Crop Species", species_options)

# --- Prepare Model Input ---
input_data = pd.DataFrame({
    "crop_species": [crop_species],
    "district": [district]
})

# Encode categorical features
input_encoded = encoder.transform(input_data)

# Add cyclic month features
month_num = season_list.index(month) + 1
input_encoded["month_sin"] = np.sin(2 * np.pi * month_num / 12)
input_encoded["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# --- Prediction ---
if st.button("🔍 Predict Production"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
