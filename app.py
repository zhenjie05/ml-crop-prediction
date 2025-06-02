import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load Model & Encoder ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # list of feature names as saved from training

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

# --- Streamlit App UI ---
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

# --- Prepare Input Data ---
input_data = pd.DataFrame({
    "crop_species": [crop_species],
    "district": [district]
})

# --- Encode categorical variables ---
encoded_array = encoder.transform(input_data)

# Convert encoded output to DataFrame if it's not already, with correct column names
if not isinstance(encoded_array, pd.DataFrame):
    # Adjust these column names if your encoder produces different names/order
    encoded_df = pd.DataFrame(encoded_array, columns=['crop_species_encoded', 'district_encoded'])
else:
    encoded_df = encoded_array.copy()

# --- Add cyclic month features ---
month_num = season_list.index(month) + 1
encoded_df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
encoded_df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# --- Add missing features with default zeros ---
for feature in feature_columns:
    if feature not in encoded_df.columns:
        encoded_df[feature] = 0

# --- Reorder columns to match model training features exactly ---
input_for_model = encoded_df[feature_columns]

# --- Predict ---
if st.button("🔍 Predict Production"):
    try:
        prediction = model.predict(input_for_model)[0]
        st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
