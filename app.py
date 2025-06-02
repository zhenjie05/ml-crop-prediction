import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load Model & Encoder ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --- Load Preprocessed Data ---
@st.cache_data
def load_preprocessed_data():
    return pd.read_csv("crop_weather_preprocessed.csv")

df = load_preprocessed_data()

# --- Build Lookup Dictionaries ---
state_district_map = df.groupby('state')['district'].unique().apply(list).to_dict()
crop_type_species_map = df.groupby('crop_type')['crop_species'].unique().apply(list).to_dict()

# --- Static Lists ---
season_list = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# --- UI Layout ---
st.set_page_config(page_title="Crop Prediction App", layout="centered")
st.title("🌾 Crop Production Prediction")
st.markdown("Predict crop production based on species, location, and season.")

# --- Input Widgets ---
month = st.selectbox("Select Month", season_list)
state = st.selectbox("Select State", sorted(state_district_map.keys()))
district = st.selectbox("Select District", sorted(state_district_map.get(state, [])))
crop_type = st.selectbox("Select Crop Type", sorted(crop_type_species_map.keys()))
crop_species = st.selectbox("Select Crop Species", sorted(crop_type_species_map.get(crop_type, [])))

# --- Prepare Input DataFrame ---
input_data = pd.DataFrame({
    "crop_species": [crop_species],
    "district": [district]
})

# --- Encode Categorical Features ---
input_encoded = encoder.transform(input_data)

# --- Add Engineered Features ---
month_num = season_list.index(month) + 1
input_encoded["month_sin"] = np.sin(2 * np.pi * month_num / 12)
input_encoded["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# Add default numeric values if necessary
default_values = {
    'temperature': df['temperature'].mean(),
    'precipitation': df['precipitation'].mean(),
    'humidity': df['humidity'].mean(),
    'radiation': df['radiation'].mean(),
    'soil_type_encoded': df['soil_type_encoded'].mode()[0],
    'irrigation': df['irrigation'].mode()[0],
    'temp_humidity_interaction': df['temperature'].mean() * df['humidity'].mean(),
    'log_production': np.log1p(df['production'].mean())
}

for feature in feature_columns:
    if feature not in input_encoded.columns:
        input_encoded[feature] = default_values.get(feature, 0)

# --- Align Input to Feature Columns ---
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# --- Make Prediction ---
if st.button("🔍 Predict Production"):
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
