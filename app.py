import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load Model & Encoder ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --- Load Preprocessed Data for Dynamic Options ---
@st.cache_data
def load_preprocessed_data():
    df = pd.read_csv("crop_weather_preprocessed.csv")
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

# Numeric inputs for features used in model
temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
precipitation = st.number_input("Precipitation (mm)", value=10.0, step=0.1)
humidity = st.number_input("Humidity (%)", value=60.0, step=0.1)
radiation = st.number_input("Radiation (MJ/m2)", value=15.0, step=0.1)

# For encoded categorical features, provide selectboxes with values from data or reasonable defaults
soil_types = sorted(df['soil_type_encoded'].dropna().unique())
soil_type_encoded = st.selectbox("Soil Type (Encoded)", soil_types)

irrigation_options = sorted(df['irrigation'].dropna().unique())
irrigation = st.selectbox("Irrigation", irrigation_options)

# --- Prepare Model Input ---

input_dict = {
    "temperature": [temperature],
    "precipitation": [precipitation],
    "humidity": [humidity],
    "radiation": [radiation],
    "soil_type_encoded": [soil_type_encoded],
    "irrigation": [irrigation],
    "crop_species": [crop_species],
    "district": [district]
}

input_df = pd.DataFrame(input_dict)

# Encode categorical columns using your saved encoder
encoded_cat = encoder.transform(input_df[['crop_species', 'district']])
input_df['crop_species_encoded'] = encoded_cat['crop_species']
input_df['district_encoded'] = encoded_cat['district']

# Add cyclic month features
month_num = season_list.index(month) + 1
input_df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
input_df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# Add interaction features
input_df["temp_humidity_interaction"] = input_df["temperature"] * input_df["humidity"]

# Drop original categorical columns after encoding
input_df = input_df.drop(columns=['crop_species', 'district'])

# Ensure all expected columns are present and in correct order
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Fill missing features with zero

st.subheader("🔍 Raw Input Data Before Reordering")
st.write(input_df)

input_df = input_df[feature_columns]

st.write("📄 Model Input Preview", input_df)

# --- Prediction ---
if st.button("🔍 Predict Production"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
