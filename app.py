import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load Model & Encoders ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")
soil_type_encoder = joblib.load("soil_type_encoder.pkl")  # load soil type encoder
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
st.divider()

st.header("📋 Select Crop and Location")
col1, col2 = st.columns(2)
with col1:
    month = st.selectbox("🌙 Select Month", season_list)
    state = st.selectbox("🗺️ Select State", sorted(state_district_map.keys()))
    district_options = sorted(state_district_map.get(state, []))
    district = st.selectbox("🏙️ Select District", district_options)
with col2:
    crop_type = st.selectbox("🌱 Select Crop Type", sorted(crop_type_species_map.keys()))
    species_options = sorted(crop_type_species_map.get(crop_type, []))
    crop_species = st.selectbox("🧬 Select Crop Species", species_options)

st.divider()

st.header("🌡️ Environmental Conditions")
col3, col4 = st.columns(2)
with col3:
    temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
    precipitation = st.number_input("Precipitation (mm)", value=10.0, step=0.1)
with col4:
    humidity = st.number_input("Humidity (%)", value=60.0, step=0.1)
    radiation = st.number_input("Radiation (MJ/m2)", value=15.0, step=0.1)

st.divider()

st.header("🌾 Soil and Irrigation")

# Show soil type names (not encoded) for user to select
soil_types = sorted(df['soil_type'].dropna().unique())
soil_type = st.selectbox("Soil Type", soil_types)

irrigation_options = sorted(df['irrigation'].dropna().unique())
irrigation = st.selectbox("Irrigation Method", irrigation_options)

st.divider()

# --- Prepare Model Input ---

# Encode soil type now
soil_type_encoded = soil_type_encoder.transform([soil_type])[0]

input_dict = {
    "temperature": [temperature],
    "precipitation": [precipitation],
    "humidity": [humidity],
    "radiation": [radiation],
    "soil_type_encoded": [soil_type_encoded],  # encoded soil_type here
    "irrigation": [irrigation],
    "crop_species": [crop_species],
    "district": [district]
}

input_df = pd.DataFrame(input_dict)

# Encode categorical columns using saved encoder
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

# Reorder columns to match saved feature_columns
input_df = input_df[feature_columns]

# --- Prediction ---
if st.button("🔍 Predict Production"):
    try:
        # Rename columns to match model's expected format (0, 1, 2, ...)
        input_df.columns = [str(i) for i in range(len(input_df.columns))]

        # Keep only the number of features the model expects
        expected_features = model.n_features_in_
        input_df = input_df.iloc[:, :expected_features]

        # Now safe to predict
        prediction = model.predict(input_df)[0]
        st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.divider()
st.caption("📌 *Note: The prediction is based on a pre-trained model and may not fully capture all seasonal or regional variations.*")
