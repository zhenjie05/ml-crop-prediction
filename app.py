import streamlit as st
import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

# --- Load Model & Encoders ---
model = joblib.load("final_model.pkl")
encoder = joblib.load("target_encoder.pkl")
soil_type_encoder = joblib.load("soil_type_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --- Load Preprocessed Data ---
@st.cache_data
def load_preprocessed_data():
    df = pd.read_csv("crop_weather_preprocessed.csv")
    return df

df = load_preprocessed_data()

# --- Build Lookup Dictionaries ---
state_district_map = df.groupby('state')['district'].unique().apply(list).to_dict()
crop_type_species_map = df.groupby('crop_type')['crop_species'].unique().apply(list).to_dict()
season_list = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# --- App UI ---
st.set_page_config(page_title="🌾 Smart Crop Production Predictor", layout="wide")

# Title and polished description
st.title("🌾 Smart Crop Production Predictor")

st.markdown("""
Welcome to the **Smart Crop Production Predictor**, an interactive tool powered by a machine learning model that estimates crop yields based on multiple environmental and regional inputs.

This app helps farmers, researchers, and agronomists simulate crop production outcomes using real-world weather, soil, and crop data.

Use the sidebar to configure your **crop type, species, and region**, then adjust environmental conditions and soil settings to explore how they affect production.
""")

# --- Sidebar for Global Selections ---
with st.sidebar:
    st.header("📌 Crop & Location Settings")
    month = st.selectbox("🌙 Month", season_list)
    state = st.selectbox("🗺️ State", sorted(state_district_map.keys()))
    
    # Title-case district names for display, use original for model
    district_options = sorted(state_district_map.get(state, []))
    district_display_map = {d.title(): d for d in district_options}
    district_display = st.selectbox("🏙️ District", list(district_display_map.keys()))
    district = district_display_map[district_display]  # use original for encoding

    # Title-case crop types for display, use original for logic
    crop_type_options = sorted(crop_type_species_map.keys())
    crop_type_display_map = {c.title(): c for c in crop_type_options}
    crop_type_display = st.selectbox("🌱 Crop Type", list(crop_type_display_map.keys()))
    crop_type = crop_type_display_map[crop_type_display]  # use original key

    crop_species = st.selectbox("🧬 Crop Species", sorted(crop_type_species_map.get(crop_type, [])))
    st.markdown("---")

# --- Environmental Conditions ---
st.subheader("🌡️ Environmental Conditions")
with st.expander("Adjust Environmental Conditions", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("🌡️ Temperature (°C)", 5.0, 45.0, 25.0, 0.1,
                                help="Typical growing temperatures range from 15°C to 35°C.")
        precipitation = st.slider("🌧️ Precipitation (mm)", 0.0, 400.0, 10.0, 0.1,
                                  help="Annual rainfall depending on region and crop requirements.")
    with col2:
        humidity = st.slider("💧 Humidity (%)", 10.0, 100.0, 60.0, 0.1)
        radiation = st.slider("☀️ Radiation (MJ/m²)", 5.0, 35.0, 15.0, 0.1)

# --- Soil and Irrigation ---
st.subheader("🌾 Soil & Irrigation Options")
with st.expander("Soil & Irrigation Details", expanded=True):
    soil_types = sorted(df['soil_type'].dropna().unique())
    soil_type = st.selectbox("🪨 Soil Type", soil_types)

    irrigation_map = {0: "No Irrigation", 1: "Irrigation"}
    irrigation_label = st.radio("🚿 Irrigation Method", list(irrigation_map.values()),
                                help="Choose whether irrigation is applied.")
    irrigation = [key for key, value in irrigation_map.items() if value == irrigation_label][0]

# --- Input Summary ---
with st.expander("📊 Review Your Inputs"):
    st.markdown(f"**🗓️ Month:** {month}")
    st.markdown(f"**🗺️ State/District:** {state} / {district}")
    st.markdown(f"**🌱 Crop Type/Species:** {crop_type} / {crop_species}")
    st.markdown(f"**🌡️ Temp:** {temperature}°C  |  💧 Humidity: {humidity}%")
    st.markdown(f"**🌧️ Precipitation:** {precipitation}mm  |  ☀️ Radiation: {radiation} MJ/m²")
    st.markdown(f"**🪨 Soil Type:** {soil_type}  |  🚿 Irrigation: {irrigation_label}")

st.divider()

# --- Prepare Model Input ---
soil_type_encoded = soil_type_encoder.transform([soil_type])[0]

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

# Encode categorical columns
encoded_cat = encoder.transform(input_df[['crop_species', 'district']])
input_df['crop_species_encoded'] = encoded_cat['crop_species']
input_df['district_encoded'] = encoded_cat['district']

# Add cyclic month features
month_num = season_list.index(month) + 1
input_df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
input_df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# Interaction feature
input_df["temp_humidity_interaction"] = input_df["temperature"] * input_df["humidity"]

# Drop original categorical columns
input_df = input_df.drop(columns=['crop_species', 'district'])

# Ensure all expected columns are present and in correct order
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# --- Prediction ---
if st.button("🔍 Predict Production"):
    try:
        input_df.columns = [str(i) for i in range(len(input_df.columns))]
        expected_features = model.n_features_in_
        input_df = input_df.iloc[:, :expected_features]
        input_df = input_df.astype(float)        
        prediction = model.predict(input_df)[0]
        st.success(f"🌱 Estimated Crop Production: **{prediction:.2f} units**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.divider()
st.caption("📌 *Note: This tool provides estimated crop production values based on machine learning models trained on historical data. Always consult local agricultural experts for critical decisions.*")
