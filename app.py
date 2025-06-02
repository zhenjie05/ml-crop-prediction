import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime

# Load the trained model
model = joblib.load('final_model.pkl')

# App Title
st.set_page_config(page_title="Crop Production Predictor", page_icon="🌾", layout="centered")
st.title("🌾 Crop Production Prediction App")
st.markdown("Predict the estimated crop production based on selected parameters.")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2714/2714429.png", width=100)
st.sidebar.header("🧾 About")
st.sidebar.markdown("""
This app uses a machine learning model to estimate crop production 
based on categorical features like crop species, district, type, and season.
""")

# Input section
st.header("📥 Enter Crop Details:")

# Dropdowns and selectors
crop_species = st.selectbox("Crop Species", ['Paddy', 'Wheat', 'Maize', 'Barley'])
district = st.selectbox("District", ['Kuala Muda', 'Kuala Selangor', 'Kemaman', 'Melaka Tengah'])
crop_type = st.selectbox("Crop Type", ['Grain', 'Leafy', 'Root', 'Fruit'])
season = st.date_input("Planting Date (Season Approximation)", datetime.date.today())

# Predict button
if st.button("🔍 Predict Production"):
    # Convert categorical inputs into features expected by model
    # (In a real implementation, use the same preprocessing pipeline as in training)

    # Create dummy encoding (mock example - replace with actual encoder or manual encoding)
    input_df = pd.DataFrame({
        'Crop_Species': [crop_species],
        'District': [district],
        'Crop_Type': [crop_type],
        'Season': [season.month]  # assuming month represents season
    })

    # (Optional) apply actual encoding if required
    # input_encoded = encoder.transform(input_df)  # if you used OneHotEncoder or similar

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Crop Production: **{prediction:.2f} units**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | UM Machine Learning Project")
