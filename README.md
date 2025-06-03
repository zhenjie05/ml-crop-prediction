# 🌾 ML Crop Production Prediction (Malaysia)

## 🧑‍🤝‍🧑 Group Members
- **Lee Zhen Jie** (24059735)
- **Saw Tian Li** (24061227)
- **Evelyn Khiu Sin Yaw** (24004572)
- **Lai Joey** (24004551)
- **Chen Shao Yee** (24004570)

---

## 📌 Project Overview
This machine learning project aims to predict **crop production** by **district in Malaysia** using various features like weather, soil, crop type, and month. We build, compare, and evaluate multiple regression models and deploy the best-performing model using **Streamlit** for interactive predictions.

---

## 📊 Dataset

| Source | Description |
|--------|-------------|
| [Crop Production by District](https://data.gov.my/data-catalogue/crops_district_production?) | Primary production dataset (2015–2021) |
| Weather Data | Historical weather data (temperature, humidity, precipitation, radiation) |
| Soil Data | District-wise soil types and irrigation information |
| Additional | Derived temporal features (e.g., `month_sin`, `month_cos`) and interactions |

---

## 🔄 Workflow

### ✅ Data Preprocessing
- Imputation of missing values
- Outlier handling using IQR
- Log transformation for skewed features
- Target encoding for categorical columns
- RobustScaler normalization
- Feature engineering: cyclical month encoding, temperature-humidity interaction

### 🧠 Model Training
- Models trained:
  - Linear Regression
  - Random Forest Regressor
  - K-Nearest Neighbors
  - Support Vector Regression (SVR)
  - XGBoost Regressor

- Evaluation Metrics:
  - 📉 **RMSE (Root Mean Squared Error)**
  - 📈 **R² Score**
  - 📊 **MAE (Mean Absolute Error)**

- Best model: **XGBoost**  
  Saved model and feature columns are serialized using `joblib`.

---

## 💻 Streamlit App

We built an interactive **Streamlit** web app that allows users to:

- Input crop, district, weather, soil, and temporal data
- Predict expected **crop production**
- View prediction results in real-time

📍 **Open the deployed app:**  
[🌐 Streamlit Crop Predictor (Demo)](https://ml-crop-prediction-czz87xcxsjmzvuvmjxskfl.streamlit.app/)

## 🧪 Project Norebook

You can open our main notebook on Google Colab:
[📔 main_project.ipynb](https://colab.research.google.com/drive/1syJ23EiOkey0Q5Slqbe0izVg8hKVXV-R?usp=sharing)

## 🎥 Demo Video
[📽️ Watch our 5-minute video here](https://youtu.be/your-demo-link)

## 📂 Project Structure
ml-crop-prediction/
├── data/
│   ├── crop_production.csv
│   ├── crop_features.csv
│   └── unique_districts.csv
├── notebooks/
│   └── main_project.ipynb
├── app/
│   ├── app.py                 # Streamlit application
│   ├── final_model.pkl        # Trained XGBoost model
│   ├── feature_columns.pkl    # Features used for model
│   └── target_encoder.pkl     # Encoder used for categorical vars
├── plots/
│   └── scaled_distributions.png
├── video/
│   └── project_demo.mp4
├── slides/
│   └── presentation.pdf
├── preprocessing_log.txt
├── README.md
└── LICENSE
        

## 📄 License
This project is licensed under the [MIT License](LICENSE).
