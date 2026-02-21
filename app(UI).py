import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AQI Forecast System", layout="centered")

st.title("🌫️ AQI Forecast System")
st.write("Predict Tomorrow's AQI Based on Historical Data")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("aqi_model.pkl")

model = load_model()

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Air_quality_data.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    cols_to_keep = ["Datetime","City","PM2.5","PM10","NO2","SO2","CO","O3","AQI"]
    df = df[cols_to_keep]

    df = df.fillna(method="ffill").dropna()

    # Target
    df["AQI_t+1"] = df["AQI"].shift(-1)

    # Lag + Rolling features (same as training)
    for col in ["AQI","PM2.5","PM10","NO2","SO2","CO","O3"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_rmean3"] = df[col].rolling(3).mean()
        df[f"{col}_rmean7"] = df[col].rolling(7).mean()

    df = pd.get_dummies(df, columns=["City"], drop_first=True)

    df = df.dropna().reset_index(drop=True)

    return df

df = load_data()

FEATURES = [c for c in df.columns if c not in ["Datetime","AQI","AQI_t+1"]]

# ----------------------------
# USER INPUT
# ----------------------------

# Extract city names properly
city_columns = [col for col in df.columns if col.startswith("City_")]
base_city = "Base City"
city_list = [base_city] + [c.replace("City_", "") for c in city_columns]

selected_city = st.selectbox("Select City", city_list)

available_dates = df["Datetime"].dt.date.unique()
selected_date = st.selectbox("Select Date", available_dates)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Tomorrow AQI"):

    # Filter by date
    row = df[df["Datetime"].dt.date == selected_date]

    # Filter by city if not base
    if selected_city != base_city:
        row = row[row[f"City_{selected_city}"] == 1]

    if row.empty:
        st.error("No data available for selected city/date.")
    else:
        X_input = row.iloc[0][FEATURES].values.reshape(1, -1)
        prediction = model.predict(X_input)[0]

        # ----------------------------
        # AQI CATEGORY
        # ----------------------------
        def get_category(aqi):
            if aqi <= 50:
                return "Good 🟢"
            elif aqi <= 100:
                return "Satisfactory 🟡"
            elif aqi <= 200:
                return "Moderate 🟠"
            elif aqi <= 300:
                return "Poor 🔴"
            elif aqi <= 400:
                return "Very Poor 🟣"
            else:
                return "Severe ⚫"

        category = get_category(prediction)

        # ----------------------------
        # DISPLAY RESULTS
        # ----------------------------
        st.success(f"Predicted AQI (Tomorrow): {round(prediction,2)}")
        st.info(f"Category: {category}")

        # ----------------------------
        # ALERT SYSTEM
        # ----------------------------
        if 300 < prediction <= 400:
            st.warning("⚠️ ALERT: Air quality is VERY POOR. Avoid outdoor activities and wear masks.")
        
        elif prediction > 400:
            st.error("🚨 SEVERE ALERT: Hazardous air quality! Stay indoors and avoid all outdoor exposure.")