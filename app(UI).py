import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("aqi_model.pkl")

st.set_page_config(page_title="AQI Prediction", layout="centered")

st.title("🌍 AQI Prediction System")

st.write("Enter pollution values to predict next-day AQI")

# Inputs
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")

if st.button("Predict AQI"):
    
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)[0]

    st.subheader(f"Predicted AQI: {round(prediction,2)}")

    # AQI Alerts
    if prediction > 400:
        st.error("🚨 Severe Pollution! Stay Indoors!")
    elif prediction > 300:
        st.error("Very Poor Air Quality")
    elif prediction > 200:
        st.warning("Poor Air Quality")
    elif prediction > 100:
        st.info("Moderate Air Quality")
    else:
        st.success("Good Air Quality")