import streamlit as st
import joblib
import numpy as np

model = joblib.load("aqi_model.pkl")

st.title("AQI Prediction System")

# Get feature names from model
feature_names = model.feature_names_in_

inputs = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

if st.button("Predict AQI"):
    input_array = np.array([inputs])
    prediction = model.predict(input_array)[0]

    st.subheader(f"Predicted AQI: {round(prediction,2)}")