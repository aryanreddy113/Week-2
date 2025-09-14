import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load the model
model = joblib.load("model.pkl")

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title(" AQI Prediction")
st.markdown("Enter pollutant levels to predict the AQI and check if air is breathable.")

# Sidebar for input fields
st.sidebar.header("Enter Pollutant Levels")

PM2_5 = st.sidebar.number_input("PM2.5", min_value=0.0, max_value=1000.0 )
PM10 = st.sidebar.number_input("PM10", min_value=0.0, max_value=1000.0)
NO = st.sidebar.number_input("NO", min_value=0.0, max_value=500.0)
NO2 = st.sidebar.number_input("NO2", min_value=0.0, max_value=500.0)
NOx = st.sidebar.number_input("NOx", min_value=0.0, max_value=500.0)
NH3 = st.sidebar.number_input("NH3", min_value=0.0, max_value=500.0)
CO = st.sidebar.number_input("CO", min_value=0.0, max_value=500.0)
SO2 = st.sidebar.number_input("SO2", min_value=0.0, max_value=500.0)
O3 = st.sidebar.number_input("O3", min_value=0.0, max_value=500.0)
Benzene = st.sidebar.number_input("Benzene", min_value=0.0, max_value=50.0)
Toluene = st.sidebar.number_input("Toluene", min_value=0.0, max_value=50.0)

if st.sidebar.button("Predict AQI"):
    input_data = pd.DataFrame({
        'PM2.5': [PM2_5],
        'PM10': [PM10],
        'NO': [NO],
        'NO2': [NO2],
        'NOx': [NOx],
        'NH3': [NH3],
        'CO': [CO],
        'SO2': [SO2],
        'O3': [O3],
        'Benzene': [Benzene],
        'Toluene': [Toluene]
    })

    prediction = model.predict(input_data)[0]

    st.subheader(f"Predicted AQI: {prediction:.2f}")

    if prediction <= 200:
        st.success("Air Quality: Breathable")
    else:
        st.error("Air Quality: Not Breathable")

st.markdown("---")
st.caption("This app uses a trained Random Forest Regressor to predict AQI based on pollutant levels.")
