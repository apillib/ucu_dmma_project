
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Extreme Rainfall Prediction App")
st.write("Predict whether a day is likely to be an extreme rainfall day.")

model = joblib.load("Apilli best model.joblib")

st.sidebar.header("Input Daily Weather Features")

temp_max_c = st.sidebar.number_input("Max Temperature (°C)", value=30.0)
temp_min_c = st.sidebar.number_input("Min Temperature (°C)", value=20.0)
temp_mean_c = st.sidebar.number_input("Mean Temperature (°C)", value=25.0)
wind_speed_max_kmh = st.sidebar.number_input("Max Wind Speed (km/h)", value=15.0)
surface_pressure_hpa = st.sidebar.number_input("Surface Pressure (hPa)", value=893.0)
relative_humidity_max_pct = st.sidebar.number_input("Max Relative Humidity (%)", value=80.0)

year = st.sidebar.number_input("Year", value=2025, step=1)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=4, step=1)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15, step=1)
day_of_year = st.sidebar.number_input("Day of Year", min_value=1, max_value=366, value=105, step=1)
week_of_year = st.sidebar.number_input("Week of Year", min_value=1, max_value=53, value=16, step=1)
quarter = st.sidebar.number_input("Quarter", min_value=1, max_value=4, value=2, step=1)

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
dayofyear_sin = np.sin(2 * np.pi * day_of_year / 365.25)
dayofyear_cos = np.cos(2 * np.pi * day_of_year / 365.25)

precip_lag1 = st.sidebar.number_input("Previous Day Rainfall (mm)", value=2.0)
precip_lag3 = st.sidebar.number_input("Rainfall 3 Days Ago (mm)", value=1.0)
temp_mean_lag1 = st.sidebar.number_input("Previous Day Mean Temperature (°C)", value=24.8)
humidity_lag1 = st.sidebar.number_input("Previous Day Humidity (%)", value=78.0)
wind_lag1 = st.sidebar.number_input("Previous Day Wind Speed (km/h)", value=14.5)

precip_roll3 = st.sidebar.number_input("3-day Avg Rainfall (mm)", value=4.0)
precip_roll7 = st.sidebar.number_input("7-day Avg Rainfall (mm)", value=5.5)
temp_roll3 = st.sidebar.number_input("3-day Avg Temperature (°C)", value=25.0)
humidity_roll3 = st.sidebar.number_input("3-day Avg Humidity (%)", value=79.0)

input_df = pd.DataFrame([{
    "temp_max_c": temp_max_c,
    "temp_min_c": temp_min_c,
    "temp_mean_c": temp_mean_c,
    "wind_speed_max_kmh": wind_speed_max_kmh,
    "surface_pressure_hpa": surface_pressure_hpa,
    "relative_humidity_max_pct": relative_humidity_max_pct,
    "year": year,
    "month": month,
    "day": day,
    "day_of_year": day_of_year,
    "week_of_year": week_of_year,
    "quarter": quarter,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "dayofyear_sin": dayofyear_sin,
    "dayofyear_cos": dayofyear_cos,
    "precip_lag1": precip_lag1,
    "precip_lag3": precip_lag3,
    "temp_mean_lag1": temp_mean_lag1,
    "humidity_lag1": humidity_lag1,
    "wind_lag1": wind_lag1,
    "precip_roll3": precip_roll3,
    "precip_roll7": precip_roll7,
    "temp_roll3": temp_roll3,
    "humidity_roll3": humidity_roll3
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0, 1]
    st.subheader("Prediction Result")
    st.write("Extreme Rain Event:", "Yes" if pred == 1 else "No")
    st.write("Probability:", round(float(prob), 4))
