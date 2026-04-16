import streamlit as st
import joblib
import pandas as pd
import sklearn
import streamlit as st


st.write("Cloud scikit-learn version:", sklearn.__version__)

st.title("🚲 Bike Rental Prediction App")

st.write("Enter the values below to predict bike rental demand.")

# Try loading the model
try:
    model = joblib.load("best_model.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.write("Error details:", e)
    st.stop()

# Input fields
season = st.selectbox("Season (1–4)", [1, 2, 3, 4])
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
windspeed = st.number_input("Wind Speed", min_value=0.0, max_value=60.0)

# Predict button
if st.button("Predict"):
    data = {
        "season": season,
        "temp": temp,
        "humidity": humidity,
        "windspeed": windspeed
    }
    df = pd.DataFrame([data])
    try:
        prediction = model.predict(df)[0]
        st.success(f"Predicted Bike Rentals: {prediction:.2f}")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Error details:", e)
