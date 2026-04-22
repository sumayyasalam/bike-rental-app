import streamlit as st
import joblib
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Load data for EDA
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")  
    return df

df = load_data()


st.write("Cloud scikit-learn version:", sklearn.__version__)
st.header("📊 Exploratory Data Analysis")
st.write("Below are some visual insights from the bike rental dataset.")
st.subheader("Distribution of Total Bike Rentals")

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['cnt'], kde=True, color='teal', ax=ax)
ax.set_title("Distribution of Total Bike Rentals")
ax.set_xlabel("Total Rentals")
ax.set_ylabel("Frequency")

st.pyplot(fig)
st.subheader("Hourly Bike Rental Trend")

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=df, x='hr', y='cnt', ax=ax)
ax.set_title("Hourly Bike Rental Trend")
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Total Rentals")

st.pyplot(fig)

st.subheader("Bike Rentals by Weekday")

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=df, x='weekday', y='cnt', palette='viridis', ax=ax)
ax.set_title("Bike Rentals by Weekday")
ax.set_xlabel("Weekday (0 = Sunday)")
ax.set_ylabel("Total Rentals")

st.pyplot(fig)



#st.write("Cloud scikit-learn version:", sklearn.__version__)

st.title("🚲 Bike Rental Prediction App")
st.write("Enter the values below to predict bike rental demand.")

# Set background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #E3F2FD;  /* Light blue */
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0); /* Transparent header */
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# Load model
try:
    model = joblib.load("best_model.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.write("Error details:", e)
    st.stop()

# Input fields
season = st.selectbox("Season (1–4)", [1, 2, 3, 4])
temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0)
windspeed = st.slider("Wind Speed", min_value=0.0, max_value=60.0)

# Predict button
if st.button("Predict"):
    data = {
        "season": season,
        "temp": temp,
        "hum": humidity,      # FIXED: model expects 'hum'
        "windspeed": windspeed
    }

    df = pd.DataFrame([data])

    try:
        prediction = model.predict(df)[0]
        st.success(f"Predicted Bike Rentals: {prediction:.2f}")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Error details:", e)
