import streamlit as st
import pickle
import pandas as pd

# Load trained model and label encoder
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🌱 Agriculture Model for Sustainable Crop Planning")

# Input fields
n = st.number_input("Nitrogen", 0, 150, 90)
p = st.number_input("Phosphorus", 0, 150, 40)
k = st.number_input("Potassium", 0, 150, 42)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("Recommend Crop"):
    sample = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                          columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    preds_proba = model.predict_proba(sample)[0]
    top_idx = preds_proba.argsort()[::-1][:3]
    results = [(le.classes_[i], preds_proba[i]) for i in top_idx]

    st.success("Recommended crops:")
    for crop, prob in results:
        st.write(f"🌾 {crop} (probability: {prob:.2f})")

