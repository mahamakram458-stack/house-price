import joblib
import pandas as pd
import streamlit as st

st.title("California House Price Predictor")

model = joblib.load("model.pkl")

col1, col2 = st.columns(2)
with col1:
    med_inc = st.number_input("Median income (in $10,000s)", 0.5, 15.0, 3.5)
    house_age = st.number_input("House age (years)", 1, 52, 25)
    ave_rooms = st.number_input("Average rooms", 1.0, 20.0, 5.0)
    ave_bedrms = st.number_input("Average bedrooms", 0.5, 5.0, 1.0)
with col2:
    population = st.number_input("Population of area", 3, 35000, 1200)
    ave_occup = st.number_input("Average occupancy", 1.0, 10.0, 3.0)
    latitude = st.number_input("Latitude", 32.5, 42.0, 34.0)
    longitude = st.number_input("Longitude", -124.4, -114.3, -118.2)

if st.button("Predict price"):
    row = pd.DataFrame([{
        "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms, "Population": population,
        "AveOccup": ave_occup, "Latitude": latitude, "Longitude": longitude,
    }])
    price = model.predict(row)[0] * 100000
    st.success(f"Predicted price: ${price:,.0f}")