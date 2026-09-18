
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

# Load trained model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")
st.write("Enter house details to predict the estimated price.")

med_inc = st.number_input(
    "Median Income",
    min_value=0.0,
    value=5.0
)

house_age = st.number_input(
    "House Age (years)",
    min_value=0.0,
    value=20.0
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    value=6.0
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0
)

ave_occup = st.number_input(
    "Average Occupants",
    min_value=0.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    value=34.0
)

longitude = st.number_input(
    "Longitude",
    value=-118.0
)

if st.button("Predict House Price"):

    new_house = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    prediction = model.predict(new_house)

    price = prediction[0] * 100000

    st.success(
        f"Estimated House Price: ${price:,.2f}"
    )
