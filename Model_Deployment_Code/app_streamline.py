import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.pkl"

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_FILE, "rb") as f:
    preprocessor = pickle.load(f)

st.title("ASG 04 MD - Farrel - Spaceship Titanic Model Deployment")
st.write("Input 13 features below to predict whether the passenger was transported.")

home_planet = st.selectbox("HomePlanet", ["Europa", "Earth", "Mars"], index=0)
cryo_sleep = st.selectbox("CryoSleep", [True, False], index=0)
cabin_deck = st.selectbox("CabinDeck", ["A", "B", "C", "D", "E", "F", "G", "Unknown"], index=5)
cabin_num = st.number_input("CabinNum", min_value=0, value=100)
cabin_side = st.selectbox("CabinSide", ["P", "S", "Unknown"], index=0)
destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"], index=0)
age = st.number_input("Age", min_value=0.0, value=27.0)
vip = st.selectbox("VIP", [True, False], index=0)
room_service = st.number_input("RoomService", min_value=0.0, value=0.0)
food_court = st.number_input("FoodCourt", min_value=0.0, value=0.0)
shopping_mall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
spa = st.number_input("Spa", min_value=0.0, value=0.0)
vr_deck = st.number_input("VRDeck", min_value=0.0, value=0.0)

input_data = pd.DataFrame(
    {
        "HomePlanet": [home_planet],
        "CryoSleep": [cryo_sleep],
        "CabinDeck": [cabin_deck],
        "CabinNum": [cabin_num],
        "CabinSide": [cabin_side],
        "Destination": [destination],
        "Age": [age],
        "VIP": [vip],
        "RoomService": [room_service],
        "FoodCourt": [food_court],
        "ShoppingMall": [shopping_mall],
        "Spa": [spa],
        "VRDeck": [vr_deck],
    }
)

if st.button("Prediction"):
    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]

    st.subheader("Prediction Results")
    st.success(f"Transported: {prediction}")