import streamlit as st
import numpy as np
import pickle

with open('airbnb_price_model_scratch.pkl', 'rb') as f:
    artifact = pickle.load(f)

weights = artifact['weights']
train_mean = artifact['train_mean']
train_std = artifact['train_std']
feature_names = artifact['feature_names']

st.title("Airbnb NYC Price Predictor")

# Basic inputs -- adjust these to match the real columns your model uses
neighbourhood_group = st.selectbox("Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=3)
availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=180)

if st.button("Predict Price"):
    # Build a feature row of all zeros, then set the ones we know
    row = {name: 0 for name in feature_names}
    row["minimum_nights"] = minimum_nights
    row["availability_365"] = availability_365

    gcol = f"neighbourhood_group_{neighbourhood_group}"
    if gcol in row:
        row[gcol] = 1
    rcol = f"room_type_{room_type}"
    if rcol in row:
        row[rcol] = 1

    X = np.array([[row[name] for name in feature_names]], dtype=float)
    X_scaled = (X - train_mean) / train_std
    X_b = np.hstack([np.ones((1, 1)), X_scaled])
    log_price_pred = X_b @ weights
    price_pred = np.exp(log_price_pred[0]) - 1

    st.success(f"Estimated Price: ${price_pred:,.2f} / night")