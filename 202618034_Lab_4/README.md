# Airbnb NYC Price Prediction

DS605 Lab 4 — End-to-End ML Project. Predicts Airbnb nightly price in NYC using a
from-scratch linear regression model (no ML libraries — NumPy/pandas only).

## Project Structure

- `Lab_04_cleaning.ipynb` — data cleaning
- `Lab_04_analysis.ipynb` — EDA, bivariate analysis, NLP on listing names
- `Lab_04_model_training_testing.ipynb` — feature engineering
- `Lab_04_model_scratch.ipynb` — model training (from scratch)
- `app.py` — Streamlit app for live predictions
- `airbnb_price_model_scratch.pkl` — saved trained model

## Results

- Test R²: 0.532
- Test RMSE ($): $60.70
- Test MAE ($): $236.93

## Key Findings

- Room type and location (borough/neighbourhood) are the strongest price predictors.
- Certain branded properties in listing names (e.g. specific buildings) show a
  consistent price premium.

## How to Run

pip install -r requirements.txt
streamlit run app.py
