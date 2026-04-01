import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
# load file
 
model=joblib.load("model1.pkl")
st.title(" EV User Default Risk Prediction")

st.markdown("""
This app predicts the likelihood of a user defaulting on payments for EV charging based on user behavior, financial and vehicle details.
""")

# inputs
st.header("Enter EV User Details")

# Demographic
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3, 4, 5])

# Vehicle
ev_type = st.selectbox("EV Type", ["Type1", "Type2", "Type3"])
battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=10, max_value=200, value=50)

# Charging
charging_sessions = st.number_input("Charging Sessions Per Month", min_value=0, max_value=100, value=10)
avg_charge_cost = st.number_input("Average Charge Cost", min_value=0.0, value=5.0)
charging_location = st.selectbox("Charging Location Type", ["Public", "Private", "Highway"])
charger_status = st.selectbox("Charger Working Status", ["Working", "Not Working"])
charging_time = st.number_input("Charging Time (Minutes)", min_value=0, max_value=600, value=60)

# Behaviour
distance_travelled = st.number_input("Distance Travelled Per Month (km)", min_value=0, max_value=5000, value=500)
tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=240, value=12)
app_usage_score = st.slider("App Usage Score (0-100)", min_value=0, max_value=100, value=50)

# Financial
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
loan_taken = st.selectbox("Loan Taken", ["No", "Yes"])
missed_payments = st.number_input("Missed Payments Last 6 Months", min_value=0, max_value=10, value=0)

# Derived (optional, can be predicted inside pipeline if you prefer)
charging_efficiency_index = st.slider("Charging Efficiency Index (0-100)", min_value=0, max_value=100, value=70)

# prediction
if st.button("Predict Default Risk"):
    # Prepare input as dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "City_Tier": [city_tier],
        "EV_Type": [ev_type],
        "Battery_Capacity_kWh": [battery_capacity],
        "Charging_Sessions_Per_Month": [charging_sessions],
        "Avg_Charge_Cost": [avg_charge_cost],
        "Distance_Travelled_Per_Month": [distance_travelled],
        "Income_Level": [income_level],
        "Loan_Taken": [1 if loan_taken=="Yes" else 0],
        "Missed_Payments_Last_6M": [missed_payments],
        "Tenure_Months": [tenure_months],
        "Charging_Location_Type": [charging_location],
        "App_Usage_Score": [app_usage_score],
        "Charger_Working_Status": [charger_status],
        "Charging_Time_Minutes": [charging_time],
        "Charging_Efficiency_Index": [charging_efficiency_index]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # probability of default

    # Display result
    if prediction == 1:
        st.error(f" High Default Risk! Probability: {prediction_proba:.2f}")
    else:
        st.success(f" Low Default Risk. Probability: {prediction_proba:.2f}")

        