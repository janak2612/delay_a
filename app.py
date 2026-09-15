import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input fields for features (based on X.columns)
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, max_value=200.0, value=20.0, step=0.1)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being high)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5, 5 being severe)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=30, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, max_value=20, value=2)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=15, value=3)
road_condition_score = st.slider('Road Condition Score (1-5, 5 being excellent)', 1, 5, 3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=200.0, value=12.0, step=0.1)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, max_value=30.0, value=12.0, step=0.1)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, max_value=120, value=60)


if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.error(f'Prediction: Delivery is likely to be DELAYED (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'Prediction: Delivery is likely to be ON TIME (Probability: {prediction_proba[0][0]:.2f})')

st.write("To run this app, save the code as `app.py` and execute `streamlit run app.py` in your terminal.")
