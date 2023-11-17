import streamlit as st
import numpy as np
import joblib

# Load the trained pipeline
loaded_pipeline = joblib.load('./models/model/log_reg_pipeline.joblib')

st.title('Model Prediction')

# User input form
feature1 = st.number_input('Feature 1', step=any, format="%.2f")
feature2 = st.number_input('Feature 2', step=any, format="%.2f")
feature3 = st.number_input('Feature 3', step=any, format="%.2f")

# Predictions
if st.button('Predict'):
    # Create a NumPy array with the user inputs
    new_data = np.array([[feature1, feature2, feature3]])

    # Make a prediction using the loaded pipeline
    prediction = loaded_pipeline.predict(new_data)

    # Display the prediction
    st.success(f'Prediction: {prediction[0]}')
