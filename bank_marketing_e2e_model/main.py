"""
This module runs the app.

It loads a local web page using streamlit that you can use
to import data, train a model, evaluate it, and use it
to make inferences on inputted data.
"""


import json
import yaml
import joblib
import pandas as pd

import streamlit as st

from src.data.raw_data_gen import FetchUCIData
from src.modelling.model_training import TrainingPipeline
from src.modelling.model_evaluation import EvaluationPipeline


# Load config
with open('config.yaml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)  # Load config file

# Page Title
st.title('Model Prediction')

# Import data button and description
st.subheader('Raw Data Import')
# Add description
st.write('Import raw data from UCI, format the data, and save.')
# Read data from UCI
if st.button('Import'):
    # Fetch data from uci and format
    try:
        uci_data = FetchUCIData(uci_repo=config['uci_repo']['uci_repo_code']).format_data()
        # Display the raw data
        st.success('Raw data successfully imported!')
        st.write(uci_data)
        # Save formatted data
        uci_data.to_csv(config['data']['raw_data_path'])
    except Exception as e:
        st.error(f"An error occurred in saving UCI data: {e}")

# Import data button and description
st.subheader('Train ML Model')
# Add description
st.write('Use formatted data to train an ML model and save.')
# Train a model
if st.button('Train'):
    try:
        # Train a model and save
        TrainingPipeline(
            hyper_params=config['model']['hyperparameters'],
            raw_data_path=config['data']['raw_data_path'],
            train_data_path=config['data']['train_data_path'],
            test_data_path=config['data']['test_data_path'],
            model_path=config['output']['model_path']
        ).save_model()
        # Success message
        st.success('New model trained!')
    except Exception as e:
        st.error(f"An error occurred in training the model: {e}")

# Import data button and description
st.subheader('Evaluate Trained Model')
# Add description
st.write('Import trained model, evaluate given metric, and save performance.')
# Evaluate trained model
if st.button('Evaluate Current Model'):
    try:
        # Evaluate current model and save evaluation metric
        EvaluationPipeline(
            model_pipeline=config['output']['model_path'],
            test_data_path=config['data']['test_data_path'],
            model_metrics_path=config['output']['model_performance_path']
        ).save_model_performance()
        # Load metrics JSON as dict
        with open(
                config['output']['model_performance_path'],
                'r',
                encoding='utf-8'
        ) as metrics_json:
            metrics_dict = json.load(metrics_json)
        # Return metrics
        st.success(f'Current model has an roc_auc score of {metrics_dict["roc_auc"]}')
    except Exception as e:
        st.error(f"An error occurred in evaluating the model: {e}")

# Import data button and description
st.subheader('Make Predictions')
# Add description
st.write('Use saved model to make predictions on new data inputs.')

# Load input schema to make a prediction
with open(
        config['app']['app_input_schema_path'],
        'r',
        encoding='utf-8'
) as input_schema_json:
    input_schema = json.load(input_schema_json)
# Add feature input buttons for numerical variables
new_data = {}  # Create empty dict to add features to
for feature in input_schema.keys():
    # Numerical features
    if input_schema[feature]['type'] == 'int64':
        feature_value = st.number_input(label=feature,
                                        step=1,
                                        format='%i')
    elif input_schema[feature]['type'] == 'object':
        feature_value = st.selectbox(
            label=feature,
            options=input_schema[feature]['properties']['categorical_column']['enum']
        )
    else:
        feature_value = pd.NA
    new_data[feature] = feature_value

# Make a prediction using these inputs
if st.button('Predict'):
    try:
        # Load pipeline
        loaded_model_pipe = joblib.load(config['output']['model_path'])
        # Create a pandas df with the user inputs
        inference_df = pd.DataFrame([new_data])
        # Make a prediction using the loaded pipeline
        prediction = loaded_model_pipe.predict(inference_df)
        # Format prediction
        if prediction[0] == 0:
            PRED_DISPLAY = 'This customer will subscribe.'
        elif prediction[0] == 1:
            PRED_DISPLAY = 'This customer will not subscribe.'
        else:
            PRED_DISPLAY = 'No prediction.'
        # Display the prediction
        st.success(PRED_DISPLAY)
    except Exception as e:
        st.error(f"An error occurred in making a prediction: {e}")
