import streamlit as st
import joblib
import yaml
import os
import json
import pandas as pd

from src.data.raw_data_gen import FetchUCIData, SaveRawTrainingData
from src.modelling.model_training import TrainingPipeline
from src.modelling.model_evaluation import EvaluationPipeline


# Load config
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Get parent dir
with open(os.path.join(parent_dir, 'config.yaml'), 'r') as config_file:
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
        st.success(f'Raw data successfully imported!')
        st.write(uci_data)
        # Save formatted data
        SaveRawTrainingData(raw_training_df=uci_data,
                            raw_data_store_path=config['data']['raw_data_path']).save_raw_training_data()
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
            model_pipeline=config['data']['model_path'],
            test_data_path=config['data']['test_data_path'],
            model_metrics_path=config['output']['model_metrics_path']
        ).save_model_performance()
        # Load metrics JSON as dict
        with open(config['output']['model_performance_path'], 'r') as metrics_json:
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
with open(config['app']['app_input_schema_path'], 'r') as input_schema_json:
    input_schema = json.load(input_schema_json)
# Add feature input buttons for numerical variables
new_data = {}  # Create empty dict to add features to
for feature in input_schema.keys():
    # Numerical features
    if input_schema[feature]['type'] == 'int64':
        feature_value = st.number_input(label=feature,
                                        step=1,
                                        format='%i')
    if input_schema[feature]['type'] == 'object':
        feature_value = st.selectbox(label=feature,
                                     options=input_schema[feature]['properties']['categorical_column']['enum'])
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
            pred_display = 'This customer will subscribe.'
        if prediction[0] == 1:
            pred_display = 'This customer will not subscribe.'
        # Display the prediction
        st.success(pred_display)
    except Exception as e:
        st.error(f"An error occurred in making a prediction: {e}")
