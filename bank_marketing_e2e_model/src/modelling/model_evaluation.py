import os
import json
from numpy import ndarray
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


class EvaluationPipeline:
    def __init__(self, model_pipeline: Pipeline, test_data_path: str, model_metrics_path: str):
        self.model_pipeline = model_pipeline
        self.test_data_path = test_data_path
        self.model_metrics_path = model_metrics_path

    '''
    A class to evaluate a logistic regression model based on a given metric
    and save model characteristics in a given directory.

    Methods:
        make_predictions: Uses an sklearn pipeline and test features to make predictions 
        evaluate_predictions: Evaluates pipeline predictions using the roc_auc metric
        save_model_performance: Saves the model evaluation score in a given directory
    '''

    def make_predictions(self) -> ndarray:
        # Read test data
        test_data = pd.read_csv(self.test_data_path)
        # Get x test to make predictions on
        x_test = test_data.drop(columns=['target'])
        # Load pipeline
        loaded_model_pipe = joblib.load(self.model_pipeline)
        # Use pipeline to make predictions
        return loaded_model_pipe.predict(x_test)

    def evaluate_predictions(self) -> float:
        # Read test data
        test_data = pd.read_csv(self.test_data_path)
        # Get y test to evaluate predictions
        y_test = test_data['target']
        # Use previous method to get predictions
        preds = self.make_predictions()
        # Use predictions to evaluate model
        return roc_auc_score(y_test, preds)

    def save_model_performance(self):
        # Create evaluation dict
        eval_dict = {
            'roc_auc': self.evaluate_predictions()
        }
        # Save in JSON format
        with open(self.model_metrics_path, 'w') as metrics_file:
            json.dump(eval_dict, metrics_file)
